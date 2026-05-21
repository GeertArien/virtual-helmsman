"""Tests for /api/documents endpoints and the documents config.

The qdrant + n8n integrations are exercised at the boundary: we stub the
:class:`httpx.AsyncClient` on the router so the tests stay hermetic and don't
need a running qdrant or n8n instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.documents import _group_documents
from voice_agent.api.events import EventBus
from voice_agent.config import DocumentsConfig


# ---------- helpers --------------------------------------------------------


def _session() -> SessionInfo:
    return SessionInfo(
        session_id="test-session",
        started_at="2026-05-21T00:00:00+00:00",
        stt_backend="parakeet_onnx",
        tts_backend="kokoro",
        vad_backend="silero",
        turn_backend="smart_turn_v3",
        simulator_backend="mock",
        llm_model="test/model",
    )


# ---------- DocumentsConfig defaults --------------------------------------


def test_documents_config_defaults_disable_endpoints():
    cfg = DocumentsConfig()
    assert cfg.n8n_upload_webhook is None
    assert cfg.qdrant_url is None
    assert cfg.qdrant_collection is None
    assert cfg.document_id_field == "document_id"
    assert cfg.scroll_limit == 10000


def test_documents_config_extra_forbid_rejects_typos():
    with pytest.raises(Exception):
        DocumentsConfig(qdrant_uri="http://x")  # type: ignore[call-arg]


# ---------- 503 when not configured ---------------------------------------


def test_list_returns_503_when_qdrant_unconfigured():
    cfg = DocumentsConfig()
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.get("/api/documents")
    assert res.status_code == 503
    assert "qdrant_url" in res.json()["detail"]


def test_upload_returns_503_when_n8n_unconfigured():
    cfg = DocumentsConfig()
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/documents/upload",
            files={"file": ("hello.txt", b"hi", "text/plain")},
        )
    assert res.status_code == 503
    assert "n8n_upload_webhook" in res.json()["detail"]


def test_delete_returns_503_when_qdrant_unconfigured():
    cfg = DocumentsConfig()
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.delete("/api/documents/some-id")
    assert res.status_code == 503
    assert "qdrant_url" in res.json()["detail"]


# ---------- _group_documents ----------------------------------------------


def test_group_documents_rolls_up_chunks_by_document_id():
    cfg = DocumentsConfig()
    points = [
        {"payload": {"document_id": "a", "title": "Alpha", "source": "a.pdf",
                     "uploaded_at": "2026-05-01"}},
        {"payload": {"document_id": "a", "title": "Alpha (later)", "source": "a.pdf"}},
        {"payload": {"document_id": "b", "title": "Bravo"}},
        # no document_id -> skipped
        {"payload": {"title": "stray"}},
    ]
    rows = _group_documents(points, cfg)
    assert [r["document_id"] for r in rows] == ["a", "b"]
    a = next(r for r in rows if r["document_id"] == "a")
    assert a["chunk_count"] == 2
    # First-seen wins for title.
    assert a["title"] == "Alpha"
    assert a["uploaded_at"] == "2026-05-01"


def test_group_documents_handles_missing_payload_keys():
    cfg = DocumentsConfig()
    points = [
        {"payload": {"document_id": "x"}},  # title/source missing
        {"payload": None},                   # no payload at all
        {},                                  # no payload key
    ]
    rows = _group_documents(points, cfg)
    assert len(rows) == 1
    assert rows[0]["title"] is None
    assert rows[0]["source"] is None
    assert rows[0]["uploaded_at"] is None
    assert rows[0]["chunk_count"] == 1


def test_group_documents_uses_configured_field_names():
    cfg = DocumentsConfig(
        document_id_field="doc",
        title_field="name",
        source_field="origin",
        uploaded_at_field="created",
    )
    points = [{"payload": {"doc": "1", "name": "n", "origin": "o", "created": "2026"}}]
    rows = _group_documents(points, cfg)
    assert rows[0] == {
        "document_id": "1",
        "title": "n",
        "source": "o",
        "uploaded_at": "2026",
        "chunk_count": 1,
    }


# ---------- happy-path with stubbed httpx ---------------------------------


@dataclass
class _FakeResponse:
    status_code: int
    body: dict[str, Any]
    text_body: str = ""

    def json(self) -> dict[str, Any]:
        return self.body

    @property
    def text(self) -> str:
        return self.text_body


class _StubClient:
    """Minimal AsyncClient stand-in capturing calls and returning canned data."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list[_FakeResponse] = []

    def queue(self, *responses: _FakeResponse) -> None:
        self.script.extend(responses)

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:  # noqa: D401
        self.calls.append({"url": url, "kwargs": kwargs})
        if not self.script:
            raise AssertionError(f"Unexpected POST to {url}")
        return self.script.pop(0)

    async def aclose(self) -> None:
        return None


def test_list_documents_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: configured qdrant URL, stubbed httpx call returns one page."""
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "result": {
                    "points": [
                        {"payload": {"document_id": "a", "title": "Alpha"}},
                        {"payload": {"document_id": "a", "title": "Alpha"}},
                        {"payload": {"document_id": "b", "title": "Bravo"}},
                    ],
                    "next_page_offset": None,
                }
            },
        )
    )
    # Patch AsyncClient so the router picks up the stub on creation.
    monkeypatch.setattr("voice_agent.api.documents.httpx.AsyncClient", lambda **_: stub)

    cfg = DocumentsConfig(qdrant_url="http://qdrant:6333", qdrant_collection="docs")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()["documents"]
    assert {d["document_id"] for d in docs} == {"a", "b"}
    a = next(d for d in docs if d["document_id"] == "a")
    assert a["chunk_count"] == 2
    # The scroll request went to the right URL with payload + no vectors.
    assert stub.calls[0]["url"].endswith("/collections/docs/points/scroll")
    assert stub.calls[0]["kwargs"]["json"]["with_payload"] is True
    assert stub.calls[0]["kwargs"]["json"]["with_vector"] is False


def test_delete_document_counts_then_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(200, {"result": {"count": 7}}),
        _FakeResponse(200, {"result": {"operation_id": 1, "status": "completed"}}),
    )
    monkeypatch.setattr("voice_agent.api.documents.httpx.AsyncClient", lambda **_: stub)

    cfg = DocumentsConfig(qdrant_url="http://qdrant:6333", qdrant_collection="docs")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.delete("/api/documents/doc-42")
    assert res.status_code == 200
    body = res.json()
    assert body == {"status": "deleted", "document_id": "doc-42", "deleted_chunks": 7}
    assert stub.calls[0]["url"].endswith("/collections/docs/points/count")
    assert stub.calls[1]["url"].endswith("/collections/docs/points/delete")
    # The delete uses the same filter as the count.
    assert stub.calls[0]["kwargs"]["json"]["filter"] == stub.calls[1]["kwargs"]["json"]["filter"]


def test_delete_document_returns_404_when_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"result": {"count": 0}}))
    monkeypatch.setattr("voice_agent.api.documents.httpx.AsyncClient", lambda **_: stub)

    cfg = DocumentsConfig(qdrant_url="http://qdrant:6333", qdrant_collection="docs")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.delete("/api/documents/missing-id")
    assert res.status_code == 404
    assert len(stub.calls) == 1  # no delete call when count is 0


def test_upload_forwards_multipart_to_n8n(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {"status": "queued", "document_id": "doc-new", "message": "Awaiting review."},
        )
    )
    monkeypatch.setattr("voice_agent.api.documents.httpx.AsyncClient", lambda **_: stub)

    cfg = DocumentsConfig(n8n_upload_webhook="http://n8n/webhook/ingest")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/documents/upload",
            files={"file": ("notes.md", b"# hello", "text/markdown")},
            data={"title": "My notes"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    assert body["document_id"] == "doc-new"
    # Right URL, multipart files passed through, title forwarded as form data.
    assert stub.calls[0]["url"] == "http://n8n/webhook/ingest"
    assert "files" in stub.calls[0]["kwargs"]
    assert stub.calls[0]["kwargs"]["data"]["title"] == "My notes"


def test_upload_502s_when_n8n_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"error": "workflow exploded"}))
    monkeypatch.setattr("voice_agent.api.documents.httpx.AsyncClient", lambda **_: stub)

    cfg = DocumentsConfig(n8n_upload_webhook="http://n8n/webhook/ingest")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/documents/upload",
            files={"file": ("notes.md", b"# hello", "text/markdown")},
        )
    assert res.status_code == 502
    detail = res.json()["detail"]
    assert detail["upstream_status"] == 500


def test_qdrant_unreachable_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomClient(_StubClient):
        async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "voice_agent.api.documents.httpx.AsyncClient", lambda **_: _BoomClient()
    )
    cfg = DocumentsConfig(qdrant_url="http://qdrant:6333", qdrant_collection="docs")
    app = create_app(event_bus=EventBus(), session=_session(), documents=cfg)
    with TestClient(app) as c:
        res = c.get("/api/documents")
    assert res.status_code == 502
    assert "qdrant unreachable" in res.json()["detail"]
