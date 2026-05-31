"""Tests for /api/review endpoints (HITL chunk-review proxy in front of n8n).

The n8n integration is stubbed via :mod:`httpx.AsyncClient` patching so the
tests stay hermetic. The shapes asserted here mirror the contract documented
in ``Chunk Review API.md`` -- if n8n's response shape changes, this file is
the canary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import EventBus
from voice_agent.config import ReviewConfig


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


@dataclass
class _FakeResponse:
    status_code: int
    body: Any
    text_body: str = ""

    def json(self) -> Any:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body

    @property
    def text(self) -> str:
        return self.text_body


class _StubClient:
    """Captures every httpx call; pops scripted responses in FIFO order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list[_FakeResponse] = []

    def queue(self, *responses: _FakeResponse) -> None:
        self.script.extend(responses)

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": "POST", "url": url, "kwargs": kwargs})
        if not self.script:
            raise AssertionError(f"Unexpected POST to {url}")
        return self.script.pop(0)

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": "GET", "url": url, "kwargs": kwargs})
        if not self.script:
            raise AssertionError(f"Unexpected GET to {url}")
        return self.script.pop(0)

    async def aclose(self) -> None:
        return None


# ---------- ReviewConfig defaults -----------------------------------------


def test_review_config_defaults_disable_endpoints():
    cfg = ReviewConfig()
    assert cfg.n8n_base_url is None
    assert cfg.upload_path == "/webhook/review/upload"
    assert cfg.pending_path == "/webhook/review/pending"
    assert cfg.audit_log_path == "/webhook/audit-log"
    assert cfg.audit_event_path == "/webhook/audit-event"
    assert cfg.default_collection_name == "maritime_hybrid"
    assert cfg.default_chunking_strategy == "paragraph_aware"


def test_review_config_extra_forbid_rejects_typos():
    with pytest.raises(Exception):
        ReviewConfig(base_url="http://x")  # type: ignore[call-arg]


def test_review_config_rejects_unknown_chunking_strategy():
    with pytest.raises(Exception):
        ReviewConfig(default_chunking_strategy="random_split")  # type: ignore[arg-type]


# ---------- 503 when not configured ---------------------------------------


def test_upload_returns_503_when_n8n_unconfigured():
    app = create_app(event_bus=EventBus(), session=_session(), review=ReviewConfig())
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("hello.pdf", b"PDF", "application/pdf")},
            data={"Document_Type": "PDF", "Collection_Name": "x"},
        )
    assert res.status_code == 503
    assert "n8n_base_url" in res.json()["detail"]


def test_pending_returns_503_when_n8n_unconfigured():
    app = create_app(event_bus=EventBus(), session=_session(), review=ReviewConfig())
    with TestClient(app) as c:
        res = c.get("/api/review/pending")
    assert res.status_code == 503


def test_resume_returns_503_when_n8n_unconfigured():
    app = create_app(event_bus=EventBus(), session=_session(), review=ReviewConfig())
    with TestClient(app) as c:
        res = c.post("/api/review/batch_x/resume", json={"decisions": []})
    assert res.status_code == 503


# ---------- Upload ---------------------------------------------------------


def test_upload_forwards_all_five_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(202, {"status": "queued", "message": "PDF received."})
    )
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
            data={
                "Document_Type": "PDF",
                "Collection_Name": "maritime_hybrid",
                "Categories": "colregs, rules",
                "Chunking_Strategy": "fixed_size",
            },
        )
    assert res.status_code == 200
    assert res.json()["status"] == "queued"
    # The upload was forwarded to the right URL on n8n.
    assert stub.calls[0]["url"] == "http://n8n:5678/webhook/review/upload"
    # All four text fields are present in the forwarded multipart body.
    forwarded_data = stub.calls[0]["kwargs"]["data"]
    assert forwarded_data == {
        "Document_Type": "PDF",
        "Collection_Name": "maritime_hybrid",
        "Categories": "colregs, rules",
        "Chunking_Strategy": "fixed_size",
    }
    # The file part is keyed "pdf" -- name is arbitrary per the contract,
    # but a stable choice keeps the n8n execution log readable.
    assert "pdf" in stub.calls[0]["kwargs"]["files"]


def test_upload_includes_configured_llm_model_as_Model_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per REVIEW_API.md the ingestion webhook accepts an optional `Model`
    form field that drives the doc-summary call. We forward whatever the
    helmsman LLM is configured to use so the two stay in sync."""
    stub = _StubClient()
    stub.queue(_FakeResponse(202, {"status": "queued"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(
        event_bus=EventBus(),
        session=_session(),
        review=cfg,
        llm_model="nvidia/nemotron-3-nano-4b",
    )
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
        )
    assert res.status_code == 200
    assert stub.calls[0]["kwargs"]["data"]["Model"] == "nvidia/nemotron-3-nano-4b"


def test_upload_form_Model_field_overrides_llm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit Model in the form wins -- per-request overrides are how
    a power user or script swaps models without rewriting the config."""
    stub = _StubClient()
    stub.queue(_FakeResponse(202, {"status": "queued"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(
        event_bus=EventBus(),
        session=_session(),
        review=cfg,
        llm_model="nvidia/nemotron-3-nano-4b",
    )
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
            data={"Model": "unsloth/gemma-4-e4b-it"},
        )
    assert res.status_code == 200
    assert stub.calls[0]["kwargs"]["data"]["Model"] == "unsloth/gemma-4-e4b-it"


def test_upload_omits_Model_field_when_neither_configured_nor_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No llm_model + no form Model -> don't send the field at all; n8n
    falls back to its own default (`unsloth/gemma-4-e4b-it` per the
    contract). Sending an empty string would be wrong because n8n's
    "empty falls back to default" rule still allows the field to *exist*
    and confuse audit logging."""
    stub = _StubClient()
    stub.queue(_FakeResponse(202, {"status": "queued"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
        )
    assert res.status_code == 200
    assert "Model" not in stub.calls[0]["kwargs"]["data"]


def test_upload_falls_back_to_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing/blank form fields use the configured defaults, not empty strings."""
    stub = _StubClient()
    stub.queue(_FakeResponse(202, {"status": "queued"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(
        n8n_base_url="http://n8n:5678",
        default_document_type="PDF",
        default_collection_name="my_coll",
        default_categories="algemeen",
        default_chunking_strategy="fixed_size",
    )
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
            # Document_Type/Collection_Name intentionally omitted.
        )
    assert res.status_code == 200
    forwarded = stub.calls[0]["kwargs"]["data"]
    assert forwarded["Document_Type"] == "PDF"
    assert forwarded["Collection_Name"] == "my_coll"
    assert forwarded["Categories"] == "algemeen"
    assert forwarded["Chunking_Strategy"] == "fixed_size"


def test_upload_502s_when_n8n_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"error": "ingestion broken"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
        )
    assert res.status_code == 502


def test_upload_502s_when_n8n_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom(_StubClient):
        async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "voice_agent.api.review.httpx.AsyncClient", lambda **_: _Boom()
    )
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/upload",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
        )
    assert res.status_code == 502
    assert "n8n unreachable" in res.json()["detail"]


# ---------- Pending --------------------------------------------------------


_SAMPLE_PENDING_BODY: dict[str, Any] = {
    "total_pending_batches": 1,
    "batches": [
        {
            "batch_id": "batch_abc",
            "filename": "test.pdf",
            "collection_name": "maritime_hybrid",
            "resume_url": "http://n8n:5678/webhook-waiting/secret/review",
            "created_at": "2026-05-21T10:00:00Z",
            "pending_chunk_count": 2,
            "chunks": [
                {"chunk_id": "chunk_000", "text": "hello", "metadata": {}},
                {"chunk_id": "chunk_001", "text": "world", "metadata": {}},
            ],
        }
    ],
}


def test_pending_strips_resume_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser must never see the n8n resume URL."""
    stub = _StubClient()
    stub.queue(_FakeResponse(200, _SAMPLE_PENDING_BODY))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/pending")
    assert res.status_code == 200
    body = res.json()
    assert body["total_pending_batches"] == 1
    assert len(body["batches"]) == 1
    batch = body["batches"][0]
    assert "resume_url" not in batch
    # Other fields are passed through, including the chunk array.
    assert batch["batch_id"] == "batch_abc"
    assert batch["filename"] == "test.pdf"
    assert len(batch["chunks"]) == 2


def test_pending_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"total_pending_batches": 0, "batches": []}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/pending")
    assert res.status_code == 200
    assert res.json() == {"total_pending_batches": 0, "batches": []}


def test_pending_maps_n8n_500_empty_quirk_to_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """n8n's pending workflow returns 500 'No item to return' when empty.
    We translate this to an empty list rather than surfacing 502, since the
    contract specifies an empty list as the empty-queue shape.
    """
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"code": 0, "message": "No item to return was found"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/pending")
    assert res.status_code == 200
    assert res.json() == {"total_pending_batches": 0, "batches": []}


# ---------- Resume ---------------------------------------------------------


def test_resume_looks_up_url_and_forwards_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(200, _SAMPLE_PENDING_BODY),     # pending lookup
        _FakeResponse(200, {"approved": 2, "rejected": 0, "edited": 0}),  # resume
    )
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    decisions = {
        "batch_id": "batch_abc",
        "decisions": [
            {"chunk_id": "chunk_000", "action": "approve"},
            {"chunk_id": "chunk_001", "action": "reject", "reason": "fluff"},
        ],
    }
    with TestClient(app) as c:
        res = c.post("/api/review/batch_abc/resume", json=decisions)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["n8n"]["approved"] == 2
    # First call was the pending lookup, second was the POST to the resume URL.
    assert stub.calls[0]["method"] == "GET"
    assert stub.calls[1]["method"] == "POST"
    assert stub.calls[1]["url"] == "http://n8n:5678/webhook-waiting/secret/review"
    # The decisions body went through verbatim.
    assert stub.calls[1]["kwargs"]["json"] == decisions


def test_resume_returns_404_when_batch_not_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"total_pending_batches": 0, "batches": []}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/gone/resume", json={"batch_id": "gone", "decisions": []})
    assert res.status_code == 404
    # No POST to the resume URL was made (only the GET for pending).
    assert all(call["method"] == "GET" for call in stub.calls)


def test_resume_returns_404_when_url_already_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """n8n's wait URL is one-shot: second POST returns 404/410."""
    stub = _StubClient()
    stub.queue(
        _FakeResponse(200, _SAMPLE_PENDING_BODY),
        _FakeResponse(410, {"error": "wait URL expired"}),
    )
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/batch_abc/resume",
            json={"batch_id": "batch_abc", "decisions": []},
        )
    assert res.status_code == 404


# ---------- Audit log ------------------------------------------------------


_SAMPLE_AUDIT_BODY: dict[str, Any] = {
    "total_in_log": 142,
    "total_returned": 2,
    "applied_filters": {"limit": 50, "actie": "ingestie_hitl", "since": None},
    "entries": [
        {
            "id": 142,
            "createdAt": "2026-05-27T09:14:22.118Z",
            "document_naam": "Albertkanaal.pdf",
            "actie": "ingestie_hitl",
            "resultaat": "Succes — HITL batch batch_abc → approved=8 / edited=1 / rejected=1",
        },
        {
            "id": 141,
            "createdAt": "2026-05-27T09:11:05.002Z",
            "document_naam": "COLREGS.pdf",
            "actie": "ingestie_hitl",
            "resultaat": "Fout — alle chunks afgewezen door reviewer",
        },
    ],
}


def test_audit_log_returns_503_when_n8n_unconfigured():
    app = create_app(event_bus=EventBus(), session=_session(), review=ReviewConfig())
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log")
    assert res.status_code == 503
    assert "n8n_base_url" in res.json()["detail"]


def test_audit_log_passes_through_body(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, _SAMPLE_AUDIT_BODY))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log")
    assert res.status_code == 200
    body = res.json()
    assert body["total_in_log"] == 142
    assert len(body["entries"]) == 2
    assert body["entries"][0]["actie"] == "ingestie_hitl"
    # No query params forwarded when none provided.
    assert stub.calls[0]["url"] == "http://n8n:5678/webhook/audit-log"
    assert stub.calls[0]["kwargs"].get("params") is None


def test_audit_log_forwards_filter_params(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, _SAMPLE_AUDIT_BODY))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get(
            "/api/review/audit-log",
            params={
                "limit": 20,
                "actie": "ingestie_hitl",
                "since": "2026-05-20T00:00:00Z",
            },
        )
    assert res.status_code == 200
    params = stub.calls[0]["kwargs"]["params"]
    assert params == {
        "limit": "20",
        "actie": "ingestie_hitl",
        "since": "2026-05-20T00:00:00Z",
    }


def test_audit_log_rejects_out_of_range_limit():
    """Pydantic guards against bad limits before we ever call n8n."""
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log", params={"limit": 0})
    assert res.status_code == 422
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log", params={"limit": 999})
    assert res.status_code == 422


def test_audit_log_502s_when_n8n_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"error": "datatable read failed"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log")
    assert res.status_code == 502


def test_audit_log_502s_when_n8n_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom(_StubClient):
        async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "voice_agent.api.review.httpx.AsyncClient", lambda **_: _Boom()
    )
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.get("/api/review/audit-log")
    assert res.status_code == 502
    assert "n8n unreachable" in res.json()["detail"]


# ---------- Audit event (write) -------------------------------------------

_ART50_EVENT = {
    "document_naam": "transparantieverklaring_v1.0",
    "actie": "art50_acknowledged",
    "resultaat": "OK",
}


def test_audit_event_returns_503_when_n8n_unconfigured():
    app = create_app(event_bus=EventBus(), session=_session(), review=ReviewConfig())
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 503
    assert "n8n_base_url" in res.json()["detail"]


def test_audit_event_forwards_body_and_passes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "status": "logged",
                "id": 142,
                "createdAt": "2026-05-31T14:22:09.117Z",
                "document_naam": "transparantieverklaring_v1.0",
                "actie": "art50_acknowledged",
            },
        )
    )
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "logged"
    assert body["id"] == 142
    # Forwarded to the right webhook with the body verbatim.
    assert stub.calls[0]["url"] == "http://n8n:5678/webhook/audit-event"
    assert stub.calls[0]["kwargs"]["json"] == _ART50_EVENT


def test_audit_event_rejects_missing_fields():
    """All three fields are required -- FastAPI/pydantic 422s before n8n is hit."""
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/audit-event",
            json={"actie": "art50_acknowledged", "resultaat": "OK"},
        )
    assert res.status_code == 422


def test_audit_event_rejects_empty_string_fields():
    """Empty strings would make n8n 500; the proxy rejects them up front."""
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post(
            "/api/review/audit-event",
            json={"document_naam": "x", "actie": "", "resultaat": "OK"},
        )
    assert res.status_code == 422


def test_audit_event_502s_when_n8n_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"message": "missing actie"}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 502


def test_audit_event_502s_when_n8n_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom(_StubClient):
        async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "voice_agent.api.review.httpx.AsyncClient", lambda **_: _Boom()
    )
    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 502
    assert "n8n unreachable" in res.json()["detail"]


# ---------- n8n Header-Auth forwarding ------------------------------------


def test_n8n_auth_header_forwarded_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N8N_API_KEY", "n8n-secret")
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"status": "logged", "id": 1}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 200
    assert stub.calls[0]["kwargs"]["headers"] == {"X-N8N-API-KEY": "n8n-secret"}


def test_n8n_auth_header_omitted_when_key_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N8N_API_KEY", raising=False)
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"status": "logged", "id": 1}))
    monkeypatch.setattr("voice_agent.api.review.httpx.AsyncClient", lambda **_: stub)

    cfg = ReviewConfig(n8n_base_url="http://n8n:5678")
    app = create_app(event_bus=EventBus(), session=_session(), review=cfg)
    with TestClient(app) as c:
        res = c.post("/api/review/audit-event", json=_ART50_EVENT)
    assert res.status_code == 200
    # No header dict forwarded (resolved_n8n_headers() -> {} -> None).
    assert stub.calls[0]["kwargs"]["headers"] is None
