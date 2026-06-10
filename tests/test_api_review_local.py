"""Tests for the local-backend /api/review endpoints + IngestionEngine.

The engine runs for real (real SQLite store, real chunking/metadata/decision
logic) with three seams stubbed so no test touches the network or needs the
optional ``langgraph`` extra:

* ``extract_fn``  -- replaces pypdf; returns scripted "PDF text".
* ``summarize_fn``-- replaces the LangChain doc-summary call.
* the httpx client -- a scripted stub asserting the Qdrant/embedding shapes.

Together with ``test_ingestion_pure.py`` / ``test_ingestion_store.py`` this
covers the full upload -> pending -> review -> upsert -> audit loop end to
end, hermetically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from voice_agent.api.review import create_review_router
from voice_agent.api.review_local import create_local_review_router
from voice_agent.config import ReviewConfig
from voice_agent.ingestion.engine import IngestionEngine

# Long enough to chunk into multiple pieces, repeated sentences keep the
# cleaning step from classifying anything as boilerplate.
_PDF_TEXT = "\n\n".join(
    " ".join(
        f"Rule {p}.{s} requires vessels to maintain a proper lookout at all times."
        for s in range(6)
    )
    for p in range(12)
)


@dataclass
class _FakeResponse:
    status_code: int = 200
    body: Any = None

    def json(self) -> Any:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


@dataclass
class _StubClient:
    script: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def queue(self, *responses: Any) -> None:
        self.script.extend(responses)

    async def _record(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        if not self.script:
            raise AssertionError(f"Unexpected {method} to {url}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await self._record("GET", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> Any:
        return await self._record("PUT", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await self._record("POST", url, **kwargs)

    async def aclose(self) -> None:
        return None


def _cfg(tmp_path: Path) -> ReviewConfig:
    return ReviewConfig(
        backend="local",
        db_path=str(tmp_path / "ingestion.db"),
        llm_base_url="http://lm:1234/v1",
        qdrant_url="http://qd:6333",
    )


def _engine(tmp_path: Path, stub: _StubClient) -> IngestionEngine:
    async def summarize(text: str, model: str) -> str:
        return "A two-sentence summary of the document."

    return IngestionEngine(
        _cfg(tmp_path),
        default_model="test/model",
        client=stub,  # type: ignore[arg-type]
        summarize_fn=summarize,
        extract_fn=lambda content: _PDF_TEXT,
    )


def _client(tmp_path: Path, stub: _StubClient) -> tuple[TestClient, IngestionEngine]:
    engine = _engine(tmp_path, stub)
    app = FastAPI()
    app.include_router(
        create_local_review_router(_cfg(tmp_path), engine=engine)
    )
    return TestClient(app), engine


def _upload(client: TestClient, **form: str) -> Any:
    return client.post(
        "/api/review/upload",
        files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
        data=form or {"Collection_Name": "maritime_hybrid"},
    )


def _wait_for_ingest(engine: IngestionEngine, timeout: float = 5.0) -> None:
    """Block until the background ingest task(s) have finished."""
    deadline = time.monotonic() + timeout
    while engine._tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not engine._tasks, "background ingest did not finish in time"


# ---------- the full HITL loop ------------------------------------------------


def test_upload_pending_resume_audit_loop(tmp_path: Path) -> None:
    stub = _StubClient()
    client, engine = _client(tmp_path, stub)
    with client:
        # 1. Upload -> immediate 202, contract message.
        res = _upload(client)
        assert res.status_code == 202
        assert res.json()["status"] == "queued"

        _wait_for_ingest(engine)

        # 2. Pending shows one batch with reviewable chunks, no resume_url.
        res = client.get("/api/review/pending")
        body = res.json()
        assert body["total_pending_batches"] == 1
        batch = body["batches"][0]
        assert batch["filename"] == "test.pdf"
        assert batch["pending_chunk_count"] == len(batch["chunks"]) > 1
        assert "resume_url" not in batch
        chunk = batch["chunks"][0]
        assert chunk["chunk_id"] == "chunk_000"
        assert chunk["metadata"]["document_summary"].startswith("A two-sentence")
        batch_id = batch["batch_id"]
        n_chunks = len(batch["chunks"])

        # 3. Resume: reject one chunk, approve the rest (by omission).
        embeddings = [
            {"index": i, "embedding": [0.1] * 4} for i in range(n_chunks - 1)
        ]
        stub.queue(
            _FakeResponse(200, {"result": {"exists": True}}),  # collection check
            _FakeResponse(200, {"data": embeddings}),  # embeddings
            _FakeResponse(200, {}),  # upsert
        )
        res = client.post(
            f"/api/review/{batch_id}/resume",
            json={
                "batch_id": batch_id,
                "decisions": [{"chunk_id": "chunk_000", "action": "reject"}],
            },
        )
        assert res.status_code == 200
        out = res.json()
        assert out["status"] == "ingested"
        assert out["rejected"] == 1
        assert out["indexed"] == n_chunks - 1

        # The upsert carried the right point shapes.
        upsert = stub.calls[-1]
        assert upsert["url"].endswith("/collections/maritime_hybrid/points?wait=true")
        points = upsert["kwargs"]["json"]["points"]
        assert len(points) == n_chunks - 1
        assert points[0]["payload"]["hitl_batch_id"] == batch_id
        assert points[0]["vector"]["bm25"]["model"] == "qdrant/bm25"

        # 4. Batch is gone; resume is one-shot (404 on second submit).
        assert client.get("/api/review/pending").json()["total_pending_batches"] == 0
        res = client.post(
            f"/api/review/{batch_id}/resume",
            json={"batch_id": batch_id, "decisions": []},
        )
        assert res.status_code == 404

        # 5. Success audit row landed with the n8n text pattern.
        res = client.get("/api/review/audit-log")
        entries = res.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["actie"] == "ingestie_hitl"
        assert entries[0]["resultaat"].startswith("Succes — HITL batch")
        assert entries[0]["document_naam"] == "test.pdf"


def test_all_rejected_short_circuits_to_audit(tmp_path: Path) -> None:
    stub = _StubClient()  # no responses queued: qdrant must NOT be called
    client, engine = _client(tmp_path, stub)
    with client:
        _upload(client)
        _wait_for_ingest(engine)
        batch = client.get("/api/review/pending").json()["batches"][0]
        decisions = [
            {"chunk_id": c["chunk_id"], "action": "reject"} for c in batch["chunks"]
        ]
        res = client.post(
            f"/api/review/{batch['batch_id']}/resume",
            json={"batch_id": batch["batch_id"], "decisions": decisions},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "rejected"
        assert res.json()["indexed"] == 0
        assert stub.calls == []  # no embeddings, no upsert

        entries = client.get("/api/review/audit-log").json()["entries"]
        assert "All rejected" in entries[0]["resultaat"]
        # Pending row cleaned up despite the rejection.
        assert client.get("/api/review/pending").json()["total_pending_batches"] == 0


# ---------- ingest-phase failure paths --------------------------------------


def test_pdf_extraction_failure_writes_audit_row(tmp_path: Path) -> None:
    from voice_agent.ingestion.pdf import PdfExtractionError

    stub = _StubClient()
    engine = _engine(tmp_path, stub)

    def bad_extract(content: bytes) -> str:
        raise PdfExtractionError("image-only PDF")

    engine._extract_fn = bad_extract
    app = FastAPI()
    app.include_router(create_local_review_router(_cfg(tmp_path), engine=engine))
    with TestClient(app) as client:
        res = _upload(client)
        assert res.status_code == 202  # still queued; failure is async
        _wait_for_ingest(engine)
        assert client.get("/api/review/pending").json()["total_pending_batches"] == 0
        entries = client.get("/api/review/audit-log").json()["entries"]
        assert entries[0]["resultaat"] == "Fout — PDF extractie mislukt"


def test_summary_llm_failure_writes_llm_error_row(tmp_path: Path) -> None:
    stub = _StubClient()
    engine = _engine(tmp_path, stub)

    async def bad_summarize(text: str, model: str) -> str:
        exc = RuntimeError("model not loaded")
        exc.status_code = 400  # type: ignore[attr-defined]
        raise exc

    engine._summarize_fn = bad_summarize
    app = FastAPI()
    app.include_router(create_local_review_router(_cfg(tmp_path), engine=engine))
    with TestClient(app) as client:
        _upload(client)
        _wait_for_ingest(engine)
        entries = client.get("/api/review/audit-log").json()["entries"]
        assert entries[0]["actie"] == "llm_error_ingestion"
        assert "error=model not loaded" in entries[0]["resultaat"]
        assert "http=400" in entries[0]["resultaat"]


def test_finalize_qdrant_failure_returns_502(tmp_path: Path) -> None:
    stub = _StubClient()
    client, engine = _client(tmp_path, stub)
    with client:
        _upload(client)
        _wait_for_ingest(engine)
        batch_id = client.get("/api/review/pending").json()["batches"][0]["batch_id"]
        stub.queue(_FakeResponse(500, {}))  # collection check blows up
        res = client.post(
            f"/api/review/{batch_id}/resume",
            json={"batch_id": batch_id, "decisions": []},
        )
        assert res.status_code == 502


# ---------- endpoint edges ----------------------------------------------------


def test_upload_503_until_configured(tmp_path: Path) -> None:
    cfg = ReviewConfig(backend="local", db_path=str(tmp_path / "i.db"))
    app = FastAPI()
    app.include_router(create_local_review_router(cfg))
    with TestClient(app) as client:
        res = _upload(client)
        assert res.status_code == 503
        assert "review.qdrant_url" in res.json()["detail"]
        # Read-only endpoints still work unconfigured.
        assert client.get("/api/review/pending").status_code == 200
        assert client.get("/api/review/audit-log").status_code == 200


def test_resume_requires_decisions_array(tmp_path: Path) -> None:
    stub = _StubClient()
    client, _ = _client(tmp_path, stub)
    with client:
        res = client.post("/api/review/b1/resume", json={"batch_id": "b1"})
        assert res.status_code == 400


def test_audit_event_roundtrip(tmp_path: Path) -> None:
    stub = _StubClient()
    client, _ = _client(tmp_path, stub)
    with client:
        res = client.post(
            "/api/review/audit-event",
            json={
                "document_naam": "transparantieverklaring_v1.0",
                "actie": "art50_acknowledged",
                "resultaat": "OK",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "logged"
        assert body["actie"] == "art50_acknowledged"
        assert isinstance(body["id"], int)

        entries = client.get(
            "/api/review/audit-log", params={"actie": "art50_acknowledged"}
        ).json()["entries"]
        assert entries[0]["document_naam"] == "transparantieverklaring_v1.0"


def test_audit_event_validates_fields(tmp_path: Path) -> None:
    stub = _StubClient()
    client, _ = _client(tmp_path, stub)
    with client:
        res = client.post(
            "/api/review/audit-event",
            json={"document_naam": "", "actie": "x", "resultaat": "y"},
        )
        assert res.status_code == 422


def test_upload_model_override_reaches_summarizer(tmp_path: Path) -> None:
    stub = _StubClient()
    seen: list[str] = []

    async def summarize(text: str, model: str) -> str:
        seen.append(model)
        return "Summary."

    engine = IngestionEngine(
        _cfg(tmp_path),
        default_model="config/default-model",
        client=stub,  # type: ignore[arg-type]
        summarize_fn=summarize,
        extract_fn=lambda content: _PDF_TEXT,
    )
    app = FastAPI()
    app.include_router(create_local_review_router(_cfg(tmp_path), engine=engine))
    with TestClient(app) as client:
        _upload(client, Model="per-request/model")
        _wait_for_ingest(engine)
        _upload(client)
        _wait_for_ingest(engine)
    assert seen == ["per-request/model", "config/default-model"]


# ---------- factory dispatch ----------------------------------------------------


def test_create_review_router_dispatches_local(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    router = create_review_router(cfg, llm_model="m")
    # The local router exposes the engine for app-shutdown cleanup.
    assert isinstance(router._http_client, IngestionEngine)  # type: ignore[attr-defined]


def test_review_config_defaults() -> None:
    cfg = ReviewConfig()
    assert cfg.backend == "n8n"
    assert cfg.db_path == "./data/ingestion.db"
    assert cfg.embedding_model == "text-embedding-bge-m3"
    assert cfg.langfuse_enabled is False


def test_review_config_qdrant_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ReviewConfig(backend="local")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    assert cfg.resolved_qdrant_headers() == {}
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    assert cfg.resolved_qdrant_headers() == {"api-key": "k"}
