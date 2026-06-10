"""Tests for the SQLite ingestion store and the Qdrant HTTP functions.

The store is exercised against a tmp-path database (real SQLite, no mocking);
the Qdrant/embedding calls run through a scripted stub client that asserts
the REST shapes match what the n8n nodes sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from voice_agent.ingestion import qdrant
from voice_agent.ingestion.store import IngestionStore


@pytest.fixture
def store(tmp_path: Path) -> IngestionStore:
    return IngestionStore(tmp_path / "ingestion.db")


def _chunk(i: int, batch: str = "b1") -> dict[str, Any]:
    return {
        "chunk_id": f"chunk_{i:03d}",
        "text": f"text {i}",
        "filename": "doc.pdf",
        "Collection_Name": "maritime_hybrid",
        "idx": i,
        "page": i // 2 + 1,
    }


# ---------- pending batches ---------------------------------------------------


def test_pending_roundtrip_groups_and_sorts(store: IngestionStore) -> None:
    store.write_pending_batch("b1", [_chunk(1), _chunk(0)])
    batches = store.list_pending_batches()
    assert len(batches) == 1
    b = batches[0]
    assert b["batch_id"] == "b1"
    assert b["filename"] == "doc.pdf"
    assert b["collection_name"] == "maritime_hybrid"
    assert b["pending_chunk_count"] == 2
    # Chunks sorted by chunk_id regardless of insert order.
    assert [c["chunk_id"] for c in b["chunks"]] == ["chunk_000", "chunk_001"]
    # Full metadata dict round-trips through the JSON column.
    assert b["chunks"][0]["metadata"]["idx"] == 0
    # No resume_url anywhere -- local mode resumes by batch_id.
    assert "resume_url" not in b


def test_pending_empty_state(store: IngestionStore) -> None:
    assert store.list_pending_batches() == []
    assert store.get_batch_chunks("nope") == []


def test_get_batch_chunks_ordered_and_one_shot(store: IngestionStore) -> None:
    store.write_pending_batch("b1", [_chunk(i) for i in range(3)])
    chunks = store.get_batch_chunks("b1")
    assert [c["chunk_id"] for c in chunks] == ["chunk_000", "chunk_001", "chunk_002"]
    store.delete_batch("b1")
    assert store.get_batch_chunks("b1") == []
    assert store.list_pending_batches() == []


def test_delete_batch_leaves_other_batches(store: IngestionStore) -> None:
    store.write_pending_batch("b1", [_chunk(0)])
    store.write_pending_batch("b2", [_chunk(0)])
    store.delete_batch("b1")
    assert [b["batch_id"] for b in store.list_pending_batches()] == ["b2"]


# ---------- audit log -----------------------------------------------------------


def test_audit_insert_and_query_shape(store: IngestionStore) -> None:
    row = store.insert_audit("doc.pdf", "ingestie_hitl", "Succes — ...")
    assert row["id"] == 1
    assert row["createdAt"].endswith("Z")

    out = store.query_audit()
    assert out["total_in_log"] == 1
    assert out["total_returned"] == 1
    assert out["applied_filters"] == {"limit": 50, "actie": None, "since": None}
    assert out["entries"][0]["document_naam"] == "doc.pdf"


def test_audit_filters_actie_since_limit(store: IngestionStore) -> None:
    store.insert_audit("a.pdf", "ingestie_hitl", "r1")
    store.insert_audit("b.pdf", "llm_error_ingestion", "r2")
    store.insert_audit("c.pdf", "ingestie_hitl", "r3")

    by_actie = store.query_audit(actie="ingestie_hitl")
    assert by_actie["total_returned"] == 2
    assert {e["document_naam"] for e in by_actie["entries"]} == {"a.pdf", "c.pdf"}

    limited = store.query_audit(limit=1)
    assert limited["total_returned"] == 1
    # Newest-first: the last insert wins the single slot.
    assert limited["entries"][0]["document_naam"] == "c.pdf"
    assert limited["total_in_log"] == 3

    # since in the future filters everything; invalid since is ignored.
    none = store.query_audit(since="2999-01-01T00:00:00Z")
    assert none["total_returned"] == 0
    all_rows = store.query_audit(since="not-a-date")
    assert all_rows["total_returned"] == 3


def test_audit_limit_clamped_to_500(store: IngestionStore) -> None:
    out = store.query_audit(limit=9999)
    assert out["applied_filters"]["limit"] == 500
    out = store.query_audit(limit=-5)
    assert out["applied_filters"]["limit"] == 50


# ---------- qdrant HTTP functions -------------------------------------------------


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


async def test_ensure_collection_creates_with_hybrid_schema() -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(200, {"result": {"exists": False}}),
        _FakeResponse(200, {}),  # create collection
        _FakeResponse(200, {}),  # index document_type
        _FakeResponse(200, {}),  # index categories
    )
    created = await qdrant.ensure_collection(
        stub, qdrant_url="http://qd:6333", collection="maritime_hybrid"
    )
    assert created is True
    create_body = stub.calls[1]["kwargs"]["json"]
    assert create_body["vectors"]["text-embedding-bge-m3"] == {
        "size": 1024,
        "distance": "Cosine",
    }
    assert create_body["sparse_vectors"]["bm25"] == {"modifier": "idf"}
    index_fields = [c["kwargs"]["json"]["field_name"] for c in stub.calls[2:]]
    assert index_fields == ["document_type", "categories"]


async def test_ensure_collection_noop_when_exists() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"result": {"exists": True}}))
    created = await qdrant.ensure_collection(
        stub, qdrant_url="http://qd:6333", collection="maritime_hybrid"
    )
    assert created is False
    assert len(stub.calls) == 1


async def test_embed_texts_batches_and_orders_by_index() -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "data": [
                    {"index": 1, "embedding": [0.2]},
                    {"index": 0, "embedding": [0.1]},
                ]
            },
        )
    )
    vecs = await qdrant.embed_texts(
        stub,
        base_url="http://lm:1234/v1",
        model="text-embedding-bge-m3",
        texts=["a", "b"],
    )
    assert vecs == [[0.1], [0.2]]
    assert stub.calls[0]["kwargs"]["json"] == {
        "input": ["a", "b"],
        "model": "text-embedding-bge-m3",
    }


def test_build_points_payload_and_vectors() -> None:
    chunk = {
        "point_id": 1750000000000000,
        "text": "rule text",
        "filename": "COLREGS.pdf",
        "page": 14,
        "chunk_id": "chunk_026",
        "total_chunks": 28,
        "start_char": 0,
        "end_char": 9,
        "chunk_length": 9,
        "words_in_text": 2,
        "document_summary": "Summary.",
        "section_title": "",
        "document_type": "PDF",
        "upload_timestamp": "2026-06-10T00:00:00Z",
        "categories": ["colregs"],
        "chunking_strategy": "paragraph_aware_sentence_boundary",
        "chunk_overlap": 75,
    }
    [point] = qdrant.build_points([chunk], [[0.1, 0.2]], 123.4, "batch_x")
    assert point["id"] == 1750000000000000
    assert point["payload"]["hitl_reviewed"] is True
    assert point["payload"]["hitl_batch_id"] == "batch_x"
    assert point["payload"]["chunk_id"] == "chunk_026"
    assert point["vector"]["text-embedding-bge-m3"] == [0.1, 0.2]
    assert point["vector"]["bm25"] == {
        "text": "rule text",
        "model": "qdrant/bm25",
        "options": {"avg_len": 123.4},
    }


def test_build_points_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        qdrant.build_points([{"point_id": 1}], [], 1.0, "b")


async def test_upsert_points_puts_with_wait() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {}))
    await qdrant.upsert_points(
        stub,
        qdrant_url="http://qd:6333",
        collection="maritime_hybrid",
        points=[{"id": 1}],
    )
    assert stub.calls[0]["url"] == (
        "http://qd:6333/collections/maritime_hybrid/points?wait=true"
    )
    assert stub.calls[0]["kwargs"]["json"] == {"points": [{"id": 1}]}
