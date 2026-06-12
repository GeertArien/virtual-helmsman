"""Unit tests for the shared Qdrant REST helpers (voice_agent/qdrant.py)."""

from __future__ import annotations

import pytest

from voice_agent import qdrant


def test_collection_url_strips_trailing_slash() -> None:
    assert (
        qdrant.collection_url("http://qd:6333/", "docs")
        == "http://qd:6333/collections/docs"
    )
    assert (
        qdrant.collection_url("http://qd:6333", "docs")
        == "http://qd:6333/collections/docs"
    )


@pytest.mark.parametrize(
    "op,expected",
    [
        ("scroll", "http://qd:6333/collections/docs/points/scroll"),
        ("query", "http://qd:6333/collections/docs/points/query"),
        ("", "http://qd:6333/collections/docs/points"),
    ],
)
def test_points_url(op: str, expected: str) -> None:
    assert qdrant.points_url("http://qd:6333", "docs", op) == expected


def test_api_key_headers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_QDRANT_KEY", "secret")
    assert qdrant.api_key_headers("MY_QDRANT_KEY") == {"api-key": "secret"}


def test_api_key_headers_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_QDRANT_KEY", raising=False)
    assert qdrant.api_key_headers("MY_QDRANT_KEY") == {}


def test_api_key_headers_blank_is_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    # A set-but-empty env var must not send an empty api-key header.
    monkeypatch.setenv("MY_QDRANT_KEY", "")
    assert qdrant.api_key_headers("MY_QDRANT_KEY") == {}


def test_vector_schema_constants() -> None:
    # Pinned to the ingested collection's named vectors; a change here is a
    # collection-format change and must be deliberate.
    assert qdrant.DENSE_VECTOR_NAME == "text-embedding-bge-m3"
    assert qdrant.DENSE_VECTOR_SIZE == 1024
    assert qdrant.SPARSE_VECTOR_NAME == "bm25"
    assert qdrant.BM25_MODEL == "qdrant/bm25"
