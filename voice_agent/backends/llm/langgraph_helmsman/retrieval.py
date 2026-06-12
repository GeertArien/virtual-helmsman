"""Qdrant hybrid retrieval + query embedding for the LangGraph backend.

These are the two network side-effects of the RAG branch, ported from the
n8n workflow's HTTP/Qdrant nodes. They use ``httpx`` (already a project
dependency) against the same REST shapes the workflow used:

* the LM Studio ``/v1/embeddings`` endpoint for the ``bge-m3`` query vector
  ("Embed Query"), and
* Qdrant's ``/points/query`` hybrid endpoint with an RRF fusion over a dense
  (``text-embedding-bge-m3``) and a sparse (``qdrant/bm25``) prefetch
  ("Query Points"), plus ``/points/scroll`` for adjacent-chunk expansion
  ("Fetch Neighbours").

An ``httpx.AsyncClient`` is passed in so callers (and tests) control its
lifecycle; nothing here opens its own connection. The pure shaping of the
responses lives in :mod:`helpers`.
"""

from __future__ import annotations

from typing import Any

import httpx

from voice_agent.qdrant import (
    BM25_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    points_url,
)


async def embed_query(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    text: str,
    headers: dict[str, str] | None = None,
) -> list[float]:
    """POST to ``<base_url>/embeddings`` and return the dense query vector.

    ``base_url`` is the LM Studio ``/v1`` root (same value as ``llm.base_url``
    for the openai_compatible backend). Raises ``httpx.HTTPStatusError`` on a
    non-2xx response so the caller maps it to an error envelope.
    """
    url = base_url.rstrip("/") + "/embeddings"
    res = await client.post(
        url, json={"input": text, "model": model}, headers=headers or None
    )
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]


async def hybrid_query(
    client: httpx.AsyncClient,
    *,
    qdrant_url: str,
    collection: str,
    embedding: list[float],
    question: str,
    top_k: int,
    embedding_vector_name: str = DENSE_VECTOR_NAME,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid dense+BM25 query with RRF fusion; return raw Qdrant points.

    Mirrors the n8n "Query Points" node: each prefetch pulls ``top_k * 2``
    candidates, RRF fuses them, and the final ``limit`` is ``top_k``. The BM25
    prefetch relies on Qdrant server-side inference (``qdrant/bm25``), exactly
    as the workflow did.
    """
    url = points_url(qdrant_url, collection, "query")
    body = {
        "prefetch": [
            {
                "query": embedding,
                "using": embedding_vector_name,
                "limit": top_k * 2,
            },
            {
                "query": {"text": question, "model": BM25_MODEL},
                "using": SPARSE_VECTOR_NAME,
                "limit": top_k * 2,
            },
        ],
        "query": {"fusion": "rrf"},
        "limit": top_k,
        "with_payload": True,
    }
    res = await client.post(url, json=body, headers=headers or None)
    res.raise_for_status()
    result = res.json().get("result") or {}
    return result.get("points") or []


async def scroll_neighbours(
    client: httpx.AsyncClient,
    *,
    qdrant_url: str,
    collection: str,
    groups: dict[str, list[str]],
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Scroll adjacent chunks for each filename group; return raw points.

    ``groups`` is ``{filename: [chunk_id, ...]}`` from
    :func:`helpers.neighbour_ids`. One scroll request per filename, filtered to
    that file's neighbour ids. A failed request for one filename is skipped
    (best-effort expansion) rather than failing the whole turn.
    """
    points: list[dict[str, Any]] = []
    url = points_url(qdrant_url, collection, "scroll")
    for filename, ids in groups.items():
        if not ids:
            continue
        body = {
            "limit": 20,
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": [
                    {"key": "filename", "match": {"value": filename}},
                    {"key": "chunk_id", "match": {"any": ids}},
                ]
            },
        }
        try:
            res = await client.post(url, json=body, headers=headers or None)
            res.raise_for_status()
        except httpx.HTTPError:
            # Best-effort: expansion is an enhancement, never block the answer.
            continue
        result = res.json().get("result") or {}
        points.extend(result.get("points") or [])
    return points
