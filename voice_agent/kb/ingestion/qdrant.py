"""Qdrant collection management, embedding, and upsert for local ingestion.

The three network side-effects of the ingestion pipeline's finalize phase,
ported from the n8n qdrant/httpRequest nodes over plain ``httpx`` (same REST
shapes; no ``qdrant-client`` dependency):

* ``Check Collection`` / ``Create Collection`` / ``Create Index: *``
  -> :func:`ensure_collection` -- named dense vector
  ``text-embedding-bge-m3`` (1024, cosine) + sparse ``bm25`` (IDF modifier),
  plus keyword payload indexes on ``document_type`` and ``categories``.
* ``LM Studio Embeddings`` -> :func:`embed_texts` -- one batched
  ``/v1/embeddings`` call for the whole batch.
* ``Upsert to Qdrant`` -> :func:`upsert_points` -- per-point payload identical
  to the workflow's, with the BM25 sparse vector supplied as a server-side
  inference document (``{"text", "model": "qdrant/bm25", "options": {"avg_len"}}``)
  exactly as the runtime query side already does.

Clients are passed in; callers own their lifecycle.
"""

from __future__ import annotations

from typing import Any

import httpx

# Vector schema + BM25 model live in the shared Qdrant helper so ingestion
# (which creates the collection) and retrieval (which queries it) can't drift.
from voice_agent.qdrant import (
    BM25_MODEL,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    SPARSE_VECTOR_NAME,
    collection_url,
)


async def ensure_collection(
    client: httpx.AsyncClient,
    *,
    qdrant_url: str,
    collection: str,
    headers: dict[str, str] | None = None,
) -> bool:
    """Create the hybrid collection + payload indexes if absent.

    Returns ``True`` if the collection was created, ``False`` if it already
    existed. Index creation only runs on creation, mirroring the n8n branch.
    """
    base = collection_url(qdrant_url, collection)
    res = await client.get(f"{base}/exists", headers=headers or None)
    res.raise_for_status()
    if (res.json().get("result") or {}).get("exists"):
        return False

    res = await client.put(
        base,
        json={
            "vectors": {
                DENSE_VECTOR_NAME: {"size": DENSE_VECTOR_SIZE, "distance": "Cosine"}
            },
            "sparse_vectors": {SPARSE_VECTOR_NAME: {"modifier": "idf"}},
        },
        headers=headers or None,
    )
    res.raise_for_status()

    for field_name in ("document_type", "categories"):
        res = await client.put(
            f"{base}/index",
            json={"field_name": field_name, "field_schema": "keyword"},
            headers=headers or None,
        )
        res.raise_for_status()
    return True


async def embed_texts(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    texts: list[str],
    headers: dict[str, str] | None = None,
) -> list[list[float]]:
    """Embed the whole batch in one ``/v1/embeddings`` call, order-preserving."""
    url = base_url.rstrip("/") + "/embeddings"
    res = await client.post(
        url, json={"input": texts, "model": model}, headers=headers or None
    )
    res.raise_for_status()
    data = res.json()["data"]
    # The API returns one entry per input with an explicit index; sort on it
    # rather than trusting response order.
    return [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]


def build_points(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    avg_len: float,
    batch_id: str,
) -> list[dict[str, Any]]:
    """Assemble the upsert point list (pure; the n8n ``Upsert to Qdrant`` body).

    Payload field set matches the workflow verbatim -- including
    ``hitl_reviewed: true`` and the batch id -- so locally-ingested chunks are
    indistinguishable from n8n-ingested ones to the runtime RAG branch and the
    Documents page.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding count mismatch: {len(chunks)} != {len(embeddings)}"
        )
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            {
                "id": chunk["point_id"],
                "payload": {
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "total_chunks": chunk["total_chunks"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "chunk_length": chunk["chunk_length"],
                    "words_in_text": chunk["words_in_text"],
                    "document_summary": chunk["document_summary"],
                    "section_title": chunk["section_title"],
                    "document_type": chunk["document_type"],
                    "upload_timestamp": chunk["upload_timestamp"],
                    "categories": chunk["categories"],
                    "chunking_strategy": chunk["chunking_strategy"],
                    "chunk_overlap": chunk["chunk_overlap"],
                    "hitl_reviewed": True,
                    "hitl_batch_id": batch_id,
                },
                "vector": {
                    DENSE_VECTOR_NAME: embedding,
                    SPARSE_VECTOR_NAME: {
                        "text": chunk["text"],
                        "model": BM25_MODEL,
                        "options": {"avg_len": avg_len},
                    },
                },
            }
        )
    return points


async def upsert_points(
    client: httpx.AsyncClient,
    *,
    qdrant_url: str,
    collection: str,
    points: list[dict[str, Any]],
    headers: dict[str, str] | None = None,
) -> None:
    """PUT the assembled points; raises on a non-2xx response."""
    url = f"{collection_url(qdrant_url, collection)}/points?wait=true"
    res = await client.put(url, json={"points": points}, headers=headers or None)
    res.raise_for_status()
