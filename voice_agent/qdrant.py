"""Shared Qdrant REST helpers.

Three subsystems talk to Qdrant over plain ``httpx`` (no ``qdrant-client``):
the Documents management API (:mod:`voice_agent.api.documents`), the ingestion
finalize phase (:mod:`voice_agent.ingestion.qdrant`), and the LangGraph RAG
retrieval branch
(:mod:`voice_agent.backends.llm.langgraph_helmsman.retrieval`). They each used
to hand-roll the ``api-key`` header, the collection URL, and the named-vector
strings. Centralising those here removes the drift risk -- in particular the
dense/sparse vector names that ingestion *creates* and retrieval *queries* must
stay identical, or recall silently breaks.

This module is intentionally tiny and dependency-free (stdlib only): it does
not own an ``httpx`` client or make requests. Callers keep their own clients
and request shapes; they just build URLs/headers and reference the vector
schema through these helpers.
"""

from __future__ import annotations

import os

# --- the hybrid collection's vector schema ---------------------------------
# Ingestion creates the collection with these named vectors; retrieval queries
# them by the same names. Keep them here so a rename can't desync the two
# sides. Pinned to bge-m3 / 1024-dim (dense) + a BM25 sparse vector.
DENSE_VECTOR_NAME = "text-embedding-bge-m3"
DENSE_VECTOR_SIZE = 1024
SPARSE_VECTOR_NAME = "bm25"
# Qdrant server-side BM25 inference model, used for both the sparse upsert
# vector and the sparse query prefetch.
BM25_MODEL = "qdrant/bm25"


def api_key_headers(api_key_env: str) -> dict[str, str]:
    """``{"api-key": <value>}`` if the named env var is set, else ``{}``.

    Qdrant authenticates with an ``api-key`` header; unauthenticated local
    instances need none, so an unset/blank env var yields no header.
    """
    key = os.environ.get(api_key_env)
    return {"api-key": key} if key else {}


def collection_url(qdrant_url: str, collection: str) -> str:
    """Base REST URL for one collection (no trailing slash)."""
    return f"{qdrant_url.rstrip('/')}/collections/{collection}"


def points_url(qdrant_url: str, collection: str, op: str) -> str:
    """URL for a ``points`` sub-endpoint.

    ``op="scroll"`` -> ``.../collections/<c>/points/scroll``. Pass ``op=""``
    for the bare ``.../points`` endpoint (upsert).
    """
    base = f"{collection_url(qdrant_url, collection)}/points"
    return f"{base}/{op}" if op else base
