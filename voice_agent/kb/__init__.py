"""The knowledge-base half of the process, behind an explicit boundary.

One process serves two functionally unrelated applications (issue #12 §6):

* the **voice runtime** -- pipeline, simulator, events WS, WebRTC signalling;
* this package -- document ingestion with HITL review, qdrant document
  management, and the audit store.

They are deliberately colocated (one port, one config file, shared LM Studio /
Langfuse plumbing for a single-user tool), but the boundary is structural, not
accidental:

* Nothing under ``voice_agent.kb`` may import the voice side
  (``backends``, ``actions``, ``pipeline``, ``api``, ``telemetry``,
  ``metrics``). Shared infrastructure (``config``, ``logging_setup``,
  ``qdrant``) is fine. Enforced by ``tests/test_kb_boundary.py``.
* The voice side touches this package at exactly **two blessed points**, both
  re-exported here: :func:`create_kb_routers` (mounted once by
  ``api.app.create_app``) and :class:`IngestionStore` (the shared SQLite
  audit log, written by the LLM service's runtime-audit hook). Anything else
  crossing the line is a boundary violation, also caught by the test.

This is what keeps a later split into separate processes trivial: the KB side
already has its own wiring entry point and its own runtime config blocks
(``DocumentsRuntime`` / ``IngestionRuntime``).
"""

from __future__ import annotations

from fastapi import APIRouter

from voice_agent.config import DocumentsRuntime, IngestionRuntime
from voice_agent.kb.ingestion.store import IngestionStore

__all__ = ["IngestionStore", "create_kb_routers"]


def create_kb_routers(
    *,
    documents: DocumentsRuntime | None = None,
    review: IngestionRuntime | None = None,
    llm_model: str | None = None,
) -> list[APIRouter]:
    """Build every knowledge-base router that is configured.

    The single wiring point between the KB side and the FastAPI app: the app
    mounts whatever comes back and otherwise knows nothing about this package.
    Each router may own a long-lived ``httpx.AsyncClient`` exposed as
    ``_http_client``; the app's lifespan closes those at shutdown.

    ``llm_model`` is the default summary model for ingestion uploads -- keeps
    the doc-summary call on the same model the helmsman LLM path uses.
    """
    routers: list[APIRouter] = []
    if documents is not None:
        from voice_agent.kb.documents import create_documents_router

        routers.append(create_documents_router(documents))
    if review is not None:
        from voice_agent.kb.review import create_review_router

        routers.append(create_review_router(review, llm_model=llm_model))
    return routers
