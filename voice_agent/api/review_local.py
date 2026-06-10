"""Local-backend HTTP endpoints for the HITL chunk-review flow.

Mounted when ``review.backend: local``; serves the same five routes (same
request/response shapes) as the n8n proxy in :mod:`voice_agent.api.review`,
but backed by the in-process :class:`~voice_agent.ingestion.engine.IngestionEngine`
instead of n8n webhooks:

* ``POST /api/review/upload``            -- 202 + background ingest.
* ``GET  /api/review/pending``           -- batches from the local store
  (no ``resume_url`` was ever present to strip).
* ``POST /api/review/{batch_id}/resume`` -- apply decisions; 404 once used.
* ``GET  /api/review/audit-log``         -- filtered local audit entries.
* ``POST /api/review/audit-event``       -- one UI-side audit row.

The frontend cannot tell the backends apart -- which is the point: flipping
``review.backend`` requires no frontend change. Endpoints 503 with a
"configure review.<field>" message until the local fields are set, matching
the n8n router's behaviour for ``n8n_base_url``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from voice_agent.api.review import AuditEventRequest
from voice_agent.config import ReviewConfig
from voice_agent.ingestion.engine import IngestionEngine
from voice_agent.logging_setup import get_logger


def _missing(field: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            f"review.{field} is not configured. Set it in config.yaml under "
            "the `review:` block to enable this endpoint."
        ),
    )


def create_local_review_router(
    cfg: ReviewConfig,
    *,
    llm_model: str | None = None,
    engine: IngestionEngine | None = None,
) -> APIRouter:
    """Build the local /api/review router.

    ``engine`` is injectable for tests; by default one is constructed from the
    config. The engine is exposed as ``router._http_client`` so
    :func:`~voice_agent.api.app.create_app`'s lifespan handler closes it at
    shutdown through the same ``aclose()`` duck type it already uses for the
    proxy routers' httpx clients.
    """
    router = APIRouter(prefix="/api/review", tags=["review"])
    log = get_logger("api.review_local")
    eng = engine or IngestionEngine(cfg, default_model=llm_model)
    router._http_client = eng  # type: ignore[attr-defined]

    def _require_local_fields() -> None:
        if not cfg.qdrant_url:
            raise _missing("qdrant_url")
        if not cfg.llm_base_url:
            raise _missing("llm_base_url")

    # ---- UPLOAD --------------------------------------------------------
    @router.post("/upload", status_code=202)
    async def upload(
        file: UploadFile = File(...),
        Document_Type: str = Form(default=""),
        Collection_Name: str = Form(default=""),
        Categories: str | None = Form(default=None),
        Chunking_Strategy: str | None = Form(default=None),
        Model: str | None = Form(default=None),
    ) -> dict[str, Any]:
        _require_local_fields()
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        eng.start_ingest(
            content=content,
            filename=file.filename or "upload.pdf",
            document_type=Document_Type or cfg.default_document_type,
            collection_name=Collection_Name or cfg.default_collection_name,
            categories=Categories if Categories is not None else cfg.default_categories,
            chunking_strategy=(
                Chunking_Strategy
                if Chunking_Strategy is not None
                else cfg.default_chunking_strategy
            ),
            model=Model or None,
        )
        log.info(
            "review_upload_accepted",
            filename=file.filename,
            size_bytes=len(content),
        )
        return {
            "status": "queued",
            "message": "PDF received. Poll /api/review/pending for the chunk-review batch.",
        }

    # ---- PENDING -------------------------------------------------------
    @router.get("/pending")
    async def pending() -> dict[str, Any]:
        batches = await asyncio.to_thread(eng.store.list_pending_batches)
        return {"total_pending_batches": len(batches), "batches": batches}

    # ---- RESUME --------------------------------------------------------
    @router.post("/{batch_id}/resume")
    async def resume(batch_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _require_local_fields()
        decisions = body.get("decisions")
        if not isinstance(decisions, list):
            raise HTTPException(
                status_code=400,
                detail='resume body must contain a "decisions" array.',
            )
        try:
            outcome = await eng.resume(batch_id, decisions)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"ingestion finalize failed: {exc}"
            ) from exc
        if outcome is None:
            raise HTTPException(
                status_code=404,
                detail=f"batch_id {batch_id!r} is no longer pending review.",
            )
        return outcome

    # ---- AUDIT LOG -----------------------------------------------------
    @router.get("/audit-log")
    async def audit_log(
        limit: int | None = None,
        actie: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            eng.store.query_audit, limit=limit, actie=actie, since=since
        )

    # ---- AUDIT EVENT (write) -------------------------------------------
    @router.post("/audit-event")
    async def audit_event(event: AuditEventRequest) -> dict[str, Any]:
        row = await asyncio.to_thread(
            eng.store.insert_audit,
            event.document_naam,
            event.actie,
            event.resultaat,
        )
        log.info("audit_event_logged", actie=event.actie, id=row["id"])
        return {
            "status": "logged",
            "id": row["id"],
            "createdAt": row["createdAt"],
            "document_naam": row["document_naam"],
            "actie": row["actie"],
        }

    return router
