"""HTTP endpoints for the in-backend HITL chunk-review flow.

Five routes mounted at ``/api/review``, backed by the in-process
:class:`~voice_agent.ingestion.engine.IngestionEngine` (see
``docs/LOCAL_INGESTION.md``):

* ``POST /api/review/upload``            -- accept a PDF, return 202, and run
  the ingest (extract -> clean -> summary -> chunk -> pending batch) in the
  background.
* ``GET  /api/review/pending``           -- batches awaiting review, read from
  the local SQLite store (no ``resume_url``: local mode resumes by batch_id).
* ``POST /api/review/{batch_id}/resume`` -- apply the reviewer's decisions and
  finalize (embed + upsert to Qdrant). 404 once the batch has been resumed.
* ``GET  /api/review/audit-log``         -- filtered audit entries (ingestion
  outcomes + runtime helmsman activity) for the UI's recent-activity feed.
* ``POST /api/review/audit-event``       -- write one UI-side audit row (e.g.
  the AI Act Art. 50 transparency acknowledgement).

Write endpoints return HTTP 503 with a "configure review.<field>" message
until ``qdrant_url`` and ``llm_base_url`` are set; the read endpoints work
immediately.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from voice_agent.config import IngestionRuntime
from voice_agent.ingestion.engine import IngestionEngine
from voice_agent.logging_setup import get_logger


class AuditEventRequest(BaseModel):
    """One UI-side audit row: three required strings, each bounded to 500 chars.

    The semantics of each field are chosen by the caller; for the Art. 50
    transparency gate the frontend sends ``actie="art50_acknowledged"``,
    ``document_naam="transparantieverklaring_v<version>"``, ``resultaat="OK"``.
    """

    document_naam: str = Field(min_length=1, max_length=500)
    actie: str = Field(min_length=1, max_length=500)
    resultaat: str = Field(min_length=1, max_length=500)


def _missing(block: str, field: str) -> HTTPException:
    """503 telling the user which shared config field to populate."""
    return HTTPException(
        status_code=503,
        detail=(
            f"{block}.{field} is not configured. Set it in config.yaml under "
            f"the `{block}:` block to enable this endpoint."
        ),
    )


def create_review_router(
    cfg: IngestionRuntime,
    *,
    llm_model: str | None = None,
    engine: IngestionEngine | None = None,
) -> APIRouter:
    """Build the /api/review router bound to an :class:`IngestionRuntime`.

    ``llm_model`` is the LM Studio chat-model identifier used as the default
    for the doc-summary call (threaded from ``LlmConfig.model`` so ingestion
    uses the same model as the helmsman LLM path); a per-request ``Model`` form
    field still wins. ``engine`` is injectable for tests; by default one is
    constructed from the config.

    The engine is exposed as ``router._http_client`` so
    :func:`~voice_agent.api.app.create_app`'s lifespan handler closes it at
    shutdown through the same ``aclose()`` duck type it uses for httpx clients.
    """
    router = APIRouter(prefix="/api/review", tags=["review"])
    log = get_logger("api.review")
    eng = engine or IngestionEngine(cfg, default_model=llm_model)
    router._http_client = eng  # type: ignore[attr-defined]

    def _require_local_fields() -> None:
        # lm_studio.base_url is required, so qdrant.url is the only gate.
        if not cfg.qdrant_url:
            raise _missing("qdrant", "url")

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
