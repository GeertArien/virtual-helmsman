"""HTTP endpoints for the HITL chunk-review flow.

Four routes mounted at ``/api/review``:

* ``POST /api/review/upload``                 -- forward multipart to
  ``<n8n>/webhook/review/upload``; returns n8n's 202 body unchanged.
* ``GET  /api/review/pending``                -- pass through
  ``<n8n>/webhook/review/pending`` with each batch's ``resume_url`` stripped
  (the frontend never sees the n8n callback URL).
* ``POST /api/review/{batch_id}/resume``      -- look up the batch's
  ``resume_url`` by re-fetching the pending list, then forward the decisions
  JSON to it. Returns 404 if the batch is no longer pending (someone else
  already submitted, or the workflow timed out).
* ``GET  /api/review/audit-log``              -- pass through
  ``<n8n>/webhook/audit-log`` with the ``limit`` / ``actie`` / ``since`` query
  params forwarded verbatim. Surfaces ingestion outcomes for the UI's
  recent-activity feed.

Each endpoint returns HTTP 503 with a "configure review.<field>" message
until ``ReviewConfig.n8n_base_url`` is set.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from voice_agent.config import ReviewConfig
from voice_agent.logging_setup import get_logger


def _missing(field: str) -> HTTPException:
    """503 telling the user which config field to populate."""
    return HTTPException(
        status_code=503,
        detail=(
            f"review.{field} is not configured. Set it in config.yaml under "
            "the `review:` block to enable this endpoint."
        ),
    )


def _strip_resume_url(batch: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one batch with ``resume_url`` removed.

    Browser code never needs the n8n callback URL -- it submits to
    ``/api/review/{batch_id}/resume`` and the backend rebinds the call to
    the right resume URL at submit time.
    """
    return {k: v for k, v in batch.items() if k != "resume_url"}


async def _fetch_pending(
    client: httpx.AsyncClient, cfg: ReviewConfig
) -> dict[str, Any]:
    """GET <n8n>/webhook/review/pending and return the parsed JSON body.

    n8n quirk: when the underlying datatable is empty, the workflow ends
    with no item to return and the webhook responds 500 with the body
    ``{"code":0,"message":"No item to return was found"}``. The contract
    says the empty case should return ``{total_pending_batches:0, batches:[]}``
    instead -- mapping it here keeps the frontend tolerant. Fix the n8n
    workflow (e.g. add an "always emit a stub item" branch) to remove the
    workaround.
    """
    assert cfg.n8n_base_url is not None
    log = get_logger("api.review")
    url = cfg.n8n_base_url.rstrip("/") + cfg.pending_path
    try:
        res = await client.get(url)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"n8n unreachable: {exc}") from exc

    parsed: Any
    try:
        parsed = res.json()
    except ValueError:
        parsed = None

    if res.status_code >= 400:
        if (
            res.status_code == 500
            and isinstance(parsed, dict)
            and parsed.get("message") == "No item to return was found"
        ):
            log.warning(
                "n8n_pending_empty_500",
                hint=(
                    "n8n's pending-review workflow returned 500 for an empty "
                    "queue; treating as []. Patch the workflow to return "
                    "{total_pending_batches:0, batches:[]} per the contract."
                ),
            )
            return {"total_pending_batches": 0, "batches": []}
        detail: Any = parsed if parsed is not None else res.text
        raise HTTPException(
            status_code=502,
            detail={"upstream_status": res.status_code, "n8n": detail},
        )

    if parsed is None:
        raise HTTPException(
            status_code=502, detail="n8n returned non-JSON for pending list."
        )
    return parsed


def create_review_router(
    cfg: ReviewConfig, *, llm_model: str | None = None
) -> APIRouter:
    """Build the /api/review router bound to a :class:`ReviewConfig`.

    The router holds a long-lived :class:`httpx.AsyncClient` exposed via
    the private ``_http_client`` attribute so :func:`create_app`'s lifespan
    handler can close it at shutdown.

    ``llm_model`` is the LM Studio chat-model identifier used as the
    default ``Model`` form field on uploads (per ``REVIEW_API.md``). It
    is normally threaded from ``LlmConfig.model`` so the doc-summary call
    inside the ingestion pipeline uses the same model the rest of the
    helmsman LLM path does. Per-request overrides via the ``Model`` form
    field still win; ``None`` means "let n8n use its own default".
    """
    router = APIRouter(prefix="/api/review", tags=["review"])
    log = get_logger("api.review")
    client = httpx.AsyncClient(timeout=cfg.request_timeout_seconds)
    router._http_client = client  # type: ignore[attr-defined]

    # ---- UPLOAD --------------------------------------------------------
    @router.post("/upload")
    async def upload(
        file: UploadFile = File(...),
        Document_Type: str = Form(default=""),
        Collection_Name: str = Form(default=""),
        Categories: str | None = Form(default=None),
        Chunking_Strategy: str | None = Form(default=None),
        Model: str | None = Form(default=None),
    ) -> dict[str, Any]:
        if not cfg.n8n_base_url:
            raise _missing("n8n_base_url")

        # The webhook treats Document_Type and Collection_Name as required.
        # When the form omits them we fall back to the configured defaults
        # rather than passing through empty strings -- empty values would
        # silently produce chunks with no document_type / wrong collection.
        document_type = Document_Type or cfg.default_document_type
        collection_name = Collection_Name or cfg.default_collection_name
        categories = Categories if Categories is not None else cfg.default_categories
        chunking_strategy = (
            Chunking_Strategy
            if Chunking_Strategy is not None
            else cfg.default_chunking_strategy
        )
        # Model precedence: explicit per-request override > configured
        # llm.model > omit (n8n falls back to its own default). Empty
        # string from the form is treated as "use the configured one"
        # rather than as an explicit empty override.
        model = Model or llm_model

        content = await file.read()
        # n8n takes the first binary part regardless of name; we use "pdf"
        # for legibility in the n8n execution view.
        files = {
            "pdf": (
                file.filename or "upload.pdf",
                content,
                file.content_type or "application/pdf",
            )
        }
        data: dict[str, str] = {
            "Document_Type": document_type,
            "Collection_Name": collection_name,
            "Categories": categories,
            "Chunking_Strategy": chunking_strategy,
        }
        if model is not None:
            data["Model"] = model

        url = cfg.n8n_base_url.rstrip("/") + cfg.upload_path
        try:
            res = await client.post(url, files=files, data=data)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"n8n unreachable: {exc}") from exc

        if res.status_code >= 400:
            try:
                detail = res.json()
            except ValueError:
                detail = res.text
            raise HTTPException(
                status_code=502,
                detail={"upstream_status": res.status_code, "n8n": detail},
            )

        try:
            body = res.json()
        except ValueError:
            body = {}
        log.info(
            "review_upload_forwarded",
            filename=file.filename,
            size_bytes=len(content),
            collection_name=collection_name,
            document_type=document_type,
            chunking_strategy=chunking_strategy,
            model=model,
            n8n_status=res.status_code,
        )
        # Surface the same shape n8n returned, with a safe default if the
        # workflow ever drops the message field.
        return {
            "status": body.get("status", "queued"),
            "message": body.get(
                "message",
                "PDF received. Poll /api/review/pending for the chunk-review batch.",
            ),
        }

    # ---- PENDING -------------------------------------------------------
    @router.get("/pending")
    async def pending() -> dict[str, Any]:
        if not cfg.n8n_base_url:
            raise _missing("n8n_base_url")
        body = await _fetch_pending(client, cfg)
        # Strip resume_url from each batch -- browsers see only batch_id.
        raw_batches = body.get("batches") or []
        batches = [_strip_resume_url(b) for b in raw_batches]
        return {
            "total_pending_batches": body.get("total_pending_batches", len(batches)),
            "batches": batches,
        }

    # ---- RESUME --------------------------------------------------------
    @router.post("/{batch_id}/resume")
    async def resume(batch_id: str, request: Request) -> Any:
        if not cfg.n8n_base_url:
            raise _missing("n8n_base_url")

        # Look up the resume URL by re-fetching pending. This costs one extra
        # GET per submit but the list is small and the call is cheap; the
        # alternative is to cache the mapping in memory and invalidate on
        # successful submit, which adds state without buying much.
        body = await _fetch_pending(client, cfg)
        match: dict[str, Any] | None = None
        for batch in body.get("batches") or []:
            if batch.get("batch_id") == batch_id:
                match = batch
                break
        if match is None:
            # Batch isn't pending anymore -- either already submitted or the
            # workflow timed out. Frontend should refresh its list.
            raise HTTPException(
                status_code=404,
                detail=f"batch_id {batch_id!r} is no longer pending review.",
            )
        resume_url = match.get("resume_url")
        if not resume_url:
            # Should never happen given the contract, but keep the failure
            # visible rather than POSTing to None.
            raise HTTPException(
                status_code=502,
                detail=f"n8n returned no resume_url for batch {batch_id!r}.",
            )

        # Forward the decisions JSON body verbatim. The frontend is expected
        # to send {"batch_id": ..., "decisions": [...]} per the contract; we
        # don't reshape it -- n8n is the schema authority.
        try:
            decisions_body = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"resume body must be JSON: {exc}"
            ) from exc

        try:
            res = await client.post(resume_url, json=decisions_body)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"n8n unreachable: {exc}") from exc

        if res.status_code == 404 or res.status_code == 410:
            # n8n's wait URL is one-shot. A 404/410 means somebody else
            # already resumed the workflow -- present this as a stale batch
            # to the caller so the UI refreshes.
            raise HTTPException(
                status_code=404,
                detail=f"batch_id {batch_id!r} resume URL has already been used.",
            )
        if res.status_code >= 400:
            try:
                detail = res.json()
            except ValueError:
                detail = res.text
            raise HTTPException(
                status_code=502,
                detail={"upstream_status": res.status_code, "n8n": detail},
            )

        log.info(
            "review_decisions_submitted",
            batch_id=batch_id,
            n8n_status=res.status_code,
        )
        # n8n's resume response is the last-node output (verbose qdrant
        # upsert in v1). Frontend just checks 2xx and refreshes the list,
        # but we surface the raw body for diagnostics.
        try:
            return {"status": "ok", "n8n": res.json()}
        except ValueError:
            return {"status": "ok", "n8n": {"raw": res.text}}

    # ---- AUDIT LOG -----------------------------------------------------
    @router.get("/audit-log")
    async def audit_log(
        limit: int | None = Query(default=None, ge=1, le=500),
        actie: str | None = Query(default=None),
        since: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Forward to ``<n8n>/webhook/audit-log``.

        n8n already validates/clamps the three query params (limit is capped
        at 500, bad ``since`` falls back to no-filter), so we just pass them
        through. Returning the body verbatim keeps the contract honest --
        the frontend renders ``entries[].actie`` / ``resultaat`` directly.
        """
        if not cfg.n8n_base_url:
            raise _missing("n8n_base_url")

        params: dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if actie:
            params["actie"] = actie
        if since:
            params["since"] = since

        url = cfg.n8n_base_url.rstrip("/") + cfg.audit_log_path
        try:
            res = await client.get(url, params=params or None)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"n8n unreachable: {exc}") from exc

        if res.status_code >= 400:
            try:
                detail = res.json()
            except ValueError:
                detail = res.text
            raise HTTPException(
                status_code=502,
                detail={"upstream_status": res.status_code, "n8n": detail},
            )

        try:
            body = res.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail=f"n8n returned non-JSON for audit-log: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=502,
                detail="n8n returned a non-object body for audit-log.",
            )
        return body

    return router
