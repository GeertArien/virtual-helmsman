"""HTTP endpoints for qdrant document management.

Two routes, both mounted at ``/api/documents``:

* ``GET    /api/documents``       -- list distinct documents in qdrant.
* ``DELETE /api/documents/{id}``  -- delete every chunk whose payload carries
  ``document_id == {id}`` (the field name is configurable).

Uploads are not handled here -- they live under ``/api/review``, which runs
the in-backend ingestion pipeline with a human-in-the-loop review step.

Each route degrades gracefully when its required config field is missing:
the call returns HTTP 503 with a body that names the exact field to set,
so the frontend can show an actionable error.

The qdrant REST surface used here is documented at
https://qdrant.tech/documentation/concepts/points/ -- specifically the
``points/scroll``, ``points/count`` and ``points/delete`` endpoints.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from voice_agent import qdrant
from voice_agent.config import DocumentsConfig
from voice_agent.logging_setup import get_logger


def _missing(field: str) -> HTTPException:
    """503 telling the user which config field to populate."""
    return HTTPException(
        status_code=503,
        detail=(
            f"documents.{field} is not configured. Set it in config.yaml under "
            "the `documents:` block to enable this endpoint."
        ),
    )


def _qdrant_headers(cfg: DocumentsConfig) -> dict[str, str]:
    """JSON content-type plus an ``api-key`` if the configured env var is set."""
    return {
        "content-type": "application/json",
        **qdrant.api_key_headers(cfg.qdrant_api_key_env),
    }


async def _qdrant_post(
    client: httpx.AsyncClient,
    cfg: DocumentsConfig,
    url: str,
    json_body: dict[str, Any],
) -> dict[str, Any]:
    """POST to qdrant, surfacing upstream errors as 502 (bad gateway).

    qdrant 4xx bodies usually carry a useful ``status.error`` string; we
    forward that verbatim so the frontend doesn't see a generic 502.
    """
    try:
        res = await client.post(url, json=json_body, headers=_qdrant_headers(cfg))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"qdrant unreachable: {exc}") from exc
    if res.status_code >= 400:
        try:
            body = res.json()
        except ValueError:
            body = {"raw": res.text}
        raise HTTPException(
            status_code=502,
            detail={"upstream_status": res.status_code, "qdrant": body},
        )
    return res.json()


def _group_documents(
    points: list[dict[str, Any]], cfg: DocumentsConfig
) -> list[dict[str, Any]]:
    """Roll up scrolled points into one row per distinct ``document_id``.

    First-seen wins for title / source / uploaded_at; chunk_count is the
    number of points sharing the same id. Points that don't carry a
    document_id payload are skipped (they aren't part of an ingested doc).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for pt in points:
        payload = pt.get("payload") or {}
        doc_id = payload.get(cfg.document_id_field)
        if doc_id is None:
            continue
        doc_id = str(doc_id)
        row = by_id.get(doc_id)
        if row is None:
            by_id[doc_id] = {
                "document_id": doc_id,
                "title": payload.get(cfg.title_field),
                "source": payload.get(cfg.source_field),
                "uploaded_at": payload.get(cfg.uploaded_at_field),
                "chunk_count": 1,
            }
        else:
            row["chunk_count"] += 1
    return sorted(by_id.values(), key=lambda r: (r.get("title") or r["document_id"]).lower())


def create_documents_router(cfg: DocumentsConfig) -> APIRouter:
    """Build the /api/documents router bound to a :class:`DocumentsConfig`.

    The router closes over ``cfg`` (and a long-lived :class:`httpx.AsyncClient`)
    so the FastAPI app stays stateless from the caller's perspective.
    """
    router = APIRouter(prefix="/api/documents", tags=["documents"])
    log = get_logger("api.documents")
    # One client per app, reused across requests. Closed when the app shuts
    # down via FastAPI's shutdown hook (registered in create_app).
    client = httpx.AsyncClient(timeout=cfg.request_timeout_seconds)
    router._http_client = client  # type: ignore[attr-defined]

    # ---- LIST ----------------------------------------------------------
    @router.get("")
    async def list_documents() -> dict[str, list[dict[str, Any]]]:
        if not cfg.qdrant_url:
            raise _missing("qdrant_url")
        if not cfg.qdrant_collection:
            raise _missing("qdrant_collection")

        # Scroll the whole collection in pages of 256, capped at scroll_limit.
        page_size = 256
        scrolled: list[dict[str, Any]] = []
        offset: Any = None
        while len(scrolled) < cfg.scroll_limit:
            body: dict[str, Any] = {
                "limit": min(page_size, cfg.scroll_limit - len(scrolled)),
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            data = await _qdrant_post(
                client,
                cfg,
                qdrant.points_url(cfg.qdrant_url, cfg.qdrant_collection, "scroll"),
                body,
            )
            result = data.get("result", {})
            points = result.get("points", []) or []
            scrolled.extend(points)
            offset = result.get("next_page_offset")
            if not offset or not points:
                break

        documents = _group_documents(scrolled, cfg)
        log.info(
            "documents_listed", count=len(documents), points_scanned=len(scrolled)
        )
        return {"documents": documents}

    # ---- DELETE --------------------------------------------------------
    @router.delete("/{document_id}")
    async def delete_document(document_id: str) -> dict[str, Any]:
        if not cfg.qdrant_url:
            raise _missing("qdrant_url")
        if not cfg.qdrant_collection:
            raise _missing("qdrant_collection")

        # Match every point whose payload[document_id_field] == document_id.
        filter_clause = {
            "must": [
                {
                    "key": cfg.document_id_field,
                    "match": {"value": document_id},
                }
            ]
        }

        # Count first so the response can report a real chunk count -- the
        # qdrant delete endpoint only returns an operation_id, not a count.
        count_data = await _qdrant_post(
            client,
            cfg,
            qdrant.points_url(cfg.qdrant_url, cfg.qdrant_collection, "count"),
            {"filter": filter_clause, "exact": True},
        )
        deleted = int(count_data.get("result", {}).get("count", 0))

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found with {cfg.document_id_field}={document_id!r}.",
            )

        await _qdrant_post(
            client,
            cfg,
            qdrant.points_url(cfg.qdrant_url, cfg.qdrant_collection, "delete"),
            {"filter": filter_clause, "wait": True},
        )
        log.info(
            "document_deleted",
            document_id=document_id,
            deleted_chunks=deleted,
        )
        return {
            "status": "deleted",
            "document_id": document_id,
            "deleted_chunks": deleted,
        }

    return router
