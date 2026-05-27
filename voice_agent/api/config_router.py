"""HTTP endpoints for viewing and editing the backend's config.yaml.

Four routes mounted at ``/api/config``:

* ``GET  /api/config``         -- current YAML contents parsed to a dict
  (raw -- env-var overrides not applied so what you read is what's on disk).
* ``GET  /api/config/schema``  -- :class:`AppConfig`'s JSON Schema, used by
  the frontend to pick input types (Literal -> select, bool -> checkbox,
  int / float -> number, str -> text).
* ``PUT  /api/config``         -- accept a dict, validate via
  :func:`parse_config` (env overrides applied for validation only), and
  write the submitted dict verbatim to ``config.yaml`` if it parses. On
  validation failure returns ``422`` with the Pydantic error list intact.
* ``POST /api/config/reload``  -- replace the running process via
  ``os.execv(sys.executable, sys.argv)``. The HTTP response is sent first
  and the exec is scheduled ~1s later so the socket releases cleanly. The
  frontend polls ``/api/health`` to detect when the new process is up.

The router stores nothing in memory; ``config.yaml`` on disk is the single
source of truth. ``GET`` re-reads it on every request.

Comments in ``config.yaml`` are lost on PUT -- PyYAML's safe_dump doesn't
preserve them. This is acceptable for now: editing via the UI is opt-in;
hand-edited installations keep their comments.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from voice_agent.config import AppConfig, parse_config
from voice_agent.logging_setup import get_logger


def create_config_router(*, config_path: Path) -> APIRouter:
    """Build the ``/api/config`` router bound to ``config_path``.

    The path is captured at construction time and used by both GET (read)
    and PUT (write). It must point to a writable file -- if the file
    cannot be read or written the corresponding endpoint returns ``500``
    with the filesystem error attached.
    """
    router = APIRouter(prefix="/api/config", tags=["config"])
    log = get_logger("api.config")

    @router.get("")
    async def read_config() -> dict[str, Any]:
        """Return the parsed contents of ``config.yaml`` as a dict.

        Env-var overrides (e.g. ``LLM_BASE_URL``) are NOT applied; the
        endpoint exposes what's on disk so edits round-trip cleanly.
        """
        if not config_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Config file not found at {config_path}",
            )
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read {config_path}: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=500,
                detail=f"{config_path} must contain a YAML mapping at the top level.",
            )
        return data

    @router.get("/schema")
    async def read_schema() -> dict[str, Any]:
        """Return the JSON Schema for :class:`AppConfig`.

        The frontend uses this to pick input widgets per field. Cached
        cheaply -- ``model_json_schema()`` is pure on the class.
        """
        # mode="serialization" matches what model_dump(mode="json") produces,
        # so the schema agrees with what GET /api/config returns.
        return AppConfig.model_json_schema(mode="serialization")

    @router.put("")
    async def write_config(body: dict[str, Any]) -> dict[str, Any]:
        """Validate ``body`` against :class:`AppConfig` and write to disk.

        On Pydantic validation failure returns ``422`` with the error list
        (the same shape FastAPI uses for malformed requests), so the
        frontend can highlight which fields are bad.
        """
        try:
            parse_config(body)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_url=False),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Write the *submitted* dict, not an env-overridden dump, so a
        # save doesn't bake env-var values into the file.
        try:
            with config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(body, fh, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not write {config_path}: {exc}",
            ) from exc

        log.info(
            "config_saved",
            path=str(config_path),
            bytes_written=config_path.stat().st_size,
        )
        return {"status": "saved", "path": str(config_path)}

    @router.post("/reload")
    async def reload_backend() -> dict[str, Any]:
        """Replace the running backend process so the new config takes effect.

        Schedules ``os.execv(sys.executable, sys.argv)`` ~1s after the
        response is queued so the HTTP 200 has time to flush and the
        listening socket releases before the new process binds it again.

        We don't gracefully tear down resources first -- ``execv`` on
        Windows spawns a fresh process anyway, so OS cleanup is automatic.
        Pipecat's audio devices / GPU contexts release when the old PID
        exits.
        """
        loop = asyncio.get_running_loop()
        log.warning(
            "backend_reload_requested",
            executable=sys.executable,
            argv=sys.argv,
        )

        def _exec_self() -> None:
            # Tiny indirection so the test suite can monkeypatch this.
            argv = [sys.executable, *sys.argv]
            os.execv(sys.executable, argv)

        loop.call_later(1.0, _exec_self)
        return {"status": "reloading", "delay_seconds": 1.0}

    return router
