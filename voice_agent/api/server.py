"""Run the FastAPI app under uvicorn as an asyncio task.

The control plane shares the event loop with the Pipecat pipeline, so this
runs uvicorn programmatically (not via the CLI) and yields a handle the caller
can cancel cleanly at shutdown.
"""

from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI

from voice_agent.logging_setup import get_logger


class ApiServer:
    """Lifecycle wrapper around a uvicorn server bound to a FastAPI app."""

    def __init__(self, app: FastAPI, *, host: str, port: int) -> None:
        # ``log_config=None`` keeps uvicorn from re-configuring logging --
        # ``voice_agent.logging_setup`` is the single source of truth for the
        # process's logging config.
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_config=None,
            lifespan="on",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task: asyncio.Task[None] | None = None
        self._log = get_logger("api")

    async def start(self) -> None:
        """Spawn the uvicorn task; returns once the server is listening."""
        self._task = asyncio.create_task(self._server.serve(), name="api-server")
        # Wait for uvicorn to flip its "started" flag before returning so the
        # frontend can't race the pipeline's first event.
        while not self._server.started and not self._task.done():
            await asyncio.sleep(0.01)
        if self._task.done():
            # serve() exited immediately -- surface the exception.
            self._task.result()
        self._log.info(
            "api_started", host=self._server.config.host, port=self._server.config.port
        )

    async def stop(self) -> None:
        """Signal uvicorn to exit and await the task."""
        if self._task is None:
            return
        self._server.should_exit = True
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._log.info("api_stopped")
        self._task = None
