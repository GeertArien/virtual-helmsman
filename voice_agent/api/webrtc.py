"""WebRTC browser-audio bridge (issue #7).

Serves the SDP signalling endpoint (``POST /api/webrtc/offer``) and runs a
STT->LLM->TTS pipeline **per browser connection** using Pipecat's
``SmallWebRTCTransport``. The heavy models live in a single
:class:`~voice_agent.pipeline.SharedBackends` built once at startup, so each
connection reuses the loaded models rather than reloading them.

Flow per connection:

1. Browser ``getUserMedia`` -> ``RTCPeerConnection`` -> POST its SDP offer to
   ``/api/webrtc/offer``.
2. We build a :class:`SmallWebRTCConnection`, answer the offer, wrap it in a
   ``SmallWebRTCTransport``, assemble a pipeline (shared backends + this
   transport), and run it in the background.
3. The browser plays the agent's TTS audio from the returned media track.
4. On disconnect the per-connection task is cancelled; the models stay loaded
   for the next connection.

Pipecat's WebRTC stack (``aiortc``) is an optional dependency -- everything
that imports it is loaded lazily inside the methods/handlers, so this module
imports cleanly without the ``webrtc`` extra (the endpoint then returns a
clear 503).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from voice_agent.config import AppConfig
from voice_agent.logging_setup import get_logger

if TYPE_CHECKING:  # avoid importing pipeline (and the local-audio/pyaudio stack)
    from voice_agent.pipeline import SharedBackends


class OfferRequest(BaseModel):
    """An SDP offer from the browser. ``pc_id`` is present on renegotiation."""

    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool = False


class WebRTCManager:
    """Owns the per-connection WebRTC pipelines over one set of shared backends."""

    def __init__(self, backends: "SharedBackends", config: AppConfig) -> None:
        self._backends = backends
        self._config = config
        self._log = get_logger("api.webrtc")
        # pc_id -> (connection, pipeline task, runner task)
        self._connections: dict[str, Any] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def available(self) -> bool:
        """True when the optional WebRTC stack (aiortc) is importable."""
        try:
            import aiortc  # noqa: F401
            import pipecat.transports.smallwebrtc.transport  # noqa: F401
        except ImportError:
            return False
        return True

    async def handle_offer(self, offer: OfferRequest) -> dict[str, Any]:
        """Answer an SDP offer, (re)starting the connection's pipeline."""
        try:
            from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise HTTPException(
                status_code=503,
                detail=(
                    "Browser audio needs the optional 'webrtc' extra "
                    '(pip install -e ".[webrtc]").'
                ),
            ) from exc

        # Renegotiation of an existing peer connection.
        if offer.pc_id and offer.pc_id in self._connections:
            conn = self._connections[offer.pc_id]
            await conn.renegotiate(
                sdp=offer.sdp, type=offer.type, restart_pc=offer.restart_pc
            )
            return conn.get_answer()

        # New connection.
        conn = SmallWebRTCConnection(self._config.audio.ice_servers)
        await conn.initialize(sdp=offer.sdp, type=offer.type)
        await self._start_pipeline(conn)
        self._connections[conn.pc_id] = conn
        self._log.info("webrtc_connection_opened", pc_id=conn.pc_id)
        return conn.get_answer()

    async def _start_pipeline(self, conn: Any) -> None:
        """Build a transport for ``conn`` and run its pipeline in the background."""
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        from voice_agent.pipeline import assemble_task

        transport = SmallWebRTCTransport(
            webrtc_connection=conn,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_t: Any, _client: Any) -> None:
            await self._cleanup(conn.pc_id)

        # Connecting/disconnecting browser audio in the dashboard IS the mic
        # control -- the pipeline carries no server-side mute.
        task, _context = assemble_task(self._backends, self._config, transport)

        async def _run() -> None:
            runner = PipelineRunner(handle_sigint=False)
            try:
                await runner.run(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - one bad peer must not crash others
                self._log.error("webrtc_pipeline_error", pc_id=conn.pc_id, error=str(exc))

        runner_task = asyncio.create_task(_run())
        self._tasks[conn.pc_id] = runner_task
        # Keep the PipelineTask so we can cancel it on disconnect.
        self._connections[conn.pc_id + ":task"] = task  # type: ignore[assignment]

    async def _cleanup(self, pc_id: str) -> None:
        """Tear down one connection's pipeline; leave the shared models loaded."""
        self._connections.pop(pc_id, None)
        task = self._connections.pop(pc_id + ":task", None)
        if task is not None:
            await task.cancel()
        runner_task = self._tasks.pop(pc_id, None)
        if runner_task is not None:
            runner_task.cancel()
        self._log.info("webrtc_connection_closed", pc_id=pc_id)

    async def close(self) -> None:
        """Close every live connection (called on app shutdown)."""
        for pc_id in [k for k in self._connections if not k.endswith(":task")]:
            conn = self._connections.get(pc_id)
            if conn is not None:
                try:
                    await conn.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            await self._cleanup(pc_id)


def create_webrtc_router(manager: WebRTCManager) -> APIRouter:
    """Build the ``/api/webrtc`` router bound to a :class:`WebRTCManager`."""
    router = APIRouter(prefix="/api/webrtc", tags=["webrtc"])

    @router.post("/offer")
    async def offer(req: OfferRequest) -> dict[str, Any]:
        if not manager.available():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Browser audio needs the optional 'webrtc' extra "
                    '(pip install -e ".[webrtc]").'
                ),
            )
        return await manager.handle_offer(req)

    return router
