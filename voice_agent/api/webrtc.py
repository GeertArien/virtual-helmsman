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
from dataclasses import dataclass
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


@dataclass
class _Connection:
    """One live browser connection: the peer connection plus its pipeline tasks.

    ``pipeline_task`` and ``runner_task`` are filled in by
    :meth:`WebRTCManager._start_pipeline` immediately after the record is
    registered; both stay ``None`` only for the brief window before the
    pipeline is wired, and :meth:`WebRTCManager._cleanup` tolerates that.
    """

    connection: Any
    pipeline_task: Any = None
    runner_task: "asyncio.Task[None] | None" = None


class WebRTCManager:
    """Owns the per-connection WebRTC pipelines over one set of shared backends."""

    def __init__(self, backends: "SharedBackends", config: AppConfig) -> None:
        self._backends = backends
        self._config = config
        self._log = get_logger("api.webrtc")
        # pc_id -> the connection and its pipeline/runner tasks.
        self._connections: dict[str, _Connection] = {}

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
            conn = self._connections[offer.pc_id].connection
            await conn.renegotiate(
                sdp=offer.sdp, type=offer.type, restart_pc=offer.restart_pc
            )
            return conn.get_answer()

        # New connection. Register the record *before* starting the pipeline so
        # an ``on_client_disconnected`` that fires during startup always finds
        # it (and the per-connection tasks are filled in by _start_pipeline).
        conn = SmallWebRTCConnection(self._config.audio.ice_servers)
        await conn.initialize(sdp=offer.sdp, type=offer.type)
        record = _Connection(connection=conn)
        self._connections[conn.pc_id] = record
        self._start_pipeline(record)
        self._log.info("webrtc_connection_opened", pc_id=conn.pc_id)
        return conn.get_answer()

    def _start_pipeline(self, record: _Connection) -> None:
        """Build a transport for ``record`` and run its pipeline in the background."""
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        from voice_agent.pipeline import assemble_task

        conn = record.connection
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
        record.pipeline_task = task

        async def _run() -> None:
            runner = PipelineRunner(handle_sigint=False)
            try:
                await runner.run(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - one bad peer must not crash others
                self._log.error("webrtc_pipeline_error", pc_id=conn.pc_id, error=str(exc))

        record.runner_task = asyncio.create_task(_run())

    async def _cleanup(self, pc_id: str) -> None:
        """Tear down one connection's pipeline; leave the shared models loaded."""
        record = self._connections.pop(pc_id, None)
        if record is None:
            return
        if record.pipeline_task is not None:
            await record.pipeline_task.cancel()
        if record.runner_task is not None:
            record.runner_task.cancel()
        self._log.info("webrtc_connection_closed", pc_id=pc_id)

    async def close(self) -> None:
        """Close every live connection (called on app shutdown)."""
        for pc_id in list(self._connections):
            record = self._connections.get(pc_id)
            if record is not None:
                try:
                    await record.connection.disconnect()
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
