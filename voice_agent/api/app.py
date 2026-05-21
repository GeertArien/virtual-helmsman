"""FastAPI app: HTTP snapshot endpoints + WebSocket event stream.

The app is built by :func:`create_app` with the live :class:`EventBus` and a
:class:`SessionInfo` snapshot. Endpoints:

* ``GET /api/health``  -- liveness ping.
* ``GET /api/session`` -- session id, start time, configured backends. Lets the
  frontend identify the run without waiting for an event.
* ``WS  /ws/events``   -- subscribes the client to the bus and streams every
  event as one JSON object per message.

CORS is permissive by default (``allow_origins=["*"]``) because the frontend's
Vite dev server runs on a different port from the agent; tighten via config in
production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from voice_agent.api.events import EventBus
from voice_agent.logging_setup import get_logger


@dataclass
class SessionInfo:
    """Static identity of a running pipeline session, exposed via /api/session."""

    session_id: str
    started_at: str
    stt_backend: str
    tts_backend: str
    vad_backend: str
    turn_backend: str
    simulator_backend: str
    llm_model: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    *,
    event_bus: EventBus,
    session: SessionInfo,
    cors_allow_origins: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to a live ``event_bus`` and session.

    The app holds no global state -- everything closes over the arguments.
    Re-creating the app per session is therefore cheap and isolation between
    sessions is clean.
    """
    app = FastAPI(title="virtual-helmsman", version="0.1.0")
    log = get_logger("api")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "ts": _now_iso()}

    @app.get("/api/session")
    async def session_endpoint() -> dict[str, object]:
        return {
            **asdict(session),
            "subscribers": event_bus.subscriber_count,
            "events_dropped": event_bus.dropped,
        }

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        queue = event_bus.subscribe()
        log.info("ws_connected", subscribers=event_bus.subscriber_count)
        try:
            while True:
                event = await queue.get()
                # Pydantic models serialise via model_dump_json -- includes the
                # ``kind`` discriminator so the client can switch on it.
                await ws.send_text(event.model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(queue)
            log.info("ws_disconnected", subscribers=event_bus.subscriber_count)

    return app
