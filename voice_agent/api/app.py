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

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from voice_agent.api.config_router import create_config_router
from voice_agent.api.control_router import TextInjector, create_control_router
from voice_agent.api.documents import create_documents_router
from voice_agent.api.events import EventBus
from voice_agent.api.review import create_review_router
from voice_agent.api.webrtc import WebRTCManager, create_webrtc_router
from voice_agent.backends.simulator.base import SimulatorClient
from voice_agent.config import DocumentsRuntime, IngestionRuntime
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
    # Always true: browser audio (WebRTC) is the only voice path, so the
    # dashboard always offers browser-side mic capture + playback. Retained as
    # a field for the frontend's session snapshot.
    browser_audio: bool = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    *,
    event_bus: EventBus,
    session: SessionInfo,
    cors_allow_origins: list[str] | None = None,
    documents: DocumentsRuntime | None = None,
    review: IngestionRuntime | None = None,
    inject_text: TextInjector | None = None,
    simulator: SimulatorClient | None = None,
    config_path: Path | None = None,
    llm_model: str | None = None,
    webrtc_manager: WebRTCManager | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to a live ``event_bus`` and session.

    The app holds no global state -- everything closes over the arguments.
    Re-creating the app per session is therefore cheap and isolation between
    sessions is clean.

    Passing ``documents`` mounts the qdrant management routes; passing
    ``review`` mounts the in-backend HITL review routes. When either is
    omitted, that family of endpoints simply isn't registered (the frontend
    gets a 404 rather than a configuration error).

    ``inject_text`` mounts the ``/api/control`` router (``POST
    /api/control/text``, the dashboard chatbox). Omitted when there is no
    pipeline task to inject into; tests pass a list-append stub for
    ``inject_text`` without standing up a real pipeline task.

    Passing ``config_path`` mounts ``/api/config`` (view + edit ``config.yaml``
    and trigger a process reload). Omit it in tests that don't need to round-
    trip through disk.
    """
    log = get_logger("api")

    # Each optional router owns a long-lived httpx.AsyncClient that must be
    # closed at shutdown. We build them before the app so the lifespan
    # handler can reference them without an attribute dance.
    docs_router: APIRouter | None = None
    if documents is not None:
        docs_router = create_documents_router(documents)
    review_router: APIRouter | None = None
    if review is not None:
        # llm_model is the default ``Model`` for ingestion uploads -- keeps
        # the doc-summary call on the same model the helmsman LLM path uses.
        review_router = create_review_router(review, llm_model=llm_model)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for r in (docs_router, review_router):
                if r is None:
                    continue
                client = getattr(r, "_http_client", None)
                if client is not None:
                    await client.aclose()
            if webrtc_manager is not None:
                await webrtc_manager.close()

    app = FastAPI(title="virtual-helmsman", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if docs_router is not None:
        app.include_router(docs_router)
    if review_router is not None:
        app.include_router(review_router)
    if inject_text is not None:
        app.include_router(
            create_control_router(
                event_bus=event_bus, inject_text=inject_text, simulator=simulator
            )
        )
    if config_path is not None:
        app.include_router(create_config_router(config_path=config_path))
    # Browser-audio (WebRTC) signalling -- mounted when a manager is supplied
    # (the API is enabled). Browser audio is the only voice input path.
    if webrtc_manager is not None:
        app.include_router(create_webrtc_router(webrtc_manager))

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
