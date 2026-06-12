"""Tests for the in-process event bus, event schemas, and the FastAPI app.

No network: the FastAPI app is exercised via Starlette's TestClient.
No GPU/audio: the ``UserTranscriptObserver`` is fed synthetic frames.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import (
    ActionDispatchedEvent,
    AssistantReplyEvent,
    EventBus,
    ShipStateEvent,
    TranscriptEvent,
    TurnMetricsEvent,
    UserTranscriptObserver,
)


# -- Event schemas -----------------------------------------------------------


def test_event_serializes_with_kind_discriminator() -> None:
    """Every event JSON-roundtrips with its ``kind`` field set."""
    ev = TranscriptEvent(text="full ahead")
    payload = json.loads(ev.model_dump_json())
    assert payload["kind"] == "transcript"
    assert payload["text"] == "full ahead"
    assert "ts" in payload


def test_each_event_kind_is_unique() -> None:
    """Discriminator values must not collide -- the client switches on them."""
    samples = [
        TranscriptEvent(text="x"),
        AssistantReplyEvent(text="x"),
        ActionDispatchedEvent(action="navigation", details={"course": 90}),
        ShipStateEvent(heading_deg=90, speed_kn=12, engine_order="full_ahead"),
        TurnMetricsEvent(turn_index=0, metrics_ms={"voice_to_voice_ms": 1500}),
    ]
    kinds = {ev.kind for ev in samples}  # type: ignore[attr-defined]
    assert len(kinds) == len(samples)


# -- EventBus pub/sub --------------------------------------------------------


async def test_publish_fans_out_to_every_subscriber() -> None:
    bus = EventBus()
    a = bus.subscribe()
    b = bus.subscribe()
    assert bus.subscriber_count == 2

    bus.publish(TranscriptEvent(text="hello"))

    ev_a = await asyncio.wait_for(a.get(), timeout=0.1)
    ev_b = await asyncio.wait_for(b.get(), timeout=0.1)
    assert isinstance(ev_a, TranscriptEvent) and ev_a.text == "hello"
    assert isinstance(ev_b, TranscriptEvent) and ev_b.text == "hello"


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    assert bus.subscriber_count == 0
    bus.publish(TranscriptEvent(text="lost"))
    assert q.empty()


async def test_full_queue_drops_only_that_subscriber() -> None:
    """A slow subscriber must not block the bus or other subscribers."""
    bus = EventBus(per_subscriber_queue_size=1)
    slow = bus.subscribe()
    fast = bus.subscribe()

    bus.publish(TranscriptEvent(text="1"))  # both queues hold one
    bus.publish(TranscriptEvent(text="2"))  # both queues full -> drop x2

    assert bus.dropped == 2
    # Fast subscriber drains and keeps receiving.
    assert (await fast.get()).text == "1"  # type: ignore[union-attr]
    bus.publish(TranscriptEvent(text="3"))
    assert (await fast.get()).text == "3"  # type: ignore[union-attr]
    # Slow subscriber still only has the very first event.
    assert (await slow.get()).text == "1"  # type: ignore[union-attr]


# -- UserTranscriptObserver --------------------------------------------------


def _push(frame: object) -> FramePushed:
    """Build a minimal FramePushed -- source/destination are unused by the observer."""
    return FramePushed(
        source=None,  # type: ignore[arg-type]
        destination=None,  # type: ignore[arg-type]
        frame=frame,  # type: ignore[arg-type]
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


async def test_observer_publishes_transcript_only() -> None:
    bus = EventBus()
    q = bus.subscribe()
    obs = UserTranscriptObserver(event_bus=bus)

    await obs.on_push_frame(_push(TranscriptionFrame("steer course 270", "", "", None)))
    await obs.on_push_frame(_push(TextFrame("not a transcript")))

    ev = await asyncio.wait_for(q.get(), timeout=0.1)
    assert isinstance(ev, TranscriptEvent)
    assert ev.text == "steer course 270"
    assert q.empty()


async def test_observer_ignores_upstream_and_dedupes() -> None:
    bus = EventBus()
    q = bus.subscribe()
    obs = UserTranscriptObserver(event_bus=bus)

    frame = TranscriptionFrame("aye sir", "", "", None)
    upstream = FramePushed(
        source=None,  # type: ignore[arg-type]
        destination=None,  # type: ignore[arg-type]
        frame=frame,
        direction=FrameDirection.UPSTREAM,
        timestamp=0,
    )
    await obs.on_push_frame(upstream)  # ignored: upstream
    assert q.empty()

    await obs.on_push_frame(_push(frame))  # first downstream sighting -> publish
    await obs.on_push_frame(_push(frame))  # repeat sighting -> dedup, no second event
    ev = await asyncio.wait_for(q.get(), timeout=0.1)
    assert ev.text == "aye sir"  # type: ignore[union-attr]
    assert q.empty()


async def test_observer_drops_empty_transcripts() -> None:
    """STT can emit whitespace-only frames; those are not user speech."""
    bus = EventBus()
    q = bus.subscribe()
    obs = UserTranscriptObserver(event_bus=bus)

    await obs.on_push_frame(_push(TranscriptionFrame("   ", "", "", None)))
    assert q.empty()


# -- FastAPI app -------------------------------------------------------------


def _session() -> SessionInfo:
    return SessionInfo(
        session_id="sess-1",
        started_at="2026-05-20T00:00:00+00:00",
        stt_backend="parakeet_onnx",
        tts_backend="kokoro",
        vad_backend="silero",
        turn_backend="smart_turn_v3",
        simulator_backend="mock",
        llm_model="nvidia/nemotron-3-nano-4b",
    )


def test_health_endpoint_returns_ok() -> None:
    app = create_app(event_bus=EventBus(), session=_session())
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_session_endpoint_reports_config_and_bus_counters() -> None:
    bus = EventBus()
    app = create_app(event_bus=bus, session=_session())
    with TestClient(app) as client:
        r = client.get("/api/session")
    body = r.json()
    assert body["session_id"] == "sess-1"
    assert body["llm_model"] == "nvidia/nemotron-3-nano-4b"
    assert body["simulator_backend"] == "mock"
    assert body["subscribers"] == 0
    assert body["events_dropped"] == 0


def test_websocket_streams_published_events() -> None:
    """A client connecting to /ws/events receives every event published after it subscribed."""
    bus = EventBus()
    app = create_app(event_bus=bus, session=_session())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            # Subscription is async inside the endpoint; nudge the loop.
            for _ in range(50):
                if bus.subscriber_count == 1:
                    break
                ws.send_text("ping")  # no-op; just yields to the server task
            bus.publish(AssistantReplyEvent(text="aye"))
            bus.publish(ShipStateEvent(heading_deg=270, speed_kn=12, engine_order="full_ahead"))

            payload = json.loads(ws.receive_text())
            assert payload["kind"] == "assistant_reply"
            assert payload["text"] == "aye"

            payload = json.loads(ws.receive_text())
            assert payload["kind"] == "ship_state"
            assert payload["heading_deg"] == 270


# -- Config integration ------------------------------------------------------


def test_default_app_config_has_api_disabled() -> None:
    """A config without an ``api`` block should still validate; default disabled."""
    from voice_agent.config import parse_config

    minimal = {
        "stt": {"model": "x"},
        "tts": {"voice": "y"},
        "llm": {"model": "nvidia/nemotron-3-nano-4b"}, "lm_studio": {"base_url": "http://x"},
    }
    cfg = parse_config(minimal)
    assert cfg.api.enabled is False
    assert cfg.api.port == 8765


def test_api_config_round_trips() -> None:
    from voice_agent.config import parse_config

    raw = {
        "stt": {"model": "x"},
        "tts": {"voice": "y"},
        "llm": {"model": "nvidia/nemotron-3-nano-4b"}, "lm_studio": {"base_url": "http://x"},
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 9000,
            "cors_allow_origins": ["http://localhost:5173"],
        },
    }
    cfg = parse_config(raw)
    assert cfg.api.enabled is True
    assert cfg.api.host == "0.0.0.0"
    assert cfg.api.port == 9000
    assert cfg.api.cors_allow_origins == ["http://localhost:5173"]


def test_api_config_rejects_unknown_field() -> None:
    """``extra=forbid`` keeps typos from being silently ignored."""
    from voice_agent.config import parse_config

    raw = {
        "stt": {"model": "x"},
        "tts": {"voice": "y"},
        "llm": {"model": "nvidia/nemotron-3-nano-4b"}, "lm_studio": {"base_url": "http://x"},
        "api": {"enabld": True},  # typo
    }
    with pytest.raises(Exception):
        parse_config(raw)
