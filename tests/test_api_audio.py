"""Tests for the browser-audio WebSocket endpoint (/ws/audio).

Phase one is a raw-PCM loopback plus a JSON ``hello``/``ready`` handshake.
Exercised through FastAPI's ``TestClient`` websocket support -- no real audio
hardware or browser involved.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voice_agent.api.audio_ws import create_audio_router
from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import EventBus
from voice_agent.config import AudioConfig


def _app(audio: AudioConfig) -> FastAPI:
    app = FastAPI()
    app.include_router(create_audio_router(audio))
    return app


def test_loopback_echoes_pcm_frames() -> None:
    client = TestClient(_app(AudioConfig(browser_enabled=True)))
    with client.websocket_connect("/ws/audio") as ws:
        frame = bytes(range(256)) * 4  # 1024 bytes of fake PCM16
        ws.send_bytes(frame)
        assert ws.receive_bytes() == frame
        # A second frame round-trips too (the loop keeps running).
        ws.send_bytes(b"\x01\x02\x03\x04")
        assert ws.receive_bytes() == b"\x01\x02\x03\x04"


def test_hello_handshake_returns_ready_with_rate() -> None:
    client = TestClient(_app(AudioConfig(browser_enabled=True, sample_rate=16000)))
    with client.websocket_connect("/ws/audio") as ws:
        ws.send_json({"type": "hello", "sample_rate": 24000})
        ready = ws.receive_json()
        assert ready == {"type": "ready", "sample_rate": 24000, "mode": "loopback"}


def test_hello_without_rate_falls_back_to_config_rate() -> None:
    client = TestClient(_app(AudioConfig(browser_enabled=True, sample_rate=16000)))
    with client.websocket_connect("/ws/audio") as ws:
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["sample_rate"] == 16000
        # A bogus rate is ignored in favour of the configured one.
        ws.send_json({"type": "hello", "sample_rate": -5})
        assert ws.receive_json()["sample_rate"] == 16000


def test_unknown_control_message_is_ignored_then_audio_still_works() -> None:
    client = TestClient(_app(AudioConfig(browser_enabled=True)))
    with client.websocket_connect("/ws/audio") as ws:
        ws.send_json({"type": "nonsense"})  # no reply expected
        ws.send_bytes(b"abcd")
        assert ws.receive_bytes() == b"abcd"


# --- mounting gate (create_app) ---------------------------------------------


def _session() -> SessionInfo:
    return SessionInfo(
        session_id="t",
        started_at="2026-06-10T00:00:00+00:00",
        stt_backend="parakeet_onnx",
        tts_backend="kokoro",
        vad_backend="silero",
        turn_backend="smart_turn_v3",
        simulator_backend="mock",
        llm_model="test/model",
    )


def test_create_app_mounts_audio_when_enabled() -> None:
    app = create_app(
        event_bus=EventBus(),
        session=_session(),
        audio=AudioConfig(browser_enabled=True),
    )
    with TestClient(app).websocket_connect("/ws/audio") as ws:
        ws.send_bytes(b"xyz")
        assert ws.receive_bytes() == b"xyz"


def test_create_app_omits_audio_by_default() -> None:
    # No audio config -> route absent -> the handshake fails to connect.
    app = create_app(event_bus=EventBus(), session=_session())
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/audio"):
            pass


def test_create_app_omits_audio_when_disabled() -> None:
    app = create_app(
        event_bus=EventBus(),
        session=_session(),
        audio=AudioConfig(browser_enabled=False),
    )
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/audio"):
            pass
