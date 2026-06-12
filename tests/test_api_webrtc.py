"""Tests for the WebRTC browser-audio signalling layer (/api/webrtc/offer).

The live audio path needs aiortc + a browser + GPU models and can't run
headlessly. What's covered here without the optional ``webrtc`` extra:

* config (``audio.ice_servers``),
* ``WebRTCManager.available()`` reporting the extra's absence,
* the endpoint returning a clear 503 when the extra is missing,
* request validation, and the ``create_app`` mount gate.

``WebRTCManager`` only touches its ``backends`` lazily (when a real connection
starts), so a placeholder stands in for ``SharedBackends`` here -- which keeps
this test off the pipeline/pyaudio import path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import EventBus
from voice_agent.api.webrtc import _Connection, WebRTCManager, create_webrtc_router
from voice_agent.config import parse_config

try:  # the "extra missing" tests only make sense without the webrtc extra
    import aiortc  # noqa: F401

    HAS_WEBRTC_EXTRA = True
except ImportError:
    HAS_WEBRTC_EXTRA = False

requires_no_webrtc_extra = pytest.mark.skipif(
    HAS_WEBRTC_EXTRA,
    reason="webrtc extra installed; the missing-extra 503 paths can't trigger",
)


def _config() -> Any:
    return parse_config(
        {
            "stt": {"model": "nvidia/parakeet-tdt-1.1b"},
            "tts": {"voice": "af_bella"},
            "llm": {
                "base_url": "http://llm:1234/v1",
                "model": "nvidia/nemotron-3-nano-4b",
            },
        }
    )


def _manager() -> WebRTCManager:
    # SharedBackends is only used lazily on a real connection; a placeholder is
    # enough for the availability/validation paths and avoids importing the
    # pipeline (and its pyaudio dependency).
    return WebRTCManager(SimpleNamespace(), _config())


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
        browser_audio=True,
    )


# ---------- config ------------------------------------------------------------


def test_audio_config_ice_defaults() -> None:
    cfg = parse_config(
        {
            "stt": {"model": "m"},
            "tts": {"voice": "v"},
            "llm": {"base_url": "http://x/v1", "model": "nvidia/nemotron-3-nano-4b"},
        }
    )
    assert cfg.audio.ice_servers == ["stun:stun.l.google.com:19302"]


def test_audio_config_custom_ice_servers() -> None:
    cfg = parse_config(
        {
            "stt": {"model": "m"},
            "tts": {"voice": "v"},
            "llm": {"base_url": "http://x/v1", "model": "nvidia/nemotron-3-nano-4b"},
            "audio": {"ice_servers": ["stun:example:3478"]},
        }
    )
    assert cfg.audio.ice_servers == ["stun:example:3478"]


# ---------- manager + router --------------------------------------------------


@requires_no_webrtc_extra
def test_manager_available_false_without_extra() -> None:
    # The sandbox has no aiortc, so the WebRTC stack reports unavailable.
    assert _manager().available() is False


@requires_no_webrtc_extra
def test_offer_returns_503_without_extra() -> None:
    app = FastAPI()
    app.include_router(create_webrtc_router(_manager()))
    res = TestClient(app).post(
        "/api/webrtc/offer", json={"sdp": "v=0...", "type": "offer"}
    )
    assert res.status_code == 503
    assert "webrtc" in res.json()["detail"].lower()


def test_offer_validates_request_body() -> None:
    app = FastAPI()
    app.include_router(create_webrtc_router(_manager()))
    # Missing required `sdp` -> 422 before the availability check.
    res = TestClient(app).post("/api/webrtc/offer", json={"type": "offer"})
    assert res.status_code == 422


# ---------- create_app mount gate ---------------------------------------------


@requires_no_webrtc_extra
def test_create_app_mounts_webrtc_when_manager_supplied() -> None:
    app = create_app(
        event_bus=EventBus(), session=_session(), webrtc_manager=_manager()
    )
    res = TestClient(app).post(
        "/api/webrtc/offer", json={"sdp": "v=0", "type": "offer"}
    )
    # Mounted, but the extra is absent in the sandbox -> 503 (not 404).
    assert res.status_code == 503


def test_create_app_omits_webrtc_by_default() -> None:
    app = create_app(event_bus=EventBus(), session=_session())
    res = TestClient(app).post(
        "/api/webrtc/offer", json={"sdp": "v=0", "type": "offer"}
    )
    assert res.status_code == 404


@pytest.mark.parametrize("body", [{}, {"sdp": "x"}, {"type": "offer"}])
def test_offer_request_requires_sdp_and_type(body: dict[str, Any]) -> None:
    app = FastAPI()
    app.include_router(create_webrtc_router(_manager()))
    assert TestClient(app).post("/api/webrtc/offer", json=body).status_code == 422


# ---------- connection teardown (no aiortc needed) ----------------------------
#
# These exercise the per-connection bookkeeping directly with stub tasks, so the
# record lifecycle is covered without a live peer connection.


class _PipelineTaskStub:
    """Stands in for a Pipecat PipelineTask -- async cancel()."""

    def __init__(self) -> None:
        self.cancelled = False

    async def cancel(self) -> None:
        self.cancelled = True


class _RunnerTaskStub:
    """Stands in for the asyncio runner task -- sync cancel()."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _ConnStub:
    def __init__(self) -> None:
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


async def test_cleanup_unknown_pc_id_is_noop() -> None:
    # A disconnect that fires for an already-removed (or never-registered)
    # connection must not raise -- guards the startup/teardown race.
    await _manager()._cleanup("ghost")


async def test_cleanup_cancels_tasks_and_drops_record() -> None:
    mgr = _manager()
    pipeline_task = _PipelineTaskStub()
    runner_task = _RunnerTaskStub()
    mgr._connections["pc1"] = _Connection(
        connection=_ConnStub(),
        pipeline_task=pipeline_task,
        runner_task=runner_task,
    )
    await mgr._cleanup("pc1")
    assert "pc1" not in mgr._connections
    assert pipeline_task.cancelled is True
    assert runner_task.cancelled is True


async def test_cleanup_tolerates_record_without_tasks() -> None:
    # The window between registering the record and _start_pipeline filling in
    # the tasks: a disconnect here must still drop the record cleanly.
    mgr = _manager()
    mgr._connections["pc1"] = _Connection(connection=_ConnStub())
    await mgr._cleanup("pc1")
    assert mgr._connections == {}


async def test_close_disconnects_and_clears_all() -> None:
    mgr = _manager()
    conn_a, conn_b = _ConnStub(), _ConnStub()
    mgr._connections["a"] = _Connection(
        connection=conn_a,
        pipeline_task=_PipelineTaskStub(),
        runner_task=_RunnerTaskStub(),
    )
    mgr._connections["b"] = _Connection(
        connection=conn_b,
        pipeline_task=_PipelineTaskStub(),
        runner_task=_RunnerTaskStub(),
    )
    await mgr.close()
    assert conn_a.disconnected is True
    assert conn_b.disconnected is True
    assert mgr._connections == {}
