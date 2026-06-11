"""Tests for /api/control endpoints and the MicGate processor.

The control router is exercised via the FastAPI test client with a list-append
stub standing in for the real ``PipelineTask.queue_frame`` callable -- this
keeps the tests hermetic (no Pipecat task to spin up) while still verifying
the exact frame the router would push onto the pipeline.

The MicGate processor is exercised separately by feeding it
``InputAudioRawFrame`` instances and asserting forwarding vs swallowing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.control import ControlState
from voice_agent.api.events import EventBus
from voice_agent.api.mic_gate import MicGate


def _session() -> SessionInfo:
    return SessionInfo(
        session_id="test-session",
        started_at="2026-05-21T00:00:00+00:00",
        stt_backend="parakeet_onnx",
        tts_backend="kokoro",
        vad_backend="silero",
        turn_backend="smart_turn_v3",
        simulator_backend="mock",
        llm_model="test/model",
    )


# ---------- helpers --------------------------------------------------------


class _InjectStub:
    """Captures every text injection the router triggers."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def __call__(self, text: str) -> None:
        self.texts.append(text)


def _build_app(
    state: ControlState | None = None, bus: EventBus | None = None
) -> tuple[TestClient, ControlState, EventBus, _InjectStub]:
    state = state or ControlState()
    bus = bus or EventBus()
    inject = _InjectStub()
    app = create_app(
        event_bus=bus,
        session=_session(),
        control_state=state,
        inject_text=inject,
    )
    return TestClient(app), state, bus, inject


# ---------- ControlState / state endpoint ----------------------------------


def test_control_state_defaults_to_mic_off():
    """A fresh ControlState starts with the mic off -- the cursist must enable
    it explicitly, after acknowledging the AI transparency gate."""
    assert ControlState().mic_enabled is False


def test_get_state_returns_current_mic_flag():
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=False))
    res = client.get("/api/control/state")
    assert res.status_code == 200
    assert res.json() == {"mic_enabled": False}


# ---------- POST /api/control/mic ------------------------------------------


def test_mic_toggle_flips_state():
    client, state, _, _ = _build_app(state=ControlState(mic_enabled=True))
    assert state.mic_enabled is True
    res = client.post("/api/control/mic", json={"enabled": False})
    assert res.status_code == 200
    assert res.json() == {"mic_enabled": False}
    assert state.mic_enabled is False


def test_mic_toggle_publishes_input_mode_changed_event():
    bus = EventBus()
    queue = bus.subscribe()
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=True), bus=bus)
    client.post("/api/control/mic", json={"enabled": False})
    # Drain exactly one event -- the queue is async but the publish was
    # sync so it's already enqueued.
    assert not queue.empty()
    event = queue.get_nowait()
    assert event.kind == "input_mode_changed"
    assert event.mic_enabled is False


def test_mic_toggle_idempotent_no_double_event():
    """Setting the flag to its current value should not spam the WS."""
    bus = EventBus()
    queue = bus.subscribe()
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=False), bus=bus)
    client.post("/api/control/mic", json={"enabled": False})
    assert queue.empty(), "no event should fire when the flag is unchanged"


def test_mic_toggle_rejects_unknown_fields():
    client, _, _, _ = _build_app()
    res = client.post("/api/control/mic", json={"enabled": True, "extra": "x"})
    assert res.status_code == 422


# ---------- POST /api/control/text -----------------------------------------


def test_text_send_returns_409_when_mic_enabled():
    client, _, _, stub = _build_app(state=ControlState(mic_enabled=True))
    res = client.post("/api/control/text", json={"text": "come to two seven zero"})
    assert res.status_code == 409
    assert "disable" in res.json()["detail"].lower()
    assert stub.texts == [], "no injection should happen when the mic is on"


def test_text_send_invokes_injector_with_trimmed_text():
    client, _, _, stub = _build_app(state=ControlState(mic_enabled=False))
    res = client.post(
        "/api/control/text", json={"text": "  come to two seven zero  "}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "queued"
    # The router trims surrounding whitespace before forwarding so the
    # context (and the conversation panel) get a clean string.
    assert stub.texts == ["come to two seven zero"]


def test_text_send_publishes_transcript_event():
    bus = EventBus()
    queue = bus.subscribe()
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=False), bus=bus)
    client.post("/api/control/text", json={"text": "hello"})
    assert not queue.empty()
    event = queue.get_nowait()
    assert event.kind == "transcript"
    assert event.text == "hello"


def test_text_send_rejects_blank_text():
    """Whitespace-only after Pydantic min_length should also 400."""
    client, _, _, stub = _build_app(state=ControlState(mic_enabled=False))
    res = client.post("/api/control/text", json={"text": "   "})
    # Pydantic accepts (min_length=1 satisfied by " "), so the explicit
    # blank-check in the handler kicks in.
    assert res.status_code == 400
    assert stub.texts == []


def test_text_send_rejects_empty_string():
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=False))
    res = client.post("/api/control/text", json={"text": ""})
    assert res.status_code == 422  # Pydantic min_length=1


def test_text_send_rejects_overlong_text():
    client, _, _, _ = _build_app(state=ControlState(mic_enabled=False))
    res = client.post(
        "/api/control/text", json={"text": "x" * 2001}
    )
    assert res.status_code == 422


# ---------- absent control plane keeps endpoints 404 -----------------------


def test_control_endpoints_404_when_control_state_missing():
    """create_app without control_state must not register /api/control."""
    app = create_app(event_bus=EventBus(), session=_session())
    with TestClient(app) as c:
        assert c.get("/api/control/state").status_code == 404
        assert c.post("/api/control/mic", json={"enabled": True}).status_code == 404


# ---------- browser-audio mode: control_state without inject_text -----------


def test_mic_toggle_works_without_text_injector():
    """Browser-audio mode passes control_state but no inject_text: the mic
    toggle must still mount and gate the per-connection MicGates. Regression
    for the mode being permanently muted with /api/control/mic returning 404."""
    state = ControlState(mic_enabled=False)
    app = create_app(event_bus=EventBus(), session=_session(), control_state=state)
    with TestClient(app) as c:
        assert c.get("/api/control/state").json() == {"mic_enabled": False}
        res = c.post("/api/control/mic", json={"enabled": True})
        assert res.status_code == 200
        assert state.mic_enabled is True


def test_text_send_503_without_text_injector():
    """No inject_text -> typed commands are unavailable (browser-audio mode),
    flagged with 503 rather than a silent 404."""
    state = ControlState(mic_enabled=False)
    app = create_app(event_bus=EventBus(), session=_session(), control_state=state)
    with TestClient(app) as c:
        res = c.post("/api/control/text", json={"text": "come to two seven zero"})
        assert res.status_code == 503
        assert "browser-audio" in res.json()["detail"]


# ---------- MicGate processor ----------------------------------------------


@pytest.mark.asyncio
async def test_mic_gate_drops_audio_when_disabled() -> None:
    state = ControlState(mic_enabled=False)
    gate = MicGate(state=state)
    pushed: list[Frame] = []

    async def fake_push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    # Swap the parent push implementation -- we don't run a full pipeline
    # in this test; we just want to assert the forwarding decision.
    gate.push_frame = fake_push  # type: ignore[method-assign]

    audio = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    await gate.process_frame(audio, FrameDirection.DOWNSTREAM)
    assert pushed == [], "audio frame must be swallowed while mic is disabled"


@pytest.mark.asyncio
async def test_mic_gate_forwards_audio_when_enabled() -> None:
    state = ControlState(mic_enabled=True)
    gate = MicGate(state=state)
    pushed: list[Frame] = []

    async def fake_push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    gate.push_frame = fake_push  # type: ignore[method-assign]

    audio = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    await gate.process_frame(audio, FrameDirection.DOWNSTREAM)
    assert pushed == [audio]


@pytest.mark.asyncio
async def test_mic_gate_passes_non_audio_through_even_when_disabled() -> None:
    """Only InputAudioRawFrame is gated; everything else must flow."""
    state = ControlState(mic_enabled=False)
    gate = MicGate(state=state)
    pushed: list[Frame] = []

    async def fake_push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    gate.push_frame = fake_push  # type: ignore[method-assign]

    text = TextFrame("hello")
    await gate.process_frame(text, FrameDirection.DOWNSTREAM)
    assert pushed == [text]


@pytest.mark.asyncio
async def test_mic_gate_state_can_flip_at_runtime() -> None:
    """Toggling the shared flag changes forwarding decisions on the fly."""
    state = ControlState(mic_enabled=True)
    gate = MicGate(state=state)
    pushed: list[Frame] = []

    async def fake_push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    gate.push_frame = fake_push  # type: ignore[method-assign]

    audio_1 = InputAudioRawFrame(audio=b"\x01\x00", sample_rate=16000, num_channels=1)
    audio_2 = InputAudioRawFrame(audio=b"\x02\x00", sample_rate=16000, num_channels=1)

    await gate.process_frame(audio_1, FrameDirection.DOWNSTREAM)
    state.mic_enabled = False
    await gate.process_frame(audio_2, FrameDirection.DOWNSTREAM)
    state.mic_enabled = True
    audio_3 = InputAudioRawFrame(audio=b"\x03\x00", sample_rate=16000, num_channels=1)
    await gate.process_frame(audio_3, FrameDirection.DOWNSTREAM)

    assert pushed == [audio_1, audio_3]


# ---------- spurious type-check sanity ------------------------------------


def test_text_send_dispatches_actual_callable_arg() -> None:
    """Sanity: the injector stub is what receives the text, not some
    global. Important because the router stores the callable by reference."""
    state = ControlState(mic_enabled=False)
    bus = EventBus()
    captured: list[str] = []

    async def my_inject(text: str) -> None:
        captured.append(text)

    app = create_app(
        event_bus=bus,
        session=_session(),
        control_state=state,
        inject_text=my_inject,
    )
    with TestClient(app) as c:
        c.post("/api/control/text", json={"text": "hi"})
    assert captured == ["hi"]


# ---------- build_text_injector (real injector contract) -------------------


@pytest.mark.asyncio
async def test_build_text_injector_appends_once_and_queues_run() -> None:
    """The real injector must add exactly one message to the *shared* context
    and queue exactly one LLMRunFrame. Regression for the double-append bug
    caused by using LLMMessagesAppendFrame (which both aggregators handled).
    """
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.processors.aggregators.llm_context import LLMContext

    from voice_agent.pipeline import build_text_injector

    context = LLMContext([{"role": "system", "content": "sys"}])

    queued: list[Any] = []

    class _TaskStub:
        async def queue_frame(self, frame: Any) -> None:
            queued.append(frame)

    inject = build_text_injector(_TaskStub(), context)
    await inject("come to two seven zero")

    # One message added, in the right role; system prompt still present.
    msgs = context.get_messages()
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "come to two seven zero"},
    ]
    # One LLMRunFrame queued -- not two, not the offending
    # LLMMessagesAppendFrame.
    assert len(queued) == 1
    assert isinstance(queued[0], LLMRunFrame)


# ---------- build_context_resetter ----------------------------------------


def test_build_context_resetter_keeps_system_prompt() -> None:
    """Reset wipes user/assistant turns but preserves the system prompt --
    that's what makes mode switching feel like 'fresh slate, same agent'."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    from voice_agent.pipeline import build_context_resetter

    context = LLMContext(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hallucinated noise"},
            {"role": "assistant", "content": "Sorry sir..."},
            {"role": "user", "content": "more noise"},
        ]
    )
    reset = build_context_resetter(context)
    reset()
    assert context.get_messages() == [{"role": "system", "content": "sys"}]


def test_build_context_resetter_clears_all_when_no_system_prompt() -> None:
    from pipecat.processors.aggregators.llm_context import LLMContext

    from voice_agent.pipeline import build_context_resetter

    context = LLMContext(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    reset = build_context_resetter(context)
    reset()
    assert context.get_messages() == []


# ---------- SingleTurnContextReset (tail-of-pipeline processor) ------------


@pytest.mark.asyncio
async def test_single_turn_reset_fires_on_assistant_timestamp() -> None:
    """The assistant-timestamp frame is the trigger: it's pushed downstream
    by the assistant aggregator *right after* it adds the assistant message
    to the shared context, so resetting here is safe and timely."""
    from pipecat.frames.frames import LLMContextAssistantTimestampFrame

    from voice_agent.pipeline import SingleTurnContextReset

    calls = 0
    pushed: list[Any] = []

    def reset() -> None:
        nonlocal calls
        calls += 1

    proc = SingleTurnContextReset(reset=reset)

    async def fake_push(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    proc.push_frame = fake_push  # type: ignore[method-assign]

    ts = LLMContextAssistantTimestampFrame(timestamp="2026-05-27T10:00:00Z")
    await proc.process_frame(ts, FrameDirection.DOWNSTREAM)
    assert calls == 1
    assert pushed == [ts], "the timestamp frame must still propagate downstream"


@pytest.mark.asyncio
async def test_single_turn_reset_does_not_fire_on_llm_end_frame() -> None:
    """Regression: the assistant aggregator swallows LLMFullResponseEndFrame
    so our processor never sees it. If a future Pipecat change starts
    forwarding it, that's fine -- but the *primary* trigger remains the
    timestamp frame; we don't want to double-fire."""
    from pipecat.frames.frames import LLMFullResponseEndFrame

    from voice_agent.pipeline import SingleTurnContextReset

    calls = 0

    def reset() -> None:
        nonlocal calls
        calls += 1

    proc = SingleTurnContextReset(reset=reset)

    async def fake_push(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        return None

    proc.push_frame = fake_push  # type: ignore[method-assign]

    await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert calls == 0


@pytest.mark.asyncio
async def test_single_turn_reset_ignores_other_frames() -> None:
    """Random frames (audio, text, etc.) must not trigger a reset."""
    from voice_agent.pipeline import SingleTurnContextReset

    calls = 0

    def reset() -> None:
        nonlocal calls
        calls += 1

    proc = SingleTurnContextReset(reset=reset)

    async def fake_push(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        return None

    proc.push_frame = fake_push  # type: ignore[method-assign]

    await proc.process_frame(TextFrame("hi"), FrameDirection.DOWNSTREAM)
    assert calls == 0


@pytest.mark.asyncio
async def test_single_turn_reset_ignores_upstream() -> None:
    """Timestamp frames travelling upstream are irrelevant; only the
    downstream emission from the assistant aggregator signals a completed
    assistant turn."""
    from pipecat.frames.frames import LLMContextAssistantTimestampFrame

    from voice_agent.pipeline import SingleTurnContextReset

    calls = 0

    def reset() -> None:
        nonlocal calls
        calls += 1

    proc = SingleTurnContextReset(reset=reset)

    async def fake_push(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        return None

    proc.push_frame = fake_push  # type: ignore[method-assign]

    await proc.process_frame(
        LLMContextAssistantTimestampFrame(timestamp="x"),
        FrameDirection.UPSTREAM,
    )
    assert calls == 0
