"""Tests for the /api/control text-injection endpoint and pipeline helpers.

The control router is exercised via the FastAPI test client with a list-append
stub standing in for the real ``PipelineTask.queue_frame`` callable -- this
keeps the tests hermetic (no Pipecat task to spin up) while still verifying
the exact frame the router would push onto the pipeline.

Voice input is the browser-audio (WebRTC) path; the only control route is the
dashboard chatbox (``POST /api/control/text``).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import EventBus


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
    bus: EventBus | None = None,
) -> tuple[TestClient, EventBus, _InjectStub]:
    bus = bus or EventBus()
    inject = _InjectStub()
    app = create_app(event_bus=bus, session=_session(), inject_text=inject)
    return TestClient(app), bus, inject


# ---------- POST /api/control/text -----------------------------------------


def test_text_send_invokes_injector_with_trimmed_text():
    client, _, stub = _build_app()
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
    client, _, _ = _build_app(bus=bus)
    client.post("/api/control/text", json={"text": "hello"})
    assert not queue.empty()
    event = queue.get_nowait()
    assert event.kind == "transcript"
    assert event.text == "hello"


def test_text_send_rejects_blank_text():
    """Whitespace-only after Pydantic min_length should also 400."""
    client, _, stub = _build_app()
    res = client.post("/api/control/text", json={"text": "   "})
    # Pydantic accepts (min_length=1 satisfied by " "), so the explicit
    # blank-check in the handler kicks in.
    assert res.status_code == 400
    assert stub.texts == []


def test_text_send_rejects_empty_string():
    client, _, _ = _build_app()
    res = client.post("/api/control/text", json={"text": ""})
    assert res.status_code == 422  # Pydantic min_length=1


def test_text_send_rejects_overlong_text():
    client, _, _ = _build_app()
    res = client.post("/api/control/text", json={"text": "x" * 2001})
    assert res.status_code == 422


def test_text_send_rejects_unknown_fields():
    client, _, _ = _build_app()
    res = client.post("/api/control/text", json={"text": "hi", "extra": "x"})
    assert res.status_code == 422


# ---------- absent text injector keeps the endpoint 404 --------------------


def test_control_endpoint_404_without_text_injector():
    """create_app without inject_text must not register /api/control."""
    app = create_app(event_bus=EventBus(), session=_session())
    with TestClient(app) as c:
        assert c.post("/api/control/text", json={"text": "hi"}).status_code == 404


def test_text_send_dispatches_actual_callable_arg() -> None:
    """Sanity: the injector stub is what receives the text, not some
    global. Important because the router stores the callable by reference."""
    bus = EventBus()
    captured: list[str] = []

    async def my_inject(text: str) -> None:
        captured.append(text)

    app = create_app(event_bus=bus, session=_session(), inject_text=my_inject)
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
