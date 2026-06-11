"""Builds the Pipecat pipeline from a validated config object.

Pipeline order::

    transport.input() -> STT -> user aggregator -> LLM -> JsonActionProcessor
        -> TTS -> transport.output() -> assistant aggregator -> ConversationLogger

The LLM answers each command with a JSON object (see
:mod:`voice_agent.actions.schema`); :class:`JsonActionProcessor` parses it,
dispatches the action to the simulator, and forwards only the spoken response
to TTS. :class:`LatencyTracker` is attached as a Pipecat *observer* (not a
pipeline processor) so it can see frames consumed mid-pipeline — the
``TranscriptionFrame`` and the real streaming ``LLMTextFrame`` s never reach
the pipeline tail.

VAD and turn detection are wired into the **user context aggregator** (Pipecat
1.2.x) via ``LLMUserAggregatorParams``. The transport itself just streams audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTimestampFrame,
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voice_agent.actions.processor import JsonActionProcessor
from voice_agent.actions.prompt import SYSTEM_PROMPT
from voice_agent.api.events import EventBus, UserTranscriptObserver
from voice_agent.backends.llm.factory import create_llm
from voice_agent.backends.simulator.base import SimulatorClient
from voice_agent.backends.simulator.factory import create_simulator
from voice_agent.backends.stt.factory import create_stt
from voice_agent.backends.tts.factory import create_tts
from voice_agent.backends.turn.factory import create_turn
from voice_agent.backends.vad.factory import create_vad
from voice_agent.config import AppConfig
from voice_agent.logging_setup import get_logger
from voice_agent.metrics import ConversationLogger, LatencyTracker


@dataclass
class SharedBackends:
    """Per-pipeline service factories plus the truly shared singletons.

    Pipecat FrameProcessors are **single-use**: once a pipeline is cancelled
    (e.g. a browser-audio disconnect) every processor in it sets
    ``_cancelling`` and silently drops all frames forever -- so service
    *instances* must never be reused across pipeline assemblies. Instead each
    assembly calls these factories for fresh instances. The expensive part --
    the STT model load -- is cached at module level in the backend
    (see ``parakeet_onnx._MODEL_CACHE``), so a factory call after warm-up is
    cheap; VAD/turn/TTS/LLM construction is fast (~1s combined).

    The simulator (the one ship) and the event bus are genuinely shared
    singletons.
    """

    vad_factory: Callable[[], Any]
    turn_factory: Callable[[], Any]
    stt_factory: Callable[[], FrameProcessor]
    tts_factory: Callable[[], FrameProcessor]
    llm_factory: Callable[[], FrameProcessor]
    simulator: SimulatorClient
    event_bus: EventBus | None
    session_id: str


@dataclass
class BuiltPipeline:
    """The runnable pipeline plus resources the caller must clean up."""

    task: PipelineTask
    simulator: SimulatorClient
    session_id: str
    event_bus: EventBus | None
    # Shared LLM context. The user/assistant aggregator pair both reference
    # this same object; mutating it via add_message is how the control
    # router injects typed commands without going through the
    # LLMMessagesAppendFrame path (which both aggregators double-handle).
    llm_context: LLMContext
    # The heavy backends, exposed so browser-audio mode can assemble a fresh
    # per-connection pipeline against the same loaded models.
    backends: SharedBackends


def build_text_injector(
    task: PipelineTask, context: LLMContext
):
    """Return an async ``inject_text(text)`` for the control router.

    The injector appends one user message to the shared ``LLMContext`` and
    queues an ``LLMRunFrame`` to trigger inference. This avoids
    :class:`LLMMessagesAppendFrame`, which Pipecat 1.2 routes through both
    the user and assistant aggregators -- each calls ``add_messages`` on
    the same context, so the message lands twice and the LLM runs twice.
    Mutating the context once + a single ``LLMRunFrame`` is the simplest
    path that mirrors the voice-input flow's end state.
    """

    async def inject(text: str) -> None:
        context.add_message({"role": "user", "content": text})
        await task.queue_frame(LLMRunFrame())

    return inject


class SingleTurnContextReset(FrameProcessor):
    """Wipe conversation history after every assistant turn.

    The helmsman is a single-turn command parser, not a chat assistant: each
    helm order is independent and must be evaluated against the system prompt
    alone, never against accumulated history. Single-turn mode is what makes
    STT hallucinations harmless -- a garbage transcript fails its own turn
    and disappears, instead of poisoning the next real command.

    Sits at the *tail* of the pipeline, after the assistant aggregator has
    finalised the spoken reply into the context. The trigger is
    :class:`LLMContextAssistantTimestampFrame`, which Pipecat's assistant
    aggregator pushes downstream *immediately after* it calls
    ``context.add_message({role: assistant, ...})``. That ordering is
    important: by the time we see this frame, the assistant message is
    already on the context, so a wipe-to-system here is safe.

    We avoid :class:`LLMFullResponseEndFrame` because the assistant
    aggregator *swallows* it -- handles the event internally and never
    pushes it downstream -- so a tail-of-pipeline processor never sees it.
    """

    def __init__(self, *, reset: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reset = reset

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, LLMContextAssistantTimestampFrame
        ):
            # Reset is a plain sync mutation on the shared list -- safe to
            # call inline. Done *after* push_frame so any further
            # downstream processors see the timestamp frame before we wipe.
            self._reset()


def build_context_resetter(context: LLMContext):
    """Return a ``reset()`` that wipes the conversation back to the system prompt.

    Called when the user flips input modes. STT hallucinations on quiet/
    noisy audio accumulate in the shared context while the mic is open,
    and they poison subsequent text turns ("Sorry sir" loops). Wiping on
    every mode change makes the mental model match the UX: switching
    modalities = fresh slate.

    The system prompt (assumed to be the first message in the context)
    survives; everything after it is dropped.
    """

    def reset() -> None:
        messages = list(context.messages)
        if messages and messages[0].get("role") == "system":
            context.set_messages([messages[0]])
        else:
            context.set_messages([])

    return reset


def build_shared_backends(config: AppConfig, session_id: str) -> SharedBackends:
    """Build the per-pipeline service factories and shared singletons.

    Every :func:`assemble_task` call -- one for local audio, one per WebRTC
    connection -- gets fresh service instances from the factories (cancelled
    Pipecat processors are unusable; see :class:`SharedBackends`). The heavy
    STT model load is warmed here so the first connection doesn't pay it.
    """
    log = get_logger("pipeline")

    # --- per-pipeline service factories ----------------------------------
    def vad_factory() -> Any:
        return create_vad(config.vad)

    def turn_factory() -> Any:
        return create_turn(config.turn_detection)

    def stt_factory() -> FrameProcessor:
        return create_stt(config.stt)

    def tts_factory() -> FrameProcessor:
        return create_tts(config.tts)

    # LLM backend is chosen by config.llm.backend: openai_compatible (LM Studio
    # + JSON-schema response_format, command-only) or langgraph (in-backend
    # command + RAG questions). Both slot into the same pipeline position.
    def llm_factory() -> FrameProcessor:
        return create_llm(config.llm)

    # Warm the model caches / downloads at startup rather than on the first
    # connection: the STT factory populates the module-level model cache, the
    # TTS factory triggers Kokoro's model-file download + provider env setup.
    # The throwaway service wrappers are cheap.
    stt_factory()
    tts_factory()

    # --- event bus (frontend observability) -----------------------------
    # Created only when the API is enabled. Processors/observers below take
    # ``event_bus=None`` as a no-op, keeping the agent runnable headless.
    event_bus: EventBus | None = EventBus() if config.api.enabled else None

    # --- simulator ------------------------------------------------------
    # One SimulatorClient, built once (the ship is a single shared entity).
    simulator = create_simulator(config.simulator)

    log.info(
        "shared_backends_built",
        session_id=session_id,
        stt=config.stt.backend,
        tts=config.tts.backend,
        vad=config.vad.backend,
        turn=config.turn_detection.backend,
        simulator=config.simulator.backend,
        llm_model=config.llm.model,
        api_enabled=config.api.enabled,
    )
    return SharedBackends(
        vad_factory=vad_factory,
        turn_factory=turn_factory,
        stt_factory=stt_factory,
        tts_factory=tts_factory,
        llm_factory=llm_factory,
        simulator=simulator,
        event_bus=event_bus,
        session_id=session_id,
    )


def assemble_task(
    backends: SharedBackends,
    config: AppConfig,
    transport: Any,
) -> tuple[PipelineTask, LLMContext]:
    """Wire the shared backends + a transport into a runnable task.

    Builds the cheap, per-run pieces (context, aggregators, action processor,
    monitors, single-turn reset) around the shared heavy backends, slotting the
    given ``transport`` at the head and tail. Called once per WebRTC browser
    connection.

    There is no server-side mic gate: a browser-audio pipeline is gated by the
    connection itself -- the user explicitly connects/disconnects browser audio
    in the dashboard, so a second server-side mute would only mean a
    deaf-looking agent.
    """
    event_bus = backends.event_bus
    json_action = JsonActionProcessor(
        simulator=backends.simulator, event_bus=event_bus
    )

    # Fresh service instances for this pipeline (see SharedBackends: cancelled
    # Pipecat processors can never be reused, so nothing audio-facing is
    # shared between assemblies).
    stt = backends.stt_factory()
    tts = backends.tts_factory()
    llm = backends.llm_factory()

    # --- context aggregator (carries VAD + turn detection) --------------
    # No tools are declared: the agent uses JSON structured output, not native
    # tool calls (a small local model emits constrained JSON far more reliably).
    # AlwaysUserMuteStrategy suppresses STT input while the bot is speaking, so
    # the agent does not transcribe its own TTS as a phantom command.
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=backends.vad_factory(),
            user_turn_strategies=UserTurnStrategies(
                stop=[backends.turn_factory()]
            ),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # --- monitors -------------------------------------------------------
    latency_tracker = LatencyTracker(
        session_id=backends.session_id,
        metrics_dir=config.logging.metrics_log_path,
        event_bus=event_bus,
    )
    conversation_logger = ConversationLogger(
        session_id=backends.session_id,
        conversation_dir=config.logging.conversation_log_path,
    )
    observers = [latency_tracker]
    if event_bus is not None:
        observers.append(UserTranscriptObserver(event_bus=event_bus))

    # Single-turn mode: wipe history after each assistant turn so the LLM only
    # ever sees [system, current_user_input].
    single_turn_reset = SingleTurnContextReset(reset=build_context_resetter(context))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            json_action,
            tts,
            transport.output(),
            context_aggregator.assistant(),
            conversation_logger,
            single_turn_reset,
        ]
    )
    # idle_timeout_secs=None disables Pipecat's 5-minute idle cancel -- a
    # helmsman is silent between commands, and long quiet stretches should not
    # tear the pipeline down.
    task = PipelineTask(pipeline, observers=observers, idle_timeout_secs=None)
    return task, context


def assemble_text_task(
    backends: SharedBackends,
    config: AppConfig,
) -> tuple[PipelineTask, LLMContext]:
    """Wire a standing text-only pipeline: chatbox commands, no audio at all.

    Browser-audio mode has no single local audio pipeline, but the dashboard
    chatbox still needs a task to inject typed commands into. This is the
    same proven shape as ``scripts/smoke.py`` -- user aggregator -> LLM ->
    JsonActionProcessor -> assistant aggregator -- plus the conversation
    logger and single-turn reset. No transport, no STT/TTS: the action drives
    the simulator and the reply surfaces in the transcript panel via the
    event bus.

    Runs alongside the per-connection WebRTC pipelines against the same
    simulator and event bus; the per-turn context wipe keeps typed and spoken
    turns from bleeding into each other. Like every assembly it gets its own
    LLM service instance from the factory -- this pipeline runs concurrently
    with the per-connection ones, and a FrameProcessor can only be linked
    into one pipeline at a time.
    """
    event_bus = backends.event_bus
    json_action = JsonActionProcessor(
        simulator=backends.simulator, event_bus=event_bus
    )
    llm = backends.llm_factory()
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    latency_tracker = LatencyTracker(
        session_id=backends.session_id,
        metrics_dir=config.logging.metrics_log_path,
        event_bus=event_bus,
    )
    conversation_logger = ConversationLogger(
        session_id=backends.session_id,
        conversation_dir=config.logging.conversation_log_path,
    )
    single_turn_reset = SingleTurnContextReset(reset=build_context_resetter(context))

    pipeline = Pipeline(
        [
            context_aggregator.user(),
            llm,
            json_action,
            context_aggregator.assistant(),
            conversation_logger,
            single_turn_reset,
        ]
    )
    task = PipelineTask(
        pipeline, observers=[latency_tracker], idle_timeout_secs=None
    )
    return task, context
