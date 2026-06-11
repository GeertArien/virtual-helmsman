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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voice_agent.actions.processor import JsonActionProcessor
from voice_agent.actions.prompt import SYSTEM_PROMPT
from voice_agent.api.control import ControlState
from voice_agent.api.events import EventBus, UserTranscriptObserver
from voice_agent.api.mic_gate import MicGate
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
    """The expensive, reusable pieces of the agent, built once at startup.

    The model-loading backends (VAD, turn detector, STT, TTS) and the LLM, the
    one simulator, and the API-facing event bus / control state are constructed
    once and reused across every pipeline assembly. For the default local-audio
    run that's a single pipeline; for browser (WebRTC) audio it's one pipeline
    per connection -- so the heavy models load once and are shared rather than
    reloaded per connect (issue #7, "shared models").
    """

    vad: Any
    turn_stop_strategy: Any
    stt: FrameProcessor
    tts: FrameProcessor
    llm: FrameProcessor
    simulator: SimulatorClient
    event_bus: EventBus | None
    control_state: ControlState | None
    session_id: str


@dataclass
class BuiltPipeline:
    """The runnable pipeline plus resources the caller must clean up."""

    task: PipelineTask
    simulator: SimulatorClient
    session_id: str
    event_bus: EventBus | None
    # Shared mic-on/off flag; the API mutates it, the pipeline's MicGate
    # reads it on every audio frame. ``None`` when the API is disabled --
    # the gate is then absent from the pipeline entirely.
    control_state: ControlState | None
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
    """Build the heavy, reusable backends once (models, LLM, simulator, bus).

    These are shared across every :func:`assemble_task` call -- one for local
    audio, one per WebRTC connection -- so the models load a single time.
    """
    log = get_logger("pipeline")

    # --- local models ---------------------------------------------------
    vad = create_vad(config.vad)
    turn_stop_strategy = create_turn(config.turn_detection)
    stt = create_stt(config.stt)
    tts = create_tts(config.tts)
    # LLM backend is chosen by config.llm.backend: openai_compatible (LM Studio
    # + JSON-schema response_format, command-only) or langgraph (in-backend
    # command + RAG questions). Both slot into the same pipeline position.
    llm = create_llm(config.llm)

    # --- event bus (frontend observability) -----------------------------
    # Created only when the API is enabled. Processors/observers below take
    # ``event_bus=None`` as a no-op, keeping the agent runnable headless.
    event_bus: EventBus | None = EventBus() if config.api.enabled else None

    # --- control state (mic gate) ---------------------------------------
    # Only meaningful when the API is enabled -- nothing else can flip the flag.
    control_state: ControlState | None = (
        ControlState() if config.api.enabled else None
    )

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
        vad=vad,
        turn_stop_strategy=turn_stop_strategy,
        stt=stt,
        tts=tts,
        llm=llm,
        simulator=simulator,
        event_bus=event_bus,
        control_state=control_state,
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
    given ``transport`` at the head and tail. Used for the local-audio pipeline
    and for each per-connection WebRTC pipeline.
    """
    event_bus = backends.event_bus
    json_action = JsonActionProcessor(
        simulator=backends.simulator, event_bus=event_bus
    )

    # --- context aggregator (carries VAD + turn detection) --------------
    # No tools are declared: the agent uses JSON structured output, not native
    # tool calls (a small local model emits constrained JSON far more reliably).
    # AlwaysUserMuteStrategy suppresses STT input while the bot is speaking, so
    # the agent does not transcribe its own TTS as a phantom command.
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=backends.vad,
            user_turn_strategies=UserTurnStrategies(
                stop=[backends.turn_stop_strategy]
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

    # MicGate sits between the transport mic and STT so it can drop inbound
    # audio frames before they hit the recogniser. Omitted in headless mode.
    mic_gate = (
        MicGate(state=backends.control_state)
        if backends.control_state is not None
        else None
    )

    # Single-turn mode: wipe history after each assistant turn so the LLM only
    # ever sees [system, current_user_input].
    single_turn_reset = SingleTurnContextReset(reset=build_context_resetter(context))

    pipeline_processors = [transport.input()]
    if mic_gate is not None:
        pipeline_processors.append(mic_gate)
    pipeline_processors += [
        backends.stt,
        context_aggregator.user(),
        backends.llm,
        json_action,
        backends.tts,
        transport.output(),
        context_aggregator.assistant(),
        conversation_logger,
        single_turn_reset,
    ]
    pipeline = Pipeline(pipeline_processors)
    # idle_timeout_secs=None disables Pipecat's 5-minute idle cancel -- a
    # helmsman is silent between commands, and long quiet stretches should not
    # tear the pipeline down.
    task = PipelineTask(pipeline, observers=observers, idle_timeout_secs=None)
    return task, context


def build_pipeline(config: AppConfig, session_id: str) -> BuiltPipeline:
    """Construct the default local-audio pipeline and supporting objects.

    Builds the shared backends and assembles a single task bound to the local
    hardware transport -- the behaviour the CLI has always had.
    """
    backends = build_shared_backends(config, session_id)
    # TODO: config.audio.input_device/output_device are accepted but not yet
    # mapped to device indices; the OS default device is used.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    task, context = assemble_task(backends, config, transport)
    return BuiltPipeline(
        task=task,
        simulator=backends.simulator,
        session_id=session_id,
        event_bus=backends.event_bus,
        control_state=backends.control_state,
        llm_context=context,
        backends=backends,
    )
