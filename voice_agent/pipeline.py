"""Builds the Pipecat pipeline from a validated config object.

Pipeline order::

    transport.input() -> STT -> user aggregator -> LLM -> JsonActionProcessor
        -> TTS -> transport.output() -> assistant aggregator
        -> LatencyTracker -> ConversationLogger

The LLM answers each command with a JSON object (see
:mod:`voice_agent.actions.schema`); :class:`JsonActionProcessor` parses it,
dispatches the action to the simulator, and forwards only the spoken response
to TTS. The two observer processors sit last so they see the full frame flow.

VAD and turn detection are wired into the **user context aggregator** (Pipecat
1.2.x) via ``LLMUserAggregatorParams``. The transport itself just streams audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voice_agent.actions.processor import JsonActionProcessor
from voice_agent.actions.prompt import SYSTEM_PROMPT
from voice_agent.actions.schema import RESPONSE_FORMAT
from voice_agent.backends.llm.openai_compatible import build_llm
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
class BuiltPipeline:
    """The runnable pipeline plus resources the caller must clean up."""

    task: PipelineTask
    simulator: SimulatorClient
    session_id: str


def build_pipeline(config: AppConfig, session_id: str) -> BuiltPipeline:
    """Construct the Pipecat pipeline and supporting objects from config."""
    log = get_logger("pipeline")

    # --- local models ---------------------------------------------------
    vad = create_vad(config.vad)
    turn_stop_strategy = create_turn(config.turn_detection)
    stt = create_stt(config.stt)
    tts = create_tts(config.tts)
    # response_format constrains the LLM to emit the helmsman JSON object.
    llm = build_llm(config.llm, extra={"response_format": RESPONSE_FORMAT})

    # --- transport ------------------------------------------------------
    # TODO: config.audio.input_device/output_device are accepted but not yet
    # mapped to device indices; the OS default device is used.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # --- simulator + action processor -----------------------------------
    # One SimulatorClient, built once, driven by the JSON action processor.
    simulator = create_simulator(config.simulator)
    json_action = JsonActionProcessor(simulator=simulator)

    # --- context aggregator (carries VAD + turn detection) --------------
    # No tools are declared: the agent uses JSON structured output, not native
    # tool calls (a small local model emits constrained JSON far more reliably).
    # AlwaysUserMuteStrategy suppresses STT input while the bot is speaking, so
    # the agent does not transcribe its own TTS as a phantom command (the local
    # mic-and-speaker setup has no hardware echo cancellation).
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=UserTurnStrategies(stop=[turn_stop_strategy]),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # --- observers ------------------------------------------------------
    latency_tracker = LatencyTracker(
        session_id=session_id,
        metrics_dir=config.logging.metrics_log_path,
    )
    conversation_logger = ConversationLogger(
        session_id=session_id,
        conversation_dir=config.logging.conversation_log_path,
    )

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
            latency_tracker,
            conversation_logger,
        ]
    )
    # Interruption handling lives in the turn strategies (VADUserTurnStartStrategy
    # enables interruptions by default), not in PipelineParams.
    task = PipelineTask(pipeline)

    log.info(
        "pipeline_built",
        session_id=session_id,
        stt=config.stt.backend,
        tts=config.tts.backend,
        vad=config.vad.backend,
        turn=config.turn_detection.backend,
        simulator=config.simulator.backend,
        llm_model=config.llm.model,
    )
    return BuiltPipeline(task=task, simulator=simulator, session_id=session_id)
