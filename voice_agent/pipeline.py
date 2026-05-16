"""Builds the Pipecat pipeline from a validated config object.

Pipeline order::

    transport.input() -> STT -> user aggregator -> LLM -> TTS
        -> transport.output() -> assistant aggregator
        -> LatencyTracker -> ConversationLogger

The two observer processors sit last so they see the full downstream frame
flow (VAD, STT, LLM, tool, and TTS frames).

VAD and turn detection are wired into the **user context aggregator** (Pipecat
1.2.x): the VAD analyzer and the user-turn stop strategy are passed via
``LLMUserAggregatorParams``. The transport itself just streams audio.
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
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voice_agent.backends.llm.openai_compatible import build_llm
from voice_agent.backends.simulator.base import SimulatorClient
from voice_agent.backends.simulator.factory import create_simulator
from voice_agent.backends.stt.factory import create_stt
from voice_agent.backends.tts.factory import create_tts
from voice_agent.backends.turn.factory import create_turn
from voice_agent.backends.vad.factory import create_vad
from voice_agent.config import AppConfig, LlmConfig
from voice_agent.logging_setup import get_logger
from voice_agent.metrics import ConversationLogger, LatencyTracker
from voice_agent.tools.schemas import build_tools_schema
from voice_agent.tools.ship import register_ship_tools

# Terse, first-person system prompt. Domain vocabulary is inline so the LLM
# corrects obvious mishearings implicitly (no separate post-correction stage).
SYSTEM_PROMPT = """\
You are the virtual helmsman on a ship simulator. The user is the captain.

Acknowledge each command in one short sentence, execute it with the appropriate \
tool, then confirm the result. Never change heading or engine order without an \
explicit command from the captain. If a command is ambiguous, ask one brief \
clarifying question instead of acting on a guess.

Headings are spoken as digits: "two seven zero" means 270 degrees. The nine \
engine telegraph orders are: full astern, half astern, slow astern, dead slow \
astern, stop, dead slow ahead, slow ahead, half ahead, full ahead. Common \
phrases: "steer course" and "come to" set a heading; "hold this heading" keeps \
the current heading; "rudder amidships" means steer the current heading.

Keep replies short. No filler.
"""


def build_system_prompt(llm_config: LlmConfig) -> str:
    """Compose the system prompt: optional model-specific prefix + domain prompt.

    ``llm.system_prompt_prefix`` carries model-specific control tokens (e.g.
    ``detailed thinking off`` for Nemotron) so the domain prompt stays portable.
    """
    prefix = llm_config.system_prompt_prefix.strip()
    return f"{prefix}\n\n{SYSTEM_PROMPT}" if prefix else SYSTEM_PROMPT


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
    llm = build_llm(config.llm)

    # --- transport ------------------------------------------------------
    # TODO: config.audio.input_device/output_device are accepted but not yet
    # mapped to device indices; the OS default device is used.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # --- simulator + tools ---------------------------------------------
    # One SimulatorClient, built once, shared by all three tool handlers.
    simulator = create_simulator(config.simulator)
    register_ship_tools(llm, simulator)

    # --- context aggregator (carries VAD + turn detection) --------------
    context = LLMContext(
        [{"role": "system", "content": build_system_prompt(config.llm)}],
        build_tools_schema(),
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=UserTurnStrategies(stop=[turn_stop_strategy]),
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
