"""Builds the Pipecat pipeline from a validated config object.

Pipeline order::

    transport.input() -> STT -> user aggregator -> LLM -> TTS
        -> transport.output() -> assistant aggregator
        -> LatencyTracker -> ConversationLogger

The two observer processors sit last so they see the full downstream frame
flow (VAD, STT, LLM, tool, and TTS frames).

Targets Pipecat 1.2.x. The framework-wiring import paths and the
``LLMContext`` / ``LLMContextAggregatorPair`` API are current as of that
release; verify against the installed package if you bump the pin.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport

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
    turn_analyzer = create_turn(config.turn_detection)
    stt = create_stt(config.stt)
    tts = create_tts(config.tts)
    llm = build_llm(config.llm)

    # --- transport ------------------------------------------------------
    # TODO: config.audio.input_device/output_device are accepted but not yet
    # mapped to device indices; the OS default device is used.
    transport_kwargs = {
        "audio_in_enabled": True,
        "audio_out_enabled": True,
        "vad_analyzer": vad,
    }
    if turn_analyzer is not None:
        transport_kwargs["turn_analyzer"] = turn_analyzer
    transport = LocalAudioTransport(TransportParams(**transport_kwargs))

    # --- simulator + tools ---------------------------------------------
    # One SimulatorClient, built once, shared by all three tool handlers.
    simulator = create_simulator(config.simulator)
    register_ship_tools(llm, simulator)

    context = LLMContext(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        build_tools_schema(),
    )
    context_aggregator = LLMContextAggregatorPair(context)

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
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

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
