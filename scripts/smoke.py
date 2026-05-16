"""Smoke test: full LLM-to-tool-to-simulator path, no audio, no real sim.

Builds a minimal pipeline (context aggregator -> remote LLM -> aggregator),
registers the three ship tools against a ``MockSimulatorClient``, injects a
fake ``TranscriptionFrame("steer course two seven zero")``, and asserts that
the LLM emits ``set_heading`` with ``degrees ~= 270`` and that the mock's
``command_history`` records it.

Requires a reachable remote LLM ($LLM_BASE_URL / config.yaml ``llm.base_url``).
Run from the repo root:

    python scripts/smoke.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.utils.time import time_now_iso8601

from voice_agent.backends.llm.openai_compatible import build_llm
from voice_agent.backends.simulator.mock import MockSimulatorClient
from voice_agent.config import load_config
from voice_agent.logging_setup import configure_logging, new_session_id
from voice_agent.pipeline import SYSTEM_PROMPT
from voice_agent.tools.schemas import build_tools_schema
from voice_agent.tools.ship import register_ship_tools

_UTTERANCE = "steer course two seven zero"
_EXPECTED_DEG = 270
_TIMEOUT_S = 30.0


async def _smoke(config_path: str) -> bool:
    config = load_config(config_path)
    configure_logging(config.logging, new_session_id())

    mock = MockSimulatorClient(log_commands=True)
    llm = build_llm(config.llm)
    register_ship_tools(llm, mock)

    context = LLMContext(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        build_tools_schema(),
    )
    aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline([aggregator.user(), llm, aggregator.assistant()])
    task = PipelineTask(pipeline)

    runner = PipelineRunner()
    run_handle = asyncio.create_task(runner.run(task))

    await task.queue_frames(
        [TranscriptionFrame(_UTTERANCE, "captain", time_now_iso8601())]
    )

    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline and not mock.command_history:
        await asyncio.sleep(0.2)

    await task.cancel()
    await run_handle

    # --- assertions -----------------------------------------------------
    heading_cmds = [c for c in mock.command_history if c.command == "set_heading"]
    if not heading_cmds:
        print(f"FAIL: no set_heading command recorded for '{_UTTERANCE}'.")
        return False

    degrees = heading_cmds[0].result.heading_deg
    if abs(degrees - _EXPECTED_DEG) > 1.0:
        print(f"FAIL: set_heading degrees={degrees}, expected ~{_EXPECTED_DEG}.")
        return False

    print(f"PASS: '{_UTTERANCE}' -> set_heading(degrees={degrees}).")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="smoke", description=__doc__)
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    ok = asyncio.run(_smoke(args.config))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
