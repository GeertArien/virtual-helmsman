"""Smoke test: full LLM -> JSON action -> simulator path, no audio.

Seeds the user utterance "port ten" into the LLM context, runs a minimal
pipeline (user aggregator -> LLM -> JsonActionProcessor -> assistant
aggregator) against a ``MockSimulatorClient``, triggers the LLM with an
``LLMRunFrame``, and asserts the processor parsed the JSON response,
dispatched ``set_rudder`` with ``angle_deg ~= -10`` (port is negative), and the
mock recorded it.

A helm order, not a course order: "steer two seven zero" is refused by design
in v1 (see ``dispatch.COURSE_ORDER_REFUSAL``), so it would exercise nothing.

The utterance is seeded straight into the context (the equivalent of a
finalized transcript): with Pipecat 1.2.x's universal aggregator a bare
``TranscriptionFrame`` is buffered into a pending turn that never completes
without audio. The exercised path -- LLM -> JSON action -> simulator -- is the
same one the live pipeline uses.

Requires a reachable LLM ($LLM_BASE_URL / config.yaml ``llm.base_url``).
Run from the repo root:

    python scripts/smoke.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from voice_agent.actions.processor import JsonActionProcessor
from voice_agent.actions.prompt import SYSTEM_PROMPT
from voice_agent.actions.schema import RESPONSE_FORMAT
from voice_agent.backends.llm.openai_compatible import build_llm
from voice_agent.backends.simulator.mock import MockSimulatorClient
from voice_agent.config import load_config
from voice_agent.logging_setup import configure_logging, new_session_id

_UTTERANCE = "port ten"
_EXPECTED_ANGLE_DEG = -10.0  # port is negative
_TIMEOUT_S = 30.0


async def _smoke(config_path: str) -> bool:
    config = load_config(config_path)
    configure_logging(config.logging, new_session_id())

    mock = MockSimulatorClient(log_commands=True)
    llm = build_llm(config.llm, extra={"response_format": RESPONSE_FORMAT})
    json_action = JsonActionProcessor(simulator=mock)

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context.add_message({"role": "user", "content": _UTTERANCE})

    aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [aggregator.user(), llm, json_action, aggregator.assistant()]
    )
    task = PipelineTask(pipeline)

    runner = PipelineRunner()
    run_handle = asyncio.create_task(runner.run(task))

    await task.queue_frames([LLMRunFrame()])

    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline and not mock.command_history:
        await asyncio.sleep(0.2)

    await task.cancel()
    await run_handle

    # --- assertions -----------------------------------------------------
    rudder_cmds = [c for c in mock.command_history if c.command == "set_rudder"]
    if not rudder_cmds:
        print(f"FAIL: no set_rudder command recorded for '{_UTTERANCE}'.")
        return False

    angle = rudder_cmds[0].result.rudder_angle_deg
    if abs(angle - _EXPECTED_ANGLE_DEG) > 1.0:
        print(f"FAIL: set_rudder angle_deg={angle}, expected ~{_EXPECTED_ANGLE_DEG}.")
        return False

    print(f"PASS: '{_UTTERANCE}' -> set_rudder(angle_deg={angle}).")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="smoke", description=__doc__)
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    ok = asyncio.run(_smoke(args.config))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
