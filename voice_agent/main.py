"""CLI entrypoint for the virtual helmsman voice agent.

Thin wrapper: parses ``--config``, loads config, configures logging, builds the
pipeline, and runs it. All real logic lives in :mod:`voice_agent.pipeline` and
the backends.
"""

from __future__ import annotations

import argparse
import asyncio

from pipecat.pipeline.runner import PipelineRunner

from voice_agent.config import load_config
from voice_agent.logging_setup import configure_logging, get_logger, new_session_id
from voice_agent.pipeline import build_pipeline


async def _run(config_path: str) -> None:
    """Load config, build the pipeline, and run it until interrupted."""
    config = load_config(config_path)
    session_id = new_session_id()
    configure_logging(config.logging, session_id)
    log = get_logger("pipeline")
    log.info("session_start", session_id=session_id, config_path=config_path)

    built = build_pipeline(config, session_id)
    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(built.task)
    finally:
        # Release simulator resources (UDP sockets / threads on the real backend).
        await built.simulator.close()
        log.info("session_end", session_id=session_id)


def main() -> None:
    """Parse args and run the agent."""
    parser = argparse.ArgumentParser(
        prog="virtual-helmsman",
        description="Modular local voice agent for a ship simulator.",
    )
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to the YAML config file (default: ./config.yaml).",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
