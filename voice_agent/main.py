"""CLI entrypoint for the virtual helmsman voice agent.

Thin wrapper: parses ``--config``, loads config, configures logging, builds the
pipeline, and runs it. All real logic lives in :mod:`voice_agent.pipeline` and
the backends.

When ``api.enabled`` is true in config, a FastAPI service is started alongside
the pipeline (same process, same event loop) so the frontend can subscribe to
events without a separate runtime.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from pipecat.pipeline.runner import PipelineRunner

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.server import ApiServer
from voice_agent.config import AppConfig, load_config
from voice_agent.logging_setup import configure_logging, get_logger, new_session_id
from voice_agent.pipeline import BuiltPipeline, build_pipeline, build_text_injector


def _session_info(config: AppConfig, session_id: str, started_at: str) -> SessionInfo:
    """Pack the static identity of this session for /api/session."""
    return SessionInfo(
        session_id=session_id,
        started_at=started_at,
        stt_backend=config.stt.backend,
        tts_backend=config.tts.backend,
        vad_backend=config.vad.backend,
        turn_backend=config.turn_detection.backend,
        simulator_backend=config.simulator.backend,
        llm_model=config.llm.model,
    )


async def _maybe_start_api(
    config: AppConfig, built: BuiltPipeline, config_path: str
) -> ApiServer | None:
    """Start the FastAPI server alongside the pipeline if configured."""
    if not config.api.enabled or built.event_bus is None:
        return None
    app = create_app(
        event_bus=built.event_bus,
        session=_session_info(
            config, built.session_id, datetime.now(timezone.utc).isoformat()
        ),
        cors_allow_origins=config.api.cors_allow_origins,
        documents=config.documents,
        review=config.review,
        control_state=built.control_state,
        inject_text=build_text_injector(built.task, built.llm_context),
        config_path=Path(config_path),
        # Shared with /api/review/upload as the default `Model` form field
        # forwarded to the n8n ingestion webhook (see REVIEW_API.md).
        llm_model=config.llm.model,
    )
    server = ApiServer(app, host=config.api.host, port=config.api.port)
    await server.start()
    return server


async def _run(config_path: str) -> None:
    """Load config, build the pipeline (and optional API), and run until interrupted."""
    config = load_config(config_path)
    session_id = new_session_id()
    configure_logging(config.logging, session_id)
    log = get_logger("pipeline")
    log.info("session_start", session_id=session_id, config_path=config_path)

    built = build_pipeline(config, session_id)
    api = await _maybe_start_api(config, built, config_path)
    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(built.task)
    finally:
        if api is not None:
            await api.stop()
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
