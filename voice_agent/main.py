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
from typing import Any

from dotenv import load_dotenv
from pipecat.pipeline.runner import PipelineRunner

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.server import ApiServer
from voice_agent.api.webrtc import WebRTCManager
from voice_agent.config import AppConfig, load_config
from voice_agent.logging_setup import configure_logging, get_logger, new_session_id
from voice_agent.pipeline import (
    BuiltPipeline,
    assemble_text_task,
    build_shared_backends,
    build_text_injector,
)


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
        browser_audio=True,
    )


async def _maybe_start_api(
    config: AppConfig,
    built: BuiltPipeline,
    config_path: str,
    *,
    webrtc_manager: WebRTCManager | None = None,
) -> ApiServer | None:
    """Start the FastAPI server alongside the pipeline if configured.

    ``webrtc_manager`` mounts the browser-audio (WebRTC) signalling endpoint.
    The chatbox text-injection route targets ``built.task`` -- the standing
    text-only pipeline.
    """
    if not config.api.enabled or built.event_bus is None:
        return None
    app = create_app(
        event_bus=built.event_bus,
        session=_session_info(
            config, built.session_id, datetime.now(timezone.utc).isoformat()
        ),
        cors_allow_origins=config.api.cors_allow_origins,
        documents=config.documents_runtime(),
        review=config.ingestion_runtime(),
        inject_text=(
            build_text_injector(built.task, built.llm_context)
            if built.task is not None
            else None
        ),
        config_path=Path(config_path),
        # Default `Model` for /api/review/upload's doc-summary call, so
        # ingestion uses the same model as the helmsman LLM path.
        llm_model=config.llm.model,
        webrtc_manager=webrtc_manager,
    )
    server = ApiServer(app, host=config.api.host, port=config.api.port)
    await server.start()
    return server


async def _run(config_path: str) -> None:
    """Load config, build the pipeline (and optional API), and run until interrupted."""
    # Load secrets from a local .env (if present) into the process environment
    # before anything reads os.environ -- API keys are referenced by env-var
    # name (config *_api_key_env). Real env vars already set
    # take precedence; override=False is python-dotenv's default.
    load_dotenv()
    config = load_config(config_path)
    session_id = new_session_id()
    configure_logging(config.logging, session_id)
    log = get_logger("pipeline")
    log.info("session_start", session_id=session_id, config_path=config_path)
    await _serve(config, config_path, session_id, log)


async def _serve(
    config: AppConfig, config_path: str, session_id: str, log: Any
) -> None:
    """Serve the control plane and run a pipeline per WebRTC browser connection.

    Voice input/output is the browser: the heavy models are loaded once into
    shared backends, and each browser connection assembles its own pipeline
    against them. Typed chatbox commands flow through a **standing text-only
    pipeline** that runs for the whole session. Both require ``api.enabled``;
    without it the agent has no inputs (we warn but still serve so a misconfig
    is obvious rather than a silent exit).
    """
    if not config.api.enabled:
        log.warning(
            "api_disabled",
            hint="Browser audio and the chatbox both need api.enabled: true; "
            "the agent has no inputs until it is set.",
        )

    backends = build_shared_backends(config, session_id)
    manager = WebRTCManager(backends, config)
    if not manager.available():
        log.warning(
            "webrtc_extra_missing",
            hint='Browser audio needs the webrtc extra: pip install -e ".[webrtc]"',
        )
    # Standing text-only pipeline: the chatbox injects into this task; the
    # reply surfaces in the transcript panel and the action drives the
    # shared simulator.
    text_task, text_context = assemble_text_task(backends, config)
    shell = BuiltPipeline(
        task=text_task,
        simulator=backends.simulator,
        session_id=session_id,
        event_bus=backends.event_bus,
        llm_context=text_context,
        backends=backends,
    )
    api = await _maybe_start_api(config, shell, config_path, webrtc_manager=manager)
    log.info("browser_audio_ready", host=config.api.host, port=config.api.port)
    runner = PipelineRunner(handle_sigint=True)
    try:
        # The text task idles between typed commands (idle timeout disabled)
        # and serves until SIGINT.
        await runner.run(text_task)
    finally:
        if api is not None:
            await api.stop()
        # Release simulator resources (UDP sockets / threads on the real backend).
        await backends.simulator.close()
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
