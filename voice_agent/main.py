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
    build_pipeline,
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
        browser_audio=config.audio.browser_enabled,
    )


async def _maybe_start_api(
    config: AppConfig,
    built: BuiltPipeline,
    config_path: str,
    *,
    webrtc_manager: WebRTCManager | None = None,
    enable_text: bool = True,
) -> ApiServer | None:
    """Start the FastAPI server alongside the pipeline if configured.

    ``webrtc_manager`` mounts the browser-audio (WebRTC) signalling endpoint.
    ``enable_text`` gates the chatbox text-injection route -- disabled in
    browser-audio mode, where there is no single local task to inject into.
    """
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
        inject_text=(
            build_text_injector(built.task, built.llm_context)
            if enable_text and built.task is not None
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

    if config.audio.browser_enabled and config.api.enabled:
        await _run_browser(config, config_path, session_id, log)
    else:
        await _run_local(config, config_path, session_id, log)


async def _run_local(
    config: AppConfig, config_path: str, session_id: str, log: Any
) -> None:
    """Default path: one pipeline bound to local hardware audio, run to completion."""
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


async def _run_browser(
    config: AppConfig, config_path: str, session_id: str, log: Any
) -> None:
    """Browser-audio path: serve the API and run a pipeline per WebRTC connection.

    The heavy models are loaded once into shared backends; each browser
    connection assembles its own pipeline against them. There is no local
    hardware audio and no single global task -- the process just serves until
    interrupted.
    """
    backends = build_shared_backends(config, session_id)
    manager = WebRTCManager(backends, config)
    if not manager.available():
        log.warning(
            "webrtc_extra_missing",
            hint='Browser audio needs the webrtc extra: pip install -e ".[webrtc]"',
        )
    # No local task: pass a BuiltPipeline carrying the shared resources but a
    # ``None`` task so the API mounts without a text-injector.
    shell = BuiltPipeline(
        task=None,  # type: ignore[arg-type]
        simulator=backends.simulator,
        session_id=session_id,
        event_bus=backends.event_bus,
        control_state=backends.control_state,
        llm_context=None,  # type: ignore[arg-type]
        backends=backends,
    )
    api = await _maybe_start_api(
        config, shell, config_path, webrtc_manager=manager, enable_text=False
    )
    log.info("browser_audio_ready", host=config.api.host, port=config.api.port)
    try:
        await asyncio.Event().wait()  # serve until SIGINT / cancellation
    finally:
        if api is not None:
            await api.stop()
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
