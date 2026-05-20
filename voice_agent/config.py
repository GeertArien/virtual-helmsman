"""Pydantic config schema, YAML loader, and environment-variable overrides.

The schema mirrors ``config.yaml`` one-to-one. ``extra="forbid"`` makes typos
in the config file fail validation rather than being silently ignored.

Environment overrides (applied before validation):

* ``LLM_BASE_URL``      -> ``llm.base_url``
* ``SIMULATOR_BACKEND`` -> ``simulator.backend``

``LLM_API_KEY`` is not a config field: the key is always read at runtime from
the environment variable named by ``llm.api_key_env`` (default ``LLM_API_KEY``)
via :meth:`LlmConfig.resolved_api_key`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from voice_agent.backends.simulator.base import EngineOrder


class _Base(BaseModel):
    """Base model: reject unknown keys so config typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class SttConfig(_Base):
    backend: Literal["parakeet_onnx", "parakeet_nemo", "whisper"] = "parakeet_onnx"
    model: str
    device: str = "cuda"
    language: str = "en"


class TtsConfig(_Base):
    backend: Literal["kokoro", "piper"] = "kokoro"
    voice: str
    device: str = "cuda"


class VadConfig(_Base):
    backend: Literal["silero"] = "silero"
    threshold: float = 0.5
    # Silence (seconds) before speech is considered ended. Pipecat's default of
    # 0.2 is aggressive enough to split a command at the pauses between words;
    # a longer value keeps a paused-but-continuing utterance as one segment.
    stop_secs: float = 0.8


class TurnConfig(_Base):
    backend: Literal["smart_turn_v3", "vad_only"] = "smart_turn_v3"
    device: str = "cpu"


class LlmConfig(_Base):
    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 1

    def resolved_api_key(self) -> str | None:
        """Return the API key from the env var named by ``api_key_env``."""
        return os.environ.get(self.api_key_env)


class SimulatorRealConfig(_Base):
    host: str = "127.0.0.1"
    port: int = 9100
    connect_timeout_seconds: float = 2.0


class SimulatorMockConfig(_Base):
    initial_heading: float = 0.0
    initial_engine_order: EngineOrder = EngineOrder.STOP
    log_commands: bool = True


class SimulatorConfig(_Base):
    backend: Literal["real", "mock"] = "mock"
    real: SimulatorRealConfig = Field(default_factory=SimulatorRealConfig)
    mock: SimulatorMockConfig = Field(default_factory=SimulatorMockConfig)


class AudioConfig(_Base):
    input_device: str = "default"
    output_device: str = "default"
    sample_rate: int = 16000


class LoggingConfig(_Base):
    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["json", "console"] = "json"
    conversation_log_path: Path = Path("./logs/conversations")
    metrics_log_path: Path = Path("./logs/metrics")


class ApiConfig(_Base):
    """Control/observability API for the frontend. Disabled by default."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    # CORS origins for the frontend. Wildcard is fine for local dev; tighten
    # to e.g. ["http://localhost:5173"] (Vite default) for stricter setups.
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])


class AppConfig(_Base):
    """Top-level config object, the single source of truth for the agent."""

    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig = Field(default_factory=VadConfig)
    turn_detection: TurnConfig = Field(default_factory=TurnConfig)
    llm: LlmConfig
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with env-var overrides applied."""
    data = copy.deepcopy(data)
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        data.setdefault("llm", {})["base_url"] = base_url
    sim_backend = os.environ.get("SIMULATOR_BACKEND")
    if sim_backend:
        data.setdefault("simulator", {})["backend"] = sim_backend
    return data


def parse_config(data: dict[str, Any]) -> AppConfig:
    """Apply env overrides to a config mapping and validate it."""
    return AppConfig(**_apply_env_overrides(data))


def load_config(path: str | Path = "./config.yaml") -> AppConfig:
    """Load, env-override, and validate the config file at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return parse_config(data)
