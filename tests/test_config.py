"""Config validation and environment-variable overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from voice_agent.config import AppConfig, load_config, parse_config

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _minimal() -> dict[str, Any]:
    """A minimal config dict that satisfies all required fields."""
    return {
        "stt": {"model": "nvidia/parakeet-tdt-1.1b"},
        "tts": {"voice": "af_bella"},
        "llm": {"model": "nvidia/nemotron-3-nano-4b"},
        "lm_studio": {"base_url": "http://llm-server:8000/v1"},
    }


def test_parse_minimal_config_applies_defaults() -> None:
    config = parse_config(_minimal())
    assert isinstance(config, AppConfig)
    assert config.stt.backend == "parakeet_onnx"  # default
    assert config.tts.backend == "kokoro"
    assert config.simulator.backend == "mock"
    assert config.vad.threshold == 0.5
    assert config.turn_detection.backend == "smart_turn_v3"


def test_load_repo_config_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SIMULATOR_BACKEND", raising=False)
    config = load_config(_REPO_ROOT / "config.yaml")
    assert config.simulator.backend == "mock"  # default config ships the mock
    assert config.stt.backend == "parakeet_onnx"


@pytest.mark.parametrize(
    "example",
    sorted((_REPO_ROOT / "config.examples").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_config_examples_still_load(
    example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shipped example must parse.

    The config models forbid extra keys, so an example that still names a
    renamed field is not a stale comment -- it is a file that raises the moment
    someone follows the README. Only config.yaml was covered before, which is
    exactly how the examples drifted.
    """
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SIMULATOR_BACKEND", raising=False)
    assert isinstance(load_config(example), AppConfig)


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(_REPO_ROOT / "does-not-exist.yaml")


def test_env_override_llm_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://override:9000/v1")
    config = parse_config(_minimal())
    assert config.lm_studio.base_url == "http://override:9000/v1"


def test_env_override_simulator_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATOR_BACKEND", "real")
    config = parse_config(_minimal())
    assert config.simulator.backend == "real"


def test_resolved_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret-token")
    config = parse_config(_minimal())
    assert config.lm_studio.resolved_api_key() == "secret-token"


def test_qdrant_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = parse_config(_minimal())
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    assert config.qdrant.resolved_headers() == {}
    monkeypatch.setenv("QDRANT_API_KEY", "qd-secret")
    assert config.qdrant.resolved_headers() == {"api-key": "qd-secret"}


def test_audit_defaults_off() -> None:
    config = parse_config(_minimal())
    assert config.llm.audit_enabled is False
    assert config.database.path == "./data/ingestion.db"


def test_n8n_backend_rejected() -> None:
    """n8n was removed -- selecting it is now a validation error."""
    data = _minimal()
    data["llm"]["backend"] = "n8n"
    with pytest.raises(ValidationError):
        parse_config(data)


def test_invalid_stt_backend_rejected() -> None:
    data = _minimal()
    data["stt"]["backend"] = "nonexistent_engine"
    with pytest.raises(ValidationError):
        parse_config(data)


def test_unknown_key_rejected() -> None:
    data = _minimal()
    data["llm"]["typo_field"] = True
    with pytest.raises(ValidationError):
        parse_config(data)


def test_missing_required_block_rejected() -> None:
    data = _minimal()
    del data["llm"]
    with pytest.raises(ValidationError):
        parse_config(data)


# --- legacy simulator.real keys (renamed in the simulator integration) ------


def _with_legacy_real_block(backend: str) -> dict[str, Any]:
    data = _minimal()
    data["simulator"] = {
        "backend": backend,
        "real": {"host": "127.0.0.1", "port": 9100, "connect_timeout_seconds": 2},
    }
    return data


def test_legacy_real_keys_are_dropped_for_mock_users() -> None:
    """A stale real-block must not stop a mock user from starting.

    Every previously shipped config carried simulator.real.host/port even with
    backend mock; those keys configured a backend that was never implemented,
    so refusing to start over them would punish exactly the users the rename
    cannot affect.
    """
    with pytest.warns(UserWarning, match="simulator.real"):
        config = parse_config(_with_legacy_real_block("mock"))
    assert config.simulator.backend == "mock"
    # The renamed fields keep their defaults (0 = "not configured": the
    # working values ship with the vendor integration, not this repo).
    assert config.simulator.real.remote_port == 0


def test_legacy_real_keys_are_a_clear_error_for_real_users() -> None:
    """A real user must get the rename explained, not pydantic's bare
    'extra inputs are not permitted'."""
    with pytest.raises(Exception, match="remote_host"):
        parse_config(_with_legacy_real_block("real"))
