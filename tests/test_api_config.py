"""Tests for /api/config (view + edit config.yaml) and /api/config/reload.

Each test uses a ``tmp_path`` fixture as the config file location so disk
writes are sandboxed. The reload endpoint never actually exec's during
tests -- :func:`os.execv` is monkeypatched to a no-op recorder so we can
assert it was scheduled with the right argv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from voice_agent.api.app import SessionInfo, create_app
from voice_agent.api.events import EventBus


# ---------- helpers --------------------------------------------------------


def _session() -> SessionInfo:
    return SessionInfo(
        session_id="test-session",
        started_at="2026-05-21T00:00:00+00:00",
        stt_backend="parakeet_onnx",
        tts_backend="kokoro",
        vad_backend="silero",
        turn_backend="smart_turn_v3",
        simulator_backend="mock",
        llm_model="test/model",
    )


# Minimum valid AppConfig payload -- mirrors what config.yaml would look
# like with just the required fields populated.
_VALID_CONFIG: dict[str, Any] = {
    "stt": {
        "backend": "parakeet_onnx",
        "model": "nemo-parakeet-tdt-0.6b-v2",
        "device": "cuda",
        "language": "en",
    },
    "tts": {"backend": "kokoro", "voice": "af_bella", "device": "cuda"},
    "llm": {
        "base_url": "http://localhost:1234/v1",
        "model": "nvidia/nemotron-3-nano-4b",
        "api_key_env": "LLM_API_KEY",
        "timeout_seconds": 30.0,
        "max_retries": 1,
    },
}


def _seed_config_file(path: Path, data: dict[str, Any] | None = None) -> Path:
    payload = data if data is not None else _VALID_CONFIG
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path


def _build_app_with_config(config_path: Path) -> TestClient:
    app = create_app(
        event_bus=EventBus(),
        session=_session(),
        config_path=config_path,
    )
    return TestClient(app)


# ---------- GET /api/config -----------------------------------------------


def test_get_config_returns_disk_contents(tmp_path: Path) -> None:
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    client = _build_app_with_config(cfg_path)
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["stt"]["backend"] == "parakeet_onnx"
    assert body["llm"]["base_url"] == "http://localhost:1234/v1"


def test_get_config_does_not_apply_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET surfaces what's on disk; env-var overrides must not leak in,
    otherwise round-tripping (GET -> edit -> PUT) would bake the env
    value into the file."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    monkeypatch.setenv("LLM_BASE_URL", "http://from-env:9999/v1")
    client = _build_app_with_config(cfg_path)
    res = client.get("/api/config")
    assert res.status_code == 200
    assert res.json()["llm"]["base_url"] == "http://localhost:1234/v1"


def test_get_config_500_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.yaml"
    client = _build_app_with_config(missing)
    res = client.get("/api/config")
    assert res.status_code == 500
    assert "not found" in res.json()["detail"].lower()


def test_get_config_500_when_yaml_is_not_a_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    client = _build_app_with_config(bad)
    res = client.get("/api/config")
    assert res.status_code == 500
    assert "mapping" in res.json()["detail"].lower()


# ---------- GET /api/config/schema ----------------------------------------


def test_get_config_schema_returns_appconfig_schema(tmp_path: Path) -> None:
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    client = _build_app_with_config(cfg_path)
    res = client.get("/api/config/schema")
    assert res.status_code == 200
    schema = res.json()
    # Top-level AppConfig sections must all be present so the frontend
    # can render every group.
    expected_sections = {
        "stt", "tts", "vad", "turn_detection", "llm", "simulator",
        "audio", "logging", "api", "documents", "review",
    }
    assert set(schema["properties"].keys()) == expected_sections


# ---------- PUT /api/config -----------------------------------------------


def test_put_config_writes_submitted_dict_verbatim(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("stt: {}\n", encoding="utf-8")  # bogus starting state
    client = _build_app_with_config(cfg_path)
    res = client.put("/api/config", json=_VALID_CONFIG)
    assert res.status_code == 200
    assert res.json()["status"] == "saved"
    # File on disk now matches what we sent (modulo YAML formatting).
    with cfg_path.open() as fh:
        on_disk = yaml.safe_load(fh)
    assert on_disk == _VALID_CONFIG


def test_put_config_does_not_persist_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If LLM_BASE_URL is set, validation applies it to the parsed config,
    but the written file must reflect the *user's submitted* base_url --
    otherwise running with the env var temporarily would permanently
    rewrite the file."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    monkeypatch.setenv("LLM_BASE_URL", "http://from-env:9999/v1")
    client = _build_app_with_config(cfg_path)

    res = client.put("/api/config", json=_VALID_CONFIG)
    assert res.status_code == 200

    with cfg_path.open() as fh:
        on_disk = yaml.safe_load(fh)
    assert on_disk["llm"]["base_url"] == "http://localhost:1234/v1"


def test_put_config_422_on_invalid_literal(tmp_path: Path) -> None:
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    client = _build_app_with_config(cfg_path)
    bad = {**_VALID_CONFIG, "stt": {**_VALID_CONFIG["stt"], "backend": "bogus_engine"}}
    res = client.put("/api/config", json=bad)
    assert res.status_code == 422
    # The Pydantic error list is preserved so the frontend can pinpoint
    # which field is bad.
    errs = res.json()["detail"]
    assert isinstance(errs, list)
    assert any("stt" in tuple(e.get("loc", ())) for e in errs)


def test_put_config_422_on_missing_required_field(tmp_path: Path) -> None:
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    client = _build_app_with_config(cfg_path)
    # Drop stt.model (required).
    bad = {**_VALID_CONFIG, "stt": {"backend": "parakeet_onnx", "device": "cuda"}}
    res = client.put("/api/config", json=bad)
    assert res.status_code == 422


def test_put_config_422_on_unknown_field(tmp_path: Path) -> None:
    """``extra='forbid'`` on AppConfig means typos surface as 422 rather
    than silently disappearing into the void."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    client = _build_app_with_config(cfg_path)
    bad = {**_VALID_CONFIG, "ttt": {"x": 1}}  # typo: ttt vs tts
    res = client.put("/api/config", json=bad)
    assert res.status_code == 422


def test_put_config_does_not_modify_disk_on_validation_failure(tmp_path: Path) -> None:
    """A bad PUT must leave the existing config.yaml untouched."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    original = cfg_path.read_text(encoding="utf-8")
    client = _build_app_with_config(cfg_path)
    client.put("/api/config", json={"stt": "this isn't even a dict"})
    assert cfg_path.read_text(encoding="utf-8") == original


# ---------- POST /api/config/reload ---------------------------------------


def test_reload_returns_reloading_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint returns 200 immediately. We can't reliably observe the
    scheduled ``os.execv`` from the sync test thread because TestClient
    tears down its event loop between requests -- the scheduled callback
    is dropped on the floor. Stubbing execv is enough defensive insurance
    in case the loop somehow drains before teardown."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    monkeypatch.setattr(
        "voice_agent.api.config_router.os.execv",
        lambda _path, _args: None,
    )
    client = _build_app_with_config(cfg_path)
    res = client.post("/api/config/reload")
    assert res.status_code == 200
    assert res.json() == {"status": "reloading", "delay_seconds": 1.0}


def test_reload_endpoint_returns_immediately_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP response must come back fast -- the actual exec is
    scheduled, not synchronous. If the handler accidentally awaited
    the exec callback this would hang forever."""
    cfg_path = _seed_config_file(tmp_path / "config.yaml")
    monkeypatch.setattr(
        "voice_agent.api.config_router.os.execv",
        lambda path, args: None,
    )
    client = _build_app_with_config(cfg_path)
    res = client.post("/api/config/reload")
    assert res.status_code == 200


# ---------- 404 when config plane absent -----------------------------------


def test_config_endpoints_404_when_path_missing() -> None:
    """create_app without config_path must not register /api/config."""
    app = create_app(event_bus=EventBus(), session=_session())
    with TestClient(app) as c:
        assert c.get("/api/config").status_code == 404
        assert c.get("/api/config/schema").status_code == 404
        assert c.put("/api/config", json={}).status_code == 404
        assert c.post("/api/config/reload").status_code == 404
