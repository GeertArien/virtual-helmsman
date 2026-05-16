"""Each factory dispatches the right backend value to the right builder.

The STT/TTS/VAD/turn builders are monkeypatched so the tests verify dispatch
logic without loading models or downloading weights. The simulator factory is
exercised for real — the mock backend is cheap to construct.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from voice_agent.backends.simulator.factory import create_simulator
from voice_agent.backends.simulator.mock import MockSimulatorClient
from voice_agent.backends.stt import factory as stt_factory
from voice_agent.backends.tts import factory as tts_factory
from voice_agent.backends.turn import factory as turn_factory
from voice_agent.backends.turn import vad_only
from voice_agent.backends.vad import factory as vad_factory


def _patch_dispatch(
    monkeypatch: pytest.MonkeyPatch, builders: dict[str, Callable[[Any], Any]]
) -> None:
    """Replace every builder with one that just echoes its backend name."""
    for name in list(builders):
        monkeypatch.setitem(builders, name, lambda _cfg, _n=name: _n)


# --- simulator factory (exercised for real) -----------------------------

def test_simulator_factory_returns_mock_instance() -> None:
    config = SimpleNamespace(
        backend="mock",
        mock=SimpleNamespace(
            initial_heading=0, initial_engine_order="stop", log_commands=False
        ),
        real=SimpleNamespace(host="127.0.0.1", port=9100, connect_timeout_seconds=2),
    )
    assert isinstance(create_simulator(config), MockSimulatorClient)


def test_simulator_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_simulator(SimpleNamespace(backend="bogus", mock=None, real=None))


# --- STT factory --------------------------------------------------------

def test_stt_factory_registers_all_v1_backends() -> None:
    assert set(stt_factory._BUILDERS) == {"parakeet_onnx", "parakeet_nemo", "whisper"}


@pytest.mark.parametrize("backend", ["parakeet_onnx", "parakeet_nemo", "whisper"])
def test_stt_factory_dispatch(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    _patch_dispatch(monkeypatch, stt_factory._BUILDERS)
    assert stt_factory.create_stt(SimpleNamespace(backend=backend)) == backend


def test_stt_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        stt_factory.create_stt(SimpleNamespace(backend="bogus"))


# --- TTS factory --------------------------------------------------------

def test_tts_factory_registers_all_v1_backends() -> None:
    assert set(tts_factory._BUILDERS) == {"kokoro", "piper"}


@pytest.mark.parametrize("backend", ["kokoro", "piper"])
def test_tts_factory_dispatch(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    _patch_dispatch(monkeypatch, tts_factory._BUILDERS)
    assert tts_factory.create_tts(SimpleNamespace(backend=backend)) == backend


def test_tts_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        tts_factory.create_tts(SimpleNamespace(backend="bogus"))


# --- VAD factory --------------------------------------------------------

def test_vad_factory_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dispatch(monkeypatch, vad_factory._BUILDERS)
    assert vad_factory.create_vad(SimpleNamespace(backend="silero")) == "silero"


def test_vad_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        vad_factory.create_vad(SimpleNamespace(backend="bogus"))


# --- turn-detection factory --------------------------------------------

def test_turn_factory_registers_all_v1_backends() -> None:
    assert set(turn_factory._BUILDERS) == {"smart_turn_v3", "vad_only"}


def test_turn_factory_vad_only_returns_none() -> None:
    # vad_only is the absence of a turn analyzer; the builder yields None.
    assert turn_factory.create_turn(SimpleNamespace(backend="vad_only")) is None
    assert vad_only.build_turn(SimpleNamespace(backend="vad_only")) is None


def test_turn_factory_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dispatch(monkeypatch, turn_factory._BUILDERS)
    assert turn_factory.create_turn(SimpleNamespace(backend="smart_turn_v3")) == (
        "smart_turn_v3"
    )


def test_turn_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        turn_factory.create_turn(SimpleNamespace(backend="bogus"))
