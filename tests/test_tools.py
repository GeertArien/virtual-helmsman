"""Tool handlers tested against MockSimulatorClient. No network calls."""

from __future__ import annotations

from typing import Any

from voice_agent.backends.simulator.base import EngineOrder, SimulatorError
from voice_agent.backends.simulator.mock import MockSimulatorClient
from voice_agent.tools.ship import build_ship_handlers
from voice_agent.tools.schemas import (
    GET_SHIP_STATE,
    SET_ENGINE_TELEGRAPH,
    SET_HEADING,
)


class FakeParams:
    """Stand-in for Pipecat's FunctionCallParams."""

    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.result: Any = None

    async def result_callback(self, result: Any, **_kwargs: Any) -> None:
        self.result = result


class FailingSimulator:
    """SimulatorClient stub whose every method raises SimulatorError."""

    async def set_heading(self, degrees: float) -> Any:
        raise SimulatorError("boom")

    async def set_engine_telegraph(self, order: EngineOrder) -> Any:
        raise SimulatorError("boom")

    async def get_state(self) -> Any:
        raise SimulatorError("boom")

    async def close(self) -> None:
        return None


def _handlers(simulator: Any) -> dict[str, Any]:
    return build_ship_handlers(simulator)


async def test_set_heading_delegates_and_records() -> None:
    sim = MockSimulatorClient(log_commands=False)
    params = FakeParams({"degrees": 270})
    await _handlers(sim)[SET_HEADING](params)

    assert params.result["heading_deg"] == 270.0
    assert [r.command for r in sim.command_history] == ["set_heading"]


async def test_set_heading_normalises_degrees() -> None:
    sim = MockSimulatorClient(log_commands=False)
    params = FakeParams({"degrees": 730})  # 730 % 360 == 10
    await _handlers(sim)[SET_HEADING](params)
    assert params.result["heading_deg"] == 10.0


async def test_set_heading_rejects_non_numeric() -> None:
    sim = MockSimulatorClient(log_commands=False)
    params = FakeParams({"degrees": "hard to port"})
    await _handlers(sim)[SET_HEADING](params)

    assert "error" in params.result
    assert sim.command_history == []  # nothing executed


async def test_set_engine_telegraph_valid_order() -> None:
    sim = MockSimulatorClient(log_commands=False)
    params = FakeParams({"order": "full_ahead"})
    await _handlers(sim)[SET_ENGINE_TELEGRAPH](params)

    assert params.result["engine_order"] == "full_ahead"
    assert params.result["speed_kn"] == 20.0


async def test_set_engine_telegraph_rejects_invalid_order() -> None:
    sim = MockSimulatorClient(log_commands=False)
    params = FakeParams({"order": "warp_nine"})
    await _handlers(sim)[SET_ENGINE_TELEGRAPH](params)

    assert "error" in params.result
    assert sim.command_history == []


async def test_get_ship_state_returns_current_state() -> None:
    sim = MockSimulatorClient(
        initial_heading=120, initial_engine_order=EngineOrder.SLOW_AHEAD,
        log_commands=False,
    )
    params = FakeParams({})
    await _handlers(sim)[GET_SHIP_STATE](params)

    assert params.result["heading_deg"] == 120.0
    assert params.result["engine_order"] == "slow_ahead"


async def test_simulator_error_surfaces_failure_phrase() -> None:
    handlers = _handlers(FailingSimulator())
    params = FakeParams({"degrees": 90})
    await handlers[SET_HEADING](params)
    assert params.result == {"error": "Lost contact with bridge"}


async def test_handlers_share_one_simulator_instance() -> None:
    sim = MockSimulatorClient(log_commands=False)
    handlers = _handlers(sim)
    await handlers[SET_HEADING](FakeParams({"degrees": 45}))
    await handlers[SET_ENGINE_TELEGRAPH](FakeParams({"order": "half_ahead"}))

    state_params = FakeParams({})
    await handlers[GET_SHIP_STATE](state_params)
    assert state_params.result["heading_deg"] == 45.0
    assert state_params.result["engine_order"] == "half_ahead"
