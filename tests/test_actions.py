"""Action schema, dispatch, and the JSON action processor. No network calls."""

from __future__ import annotations

from typing import Any

import pytest

from voice_agent.actions.dispatch import BRIDGE_LOST, dispatch_action
from voice_agent.actions.processor import UNPARSEABLE, JsonActionProcessor
from voice_agent.actions.schema import (
    ActionParseError,
    ErrorAction,
    GetShipStateAction,
    SetEngineTelegraphAction,
    SetHeadingAction,
    parse_response,
)
from voice_agent.backends.simulator.base import EngineOrder, SimulatorError
from voice_agent.backends.simulator.mock import MockSimulatorClient


class _FailingSimulator:
    """SimulatorClient stub whose every command raises SimulatorError."""

    async def set_heading(self, degrees: float) -> Any:
        raise SimulatorError("boom")

    async def set_engine_telegraph(self, order: EngineOrder) -> Any:
        raise SimulatorError("boom")

    async def get_state(self) -> Any:
        raise SimulatorError("boom")

    async def close(self) -> None:
        return None


# --- parse_response -----------------------------------------------------


def test_parse_set_heading() -> None:
    parsed = parse_response(
        '{"action": {"type": "set_heading", "degrees": 270}, "response": "Aye."}'
    )
    assert isinstance(parsed.action, SetHeadingAction)
    assert parsed.action.degrees == 270
    assert parsed.response == "Aye."


def test_parse_set_engine_telegraph() -> None:
    parsed = parse_response(
        '{"action": {"type": "set_engine_telegraph", "order": "full_ahead"}, '
        '"response": "Aye."}'
    )
    assert isinstance(parsed.action, SetEngineTelegraphAction)
    assert parsed.action.order is EngineOrder.FULL_AHEAD


def test_parse_get_ship_state() -> None:
    parsed = parse_response(
        '{"action": {"type": "get_ship_state"}, "response": "Checking."}'
    )
    assert isinstance(parsed.action, GetShipStateAction)


def test_parse_error_action() -> None:
    parsed = parse_response(
        '{"action": {"type": "error", "error_type": "out_of_scope", '
        '"reason": "r", "suggestion": "s"}, "response": "No, sir."}'
    )
    assert isinstance(parsed.action, ErrorAction)
    assert parsed.action.error_type == "out_of_scope"


def test_parse_strips_code_fence() -> None:
    raw = '```json\n{"action": {"type": "get_ship_state"}, "response": "Aye."}\n```'
    assert isinstance(parse_response(raw).action, GetShipStateAction)


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(ActionParseError):
        parse_response("the model rambled instead of returning JSON")


def test_parse_rejects_empty() -> None:
    with pytest.raises(ActionParseError):
        parse_response("   ")


def test_parse_rejects_unknown_action_type() -> None:
    with pytest.raises(ActionParseError):
        parse_response('{"action": {"type": "fire_torpedo"}, "response": "x"}')


def test_parse_rejects_missing_degrees() -> None:
    with pytest.raises(ActionParseError):
        parse_response('{"action": {"type": "set_heading"}, "response": "x"}')


def test_parse_rejects_invalid_engine_order() -> None:
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "set_engine_telegraph", "order": "warp_nine"}, '
            '"response": "x"}'
        )


# --- dispatch_action ----------------------------------------------------


async def test_dispatch_set_heading_commands_simulator() -> None:
    sim = MockSimulatorClient(log_commands=True)
    parsed = parse_response(
        '{"action": {"type": "set_heading", "degrees": 270}, '
        '"response": "Coming to two seven zero, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ok
    assert result.spoken == "Coming to two seven zero, aye."
    assert [c.command for c in sim.command_history] == ["set_heading"]
    assert result.ship_state is not None
    assert result.ship_state.heading_deg == 270.0


async def test_dispatch_normalises_heading() -> None:
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "set_heading", "degrees": 450}, "response": "x"}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ship_state is not None
    assert result.ship_state.heading_deg == 90.0  # 450 % 360


async def test_dispatch_set_engine_telegraph() -> None:
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "set_engine_telegraph", "order": "half_ahead"}, '
        '"response": "Half ahead, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ship_state is not None
    assert result.ship_state.engine_order is EngineOrder.HALF_AHEAD


async def test_dispatch_get_ship_state_appends_live_readings() -> None:
    sim = MockSimulatorClient(
        initial_heading=120,
        initial_engine_order=EngineOrder.SLOW_AHEAD,
        log_commands=False,
    )
    parsed = parse_response(
        '{"action": {"type": "get_ship_state"}, "response": "Checking, sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.spoken.startswith("Checking, sir.")
    assert "120 degrees" in result.spoken
    assert "slow ahead" in result.spoken


async def test_dispatch_error_action_skips_simulator() -> None:
    sim = MockSimulatorClient(log_commands=True)
    parsed = parse_response(
        '{"action": {"type": "error", "error_type": "ambiguous_command", '
        '"reason": "r", "suggestion": "s"}, "response": "Request clarification, sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.spoken == "Request clarification, sir."
    assert sim.command_history == []


async def test_dispatch_simulator_error_reports_bridge_lost() -> None:
    parsed = parse_response(
        '{"action": {"type": "set_heading", "degrees": 90}, "response": "x"}'
    )
    result = await dispatch_action(parsed, _FailingSimulator())
    assert not result.ok
    assert result.spoken == BRIDGE_LOST


# --- JsonActionProcessor ------------------------------------------------


async def test_processor_resolves_command_and_speaks() -> None:
    sim = MockSimulatorClient(log_commands=True)
    proc = JsonActionProcessor(simulator=sim)
    spoken = await proc._resolve(
        '{"action": {"type": "set_heading", "degrees": 90}, '
        '"response": "Coming to zero nine zero, aye."}'
    )
    assert spoken == "Coming to zero nine zero, aye."
    assert [c.command for c in sim.command_history] == ["set_heading"]


async def test_processor_resolve_handles_unparseable_output() -> None:
    sim = MockSimulatorClient(log_commands=False)
    proc = JsonActionProcessor(simulator=sim)
    spoken = await proc._resolve("the model did not return JSON")
    assert spoken == UNPARSEABLE
    assert sim.command_history == []
