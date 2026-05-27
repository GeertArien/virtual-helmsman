"""Action schema, dispatch, and the JSON action processor. No network calls."""

from __future__ import annotations

from typing import Any

import pytest

from voice_agent.actions.dispatch import (
    BRIDGE_LOST,
    _knots_to_telegraph,
    dispatch_action,
)
from voice_agent.actions.processor import UNPARSEABLE, JsonActionProcessor
from voice_agent.actions.schema import (
    ActionParseError,
    AnchorAction,
    AutopilotAction,
    ErrorAction,
    NavigationAction,
    RudderAction,
    StatusQueryAction,
    ThrottleAction,
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


def test_parse_rudder() -> None:
    parsed = parse_response(
        '{"action": {"type": "rudder", "direction": "starboard", "degrees": 20}, '
        '"response": "Starboard twenty, aye."}'
    )
    assert isinstance(parsed.action, RudderAction)
    assert parsed.action.direction == "starboard"
    assert parsed.action.degrees == 20


def test_parse_throttle() -> None:
    parsed = parse_response(
        '{"action": {"type": "throttle", "speed": 15, "unit": "knots"}, '
        '"response": "Making turns for fifteen, aye."}'
    )
    assert isinstance(parsed.action, ThrottleAction)
    assert parsed.action.speed == 15
    assert parsed.action.unit == "knots"


def test_parse_throttle_unit_defaults_to_knots() -> None:
    """unit is optional; defaults to knots per the schema's Literal default."""
    parsed = parse_response(
        '{"action": {"type": "throttle", "speed": 10}, "response": "x"}'
    )
    assert isinstance(parsed.action, ThrottleAction)
    assert parsed.action.unit == "knots"


def test_parse_navigation() -> None:
    parsed = parse_response(
        '{"action": {"type": "navigation", "course": 270}, "response": "Steering."}'
    )
    assert isinstance(parsed.action, NavigationAction)
    assert parsed.action.course == 270


def test_parse_autopilot() -> None:
    parsed = parse_response(
        '{"action": {"type": "autopilot", "state": "engaged"}, '
        '"response": "Autopilot engaged, aye."}'
    )
    assert isinstance(parsed.action, AutopilotAction)
    assert parsed.action.state == "engaged"


def test_parse_anchor_with_chain_length() -> None:
    parsed = parse_response(
        '{"action": {"type": "anchor", "operation": "let_out_chain", '
        '"chain_length": 30}, "response": "Letting out chain, aye."}'
    )
    assert isinstance(parsed.action, AnchorAction)
    assert parsed.action.operation == "let_out_chain"
    assert parsed.action.chain_length == 30


def test_parse_anchor_chain_length_optional_for_drop() -> None:
    parsed = parse_response(
        '{"action": {"type": "anchor", "operation": "drop"}, '
        '"response": "Dropping anchor, aye."}'
    )
    assert isinstance(parsed.action, AnchorAction)
    assert parsed.action.chain_length is None


def test_parse_status_query() -> None:
    parsed = parse_response(
        '{"action": {"type": "status_query", "query": "heading"}, '
        '"response": "Checking heading, sir."}'
    )
    assert isinstance(parsed.action, StatusQueryAction)
    assert parsed.action.query == "heading"


def test_parse_error_action() -> None:
    parsed = parse_response(
        '{"action": {"type": "error", "error_type": "out_of_scope", '
        '"reason": "r", "suggestion": "s"}, "response": "No, sir."}'
    )
    assert isinstance(parsed.action, ErrorAction)
    assert parsed.action.error_type == "out_of_scope"


def test_parse_strips_code_fence() -> None:
    raw = (
        '```json\n{"action": {"type": "status_query", "query": "speed"}, '
        '"response": "Aye."}\n```'
    )
    assert isinstance(parse_response(raw).action, StatusQueryAction)


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(ActionParseError):
        parse_response("the model rambled instead of returning JSON")


def test_parse_rejects_empty() -> None:
    with pytest.raises(ActionParseError):
        parse_response("   ")


def test_parse_rejects_unknown_action_type() -> None:
    with pytest.raises(ActionParseError):
        parse_response('{"action": {"type": "fire_torpedo"}, "response": "x"}')


def test_parse_rejects_multi_step() -> None:
    """multi_step was intentionally dropped from the v1 vocabulary."""
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "multi_step", "steps": []}, "response": "x"}'
        )


def test_parse_rejects_rudder_over_max() -> None:
    """Safety limit: rudder max 45 degrees."""
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "rudder", "direction": "starboard", '
            '"degrees": 60}, "response": "x"}'
        )


def test_parse_rejects_speed_over_max() -> None:
    """Safety limit: speed max 30 knots."""
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "throttle", "speed": 50, "unit": "knots"}, '
            '"response": "x"}'
        )


def test_parse_rejects_course_out_of_range() -> None:
    """Safety limit: heading 0-359."""
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "navigation", "course": 400}, "response": "x"}'
        )


def test_parse_rejects_invalid_rudder_direction() -> None:
    with pytest.raises(ActionParseError):
        parse_response(
            '{"action": {"type": "rudder", "direction": "up", "degrees": 10}, '
            '"response": "x"}'
        )


# --- _knots_to_telegraph -------------------------------------------------


@pytest.mark.parametrize(
    ("knots", "expected_order"),
    [
        (0, EngineOrder.STOP),
        (2, EngineOrder.DEAD_SLOW_AHEAD),
        (6, EngineOrder.SLOW_AHEAD),
        (10, EngineOrder.HALF_AHEAD),
        (15, EngineOrder.FULL_AHEAD),
        (30, EngineOrder.FULL_AHEAD),
        (-2, EngineOrder.DEAD_SLOW_ASTERN),
        (-6, EngineOrder.SLOW_ASTERN),
        (-10, EngineOrder.HALF_ASTERN),
        (-15, EngineOrder.FULL_ASTERN),
    ],
)
def test_knots_to_telegraph_bands(knots: float, expected_order: EngineOrder) -> None:
    assert _knots_to_telegraph(knots) is expected_order


# --- dispatch_action ----------------------------------------------------


async def test_dispatch_navigation_sets_absolute_heading() -> None:
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "navigation", "course": 270}, '
        '"response": "Steering two seven zero, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ok
    assert result.spoken == "Steering two seven zero, aye."
    assert [c.command for c in sim.command_history] == ["set_heading"]
    assert result.ship_state is not None
    assert result.ship_state.heading_deg == 270.0


async def test_dispatch_rudder_starboard_adds_delta_to_current() -> None:
    sim = MockSimulatorClient(initial_heading=100, log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "rudder", "direction": "starboard", "degrees": 30}, '
        '"response": "Starboard thirty, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ship_state is not None
    assert result.ship_state.heading_deg == 130.0


async def test_dispatch_rudder_port_subtracts() -> None:
    sim = MockSimulatorClient(initial_heading=10, log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "rudder", "direction": "port", "degrees": 30}, '
        '"response": "Port thirty, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    # 10 - 30 = -20, wrapped to 340
    assert result.ship_state is not None
    assert result.ship_state.heading_deg == 340.0


async def test_dispatch_throttle_maps_to_engine_telegraph() -> None:
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "throttle", "speed": 15, "unit": "knots"}, '
        '"response": "All ahead full, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ship_state is not None
    assert result.ship_state.engine_order is EngineOrder.FULL_AHEAD


async def test_dispatch_status_query_heading_appends_reading() -> None:
    sim = MockSimulatorClient(initial_heading=87, log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "status_query", "query": "heading"}, '
        '"response": "Checking heading, sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert "Heading 87 degrees" in result.spoken
    assert result.spoken.startswith("Checking heading, sir.")


async def test_dispatch_status_query_speed_appends_reading() -> None:
    sim = MockSimulatorClient(
        initial_engine_order=EngineOrder.HALF_AHEAD, log_commands=False
    )
    parsed = parse_response(
        '{"action": {"type": "status_query", "query": "speed"}, '
        '"response": "Checking speed, sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert "12.0 knots" in result.spoken


async def test_dispatch_status_query_position_is_unavailable() -> None:
    """The simulator doesn't track position; we say so explicitly."""
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "status_query", "query": "position"}, '
        '"response": "Checking position, sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert "not available" in result.spoken.lower()


async def test_dispatch_autopilot_acks_without_touching_simulator() -> None:
    """v1: autopilot/anchor are ack-only -- no simulator state change."""
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "autopilot", "state": "engaged"}, '
        '"response": "Autopilot engaged, aye sir."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ok
    assert result.spoken == "Autopilot engaged, aye sir."
    assert sim.command_history == [], "autopilot must not drive the simulator yet"
    assert result.ship_state is None


async def test_dispatch_anchor_acks_without_touching_simulator() -> None:
    sim = MockSimulatorClient(log_commands=False)
    parsed = parse_response(
        '{"action": {"type": "anchor", "operation": "drop"}, '
        '"response": "Dropping anchor, aye."}'
    )
    result = await dispatch_action(parsed, sim)
    assert result.ok
    assert sim.command_history == []


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
        '{"action": {"type": "navigation", "course": 90}, "response": "x"}'
    )
    result = await dispatch_action(parsed, _FailingSimulator())
    assert not result.ok
    assert result.spoken == BRIDGE_LOST


# --- JsonActionProcessor ------------------------------------------------


async def test_processor_resolves_navigation_and_speaks() -> None:
    sim = MockSimulatorClient(log_commands=False)
    proc = JsonActionProcessor(simulator=sim)
    spoken = await proc._resolve(
        '{"action": {"type": "navigation", "course": 90}, '
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
