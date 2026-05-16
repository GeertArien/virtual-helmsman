"""Mock simulator: sequences of commands produce the expected state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voice_agent.backends.simulator.base import EngineOrder
from voice_agent.backends.simulator.mock import MockSimulatorClient, build_simulator


def _client(**kwargs) -> MockSimulatorClient:
    kwargs.setdefault("log_commands", False)
    return MockSimulatorClient(**kwargs)


async def test_initial_state() -> None:
    client = _client(initial_heading=42.0, initial_engine_order=EngineOrder.SLOW_AHEAD)
    state = await client.get_state()
    assert state.heading_deg == 42.0
    assert state.engine_order is EngineOrder.SLOW_AHEAD
    assert state.speed_kn == 6.0


async def test_set_heading_normalises_to_0_360() -> None:
    client = _client()
    assert (await client.set_heading(370)).heading_deg == 10.0
    assert (await client.set_heading(-10)).heading_deg == 350.0
    assert (await client.set_heading(360)).heading_deg == 0.0
    assert (await client.set_heading(270)).heading_deg == 270.0


@pytest.mark.parametrize(
    ("order", "expected_speed"),
    [
        (EngineOrder.FULL_AHEAD, 20.0),
        (EngineOrder.HALF_AHEAD, 12.0),
        (EngineOrder.SLOW_AHEAD, 6.0),
        (EngineOrder.DEAD_SLOW_AHEAD, 3.0),
        (EngineOrder.STOP, 0.0),
        (EngineOrder.DEAD_SLOW_ASTERN, -3.0),
        (EngineOrder.SLOW_ASTERN, -6.0),
        (EngineOrder.HALF_ASTERN, -12.0),
        (EngineOrder.FULL_ASTERN, -20.0),
    ],
)
async def test_engine_order_maps_to_speed(order: EngineOrder, expected_speed: float) -> None:
    client = _client()
    state = await client.set_engine_telegraph(order)
    assert state.engine_order is order
    assert state.speed_kn == expected_speed


async def test_command_sequence_accumulates_state() -> None:
    client = _client()
    await client.set_heading(90)
    await client.set_engine_telegraph(EngineOrder.HALF_AHEAD)
    state = await client.get_state()
    assert state.heading_deg == 90.0
    assert state.engine_order is EngineOrder.HALF_AHEAD
    assert state.speed_kn == 12.0


async def test_command_history_records_commands_in_order() -> None:
    client = _client()
    await client.set_heading(180)
    await client.set_engine_telegraph(EngineOrder.SLOW_AHEAD)
    await client.get_state()  # queries are not recorded

    history = client.command_history
    assert [rec.command for rec in history] == ["set_heading", "set_engine_telegraph"]
    assert history[0].arguments == {"degrees": 180}
    assert history[1].arguments == {"order": "slow_ahead"}
    assert history[1].result.heading_deg == 180.0


async def test_close_is_noop() -> None:
    client = _client()
    await client.close()  # must not raise


def test_build_simulator_from_config_block() -> None:
    config = SimpleNamespace(
        initial_heading=270,
        initial_engine_order="half_ahead",
        log_commands=False,
    )
    client = build_simulator(config)
    assert isinstance(client, MockSimulatorClient)
