"""The periodic ship-state publisher: cadence, gating, and fault absorption."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from voice_agent.api.events import EventBus, ShipStateEvent
from voice_agent.backends.simulator.base import (
    ConnectionState,
    EngineOrder,
    ShipState,
    SimulatorError,
)
from voice_agent.backends.simulator.mock import MockSimulatorClient
from voice_agent.telemetry import run_ship_state_publisher


class _FlakySimulator:
    """Scriptable stand-in: a fixed connection state and a scriptable read."""

    def __init__(self, state: ConnectionState, *, raises: bool = False) -> None:
        self._state = state
        self._raises = raises
        self.reads = 0

    @property
    def connection_state(self) -> ConnectionState:
        return self._state

    async def get_state(self) -> ShipState:
        self.reads += 1
        if self._raises:
            raise SimulatorError("link died mid-read")
        return ShipState(
            heading_deg=90.0,
            speed_kn=5.0,
            engine_order=EngineOrder.HALF_AHEAD,
            rudder_angle_deg=-10.0,
            timestamp=datetime.now(timezone.utc),
            sim_time_s=1234.5,
            lat_deg=49.30,
            lon_deg=2.31,
        )


async def _run_publisher_briefly(simulator, bus: EventBus, ticks: int = 3) -> list:
    queue = bus.subscribe()
    task = asyncio.create_task(
        run_ship_state_publisher(simulator, bus, period_s=0.01)
    )
    await asyncio.sleep(0.01 * ticks + 0.05)
    task.cancel()
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    bus.unsubscribe(queue)
    return events


async def test_publishes_ship_state_while_connected() -> None:
    sim = _FlakySimulator(ConnectionState.CONNECTED)
    events = await _run_publisher_briefly(sim, EventBus())
    assert len(events) >= 2  # repeated, not a one-shot
    event = events[0]
    assert isinstance(event, ShipStateEvent)
    assert event.heading_deg == 90.0
    assert event.sim_time_s == 1234.5
    assert event.lat_deg == 49.30
    assert event.lon_deg == 2.31


async def test_mock_simulator_publishes_with_null_extras() -> None:
    """The mock has no exercise clock or position; the event must say so."""
    sim = MockSimulatorClient(initial_heading=10.0)
    events = await _run_publisher_briefly(sim, EventBus())
    assert events, "mock is always connected, so telemetry must flow"
    assert events[0].sim_time_s is None
    assert events[0].lat_deg is None


async def test_silent_while_not_connected() -> None:
    sim = _FlakySimulator(ConnectionState.CONNECTING)
    events = await _run_publisher_briefly(sim, EventBus())
    assert events == []
    assert sim.reads == 0  # gated on state, never even reads


async def test_read_failure_skips_tick_and_keeps_running() -> None:
    """A SimulatorError mid-read must not kill the publisher task."""
    sim = _FlakySimulator(ConnectionState.CONNECTED, raises=True)
    events = await _run_publisher_briefly(sim, EventBus())
    assert events == []
    assert sim.reads >= 2  # kept trying after the failure
