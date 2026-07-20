"""Periodic ship-state telemetry for the dashboard.

Dispatch-driven :class:`~voice_agent.api.events.ShipStateEvent` publishes only
capture the moment an order executes; between orders the ship keeps swinging
and the panel would freeze at the last dispatch snapshot. This publisher reads
the simulator at a fixed cadence while the link is up and pushes the same
event, so the panel tracks the ship continuously.

Runs as one background task owned by ``main._serve``. Failures are absorbed:
telemetry must never take the agent down, and a broken read simply means no
update this tick -- the link supervisor is the component responsible for
noticing and reporting a dead link.
"""

from __future__ import annotations

import asyncio

from voice_agent.api.events import EventBus, ShipStateEvent
from voice_agent.backends.simulator.base import (
    ConnectionState,
    SimulatorClient,
    SimulatorError,
)

# 2 Hz: fast enough that heading/position visibly track the ship, slow enough
# that the WebSocket traffic stays negligible (~100 bytes per event).
DEFAULT_PERIOD_S = 0.5


async def run_ship_state_publisher(
    simulator: SimulatorClient,
    event_bus: EventBus,
    *,
    period_s: float = DEFAULT_PERIOD_S,
) -> None:
    """Publish a ``ShipStateEvent`` every ``period_s`` while connected.

    Runs until cancelled. Skips a tick (rather than raising) when the link is
    down or the read fails -- the connection-state event stream already tells
    the dashboard *why* the telemetry went quiet.
    """
    while True:
        if simulator.connection_state is ConnectionState.CONNECTED:
            try:
                state = await simulator.get_state()
            except SimulatorError:
                pass
            else:
                event_bus.publish(
                    ShipStateEvent(
                        heading_deg=state.heading_deg,
                        speed_kn=state.speed_kn,
                        engine_order=state.engine_order.value,
                        rudder_angle_deg=state.rudder_angle_deg,
                        sim_time_s=state.sim_time_s,
                        lat_deg=state.lat_deg,
                        lon_deg=state.lon_deg,
                    )
                )
        await asyncio.sleep(period_s)
