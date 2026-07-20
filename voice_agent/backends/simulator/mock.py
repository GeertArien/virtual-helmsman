"""Mock simulator backend: in-memory ``SimulatorClient`` (default for dev/tests).

Holds heading, rudder angle, engine order, and a speed derived by a simple
static mapping from the engine telegraph order. No ship dynamics — no inertia,
no turn rate, and **no steering**: an ordered rudder angle is recorded but the
heading never changes as a result. That is deliberate. The mock exists to prove
the command path (LLM -> dispatch -> client), not to fly the ship; modelling a
turn here would invent numbers the real simulator is the only authority on.
Every command is appended to ``command_history`` for test assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from voice_agent.backends.simulator.base import (
    ConnectionState,
    EngineOrder,
    ShipState,
    StateListener,
)

# Engine telegraph order -> speed in knots. Static mock mapping; astern orders
# are negative. The real simulator derives speed from modelled dynamics.
_SPEED_BY_ORDER: dict[EngineOrder, float] = {
    EngineOrder.FULL_ASTERN: -20.0,
    EngineOrder.HALF_ASTERN: -12.0,
    EngineOrder.SLOW_ASTERN: -6.0,
    EngineOrder.DEAD_SLOW_ASTERN: -3.0,
    EngineOrder.STOP: 0.0,
    EngineOrder.DEAD_SLOW_AHEAD: 3.0,
    EngineOrder.SLOW_AHEAD: 6.0,
    EngineOrder.HALF_AHEAD: 12.0,
    EngineOrder.FULL_AHEAD: 20.0,
}


@dataclass(slots=True)
class CommandRecord:
    """One command recorded by the mock, for test inspection."""

    ts: datetime
    command: str
    arguments: dict[str, Any]
    result: ShipState


class MockSimulatorClient:
    """In-memory implementation of the ``SimulatorClient`` protocol.

    The default simulator backend for development and tests: the full pipeline
    can run without a real simulator. State updates are synchronous and
    in-process; the async methods never block.
    """

    def __init__(
        self,
        *,
        initial_heading: float = 0.0,
        initial_engine_order: EngineOrder = EngineOrder.STOP,
        log_commands: bool = True,
    ) -> None:
        self._heading_deg: float = initial_heading % 360
        self._engine_order: EngineOrder = initial_engine_order
        self._rudder_angle_deg: float = 0.0
        self._log_commands: bool = log_commands
        # Exposed for test assertions; one entry per executed command.
        self.command_history: list[CommandRecord] = []
        self._log = structlog.get_logger().bind(component="simulator")

    @property
    def connection_state(self) -> ConnectionState:
        """Always connected: there is no link to lose."""
        return ConnectionState.CONNECTED

    def set_state_listener(self, listener: StateListener | None) -> None:
        """Accepted and never called: the mock's state cannot change.

        Not an oversight -- a listener reports transitions, and there are none.
        Consumers read :attr:`connection_state` for the (constant) value.
        """
        return None

    def _snapshot(self) -> ShipState:
        """Build a fresh ShipState from current in-memory state."""
        return ShipState(
            heading_deg=self._heading_deg,
            speed_kn=_SPEED_BY_ORDER[self._engine_order],
            engine_order=self._engine_order,
            rudder_angle_deg=self._rudder_angle_deg,
            timestamp=datetime.now(timezone.utc),
        )

    def _record(self, command: str, arguments: dict[str, Any], result: ShipState) -> None:
        """Append a command to history and optionally log it at INFO."""
        self.command_history.append(
            CommandRecord(
                ts=result.timestamp,
                command=command,
                arguments=arguments,
                result=result,
            )
        )
        if self._log_commands:
            self._log.info(
                "simulator_command",
                command=command,
                arguments=arguments,
                heading_deg=result.heading_deg,
                speed_kn=result.speed_kn,
                engine_order=result.engine_order.value,
                rudder_angle_deg=result.rudder_angle_deg,
            )

    async def connect(self) -> None:
        """No-op: the mock has no link to establish."""
        return None

    async def disconnect(self) -> None:
        """No-op: the mock has no link to tear down."""
        return None

    async def set_rudder(self, angle_deg: float) -> ShipState:
        """Record the ordered rudder angle (negative port, positive starboard).

        The angle is applied instantly and the heading does not follow -- see
        the module docstring on why the mock does not model the turn.
        """
        self._rudder_angle_deg = angle_deg
        state = self._snapshot()
        self._record("set_rudder", {"angle_deg": angle_deg}, state)
        return state

    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState:
        """Set the engine telegraph order and return new state."""
        self._engine_order = order
        state = self._snapshot()
        self._record("set_engine_telegraph", {"order": order.value}, state)
        return state

    async def get_state(self) -> ShipState:
        """Return the current ship state. Not recorded in command_history."""
        return self._snapshot()

    async def close(self) -> None:
        """No-op: the mock holds no external resources."""
        return None


def build_simulator(config: Any) -> MockSimulatorClient:
    """Build the mock ``SimulatorClient`` from the ``simulator.mock`` config block."""
    order = config.initial_engine_order
    if isinstance(order, str):
        order = EngineOrder(order)
    return MockSimulatorClient(
        initial_heading=float(config.initial_heading),
        initial_engine_order=order,
        log_commands=bool(config.log_commands),
    )
