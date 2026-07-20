"""Simulator client contract: protocol, ship state, engine telegraph orders.

Tool handlers depend on the ``SimulatorClient`` protocol defined here, never on
a concrete implementation.

The vocabulary is deliberately that of a *conning order*, not an autopilot: the
helmsman is the rating on the wheel, so it puts the **rudder** to an ordered
angle and holds it until countermanded, and moves the **engine telegraph**.
There is no ``set_heading``: steering a compass course is a control loop that,
in real pilotage, is the helmsman's own work against the compass. That belongs
in a later "steering skill" on top of ``set_rudder`` -- see issue #29.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

# Called on every connection-state transition. Sync and must not block: the
# real backend invokes it from its supervisor loop. Taking a plain callable
# rather than the event bus keeps this package from importing the API layer.
StateListener = Callable[["ConnectionState"], None]


class EngineOrder(str, Enum):
    """The nine valid engine telegraph positions."""

    FULL_ASTERN = "full_astern"
    HALF_ASTERN = "half_astern"
    SLOW_ASTERN = "slow_astern"
    DEAD_SLOW_ASTERN = "dead_slow_astern"
    STOP = "stop"
    DEAD_SLOW_AHEAD = "dead_slow_ahead"
    SLOW_AHEAD = "slow_ahead"
    HALF_AHEAD = "half_ahead"
    FULL_AHEAD = "full_ahead"


class ConnectionState(str, Enum):
    """Health of the link to the simulator.

    ``STALE`` is distinct from ``DISCONNECTED``: the session still exists and a
    reconnect is being attempted, but no fresh data has arrived recently. Both
    refuse commands -- a helm order must never be executed against a dead link
    (or, worse, minutes late once it revives).
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STALE = "stale"


class SimulatorError(Exception):
    """Raised when a simulator backend cannot complete an operation.

    Tool handlers catch this to surface a short spoken failure phrase
    ("Lost contact with the bridge") instead of crashing the pipeline. Backends
    raise it on connection loss, timeouts, and wrapper errors.
    """


@dataclass(slots=True)
class ShipState:
    """A snapshot of the ship's navigational state.

    ``rudder_angle_deg`` follows the helm convention: **negative is port,
    positive is starboard**, matching :meth:`SimulatorClient.set_rudder`. It is
    the *actual* angle, which lags an order -- a real rudder slews at only a few
    degrees per second.

    ``sim_time_s`` (the exercise clock, seconds) and ``lat_deg``/``lon_deg``
    (GPS position, signed decimal degrees) are ``None`` when the backend cannot
    provide them -- the mock has no exercise clock and no world to be in.
    """

    heading_deg: float
    speed_kn: float
    engine_order: EngineOrder
    rudder_angle_deg: float
    timestamp: datetime
    sim_time_s: float | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None


@runtime_checkable
class SimulatorClient(Protocol):
    """Async interface every simulator backend must implement."""

    @property
    def connection_state(self) -> ConnectionState:
        """Current link health. The mock is always ``CONNECTED``."""
        ...

    def set_state_listener(self, listener: StateListener | None) -> None:
        """Register a callback for connection-state transitions.

        Push, not poll: a link can drop at any moment, and the operator needs to
        know before giving an order, not after hearing "lost contact". Callers
        that need a starting value should read :attr:`connection_state` -- a
        listener only reports *changes*, and a backend that never changes state
        (the mock) will never call it.
        """
        ...

    async def connect(self) -> None:
        """Establish the session and start supervising it.

        Must not raise if the simulator is simply not running: the agent has to
        start (and stay useful for questions) with no simulator present. The
        state stays ``CONNECTING`` and the backend keeps trying.
        """
        ...

    async def disconnect(self) -> None:
        """Tear the session down. Idempotent, and repeatable with ``connect``."""
        ...

    async def set_rudder(self, angle_deg: float) -> ShipState:
        """Order the rudder to ``angle_deg`` (negative port, positive starboard).

        The order is *held* until countermanded; ``0`` is midships.
        """
        ...

    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState: ...

    async def get_state(self) -> ShipState: ...

    async def close(self) -> None: ...
