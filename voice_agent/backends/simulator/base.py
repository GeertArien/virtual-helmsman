"""Simulator client contract: protocol, ship state, engine telegraph orders.

Tool handlers depend on the ``SimulatorClient`` protocol defined here, never on
a concrete implementation. Two implementations exist: ``real`` (wraps the
vendored pythonnet wrappers) and ``mock`` (in-memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable


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


class SimulatorError(Exception):
    """Raised when a simulator backend cannot complete an operation.

    Tool handlers catch this to surface a short spoken failure phrase
    ("Lost contact with bridge") instead of crashing the pipeline. The mock
    backend never raises it; the real backend raises it on connection loss,
    timeouts, and wrapper errors.
    """


@dataclass(slots=True)
class ShipState:
    """A snapshot of the ship's navigational state.

    Extend with additional fields as the real wrapper library exposes them.
    """

    heading_deg: float
    speed_kn: float
    engine_order: EngineOrder
    timestamp: datetime


@runtime_checkable
class SimulatorClient(Protocol):
    """Async interface every simulator backend must implement."""

    async def set_heading(self, degrees: float) -> ShipState: ...

    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState: ...

    async def get_state(self) -> ShipState: ...

    async def close(self) -> None: ...
