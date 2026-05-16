"""Real simulator backend: adapter around the vendored pythonnet wrappers.

The in-house wrapper ``.py`` files and the managed .NET DLL are dropped by the
user into ``voice_agent/backends/simulator/vendor/``. This adapter maps the
``SimulatorClient`` protocol onto the wrapper API.

Async/sync bridge: the wrapper's concurrency model is not yet fixed. If its API
is synchronous and blocks on UDP I/O, every call is offloaded with
``asyncio.to_thread()`` so the Pipecat event loop is never blocked. If a future
threaded wrapper is already async-safe, the ``to_thread()`` hops become cheap
no-ops and can be dropped.

Windows-only: the .NET DLL is loaded via ``pythonnet``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from voice_agent.backends.simulator.base import EngineOrder, ShipState, SimulatorError

_VENDOR_DIR = Path(__file__).parent / "vendor"


class RealSimulatorClient:
    """Adapts the vendored UDP-sync wrapper to the ``SimulatorClient`` protocol.

    Kept deliberately tiny: all wrapper-specific calls live in the ``_*_sync``
    helpers below, each marked with a ``TODO(integration)`` where the actual
    vendored method name goes.
    """

    def __init__(self, host: str, port: int, connect_timeout_seconds: float) -> None:
        self._log = structlog.get_logger().bind(component="simulator")
        self._host = host
        self._port = port
        self._connect_timeout = connect_timeout_seconds
        self._wrapper: Any = self._connect()

    # -- connection -------------------------------------------------------

    def _connect(self) -> Any:
        """Load the vendored wrapper and construct the underlying client.

        TODO(integration): replace the body with the real wrapper import and
        constructor. The wrapper module typically runs ``import clr;
        clr.AddReference(<dll>)`` at import time, e.g.::

            from voice_agent.backends.simulator.vendor.ship_bridge import ShipBridge
            return ShipBridge(host=self._host, port=self._port,
                              timeout=self._connect_timeout)

        See ``backends/simulator/vendor/README.md``.
        """
        try:
            raise NotImplementedError(
                "Real simulator wrapper not yet vendored. Drop the in-house "
                f"wrapper .py files and .NET DLL into {_VENDOR_DIR} and wire up "
                "RealSimulatorClient._connect "
                "(see backends/simulator/vendor/README.md)."
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # DLL load / socket / timeout
            self._log.error(
                "simulator_connect_failed",
                host=self._host,
                port=self._port,
                error=str(exc),
            )
            raise SimulatorError(
                f"Could not connect to simulator at {self._host}:{self._port}"
            ) from exc

    # -- sync helpers (run off the event loop) ----------------------------

    def _to_ship_state(self, raw: Any) -> ShipState:
        """Convert a wrapper state object into a ``ShipState``.

        TODO(integration): map the wrapper's fields onto ShipState. Field names
        below are placeholders for the real wrapper attributes.
        """
        return ShipState(
            heading_deg=float(raw.heading),  # TODO(integration): real attr name
            speed_kn=float(raw.speed),  # TODO(integration): real attr name
            engine_order=EngineOrder(raw.engine_order),  # TODO(integration)
            timestamp=datetime.now(timezone.utc),
        )

    def _set_heading_sync(self, degrees: float) -> ShipState:
        # TODO(integration): self._wrapper.set_heading(degrees); read back state.
        raise NotImplementedError("RealSimulatorClient._set_heading_sync")

    def _set_engine_telegraph_sync(self, order: EngineOrder) -> ShipState:
        # TODO(integration): self._wrapper.set_telegraph(order.value); read back.
        raise NotImplementedError("RealSimulatorClient._set_engine_telegraph_sync")

    def _get_state_sync(self) -> ShipState:
        # TODO(integration): read the wrapper's last-synced state mirror.
        # A UDP-sync library keeps a local mirror, so this is a cheap read with
        # no round-trip; setters above may lag it by one sync tick.
        raise NotImplementedError("RealSimulatorClient._get_state_sync")

    def _close_sync(self) -> None:
        # TODO(integration): self._wrapper.close() / dispose sockets / join threads.
        raise NotImplementedError("RealSimulatorClient._close_sync")

    # -- SimulatorClient protocol -----------------------------------------

    async def set_heading(self, degrees: float) -> ShipState:
        try:
            return await asyncio.to_thread(self._set_heading_sync, degrees)
        except Exception as exc:
            self._log.error("simulator_set_heading_failed", error=str(exc))
            raise SimulatorError("set_heading failed") from exc

    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState:
        try:
            return await asyncio.to_thread(self._set_engine_telegraph_sync, order)
        except Exception as exc:
            self._log.error("simulator_set_telegraph_failed", error=str(exc))
            raise SimulatorError("set_engine_telegraph failed") from exc

    async def get_state(self) -> ShipState:
        try:
            return await asyncio.to_thread(self._get_state_sync)
        except Exception as exc:
            self._log.error("simulator_get_state_failed", error=str(exc))
            raise SimulatorError("get_state failed") from exc

    async def close(self) -> None:
        try:
            await asyncio.to_thread(self._close_sync)
        except Exception as exc:  # cleanup failure is logged, not raised
            self._log.warning("simulator_close_failed", error=str(exc))


def build_simulator(config: Any) -> RealSimulatorClient:
    """Build the real ``SimulatorClient`` from the ``simulator.real`` config block."""
    return RealSimulatorClient(
        host=config.host,
        port=int(config.port),
        connect_timeout_seconds=float(config.connect_timeout_seconds),
    )
