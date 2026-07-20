"""The contract between the adapter and the vendored simulator wrapper.

Two implementations exist:

* ``vendor/ship_bridge.py`` -- the real one. Not in this repository (see
  ``vendor/README.md``): it speaks the in-house simulator's API, so it is
  hand-dropped per machine and imported lazily by
  :mod:`voice_agent.backends.simulator.real`.
* the fakes in ``tests/`` -- which is why every lifecycle rule below can be
  tested on any platform, with no simulator and no network.

This module is deliberately written in *our* vocabulary rather than the
simulator's. Nothing here names a vendor type, so the interesting logic --
lazy connect, link supervision, reconnect, fail-fast -- lives in `real.py`
where it can be read and tested, and the vendored file stays a thin, dumb
translation layer.

Threading contract
------------------
A session owns one background thread that does *all* work with the simulator
libraries. Everything here is therefore synchronous and must be cheap: the
adapter calls it from ``asyncio.to_thread``. ``snapshot`` and the ``order_*``
methods are called from other threads and must be safe to do so.

Setpoints are *staged*, not sent: an ``order_rudder`` stores the value, and the
session thread re-asserts it on every tick until it is countermanded. That
matches how the transport works -- writes only reach the simulator while frames
are flowing, and the simulator's own broadcast can otherwise overwrite a held
value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class WrapperSnapshot:
    """One consistent read of the wrapper's local state mirror.

    ``frame_age_s`` is how long ago the last update arrived, and is the only
    link-health signal available: the managed simulator wrapper does not expose
    a communication-status API, and the transport is connectionless -- silence
    is indistinguishable from a healthy-but-quiet link at the socket level. A
    live session delivers frames continuously (~40/s), so a frame age of more
    than a few ticks means the link is gone. ``inf`` means nothing has ever
    arrived.

    ``frame_count`` is the total number of updates this session has received.
    It exists because age alone cannot prove *delivery*: staged setpoints only
    reach the simulator while frames flow, so the adapter acknowledges an order
    only after seeing the counter advance past the staging point -- frames that
    arrived after the order was staged are the proof it went out.

    ``initialized`` goes true once the simulator has published a scenario and
    the controls have been engaged; until then the values below are meaningless.

    ``sim_time_s`` is the exercise clock in seconds (wall-clock rate on a
    real-time host). ``lat_deg``/``lon_deg`` are the own ship's GPS position in
    signed decimal **degrees** (the wrapper converts from whatever the
    simulator's native unit is); ``None`` when the session cannot provide them.
    """

    initialized: bool
    frame_age_s: float
    frame_count: int
    heading_deg: float
    speed_kn: float
    yaw_rate_deg_s: float
    rudder_angle_deg: float
    engine_lever: float
    sim_time_s: float = 0.0
    lat_deg: float | None = None
    lon_deg: float | None = None


@runtime_checkable
class SimulatorWrapper(Protocol):
    """A live session with the simulator, driven by its own thread."""

    def start(self) -> bool:
        """Join the simulator's channel and start the session thread.

        Returns whether the session came up. This does **not** mean the
        simulator is running: the transport is connectionless, so a session
        starts happily with nothing on the other end and simply receives no
        frames. Callers decide that via :attr:`WrapperSnapshot.frame_age_s`.
        """
        ...

    def stop(self) -> None:
        """Stop the session thread and release the channel. Idempotent."""
        ...

    def snapshot(self) -> WrapperSnapshot:
        """Read the state mirror. Never blocks on the simulator."""
        ...

    def order_rudder(self, angle_deg: float) -> None:
        """Stage a rudder angle (negative port, positive starboard) to hold."""
        ...

    def order_engine_lever(self, value: float) -> None:
        """Stage an engine lever value in ``[-1.0, 1.0]`` to hold."""
        ...


@runtime_checkable
class WrapperFactory(Protocol):
    """Builds a fresh :class:`SimulatorWrapper` for one session.

    Reconnecting means dropping a session and building a new one, so the
    adapter never reuses a stopped wrapper. Injecting this is also how the
    tests replace the whole simulator with a fake.
    """

    def __call__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        local_host: str,
        local_port: int,
        poll_hz: float,
    ) -> SimulatorWrapper: ...
