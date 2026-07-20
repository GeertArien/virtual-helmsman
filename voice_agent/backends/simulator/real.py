"""Real simulator backend: lifecycle around the vendored wrapper.

This adapter owns *when* there is a link; the wrapper
(:mod:`voice_agent.backends.simulator.wrapper_api`) owns *how* to talk over it.
Everything interesting is here, in code that runs anywhere and is tested against
a fake wrapper -- the vendored file is a thin translation layer that cannot be
imported off Windows (see ``vendor/README.md``).

The lifecycle exists because the link has two awkward properties:

* **It is connectionless.** Joining costs nothing and never fails, even with no
  simulator running, so "connected" cannot be answered by the transport. It is
  inferred from whether frames are arriving.
* **Loss is silent.** Nothing raises, nothing fires. A dead link looks exactly
  like a live one until you notice the last frame is old.

So: connect lazily and never fail startup (the agent must be usable for
questions with no simulator present), watch the frame age, and rebuild the
session whenever it goes quiet. Commands fail fast while the link is down --
a helm order must never be silently queued and then executed minutes later.

Two rules keep the acknowledgment honest, because "the state said CONNECTED"
is not the same as "the order reached the ship":

* **One session owner at a time.** Every session mutation -- commands staging
  orders, the supervisor dropping or rebuilding -- happens under one lock, so
  an order can never be staged into a session the supervisor is concurrently
  discarding.
* **Acknowledge on delivery, not on staging.** Setpoints only reach the
  simulator while frames flow, so a command completes only after the wrapper
  has received further frames *after* staging. If they do not come, the order
  is reported failed and the session (with the undelivered order in it) is
  dropped on the spot -- it must never fire later when the link revives.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from voice_agent.backends.simulator.base import (
    ConnectionState,
    EngineOrder,
    ShipState,
    SimulatorError,
    StateListener,
)
from voice_agent.backends.simulator.wrapper_api import (
    SimulatorWrapper,
    WrapperFactory,
    WrapperSnapshot,
)

# Engine telegraph position -> lever value in [-1.0, 1.0].
#
# This mapping is ours to choose: the simulator's lever is continuous and does
# not snap to detents (setting 0.7 reads back 0.7), so nothing on the far side
# defines what "half ahead" is worth. The curve is deliberately not linear --
# the gap between dead slow and slow matters more to a ship handler than the
# gap between half and full.
_LEVER_BY_ORDER: dict[EngineOrder, float] = {
    EngineOrder.FULL_ASTERN: -1.0,
    EngineOrder.HALF_ASTERN: -0.7,
    EngineOrder.SLOW_ASTERN: -0.45,
    EngineOrder.DEAD_SLOW_ASTERN: -0.2,
    EngineOrder.STOP: 0.0,
    EngineOrder.DEAD_SLOW_AHEAD: 0.2,
    EngineOrder.SLOW_AHEAD: 0.45,
    EngineOrder.HALF_AHEAD: 0.7,
    EngineOrder.FULL_AHEAD: 1.0,
}

# Frames that must arrive after staging before an order counts as delivered.
# Two, not one: the setpoint is written during one frame's update callback and
# flushed to the simulator on a received frame, so only after two further
# frames has a full receive cycle provably run with the order in place.
_DELIVERY_FRAMES = 2


def _order_from_lever(value: float) -> EngineOrder:
    """Nearest telegraph position to a lever reading.

    The lever is continuous and other consoles can move it, so a reading rarely
    lands exactly on one of ours; report the closest position rather than
    pretending the value is invalid.
    """
    return min(_LEVER_BY_ORDER, key=lambda o: abs(_LEVER_BY_ORDER[o] - value))


def _default_wrapper_factory(**kwargs: Any) -> SimulatorWrapper:
    """Import the vendored wrapper lazily and build a session.

    Deferred to call time on purpose: the vendor integration exists only on a
    machine it has been hand-dropped onto. The mock backend, the test suite and
    every other developer must never trip over that import.
    """
    try:
        from voice_agent.backends.simulator.vendor.ship_bridge import ShipBridge
    except ImportError as exc:
        raise SimulatorError(
            "The simulator wrapper is not vendored on this machine. Drop the "
            "vendor integration files into voice_agent/backends/simulator/"
            "vendor/ (see that directory's README.md), or set "
            f"simulator.backend: mock. Import failed: {exc}"
        ) from exc
    return ShipBridge(**kwargs)


class RealSimulatorClient:
    """``SimulatorClient`` over a supervised wrapper session."""

    def __init__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        local_host: str,
        local_port: int,
        connect_timeout_seconds: float,
        expected_fps: float,
        stale_after_missed_frames: float,
        reconnect_initial_seconds: float,
        reconnect_max_seconds: float,
        wrapper_factory: WrapperFactory | None = None,
    ) -> None:
        self._log = structlog.get_logger().bind(component="simulator")
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._local_host = local_host
        self._local_port = local_port
        self._connect_timeout = connect_timeout_seconds
        self._expected_fps = expected_fps
        self._reconnect_initial = reconnect_initial_seconds
        self._reconnect_max = reconnect_max_seconds
        self._factory: WrapperFactory = wrapper_factory or _default_wrapper_factory

        # A link is stale once the newest frame is older than this. Derived from
        # the frame rate so it scales with the simulator rather than being a
        # magic number. Doubles as the delivery-confirmation budget: an order
        # whose proof-of-delivery frames do not arrive within it has failed.
        self._stale_after_s = stale_after_missed_frames / max(expected_fps, 1.0)

        self._wrapper: SimulatorWrapper | None = None
        self._supervisor: asyncio.Task[None] | None = None
        # A session build the supervisor was cancelled out of. The thread
        # cannot be interrupted, so disconnect() reaps it (see _build_guarded).
        self._pending_build: asyncio.Task[SimulatorWrapper] | None = None
        self._state = ConnectionState.DISCONNECTED
        self._listener: StateListener | None = None
        # Session ownership: held by commands for stage-and-confirm and by the
        # supervisor for drop/rebuild, so neither can pull the session out from
        # under the other. Everything held under it is bounded (no open-ended
        # waits), so contention is at worst one confirmation budget.
        self._lock = asyncio.Lock()
        # Set by connect() while a supervisor is already running: skip the
        # remaining reconnect backoff and try again now.
        self._retry_now = asyncio.Event()

    # -- state ------------------------------------------------------------

    @property
    def connection_state(self) -> ConnectionState:
        return self._state

    def set_state_listener(self, listener: StateListener | None) -> None:
        self._listener = listener

    def _set_state(self, state: ConnectionState) -> None:
        if state is self._state:
            return
        self._log.info(
            "simulator_connection_state", previous=self._state.value, state=state.value
        )
        self._state = state
        if self._listener is None:
            return
        try:
            self._listener(state)
        except Exception as exc:
            # A listener is an observer. Whatever it is doing (publishing to a
            # dashboard, say), it must never take the supervisor down with it --
            # losing the link is bad; losing the thing that reconnects it is worse.
            self._log.warning("simulator_state_listener_failed", error=str(exc))

    # -- session ----------------------------------------------------------

    def _build_session(self) -> SimulatorWrapper:
        wrapper = self._factory(
            remote_host=self._remote_host,
            remote_port=self._remote_port,
            local_host=self._local_host,
            local_port=self._local_port,
            poll_hz=self._expected_fps,
        )
        if not wrapper.start():
            wrapper.stop()
            raise SimulatorError(
                f"Could not open a simulator session on {self._local_host}:"
                f"{self._local_port}"
            )
        return wrapper

    async def _build_guarded(self) -> SimulatorWrapper:
        """Build a session without ever losing track of the result.

        The build runs in a thread, and a thread cannot be cancelled: if the
        supervisor is cancelled at this await, the build finishes anyway and
        would otherwise return a *live* session -- thread running, local port
        bound, ship controls engaged -- that nothing holds a reference to. That
        orphan keeps the port, so every later connect fails until the process
        restarts. Shielding keeps the build task intact and ``_pending_build``
        hands it to :meth:`disconnect`, which awaits it and stops the result.
        """
        build = asyncio.create_task(asyncio.to_thread(self._build_session))
        self._pending_build = build
        try:
            wrapper = await asyncio.shield(build)
        except asyncio.CancelledError:
            # Leave _pending_build set: disconnect() reaps it.
            raise
        except BaseException:
            self._pending_build = None
            raise
        self._pending_build = None
        return wrapper

    def _stop_wrapper(self, wrapper: SimulatorWrapper) -> None:
        try:
            wrapper.stop()
        except Exception as exc:  # a failed teardown must not block a rebuild
            self._log.warning("simulator_session_stop_failed", error=str(exc))

    def _drop_session(self) -> None:
        wrapper, self._wrapper = self._wrapper, None
        if wrapper is not None:
            self._stop_wrapper(wrapper)

    async def connect(self) -> None:
        """Start the session and its supervisor.

        Returns once the link is up *or* the timeout expires -- not raising in
        the latter case is the point: the simulator is often started after the
        agent, and the agent stays useful meanwhile. The supervisor keeps
        trying either way.

        Calling this while already supervising is the operator's "try again
        now": it skips whatever remains of the reconnect backoff, so fixing the
        simulator and clicking Connect takes effect immediately instead of
        after up to the backoff cap.
        """
        async with self._lock:
            if self._supervisor is not None and not self._supervisor.done():
                self._retry_now.set()
            else:
                self._set_state(ConnectionState.CONNECTING)
                self._supervisor = asyncio.create_task(
                    self._supervise(), name="simulator-supervisor"
                )

        # Give the link a chance to come up so a normal startup (simulator
        # already running) reports "connected" rather than "connecting".
        try:
            await asyncio.wait_for(self._wait_for_connected(), self._connect_timeout)
        except asyncio.TimeoutError:
            self._log.info(
                "simulator_connect_timeout",
                seconds=self._connect_timeout,
                note="no frames yet; supervisor keeps retrying",
            )

    async def _wait_for_connected(self) -> None:
        while self._state is not ConnectionState.CONNECTED:
            await asyncio.sleep(0.05)

    async def disconnect(self) -> None:
        """Stop supervising and tear the session down. Idempotent.

        Also reaps a build the supervisor was cancelled out of: the build
        thread finishes regardless, so this waits for it (bounded -- a session
        open is quick) and stops the resulting session. Without that, the
        orphan would keep the local port and block every future connect.
        """
        async with self._lock:
            supervisor, self._supervisor = self._supervisor, None
            if supervisor is not None:
                supervisor.cancel()
                try:
                    await supervisor
                except asyncio.CancelledError:
                    pass
            pending, self._pending_build = self._pending_build, None
            if pending is not None:
                orphan: SimulatorWrapper | None
                try:
                    orphan = await pending
                except asyncio.CancelledError:
                    orphan = None
                except Exception:
                    orphan = None  # failed builds stop their own wrapper
                if orphan is not None:
                    await asyncio.to_thread(self._stop_wrapper, orphan)
            await asyncio.to_thread(self._drop_session)
            self._set_state(ConnectionState.DISCONNECTED)

    async def _supervise(self) -> None:
        """Keep a live session, forever.

        One loop covers first connect, link loss and simulator restarts: they
        are the same event -- no fresh frames -- and the same cure: drop the
        session and build a new one.
        """
        backoff = self._reconnect_initial
        while True:
            try:
                async with self._lock:
                    if self._wrapper is None:
                        self._set_state(ConnectionState.CONNECTING)
                        self._wrapper = await self._build_guarded()

                    snapshot = await asyncio.to_thread(self._snapshot_sync)
                    healthy = (
                        snapshot.initialized
                        and snapshot.frame_age_s <= self._stale_after_s
                    )

                    if healthy:
                        self._set_state(ConnectionState.CONNECTED)
                        backoff = self._reconnect_initial
                        self._retry_now.clear()  # a stale kick must not skip a future backoff
                    elif self._state is ConnectionState.CONNECTED:
                        # Was live, has gone quiet: the simulator stopped, died,
                        # or the exercise ended. Rebuild rather than wait for a
                        # recovery that the transport will never announce.
                        self._log.warning(
                            "simulator_link_stale",
                            frame_age_s=round(snapshot.frame_age_s, 2),
                            threshold_s=round(self._stale_after_s, 2),
                        )
                        self._set_state(ConnectionState.STALE)
                        await asyncio.to_thread(self._drop_session)

                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Includes a wrapper that cannot even be built (not vendored,
                # DLLs missing). Retrying is harmless and costs one log line
                # per backoff, and the operator may yet fix it underneath us.
                self._log.warning(
                    "simulator_session_failed",
                    error=str(exc),
                    retry_in_s=round(backoff, 1),
                )
                async with self._lock:
                    await asyncio.to_thread(self._drop_session)
                self._set_state(ConnectionState.CONNECTING)
                await self._sleep_backoff(backoff)
                backoff = min(backoff * 2, self._reconnect_max)

    async def _sleep_backoff(self, seconds: float) -> None:
        """Back off, but let a manual connect() cut the wait short."""
        try:
            await asyncio.wait_for(self._retry_now.wait(), timeout=seconds)
            self._log.info("simulator_retry_requested", skipped_backoff_s=seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            self._retry_now.clear()

    # -- sync helpers (run off the event loop) ----------------------------

    def _snapshot_sync(self) -> WrapperSnapshot:
        wrapper = self._wrapper
        if wrapper is None:
            raise SimulatorError("no simulator session")
        return wrapper.snapshot()

    def _to_ship_state(self, snapshot: WrapperSnapshot) -> ShipState:
        return ShipState(
            heading_deg=snapshot.heading_deg,
            speed_kn=snapshot.speed_kn,
            engine_order=_order_from_lever(snapshot.engine_lever),
            rudder_angle_deg=snapshot.rudder_angle_deg,
            timestamp=datetime.now(timezone.utc),
            sim_time_s=snapshot.sim_time_s,
            lat_deg=snapshot.lat_deg,
            lon_deg=snapshot.lon_deg,
        )

    def _require_live(self) -> SimulatorWrapper:
        """The wrapper, or an error naming why there is no link.

        Every command goes through here. Refusing outright is deliberate: a
        staged order would sit in the wrapper and fire the moment the link
        returned, which for a helm order is worse than not executing at all.
        """
        if self._state is not ConnectionState.CONNECTED or self._wrapper is None:
            raise SimulatorError(
                f"no link to the simulator (state: {self._state.value})"
            )
        return self._wrapper

    # -- command path -------------------------------------------------------

    async def _stage_and_confirm(
        self, stage: Callable[[SimulatorWrapper], None]
    ) -> ShipState:
        """Stage an order and return only once its delivery is proven.

        The state enum lags reality by up to the stale threshold, so it alone
        cannot back an acknowledgment. Under the session lock (the supervisor
        cannot swap the session out mid-command):

        1. refuse if the newest frame is already older than the threshold --
           the link is dead, whatever the enum still says;
        2. stage the order;
        3. wait for ``_DELIVERY_FRAMES`` further frames -- setpoints ride
           received frames, so their arrival proves the order went out;
        4. if they never come, drop the session *with the undelivered order
           still in it* and report failure. The order must die with the
           session: reporting "lost contact" and then having the rudder move
           anyway (link revives, staged value delivered) would be worse than
           either outcome alone.
        """
        async with self._lock:
            wrapper = self._require_live()
            before = await asyncio.to_thread(wrapper.snapshot)
            if before.frame_age_s > self._stale_after_s:
                raise SimulatorError(
                    "no link to the simulator (link just went quiet)"
                )

            await asyncio.to_thread(stage, wrapper)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._stale_after_s
            while True:
                snapshot = await asyncio.to_thread(wrapper.snapshot)
                if snapshot.frame_count >= before.frame_count + _DELIVERY_FRAMES:
                    return self._to_ship_state(snapshot)
                if loop.time() >= deadline:
                    self._log.warning(
                        "simulator_order_undelivered",
                        frames_seen=snapshot.frame_count - before.frame_count,
                        budget_s=round(self._stale_after_s, 2),
                    )
                    self._set_state(ConnectionState.STALE)
                    await asyncio.to_thread(self._drop_session)
                    raise SimulatorError(
                        "order not delivered: the link went quiet"
                    )
                await asyncio.sleep(0.01)

    # -- SimulatorClient protocol -----------------------------------------

    async def set_rudder(self, angle_deg: float) -> ShipState:
        try:
            # The returned snapshot is a few frames after staging: the rudder
            # slews at a few degrees per second, so it shows the helm answering,
            # not yet at the ordered angle. The spoken read-back quotes the
            # order itself, which is why that is not a lie.
            return await self._stage_and_confirm(
                lambda wrapper: wrapper.order_rudder(angle_deg)
            )
        except SimulatorError:
            raise
        except Exception as exc:
            self._log.error("simulator_set_rudder_failed", error=str(exc))
            raise SimulatorError("set_rudder failed") from exc

    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState:
        try:
            lever = _LEVER_BY_ORDER[order]
            return await self._stage_and_confirm(
                lambda wrapper: wrapper.order_engine_lever(lever)
            )
        except SimulatorError:
            raise
        except Exception as exc:
            self._log.error("simulator_set_telegraph_failed", error=str(exc))
            raise SimulatorError("set_engine_telegraph failed") from exc

    async def get_state(self) -> ShipState:
        try:
            async with self._lock:
                wrapper = self._require_live()
                snapshot = await asyncio.to_thread(wrapper.snapshot)
            if not snapshot.initialized or snapshot.frame_age_s > self._stale_after_s:
                # The mirror holds the last values a dead link left behind;
                # reading them out as the ship's current state would be
                # fabricating a report. Refuse instead.
                raise SimulatorError("no fresh data from the simulator")
            return self._to_ship_state(snapshot)
        except SimulatorError:
            raise
        except Exception as exc:
            self._log.error("simulator_get_state_failed", error=str(exc))
            raise SimulatorError("get_state failed") from exc

    async def close(self) -> None:
        try:
            await self.disconnect()
        except Exception as exc:  # cleanup failure is logged, not raised
            self._log.warning("simulator_close_failed", error=str(exc))


def build_simulator(config: Any) -> RealSimulatorClient:
    """Build the real ``SimulatorClient`` from the ``simulator.real`` config block.

    Constructing it does not touch the network or load the vendor integration
    -- that starts at :meth:`RealSimulatorClient.connect`, so a misconfigured
    or absent simulator cannot stop the agent from starting.
    """
    unset = [
        name
        for name, value in (
            ("remote_port", config.remote_port),
            ("local_port", config.local_port),
            ("expected_fps", config.expected_fps),
        )
        if float(value) <= 0
    ]
    if unset:
        # The shipped defaults are placeholders on purpose: the working values
        # are properties of the simulator installation and are deliberately
        # not recorded in this repository.
        raise ValueError(
            f"simulator.real.{'/'.join(unset)} not configured. These values "
            "come with the vendor integration notes -- set them in a local "
            "config file (never commit them)."
        )
    return RealSimulatorClient(
        remote_host=config.remote_host,
        remote_port=int(config.remote_port),
        local_host=config.local_host,
        local_port=int(config.local_port),
        connect_timeout_seconds=float(config.connect_timeout_seconds),
        expected_fps=float(config.expected_fps),
        stale_after_missed_frames=float(config.stale_after_missed_frames),
        reconnect_initial_seconds=float(config.reconnect_initial_seconds),
        reconnect_max_seconds=float(config.reconnect_max_seconds),
    )
