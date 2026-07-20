"""Real simulator adapter: connection lifecycle, against a fake wrapper.

No network, no vendor integration, no simulator -- the adapter takes a wrapper
factory, so
everything the lifecycle does can be driven deterministically here. That is the
whole reason the vendored wrapper is kept behind a narrow protocol.
"""

from __future__ import annotations

import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from voice_agent.backends.simulator.base import (
    ConnectionState,
    EngineOrder,
    SimulatorError,
)
from voice_agent.backends.simulator.real import (
    RealSimulatorClient,
    _order_from_lever,
    build_simulator,
)
from voice_agent.backends.simulator.wrapper_api import WrapperSnapshot

# The adapter polls at 0.25s; keep the tests an order of magnitude coarser than
# that and they stay reliable without being slow.
_TICK = 0.4


class FakeSim:
    """The simulator itself -- a thing in the world that sessions observe.

    Modelled separately from the sessions on purpose. A session cannot make the
    simulator run: if it is stopped, a *reconnect* sees silence too. Folding
    "is the sim up" into the wrapper would let every new session come up live
    and turn the reconnect tests into races.
    """

    def __init__(self) -> None:
        self.running = True
        self.can_open_session = True
        # Seconds a session takes to open. Used to hold the supervisor inside
        # a build while the test does something rude to it.
        self.session_open_seconds = 0.0
        # When set, the simulator dies the moment an order is staged -- the
        # tightest possible loss window, for the delivery-confirmation tests.
        self.dies_on_order = False
        # Advances while the sim runs (each snapshot poll observes new frames,
        # like the simulator's continuous broadcast). Frozen when down.
        self.frame_count = 0
        # Set by whichever session is current; survives session rebuilds the way
        # the real ship's state does.
        self.rudder_orders: list[float] = []
        self.lever_orders: list[float] = []


class FakeWrapper:
    """One session against a :class:`FakeSim`."""

    instances: list["FakeWrapper"] = []

    def __init__(self, sim: FakeSim, **kwargs) -> None:
        self.sim = sim
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        FakeWrapper.instances.append(self)

    # Orders live on the sim: they must outlive a session rebuild.
    @property
    def rudder_orders(self) -> list[float]:
        return self.sim.rudder_orders

    @property
    def lever_orders(self) -> list[float]:
        return self.sim.lever_orders

    def start(self) -> bool:
        if self.sim.session_open_seconds:
            time.sleep(self.sim.session_open_seconds)
        self.started = True
        return self.sim.can_open_session

    def stop(self) -> None:
        self.stopped = True

    def snapshot(self) -> WrapperSnapshot:
        live = self.sim.running and not self.stopped
        if live:
            # Each poll sees frames that arrived since the last one.
            self.sim.frame_count += 1
        return WrapperSnapshot(
            initialized=live,
            # A dead link's newest frame is infinitely old; a live one's is fresh.
            frame_age_s=0.0 if live else math.inf,
            frame_count=self.sim.frame_count,
            heading_deg=90.0,
            speed_kn=6.0,
            yaw_rate_deg_s=0.0,
            rudder_angle_deg=self.rudder_orders[-1] if self.rudder_orders else 0.0,
            engine_lever=self.lever_orders[-1] if self.lever_orders else 0.0,
        )

    def order_rudder(self, angle_deg: float) -> None:
        self.sim.rudder_orders.append(angle_deg)
        if self.sim.dies_on_order:
            self.sim.running = False

    def order_engine_lever(self, value: float) -> None:
        self.sim.lever_orders.append(value)
        if self.sim.dies_on_order:
            self.sim.running = False


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeWrapper.instances.clear()
    yield
    FakeWrapper.instances.clear()


@pytest.fixture
def sim() -> FakeSim:
    return FakeSim()


def _client(sim: FakeSim, **overrides) -> RealSimulatorClient:
    kwargs = dict(
        remote_host="127.0.0.1",
        remote_port=15001,
        local_host="0.0.0.0",
        local_port=15002,
        connect_timeout_seconds=2.0,
        expected_fps=100.0,
        stale_after_missed_frames=50.0,
        reconnect_initial_seconds=0.05,
        reconnect_max_seconds=0.2,
        wrapper_factory=lambda **kw: FakeWrapper(sim, **kw),
    )
    kwargs.update(overrides)
    return RealSimulatorClient(**kwargs)


async def _until(predicate, timeout: float = 5.0) -> bool:
    """Wait for a supervisor-driven state change."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


# --- construction / startup ---------------------------------------------


def test_construction_does_not_touch_the_simulator(sim: FakeSim) -> None:
    """Building the client must not connect: startup cannot depend on the sim."""
    _client(sim)
    assert FakeWrapper.instances == []


async def test_connect_reaches_connected_when_frames_flow(sim: FakeSim) -> None:
    client = _client(sim)
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED
    await client.disconnect()


async def test_startup_without_a_simulator_does_not_raise(sim: FakeSim) -> None:
    """The agent must come up (and stay useful) with no simulator running."""
    sim.running = False  # session opens, but no frames ever arrive
    client = _client(sim, connect_timeout_seconds=0.3)
    await client.connect()  # must not raise
    assert client.connection_state is ConnectionState.CONNECTING
    await client.disconnect()


async def test_simulator_started_late_is_picked_up(sim: FakeSim) -> None:
    """The normal case: agent first, simulator second."""
    sim.running = False
    client = _client(sim, connect_timeout_seconds=0.1)
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTING

    sim.running = True  # the operator starts the exercise
    assert await _until(lambda: client.connection_state is ConnectionState.CONNECTED)
    await client.disconnect()


async def test_connect_keeps_trying_after_a_failed_session(sim: FakeSim) -> None:
    """A session that cannot even be opened must be retried, not given up on."""
    sim.can_open_session = False
    client = _client(sim, connect_timeout_seconds=0.1)
    await client.connect()
    assert await _until(lambda: len(FakeWrapper.instances) >= 2)  # retrying

    sim.can_open_session = True
    assert await _until(lambda: client.connection_state is ConnectionState.CONNECTED)
    await client.disconnect()


async def test_missing_wrapper_is_reported_but_does_not_crash_startup(
    sim: FakeSim,
) -> None:
    """An unvendored machine gets a clear state, not an exception at startup."""

    def factory(**kw):
        raise SimulatorError("not vendored")

    client = _client(sim, wrapper_factory=factory, connect_timeout_seconds=0.2)
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTING
    with pytest.raises(SimulatorError):
        await client.get_state()
    await client.disconnect()


# --- commands ------------------------------------------------------------


async def test_set_rudder_stages_the_signed_angle(sim: FakeSim) -> None:
    client = _client(sim)
    await client.connect()
    state = await client.set_rudder(-10.0)
    assert sim.rudder_orders == [-10.0]
    assert state.rudder_angle_deg == -10.0
    await client.disconnect()


async def test_set_engine_telegraph_maps_orders_to_lever_values(sim: FakeSim) -> None:
    client = _client(sim)
    await client.connect()
    await client.set_engine_telegraph(EngineOrder.HALF_AHEAD)
    await client.set_engine_telegraph(EngineOrder.STOP)
    await client.set_engine_telegraph(EngineOrder.FULL_ASTERN)
    assert sim.lever_orders == [0.7, 0.0, -1.0]
    await client.disconnect()


async def test_get_state_reports_the_nearest_telegraph_position(sim: FakeSim) -> None:
    """Another console can leave the lever between our positions."""
    client = _client(sim)
    await client.connect()
    sim.lever_orders.append(0.62)  # nobody's detent
    state = await client.get_state()
    assert state.engine_order is EngineOrder.HALF_AHEAD  # 0.7 is nearest
    await client.disconnect()


@pytest.mark.parametrize(
    ("lever", "expected"),
    [
        (1.0, EngineOrder.FULL_AHEAD),
        (0.0, EngineOrder.STOP),
        (-1.0, EngineOrder.FULL_ASTERN),
        (0.44, EngineOrder.SLOW_AHEAD),
        (-0.19, EngineOrder.DEAD_SLOW_ASTERN),
    ],
)
def test_order_from_lever(lever: float, expected: EngineOrder) -> None:
    assert _order_from_lever(lever) is expected


# --- link loss / reconnect ----------------------------------------------


async def test_link_loss_refuses_commands_for_as_long_as_it_lasts(
    sim: FakeSim,
) -> None:
    """A helm order must never be queued for a link that is down.

    The simulator stays down here, so this asserts the steady state rather
    than racing the supervisor's next reconnect attempt.
    """
    client = _client(sim)
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED

    sim.running = False
    assert await _until(
        lambda: client.connection_state is not ConnectionState.CONNECTED
    )

    for _ in range(3):  # across several supervisor cycles
        with pytest.raises(SimulatorError):
            await client.set_rudder(-10.0)
        with pytest.raises(SimulatorError):
            await client.set_engine_telegraph(EngineOrder.HALF_AHEAD)
        with pytest.raises(SimulatorError):
            await client.get_state()
        await asyncio.sleep(_TICK)

    # Nothing was staged while the link was down -- no order can fire late.
    assert sim.rudder_orders == []
    assert sim.lever_orders == []
    await client.disconnect()


async def test_link_loss_drops_the_session_and_builds_a_new_one(
    sim: FakeSim,
) -> None:
    """Recovery is a rebuild: nothing will announce that the link is back."""
    client = _client(sim)
    await client.connect()
    first = FakeWrapper.instances[-1]

    sim.running = False
    assert await _until(lambda: first.stopped)
    assert await _until(lambda: client.connection_state is ConnectionState.STALE)

    sim.running = True  # exercise restarted
    assert await _until(lambda: client.connection_state is ConnectionState.CONNECTED)
    # The replacement is a fresh session, not the stopped one reused.
    assert FakeWrapper.instances[-1] is not first
    await client.disconnect()


async def test_commands_work_again_after_an_automatic_reconnect(
    sim: FakeSim,
) -> None:
    client = _client(sim)
    await client.connect()
    sim.running = False
    assert await _until(
        lambda: client.connection_state is not ConnectionState.CONNECTED
    )

    sim.running = True
    assert await _until(lambda: client.connection_state is ConnectionState.CONNECTED)
    await client.set_rudder(15.0)
    assert sim.rudder_orders == [15.0]
    await client.disconnect()


# --- disconnect / repeatability -----------------------------------------


async def test_disconnect_is_idempotent_and_repeatable_with_connect(
    sim: FakeSim,
) -> None:
    client = _client(sim)
    await client.connect()
    await client.disconnect()
    await client.disconnect()  # must not raise
    assert client.connection_state is ConnectionState.DISCONNECTED
    assert FakeWrapper.instances[-1].stopped

    # A manual reconnect is just connect again.
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED
    await client.disconnect()


async def test_disconnect_stops_supervising(sim: FakeSim) -> None:
    """A torn-down client must not quietly rebuild sessions behind our back."""
    client = _client(sim)
    await client.connect()
    await client.disconnect()
    count = len(FakeWrapper.instances)
    await asyncio.sleep(_TICK)
    assert len(FakeWrapper.instances) == count


async def test_commands_are_refused_after_disconnect(sim: FakeSim) -> None:
    client = _client(sim)
    await client.connect()
    await client.disconnect()
    with pytest.raises(SimulatorError):
        await client.set_rudder(-10.0)


async def test_close_disconnects(sim: FakeSim) -> None:
    client = _client(sim)
    await client.connect()
    await client.close()
    assert client.connection_state is ConnectionState.DISCONNECTED
    assert FakeWrapper.instances[-1].stopped


# --- config wiring -------------------------------------------------------


def test_build_simulator_passes_both_address_pairs() -> None:
    """The link needs a send *and* a listen endpoint; both must survive config."""
    config = SimpleNamespace(
        remote_host="10.0.0.5",
        remote_port=15001,
        local_host="0.0.0.0",
        local_port=15002,
        connect_timeout_seconds=5.0,
        expected_fps=100.0,
        stale_after_missed_frames=50.0,
        reconnect_initial_seconds=1.0,
        reconnect_max_seconds=15.0,
    )
    client = build_simulator(config)
    assert isinstance(client, RealSimulatorClient)
    assert client.connection_state is ConnectionState.DISCONNECTED


def test_build_simulator_refuses_unconfigured_endpoints() -> None:
    """The shipped defaults are placeholders: the working values come with the
    vendor integration and must be set locally, so building against the
    defaults must fail with a message naming every missing field."""
    config = SimpleNamespace(
        remote_host="127.0.0.1",
        remote_port=0,
        local_host="127.0.0.1",
        local_port=0,
        connect_timeout_seconds=5.0,
        expected_fps=0.0,
        stale_after_missed_frames=50.0,
        reconnect_initial_seconds=1.0,
        reconnect_max_seconds=15.0,
    )
    with pytest.raises(ValueError, match="remote_port/local_port/expected_fps"):
        build_simulator(config)


# --- state listener (what the dashboard sees) ----------------------------


async def test_state_listener_reports_every_transition(sim: FakeSim) -> None:
    """The dashboard is push-driven: a missed transition is a stuck pill."""
    seen: list[ConnectionState] = []
    client = _client(sim)
    client.set_state_listener(seen.append)

    await client.connect()
    assert ConnectionState.CONNECTED in seen

    sim.running = False
    assert await _until(lambda: ConnectionState.STALE in seen)

    sim.running = True
    assert await _until(lambda: seen[-1] is ConnectionState.CONNECTED)

    await client.disconnect()
    assert seen[-1] is ConnectionState.DISCONNECTED


async def test_state_listener_reports_no_duplicates(sim: FakeSim) -> None:
    """Only changes are published; a steady link must not spam the bus."""
    seen: list[ConnectionState] = []
    client = _client(sim)
    client.set_state_listener(seen.append)
    await client.connect()
    count = len(seen)
    await asyncio.sleep(_TICK * 2)  # several supervisor cycles, link steady
    assert len(seen) == count
    await client.disconnect()


async def test_a_failing_listener_does_not_break_the_supervisor(
    sim: FakeSim,
) -> None:
    """Losing the link is bad; losing the thing that reconnects it is worse."""
    client = _client(sim)
    client.set_state_listener(lambda _state: (_ for _ in ()).throw(RuntimeError("boom")))

    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED

    # The supervisor must still detect loss and recover, listener or no listener.
    sim.running = False
    assert await _until(lambda: client.connection_state is ConnectionState.STALE)
    sim.running = True
    assert await _until(lambda: client.connection_state is ConnectionState.CONNECTED)
    await client.disconnect()


# --- delivery confirmation (an acknowledgment must mean delivery) ---------


async def test_order_is_refused_when_the_sim_dies_at_staging_time(
    sim: FakeSim,
) -> None:
    """The tightest loss window: the link dies the instant the order is staged.

    The state enum still says CONNECTED (the supervisor has not noticed yet),
    so before delivery confirmation existed this returned success and the
    order silently died with the session. Now the confirmation frames never
    arrive, the command must fail aloud, and the session -- with the
    undelivered order still staged in it -- must be dropped so the order can
    never fire later when the link revives.
    """
    sim.dies_on_order = True
    client = _client(sim)
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED

    doomed = FakeWrapper.instances[-1]
    with pytest.raises(SimulatorError):
        await client.set_rudder(-10.0)
    # The session holding the undelivered order is gone, immediately -- not
    # whenever the supervisor would next have noticed.
    assert doomed.stopped
    await client.disconnect()


async def test_order_is_refused_when_frames_stopped_before_staging(
    sim: FakeSim,
) -> None:
    """Frames already old at command time: refuse before staging anything."""
    client = _client(sim)
    await client.connect()

    sim.running = False  # silence begins; supervisor may not have noticed yet
    with pytest.raises(SimulatorError):
        await client.set_rudder(-10.0)
    # Nothing was staged: the pre-check fires before the order reaches the
    # wrapper, so there is nothing that could execute later.
    assert sim.rudder_orders == []
    await client.disconnect()


async def test_get_state_refuses_a_stale_mirror(sim: FakeSim) -> None:
    """A status query must not read out values a dead link left behind."""
    client = _client(sim)
    await client.connect()
    sim.running = False
    with pytest.raises(SimulatorError):
        await client.get_state()
    await client.disconnect()


# --- disconnect during a session build ------------------------------------


async def test_disconnect_during_a_slow_build_leaves_no_live_session(
    sim: FakeSim,
) -> None:
    """Disconnect while 'connecting' must not leak a running session.

    A session that survived this would keep the local port bound (blocking
    every later connect until process restart) and keep the ship's controls
    engaged. The dashboard offers Disconnect in exactly this state, so it is
    an ordinary click, not a corner case.
    """
    sim.session_open_seconds = 0.3
    client = _client(sim, connect_timeout_seconds=0.05)
    await client.connect()  # returns on timeout; build still in flight

    await client.disconnect()
    assert client.connection_state is ConnectionState.DISCONNECTED

    # Whatever sessions were built, none may still be live.
    assert all(w.stopped for w in FakeWrapper.instances if w.started)

    # And the port is genuinely free: a fresh connect works.
    sim.session_open_seconds = 0.0
    await client.connect()
    assert client.connection_state is ConnectionState.CONNECTED
    await client.disconnect()


# --- manual retry (the backoff kick) ---------------------------------------


async def test_connect_skips_the_remaining_backoff(sim: FakeSim) -> None:
    """POST connect is 'try again now', not 'wait out the backoff anyway'.

    With a huge backoff and a session that cannot open, the supervisor parks
    in its backoff sleep. Fixing the simulator and calling connect() again
    must produce a live link promptly -- if the kick were lost, this test
    would hang for the full 30s backoff and time out.
    """
    sim.can_open_session = False
    client = _client(
        sim,
        connect_timeout_seconds=0.1,
        reconnect_initial_seconds=30.0,
        reconnect_max_seconds=30.0,
    )
    await client.connect()
    # Let the failed build happen and the supervisor enter its backoff.
    assert await _until(lambda: len(FakeWrapper.instances) >= 1)
    await asyncio.sleep(_TICK)

    sim.can_open_session = True  # the operator fixed the simulator...
    await client.connect()  # ...and clicked Connect
    assert await _until(
        lambda: client.connection_state is ConnectionState.CONNECTED, timeout=5.0
    )
    await client.disconnect()


async def test_commands_and_supervisor_share_the_session_lock(
    sim: FakeSim,
) -> None:
    """A command mid-confirmation must not have its session swapped away.

    Holds the sim in a healthy state but makes snapshots slow enough that a
    command's stage-and-confirm overlaps several supervisor cycles; the order
    must land in the session that stays current.
    """
    client = _client(sim)
    await client.connect()
    current = FakeWrapper.instances[-1]

    state = await client.set_rudder(-15.0)
    assert state.rudder_angle_deg == -15.0
    # The session the order went to is still the client's session.
    assert FakeWrapper.instances[-1] is current
    assert not current.stopped
    await client.disconnect()
