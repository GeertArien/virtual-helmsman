"""Translate a parsed helmsman action into simulator calls.

The :class:`~voice_agent.backends.simulator.base.SimulatorClient` protocol
speaks conning orders (``set_rudder`` / ``set_engine_telegraph`` /
``get_state``). The LLM speaks the richer n8n vocabulary (``rudder``,
``throttle``, ``navigation``, ``autopilot``, ``anchor``, ``status_query``,
``error``). This module is the translation layer.

Mapping summary:

* ``rudder`` -> ``set_rudder(±degrees)``: port negative, starboard positive.
  This is a *helm order* -- the rudder goes to that angle and stays there
  until countermanded ("midships" is simply ``degrees: 0``).
* ``throttle`` -> map ``speed`` knots to the nearest ``EngineOrder`` and
  call ``set_engine_telegraph`` (see :func:`_knots_to_telegraph`).
* ``navigation`` -> **refused in v1** (see :data:`COURSE_ORDER_REFUSAL`).
  A course order ("steer 090") asks for a closed loop against the compass;
  the simulator exposes no heading setpoint, and in real pilotage that loop
  *is* the helmsman's own work. Implementing it as a steering skill on top of
  ``set_rudder`` is a follow-up, not a v1 requirement -- so we say so plainly
  rather than half-executing the order.
* ``status_query`` -> ``get_state``; the spoken response is augmented with
  the requested field (heading / speed / position-not-available).
* ``autopilot`` / ``anchor`` -> not supported by the simulator interface;
  acknowledged verbally, logged as ``simulator_skip_unsupported``, no
  state change. The action still publishes an ``action_dispatched`` event
  (consumers can render an audit row) -- the operator sees that the
  command was *recognised*, just not executed.
* ``error`` -> no simulator call; refusal event downstream.

Depends only on the ``SimulatorClient`` protocol, never on a concrete
backend, so swapping ``real``/``mock`` stays a pure config change.
"""

from __future__ import annotations

from dataclasses import dataclass

from voice_agent.actions.schema import (
    AnchorAction,
    AnswerAction,
    AutopilotAction,
    ErrorAction,
    HelmsmanResponse,
    NavigationAction,
    RudderAction,
    StatusQueryAction,
    ThrottleAction,
)
from voice_agent.backends.simulator.base import (
    EngineOrder,
    ShipState,
    SimulatorClient,
    SimulatorError,
)
from voice_agent.logging_setup import get_logger

# Spoken when the simulator backend fails -- voiced verbatim to the captain.
BRIDGE_LOST = "Lost contact with the bridge, sir. Unable to comply."

# Spoken when a course order is given. v1 executes helm orders (rudder angles)
# and engine orders, not course-keeping -- so the helmsman says what it can do
# instead of silently doing something else.
COURSE_ORDER_REFUSAL = (
    "Unable to steer a course, sir. I can answer the helm on rudder orders "
    "and the telegraph. Request a helm order, such as port ten."
)


@dataclass
class DispatchResult:
    """Outcome of dispatching one action: what to speak, and whether it ran."""

    spoken: str
    ok: bool = True
    ship_state: ShipState | None = None


def _knots_to_telegraph(speed: float) -> EngineOrder:
    """Coarse mapping from a knots setpoint to the nearest telegraph order.

    The bands are intentionally wide: a 9-position telegraph cannot encode
    every requested speed exactly, and the LLM is told to issue throttle in
    knots. Negative speed selects the equivalent astern order. The thresholds
    follow the mock simulator's static ``_SPEED_BY_ORDER`` mapping so
    requested 6 knots in lands on ``slow_ahead`` (which the mock then reports
    back as 6 knots).
    """
    if speed >= 13:
        return EngineOrder.FULL_AHEAD
    if speed >= 8:
        return EngineOrder.HALF_AHEAD
    if speed >= 4:
        return EngineOrder.SLOW_AHEAD
    if speed >= 1:
        return EngineOrder.DEAD_SLOW_AHEAD
    if speed > -1:
        return EngineOrder.STOP
    if speed > -4:
        return EngineOrder.DEAD_SLOW_ASTERN
    if speed > -8:
        return EngineOrder.SLOW_ASTERN
    if speed > -13:
        return EngineOrder.HALF_ASTERN
    return EngineOrder.FULL_ASTERN


def _format_status(query: str, base: str, state: ShipState) -> str:
    """Append the requested field to the spoken acknowledgement.

    The model emits a short "Checking heading, sir." style line; we append
    the live reading the LLM cannot know. ``position`` is unsupported by
    the current simulator; we say so explicitly rather than fabricating.
    """
    base = base.strip()
    if query == "heading":
        return f"{base} Heading {round(state.heading_deg)} degrees.".strip()
    if query == "speed":
        return f"{base} Speed {round(state.speed_kn, 1)} knots.".strip()
    if query == "position":
        # The current simulator does not surface position. Don't invent one.
        return f"{base} Position is not available from the helm.".strip()
    return base  # unreachable -- Literal exhausts the cases


async def dispatch_action(
    parsed: HelmsmanResponse, simulator: SimulatorClient
) -> DispatchResult:
    """Carry out ``parsed.action`` against the simulator; return what to speak.

    A simulator failure is reported with :data:`BRIDGE_LOST`. An ``error``
    action is the LLM refusing or asking for clarification: no sim call,
    just speak the response.
    """
    log = get_logger("actions")
    action = parsed.action

    if isinstance(action, ErrorAction):
        log.info(
            "action_refused", error_type=action.error_type, reason=action.reason
        )
        return DispatchResult(spoken=parsed.response)

    if isinstance(action, AnswerAction):
        # n8n question-branch reply: just speak the RAG answer. No
        # simulator side-effects, no event-bus dispatch beyond the
        # assistant reply -- it's information, not a command.
        log.info("answer_returned")
        return DispatchResult(spoken=parsed.response)

    if isinstance(action, NavigationAction):
        # Not a simulator failure -- a capability we deliberately do not have
        # yet, so it is reported like a refusal rather than a lost link.
        log.warning(
            "simulator_skip_unsupported", action="navigation", course=action.course
        )
        return DispatchResult(spoken=COURSE_ORDER_REFUSAL, ok=False)

    try:
        if isinstance(action, RudderAction):
            # The action already *is* a rudder-angle order; only the sign is
            # ours to resolve. No read-modify-write of the heading.
            angle = action.degrees if action.direction == "starboard" else -action.degrees
            state = await simulator.set_rudder(angle)
            log.info(
                "action_dispatched",
                action="rudder",
                direction=action.direction,
                degrees=action.degrees,
                ordered_angle_deg=angle,
            )
            return DispatchResult(spoken=parsed.response, ship_state=state)

        if isinstance(action, ThrottleAction):
            # A named telegraph order is exact; knots are a request the
            # 9-position telegraph can only approximate. Prefer the former.
            if action.order is not None:
                order = EngineOrder(action.order)
            else:
                # The schema guarantees one of the two is set.
                order = _knots_to_telegraph(float(action.speed))
            state = await simulator.set_engine_telegraph(order)
            log.info(
                "action_dispatched",
                action="throttle",
                order=action.order,
                speed=action.speed,
                resolved_order=order.value,
            )
            return DispatchResult(spoken=parsed.response, ship_state=state)

        if isinstance(action, StatusQueryAction):
            state = await simulator.get_state()
            log.info("action_dispatched", action="status_query", query=action.query)
            return DispatchResult(
                spoken=_format_status(action.query, parsed.response, state),
                ship_state=state,
            )

        if isinstance(action, AutopilotAction):
            # No simulator support yet -- acknowledge verbally, log clearly.
            log.warning(
                "simulator_skip_unsupported",
                action="autopilot",
                state=action.state,
            )
            return DispatchResult(spoken=parsed.response)

        if isinstance(action, AnchorAction):
            log.warning(
                "simulator_skip_unsupported",
                action="anchor",
                operation=action.operation,
                chain_length=action.chain_length,
            )
            return DispatchResult(spoken=parsed.response)
    except SimulatorError as exc:
        log.error("action_failed", action=action.type, error=str(exc))
        return DispatchResult(spoken=BRIDGE_LOST, ok=False)

    # The discriminated union is exhaustive; pyright/mypy will catch a new
    # variant that misses a branch above.
    raise AssertionError(f"unhandled action type: {action.type}")
