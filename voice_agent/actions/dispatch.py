"""Dispatch a parsed helmsman action to the simulator.

A pure step between :func:`voice_agent.actions.schema.parse_response` and the
:class:`~voice_agent.actions.processor.JsonActionProcessor`: it maps a validated
action onto a :class:`SimulatorClient` call and decides what TTS should speak.

Depends only on the ``SimulatorClient`` protocol, never on a concrete backend,
so swapping ``real``/``mock`` stays a pure config change.
"""

from __future__ import annotations

from dataclasses import dataclass

from voice_agent.actions.schema import (
    ErrorAction,
    GetShipStateAction,
    HelmsmanResponse,
    SetEngineTelegraphAction,
    SetHeadingAction,
)
from voice_agent.backends.simulator.base import (
    ShipState,
    SimulatorClient,
    SimulatorError,
)
from voice_agent.logging_setup import get_logger

# Spoken when the simulator backend fails -- voiced verbatim to the captain.
BRIDGE_LOST = "Lost contact with the bridge, sir. Unable to comply."


@dataclass
class DispatchResult:
    """Outcome of dispatching one action: what to speak, and whether it ran."""

    spoken: str
    ok: bool = True
    ship_state: ShipState | None = None


def _describe(state: ShipState) -> str:
    """A spoken status read-back of the ship's current state."""
    order = state.engine_order.value.replace("_", " ")
    return (
        f"Heading {round(state.heading_deg)} degrees, "
        f"speed {round(state.speed_kn, 1)} knots, engines {order}."
    )


async def dispatch_action(
    parsed: HelmsmanResponse, simulator: SimulatorClient
) -> DispatchResult:
    """Carry out ``parsed.action`` against the simulator; return what to speak.

    The model's ``response`` is the spoken acknowledgement for command actions.
    For ``get_ship_state`` the live readings are appended (the model cannot know
    them). A simulator failure is reported with :data:`BRIDGE_LOST`.
    """
    log = get_logger("actions")
    action = parsed.action

    # An error action is the model refusing / asking for clarification -- there
    # is nothing to send to the simulator; just speak its response.
    if isinstance(action, ErrorAction):
        log.info(
            "action_refused", error_type=action.error_type, reason=action.reason
        )
        return DispatchResult(spoken=parsed.response)

    try:
        if isinstance(action, SetHeadingAction):
            heading = action.degrees % 360
            state = await simulator.set_heading(heading)
            log.info("action_dispatched", action="set_heading", degrees=heading)
            return DispatchResult(spoken=parsed.response, ship_state=state)

        if isinstance(action, SetEngineTelegraphAction):
            state = await simulator.set_engine_telegraph(action.order)
            log.info(
                "action_dispatched",
                action="set_engine_telegraph",
                order=action.order.value,
            )
            return DispatchResult(spoken=parsed.response, ship_state=state)

        if isinstance(action, GetShipStateAction):
            state = await simulator.get_state()
            log.info("action_dispatched", action="get_ship_state")
            spoken = f"{parsed.response.strip()} {_describe(state)}".strip()
            return DispatchResult(spoken=spoken, ship_state=state)
    except SimulatorError as exc:
        log.error("action_failed", action=action.type, error=str(exc))
        return DispatchResult(spoken=BRIDGE_LOST, ok=False)

    # The discriminated union has no other members.
    raise AssertionError(f"unhandled action type: {action.type}")
