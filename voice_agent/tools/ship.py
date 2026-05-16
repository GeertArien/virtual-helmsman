"""Ship tool handlers.

Three thin handlers that delegate to an injected ``SimulatorClient``. Handlers
carry only input validation and error translation; all ship state lives in the
``SimulatorClient``. The same ``SimulatorClient`` instance is shared by all
three handlers (closed over by :func:`build_ship_handlers`).

A handler depends only on the ``SimulatorClient`` protocol, never on the
``real``/``mock`` concrete classes — so swapping the simulator backend is a
pure config change.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import structlog

from voice_agent.backends.simulator.base import (
    EngineOrder,
    ShipState,
    SimulatorClient,
    SimulatorError,
)
from voice_agent.tools.schemas import (
    GET_SHIP_STATE,
    SET_ENGINE_TELEGRAPH,
    SET_HEADING,
)

# A Pipecat tool handler: async, receives a FunctionCallParams-like object with
# `.arguments` (dict) and an awaitable `.result_callback`.
ShipToolHandler = Callable[[Any], Awaitable[None]]

# Spoken when the simulator backend fails; the LLM voices this error verbatim.
_BRIDGE_LOST = "Lost contact with bridge"


def _ship_state_to_result(state: ShipState) -> dict[str, Any]:
    """Serialise a ``ShipState`` into the dict returned to the LLM."""
    return {
        "heading_deg": round(state.heading_deg, 1),
        "speed_kn": round(state.speed_kn, 1),
        "engine_order": state.engine_order.value,
        "timestamp": state.timestamp.isoformat(),
    }


def build_ship_handlers(simulator: SimulatorClient) -> dict[str, ShipToolHandler]:
    """Build the three tool handlers bound to one shared ``SimulatorClient``.

    Exposed separately from :func:`register_ship_tools` so tests can drive the
    handlers directly against a ``MockSimulatorClient``.
    """
    log = structlog.get_logger().bind(component="tools")

    async def set_heading(params: Any) -> None:
        raw = params.arguments.get("degrees")
        try:
            degrees = float(raw)
        except (TypeError, ValueError):
            log.warning("tool_invalid_argument", tool=SET_HEADING, degrees=raw)
            await params.result_callback({"error": f"Invalid heading: {raw!r}"})
            return
        normalised = degrees % 360
        log.info("tool_call", tool=SET_HEADING, degrees=normalised)
        try:
            state = await simulator.set_heading(normalised)
        except SimulatorError as exc:
            log.error("tool_failed", tool=SET_HEADING, error=str(exc))
            await params.result_callback({"error": _BRIDGE_LOST})
            return
        log.info("tool_result", tool=SET_HEADING, **_ship_state_to_result(state))
        await params.result_callback(_ship_state_to_result(state))

    async def set_engine_telegraph(params: Any) -> None:
        raw = params.arguments.get("order")
        try:
            order = EngineOrder(raw)
        except ValueError:
            log.warning("tool_invalid_argument", tool=SET_ENGINE_TELEGRAPH, order=raw)
            await params.result_callback({"error": f"Invalid engine order: {raw!r}"})
            return
        log.info("tool_call", tool=SET_ENGINE_TELEGRAPH, order=order.value)
        try:
            state = await simulator.set_engine_telegraph(order)
        except SimulatorError as exc:
            log.error("tool_failed", tool=SET_ENGINE_TELEGRAPH, error=str(exc))
            await params.result_callback({"error": _BRIDGE_LOST})
            return
        log.info("tool_result", tool=SET_ENGINE_TELEGRAPH, **_ship_state_to_result(state))
        await params.result_callback(_ship_state_to_result(state))

    async def get_ship_state(params: Any) -> None:
        log.info("tool_call", tool=GET_SHIP_STATE)
        try:
            state = await simulator.get_state()
        except SimulatorError as exc:
            log.error("tool_failed", tool=GET_SHIP_STATE, error=str(exc))
            await params.result_callback({"error": _BRIDGE_LOST})
            return
        log.info("tool_result", tool=GET_SHIP_STATE, **_ship_state_to_result(state))
        await params.result_callback(_ship_state_to_result(state))

    return {
        SET_HEADING: set_heading,
        SET_ENGINE_TELEGRAPH: set_engine_telegraph,
        GET_SHIP_STATE: get_ship_state,
    }


def register_ship_tools(llm: Any, simulator: SimulatorClient) -> None:
    """Register the three ship tool handlers on the LLM service.

    ``llm`` is a Pipecat ``LLMService``; ``simulator`` is the single
    ``SimulatorClient`` built once at pipeline startup.
    """
    for name, handler in build_ship_handlers(simulator).items():
        llm.register_function(name, handler)
