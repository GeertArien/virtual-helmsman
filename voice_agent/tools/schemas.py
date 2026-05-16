"""Pipecat ``FunctionSchema`` definitions for the three ship tools.

Tool names are exported as constants so the schema and the handler registration
in :mod:`voice_agent.tools.ship` cannot drift apart.
"""

from __future__ import annotations

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from voice_agent.backends.simulator.base import EngineOrder

SET_HEADING = "set_heading"
SET_ENGINE_TELEGRAPH = "set_engine_telegraph"
GET_SHIP_STATE = "get_ship_state"

# The nine valid telegraph positions, as the LLM-facing enum values.
ENGINE_ORDER_VALUES: list[str] = [order.value for order in EngineOrder]


def build_tools_schema() -> ToolsSchema:
    """Return the ``ToolsSchema`` declaring the three ship tools."""
    set_heading = FunctionSchema(
        name=SET_HEADING,
        description=(
            "Steer the ship to a compass heading. Use for course and heading "
            "commands once the captain has stated an explicit heading."
        ),
        properties={
            "degrees": {
                "type": "number",
                "description": "Target compass heading in degrees, 0-359.",
            },
        },
        required=["degrees"],
    )
    set_engine_telegraph = FunctionSchema(
        name=SET_ENGINE_TELEGRAPH,
        description="Set the engine telegraph to one of the nine standard orders.",
        properties={
            "order": {
                "type": "string",
                "enum": ENGINE_ORDER_VALUES,
                "description": "Engine telegraph position.",
            },
        },
        required=["order"],
    )
    get_ship_state = FunctionSchema(
        name=GET_SHIP_STATE,
        description="Report the ship's current heading, speed, and engine order.",
        properties={},
        required=[],
    )
    return ToolsSchema(
        standard_tools=[set_heading, set_engine_telegraph, get_ship_state]
    )
