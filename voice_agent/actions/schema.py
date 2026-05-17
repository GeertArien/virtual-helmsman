"""Structured-action schema for the helmsman's JSON replies.

The LLM answers every command with one JSON object: an ``action`` (one of the
four types below) plus a ``response`` -- the spoken acknowledgement. The pydantic
models here validate that object; :data:`RESPONSE_FORMAT` is the JSON schema
handed to the LLM server so it constrains decoding to the right shape.

This replaces native OpenAI tool calling: a small local model emits a
constrained JSON object far more reliably than it emits ``tool_calls``.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError

from voice_agent.backends.simulator.base import EngineOrder

# The nine telegraph orders, as the LLM-facing string values.
ENGINE_ORDER_VALUES: list[str] = [order.value for order in EngineOrder]


class ActionParseError(Exception):
    """Raised when LLM output is not a valid helmsman response object."""


class SetHeadingAction(BaseModel):
    """Steer the ship to an absolute compass heading."""

    type: Literal["set_heading"]
    degrees: float


class SetEngineTelegraphAction(BaseModel):
    """Set the engine telegraph to one of the nine standard orders."""

    type: Literal["set_engine_telegraph"]
    order: EngineOrder


class GetShipStateAction(BaseModel):
    """Report the ship's current heading, speed, and engine order."""

    type: Literal["get_ship_state"]


class ErrorAction(BaseModel):
    """A refused, ambiguous, or out-of-scope command -- no simulator call."""

    type: Literal["error"]
    error_type: str
    reason: str
    suggestion: str = ""


# Discriminated on ``type`` -- pydantic selects the right model per object.
HelmsmanAction = Annotated[
    Union[
        SetHeadingAction,
        SetEngineTelegraphAction,
        GetShipStateAction,
        ErrorAction,
    ],
    Field(discriminator="type"),
]


class HelmsmanResponse(BaseModel):
    """The full JSON object the LLM must return for every command."""

    action: HelmsmanAction
    response: str


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping markdown code fence (```` ```json ... ``` ````) if present."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]  # drop the opening fence and its optional language tag line
    newline = s.find("\n")
    if newline != -1:
        s = s[newline + 1 :]
    s = s.rstrip()
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def parse_response(text: str) -> HelmsmanResponse:
    """Parse raw LLM text into a :class:`HelmsmanResponse`.

    Raises :class:`ActionParseError` if the text is not valid JSON or does not
    match the action schema.
    """
    cleaned = _strip_code_fence(text)
    if not cleaned:
        raise ActionParseError("empty response")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"not valid JSON: {exc}") from exc
    try:
        return HelmsmanResponse.model_validate(data)
    except ValidationError as exc:
        raise ActionParseError(f"does not match the action schema: {exc}") from exc


# JSON schema handed to the LLM server via ``response_format``. Deliberately
# loose -- only ``type`` is required inside ``action`` -- so a grammar converter
# accepts it; the pydantic models above enforce the per-action fields after
# decoding. The prompt, not the schema, teaches which fields each action needs.
RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "helmsman_response",
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "set_heading",
                                "set_engine_telegraph",
                                "get_ship_state",
                                "error",
                            ],
                        },
                        "degrees": {"type": "number"},
                        "order": {"type": "string", "enum": ENGINE_ORDER_VALUES},
                        "error_type": {"type": "string"},
                        "reason": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["type"],
                },
                "response": {"type": "string"},
            },
            "required": ["action", "response"],
        },
    },
}
