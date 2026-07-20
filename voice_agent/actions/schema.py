"""Structured-action schema for the helmsman's JSON replies.

The LLM answers every command with one JSON object: an ``action`` (one of the
six command types or an ``error``) plus a ``response`` -- the spoken
acknowledgement. The Pydantic models here validate that object;
:data:`RESPONSE_FORMAT` is the JSON schema handed to the LLM server so it
grammar-constrains decoding to the right shape.

The vocabulary is defined by the helmsman system prompt
(:data:`voice_agent.actions.prompt.SYSTEM_PROMPT`). Both LLM backends produce
the same on-the-wire action shape; the `langgraph` backend wraps the RAG
branch in a synthetic ``answer`` action (see :class:`AnswerAction`). The
simulator-side translation lives in :mod:`voice_agent.actions.dispatch`; the
simulator client itself speaks conning orders
(``set_rudder`` / ``set_engine_telegraph`` / ``get_state``).

``multi_step`` from the prompt is intentionally omitted -- v1 dispatches one
action per turn.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

# The nine telegraph positions, as the LLM may name them. Kept as a Literal
# rather than importing the simulator's EngineOrder enum: this module defines
# the LLM-facing contract and must not depend on a backend package. Dispatch
# maps one to the other, and a mismatch fails loudly there.
EngineOrderLiteral = Literal[
    "full_astern",
    "half_astern",
    "slow_astern",
    "dead_slow_astern",
    "stop",
    "dead_slow_ahead",
    "slow_ahead",
    "half_ahead",
    "full_ahead",
]

# Safety limits, lifted verbatim from the n8n system prompt. Repeating them
# here as Pydantic constraints is belt-and-suspenders: the prompt teaches the
# LLM to emit an `error` action with `error_type: "safety_limit_exceeded"`
# when these are breached, and if it slips up anyway the model_validate call
# rejects the action.
MAX_RUDDER_DEGREES: float = 45.0
MAX_SPEED_KNOTS: float = 30.0
HEADING_MAX: int = 359


class ActionParseError(Exception):
    """Raised when LLM output is not a valid helmsman response object."""


class RudderAction(BaseModel):
    """Relative steering command: turn `degrees` to port or starboard."""

    type: Literal["rudder"]
    direction: Literal["port", "starboard"]
    degrees: Annotated[float, Field(ge=0, le=MAX_RUDDER_DEGREES)]


class ThrottleAction(BaseModel):
    """An engine order.

    Two ways to say it, in order of preference:

    * ``order`` -- a telegraph position ("half ahead"). This is what a conning
      officer actually orders, and what the telegraph can actually represent,
      so the prompt teaches it first.
    * ``speed`` -- knots ("make turns for fifteen knots"). A real order too,
      but the 9-position telegraph cannot encode every speed, so the dispatcher
      picks the nearest position (see ``dispatch._knots_to_telegraph``).

    At least one must be present; ``order`` wins if both are. ``unit`` is fixed
    to ``"knots"`` -- the prompt teaches no other unit, and a freeform unit
    field would be a footgun.
    """

    type: Literal["throttle"]
    order: EngineOrderLiteral | None = None
    speed: Annotated[float, Field(ge=-MAX_SPEED_KNOTS, le=MAX_SPEED_KNOTS)] | None = None
    unit: Literal["knots"] = "knots"

    @model_validator(mode="after")
    def _require_order_or_speed(self) -> ThrottleAction:
        if self.order is None and self.speed is None:
            raise ValueError("throttle needs either 'order' (telegraph) or 'speed' (knots)")
        return self


class NavigationAction(BaseModel):
    """A course order: steer and hold an absolute compass course (0-359).

    Still parsed -- the LLM must be able to *recognise* "steer zero-nine-zero"
    -- but the dispatcher refuses it in v1: holding a course is a closed loop
    against the compass that the simulator has no setpoint for, and which in
    real pilotage is the helmsman's own work. See
    :data:`voice_agent.actions.dispatch.COURSE_ORDER_REFUSAL`.
    """

    type: Literal["navigation"]
    course: Annotated[int, Field(ge=0, le=HEADING_MAX)]


class AutopilotAction(BaseModel):
    """Engage or disengage the autopilot.

    The current simulator does not yet implement autopilot; the dispatcher
    logs the request and acknowledges verbally without touching ship state.
    """

    type: Literal["autopilot"]
    state: Literal["engaged", "disengaged"]


class AnchorAction(BaseModel):
    """Anchor operations. Same v1 status as autopilot: ack-only, no sim call.

    ``chain_length`` is meaningful only for ``let_out_chain``; left optional
    so the model can omit it for drop/raise (matches the n8n schema).
    """

    type: Literal["anchor"]
    operation: Literal["drop", "raise", "let_out_chain"]
    chain_length: float | None = None


class StatusQueryAction(BaseModel):
    """Read-back of one field of the ship state."""

    type: Literal["status_query"]
    query: Literal["heading", "speed", "position"]


class ErrorAction(BaseModel):
    """A refused, ambiguous, or out-of-scope command -- no simulator call."""

    type: Literal["error"]
    error_type: str
    reason: str
    suggestion: str = ""


class AnswerAction(BaseModel):
    """A RAG-style information answer -- no simulator call, just speak.

    Not part of the LLM-facing action vocabulary in
    :data:`voice_agent.actions.prompt.SYSTEM_PROMPT` (the LLM never emits
    this). Synthesised by the `langgraph` backend on a ``question`` turn --
    the RAG answer goes in ``response`` and nothing drives the simulator.

    The `openai_compatible` backend never produces this type (command-only
    by design).
    """

    type: Literal["answer"]


# Discriminated on ``type`` -- pydantic selects the right model per object.
HelmsmanAction = Annotated[
    Union[
        RudderAction,
        ThrottleAction,
        NavigationAction,
        AutopilotAction,
        AnchorAction,
        StatusQueryAction,
        ErrorAction,
        AnswerAction,
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
# loose -- only ``type`` is required inside ``action``, and the per-action
# properties are all listed flat without ``oneOf`` branches -- so any grammar
# converter accepts it. The pydantic models above enforce the per-action
# fields after decoding. The prompt, not the schema, teaches which fields
# each action needs.
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
                                "rudder",
                                "throttle",
                                "navigation",
                                "autopilot",
                                "anchor",
                                "status_query",
                                "error",
                                # "answer" is not in the LLM-facing enum --
                                # only the n8n adapter synthesises it, never
                                # an LLM. Listing it here would be misleading.
                            ],
                        },
                        # rudder
                        "direction": {
                            "type": "string",
                            "enum": ["port", "starboard"],
                        },
                        "degrees": {"type": "number"},
                        # throttle -- `order` is preferred over `speed`
                        "order": {
                            "type": "string",
                            "enum": [
                                "full_astern",
                                "half_astern",
                                "slow_astern",
                                "dead_slow_astern",
                                "stop",
                                "dead_slow_ahead",
                                "slow_ahead",
                                "half_ahead",
                                "full_ahead",
                            ],
                        },
                        "speed": {"type": "number"},
                        "unit": {"type": "string", "enum": ["knots"]},
                        # navigation
                        "course": {"type": "integer"},
                        # autopilot
                        "state": {
                            "type": "string",
                            "enum": ["engaged", "disengaged"],
                        },
                        # anchor
                        "operation": {
                            "type": "string",
                            "enum": ["drop", "raise", "let_out_chain"],
                        },
                        "chain_length": {"type": "number"},
                        # status_query
                        "query": {
                            "type": "string",
                            "enum": ["heading", "speed", "position"],
                        },
                        # error
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
