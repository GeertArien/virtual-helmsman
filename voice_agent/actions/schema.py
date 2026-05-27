"""Structured-action schema for the helmsman's JSON replies.

The LLM answers every command with one JSON object: an ``action`` (one of the
six command types or an ``error``) plus a ``response`` -- the spoken
acknowledgement. The Pydantic models here validate that object;
:data:`RESPONSE_FORMAT` is the JSON schema handed to the LLM server so it
grammar-constrains decoding to the right shape.

The vocabulary mirrors the n8n helmsman workflow (see ``n8n_system_prompt.txt``
in the repo root and ``API.md``). The local LM Studio backend and the n8n
backend therefore produce the same on-the-wire action shape -- only their
"envelope" differs (n8n adds an ``intent`` / ``output`` / ``source`` layer for
its RAG branch). The simulator-side translation lives in
:mod:`voice_agent.actions.dispatch`; the simulator client itself still speaks
its narrower native protocol (``set_heading`` / ``set_engine_telegraph`` /
``get_state``).

``multi_step`` from the n8n prompt is intentionally omitted -- v1 dispatches
one action per turn.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError


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
    """Set the ship's speed. The dispatcher maps knots to a telegraph order.

    ``unit`` is fixed to ``"knots"`` in v1 -- the prompt doesn't teach the LLM
    any other unit, and a freeform unit field would be a footgun.
    """

    type: Literal["throttle"]
    speed: Annotated[float, Field(ge=-MAX_SPEED_KNOTS, le=MAX_SPEED_KNOTS)]
    unit: Literal["knots"] = "knots"


class NavigationAction(BaseModel):
    """Steer to an absolute compass course (0-359)."""

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

    Not part of the n8n LLM-side action vocabulary in ``n8n_system_prompt.txt``
    (the LLM never emits this). Synthesised by the n8n *adapter* when the
    workflow envelope reports ``intent: "question"`` -- in that branch the
    envelope's ``output`` is a RAG answer, the ``action`` field is null, and
    nothing needs to drive the simulator.

    The local LM Studio backend never produces this type because its prompt
    doesn't teach it -- LM Studio is command-only by design.
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
                        # throttle
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
