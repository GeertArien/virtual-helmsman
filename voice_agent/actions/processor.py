"""Pipecat ``FrameProcessor`` that turns the LLM's JSON action into speech.

Sits between the LLM service and TTS. The LLM emits one JSON object
(``{"action": ..., "response": ...}``) as streamed text frames; this processor
buffers that text, parses it, dispatches the action to the simulator, and
forwards only the spoken ``response`` downstream -- so the raw JSON never
reaches TTS or the conversation context.

The LLM's ``LLMFullResponseStartFrame`` / ``LLMTextFrame`` / ``...EndFrame``
triple is consumed and replaced with a fresh triple carrying the spoken line.

If an :class:`~voice_agent.api.events.EventBus` is provided, the processor also
publishes one event per turn so the frontend can render the conversation: the
assistant reply, the action dispatched (or refused), and the new ship state.
"""

from __future__ import annotations

from typing import Any

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.actions.dispatch import DispatchResult, dispatch_action
from voice_agent.actions.schema import (
    ActionParseError,
    AnchorAction,
    AnswerAction,
    AutopilotAction,
    ErrorAction,
    HelmsmanResponse,
    NavigationAction,
    RudderAction,
    StatusQueryAction,
    ThrottleAction,
    parse_response,
)
from voice_agent.api.events import (
    ActionDispatchedEvent,
    ActionRefusedEvent,
    AssistantReplyEvent,
    EventBus,
    ShipStateEvent,
)
from voice_agent.backends.simulator.base import SimulatorClient
from voice_agent.logging_setup import get_logger

# Spoken when the LLM output cannot be parsed into a valid action.
UNPARSEABLE = "Sorry sir, I did not catch that order. Please say again."


class JsonActionProcessor(FrameProcessor):
    """Parse the LLM's JSON response, run the action, speak the acknowledgement."""

    def __init__(
        self,
        *,
        simulator: SimulatorClient,
        event_bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._simulator = simulator
        self._event_bus = event_bus
        self._log = get_logger("actions")
        self._parts: list[str] = []
        self._capturing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Only the LLM's own downstream response is rewritten; control frames
        # and any upstream frames pass straight through.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._parts = []
            self._capturing = True
            return  # swallowed; a fresh triple is emitted on the End frame
        if isinstance(frame, LLMTextFrame) and self._capturing:
            self._parts.append(frame.text)
            return  # swallowed -- raw JSON must not reach TTS
        if isinstance(frame, LLMFullResponseEndFrame) and self._capturing:
            self._capturing = False
            raw = "".join(self._parts).strip()
            self._parts = []
            spoken = await self._resolve(raw)
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            await self.push_frame(LLMTextFrame(spoken), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)
            return

        await self.push_frame(frame, direction)

    async def _resolve(self, raw: str) -> str:
        """Parse and dispatch ``raw``; return the line for TTS to speak."""
        try:
            parsed = parse_response(raw)
        except ActionParseError as exc:
            self._log.warning("action_parse_failed", error=str(exc), raw=raw[:300])
            self._publish_reply(UNPARSEABLE)
            return UNPARSEABLE
        result = await dispatch_action(parsed, self._simulator)
        spoken = result.spoken.strip() or UNPARSEABLE
        self._publish_turn_events(parsed, result, spoken)
        return spoken

    def _publish_reply(self, text: str) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(AssistantReplyEvent(text=text))

    def _publish_turn_events(
        self, parsed: HelmsmanResponse, result: DispatchResult, spoken: str
    ) -> None:
        """Push the assistant reply, the action (or refusal), and the ship state.

        Order matters for the UI: the reply lands first so the conversation
        view updates immediately; the action and ship-state follow so a heading
        change is visibly tied to the line the helmsman just spoke.
        """
        if self._event_bus is None:
            return

        self._event_bus.publish(AssistantReplyEvent(text=spoken))

        action = parsed.action
        if isinstance(action, ErrorAction):
            self._event_bus.publish(
                ActionRefusedEvent(
                    error_type=action.error_type,
                    reason=action.reason,
                    suggestion=action.suggestion,
                )
            )
            return

        if isinstance(action, AnswerAction):
            # Question-intent reply -- the assistant_reply event we already
            # published carries the spoken answer. Nothing to dispatch.
            return

        if not result.ok:
            # Recognised but NOT carried out -- a course order the helm cannot
            # steer, or a simulator failure behind the spoken BRIDGE_LOST. The
            # audit trail must say so: publishing these as "dispatched" would
            # show an incident reviewer an order recorded as executed that
            # never reached the ship. The reason is exactly the phrase the
            # operator heard, so log and speech stay consistent.
            self._event_bus.publish(
                ActionRefusedEvent(
                    error_type="not_executed",
                    reason=spoken,
                    suggestion="",
                )
            )
            return

        details: dict[str, Any]
        if isinstance(action, RudderAction):
            details = {"direction": action.direction, "degrees": action.degrees}
        elif isinstance(action, ThrottleAction):
            # Either form may be absent: `order` is the telegraph position, and
            # `speed` the knots fallback. Report whichever was ordered.
            details = {}
            if action.order is not None:
                details["order"] = action.order
            if action.speed is not None:
                details["speed"] = action.speed
                details["unit"] = action.unit
        elif isinstance(action, NavigationAction):
            details = {"course": action.course}
        elif isinstance(action, AutopilotAction):
            details = {"state": action.state}
        elif isinstance(action, AnchorAction):
            details = {"operation": action.operation}
            if action.chain_length is not None:
                details["chain_length"] = action.chain_length
        elif isinstance(action, StatusQueryAction):
            details = {"query": action.query}
        else:  # pragma: no cover -- discriminated union is exhaustive
            details = {}

        self._event_bus.publish(
            ActionDispatchedEvent(action=action.type, details=details)
        )

        state = result.ship_state
        if state is not None:
            self._event_bus.publish(
                ShipStateEvent(
                    heading_deg=state.heading_deg,
                    speed_kn=state.speed_kn,
                    engine_order=state.engine_order.value,
                    rudder_angle_deg=state.rudder_angle_deg,
                    sim_time_s=state.sim_time_s,
                    lat_deg=state.lat_deg,
                    lon_deg=state.lon_deg,
                )
            )
