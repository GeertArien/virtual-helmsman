"""Pipecat ``FrameProcessor`` that turns the LLM's JSON action into speech.

Sits between the LLM service and TTS. The LLM emits one JSON object
(``{"action": ..., "response": ...}``) as streamed text frames; this processor
buffers that text, parses it, dispatches the action to the simulator, and
forwards only the spoken ``response`` downstream -- so the raw JSON never
reaches TTS or the conversation context.

The LLM's ``LLMFullResponseStartFrame`` / ``LLMTextFrame`` / ``...EndFrame``
triple is consumed and replaced with a fresh triple carrying the spoken line.
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

from voice_agent.actions.dispatch import dispatch_action
from voice_agent.actions.schema import ActionParseError, parse_response
from voice_agent.backends.simulator.base import SimulatorClient
from voice_agent.logging_setup import get_logger

# Spoken when the LLM output cannot be parsed into a valid action.
UNPARSEABLE = "Sorry sir, I did not catch that order. Please say again."


class JsonActionProcessor(FrameProcessor):
    """Parse the LLM's JSON response, run the action, speak the acknowledgement."""

    def __init__(self, *, simulator: SimulatorClient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._simulator = simulator
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
            return UNPARSEABLE
        result = await dispatch_action(parsed, self._simulator)
        return result.spoken.strip() or UNPARSEABLE
