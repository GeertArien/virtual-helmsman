"""Pipeline processor that gates server-side mic input.

Sits immediately after ``transport.input()`` in the pipeline so it can drop
microphone audio before it ever reaches STT, the VAD analyzer, or the turn
detector. When :attr:`ControlState.mic_enabled` is False every inbound
:class:`InputAudioRawFrame` (and its subclasses, e.g. ``UserAudioRawFrame``)
is silently swallowed; every other frame -- control frames, transcription
frames, system frames -- passes through untouched.

Dropping audio at the head of the pipeline is the cheapest way to "mute" the
mic from Python without reaching into the OS audio driver: STT receives no
samples, so it produces no transcripts, so the LLM is not triggered. The
LocalAudioTransport keeps polling the device (so toggling back on is
instant), it just has no downstream effect.
"""

from __future__ import annotations

from typing import Any

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.api.control import ControlState


class MicGate(FrameProcessor):
    """Drop ``InputAudioRawFrame``s when ``state.mic_enabled`` is False."""

    def __init__(self, *, state: ControlState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Always let Pipecat run its own bookkeeping first.
        await super().process_frame(frame, direction)

        # Only inbound audio is gated; upstream / control frames are passed
        # through verbatim so the rest of the pipeline keeps working (e.g.
        # bot-speaking signals, system frames).
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InputAudioRawFrame)
            and not self._state.mic_enabled
        ):
            return  # swallow

        await self.push_frame(frame, direction)
