"""Event types and the in-process pub/sub bus that feeds the WebSocket.

The pipeline emits events at three points:

* :class:`~voice_agent.actions.processor.JsonActionProcessor` -- assistant
  reply, action dispatched/refused, ship-state changed.
* :class:`~voice_agent.metrics.LatencyTracker` -- per-turn metrics.
* :class:`UserTranscriptObserver` -- the user's STT transcripts (a Pipecat
  observer, kept here rather than in ``metrics.py`` because publishing is its
  only job).

:class:`EventBus` is a tiny async fan-out: every subscriber gets its own bounded
queue. A slow WebSocket client cannot back-pressure the pipeline -- if its queue
fills, events for *that subscriber* are dropped, the rest are unaffected.

Each event is a Pydantic model with a literal ``kind`` discriminator so the
TypeScript client can switch on it without runtime type tags.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pydantic import BaseModel, Field

from voice_agent.logging_setup import get_logger


def _now_iso() -> str:
    """Wall-clock timestamp, ISO 8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


class _BaseEvent(BaseModel):
    """Shared fields. ``ts`` is set on construction; ``kind`` discriminates."""

    ts: str = Field(default_factory=_now_iso)


class TranscriptEvent(_BaseEvent):
    """User's spoken command, as the STT transcribed it."""

    kind: Literal["transcript"] = "transcript"
    text: str


class AssistantReplyEvent(_BaseEvent):
    """The spoken line the helmsman sent to TTS for this turn."""

    kind: Literal["assistant_reply"] = "assistant_reply"
    text: str


class ActionDispatchedEvent(_BaseEvent):
    """An action was carried out against the simulator."""

    kind: Literal["action_dispatched"] = "action_dispatched"
    action: str  # set_heading | set_engine_telegraph | get_ship_state
    details: dict[str, Any] = Field(default_factory=dict)


class ActionRefusedEvent(_BaseEvent):
    """The LLM returned an error action (ambiguous / out-of-scope / invalid)."""

    kind: Literal["action_refused"] = "action_refused"
    error_type: str
    reason: str
    suggestion: str = ""


class ShipStateEvent(_BaseEvent):
    """Latest known ship state (after a dispatch that returned it)."""

    kind: Literal["ship_state"] = "ship_state"
    heading_deg: float
    speed_kn: float
    engine_order: str


class TurnMetricsEvent(_BaseEvent):
    """Per-turn latency breakdown produced by ``LatencyTracker``."""

    kind: Literal["turn_metrics"] = "turn_metrics"
    turn_index: int
    metrics_ms: dict[str, float]


class InputModeChangedEvent(_BaseEvent):
    """Server-side mic was toggled on/off via ``POST /api/control/mic``.

    Every connected browser tab listens for this event so the chat UI's mic
    toggle stays in sync without having to poll. ``mic_enabled=True`` means
    audio input is live and the chatbox should be locked; False means the
    operator is in text mode.
    """

    kind: Literal["input_mode_changed"] = "input_mode_changed"
    mic_enabled: bool


# Union for serialization on the wire. Pydantic does not need a discriminated
# union here -- we always know the concrete type at publish time -- but the
# Literal ``kind`` makes the client side unambiguous.
Event = (
    TranscriptEvent
    | AssistantReplyEvent
    | ActionDispatchedEvent
    | ActionRefusedEvent
    | ShipStateEvent
    | TurnMetricsEvent
    | InputModeChangedEvent
)


class EventBus:
    """Async fan-out pub/sub. One bounded queue per subscriber.

    ``publish()`` never awaits anything that depends on a subscriber, so it is
    safe to call from any pipeline processor or observer. If a subscriber's
    queue is full -- a slow client -- its event is silently dropped for that
    subscriber only; the bus keeps a counter so the dropped count is visible.
    """

    def __init__(self, *, per_subscriber_queue_size: int = 256) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._queue_size = per_subscriber_queue_size
        self._dropped = 0
        self._log = get_logger("api")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def dropped(self) -> int:
        """Total events dropped across the bus lifetime (slow-subscriber back-pressure)."""
        return self._dropped

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register a new subscriber and return its queue."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        """Drop a subscriber. Idempotent."""
        self._subscribers.discard(q)

    def publish(self, event: Event) -> None:
        """Fan ``event`` out to every subscriber without awaiting.

        Uses ``put_nowait``: a full queue means a slow client, so we drop the
        event for that subscriber and continue. This keeps the pipeline immune
        to client back-pressure.
        """
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped += 1


class UserTranscriptObserver(BaseObserver):
    """Pipecat observer that publishes user transcripts to an EventBus.

    Sits separate from :class:`~voice_agent.metrics.LatencyTracker` because the
    two have different responsibilities; sharing one observer would couple them.
    Only ``TranscriptionFrame`` is needed -- the assistant reply is published by
    :class:`~voice_agent.actions.processor.JsonActionProcessor`, which has the
    *final* spoken text after action dispatch.
    """

    def __init__(self, *, event_bus: EventBus, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bus = event_bus
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame = data.frame
        if frame.id in self._seen:
            return
        self._seen.add(frame.id)
        # Bound the dedup set; transcripts are rare so a coarse cap is fine.
        if len(self._seen) > 4096:
            self._seen = {frame.id}

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                self._bus.publish(TranscriptEvent(text=text))
