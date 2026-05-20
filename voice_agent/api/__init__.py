"""In-process control & observability API for the virtual-helmsman frontend.

A thin FastAPI service that runs alongside the Pipecat pipeline in the same
Python process. It exposes:

* ``GET  /api/health``    -- liveness ping.
* ``GET  /api/session``   -- current session snapshot (config + last ship state
  + last metrics + recent turns).
* ``WS   /ws/events``     -- live stream of events: user transcripts, assistant
  replies, action dispatches, ship-state updates, per-turn metrics.

Events flow from the pipeline to the WebSocket via
:class:`voice_agent.api.events.EventBus`. Three publishers feed it:
:class:`~voice_agent.actions.processor.JsonActionProcessor` (assistant replies,
actions, ship state), :class:`~voice_agent.metrics.LatencyTracker` (per-turn
metrics), and :class:`~voice_agent.api.events.UserTranscriptObserver` (the
user's STT transcripts).
"""

from voice_agent.api.events import (
    ActionDispatchedEvent,
    ActionRefusedEvent,
    AssistantReplyEvent,
    Event,
    EventBus,
    ShipStateEvent,
    TranscriptEvent,
    TurnMetricsEvent,
    UserTranscriptObserver,
)

__all__ = [
    "ActionDispatchedEvent",
    "ActionRefusedEvent",
    "AssistantReplyEvent",
    "Event",
    "EventBus",
    "ShipStateEvent",
    "TranscriptEvent",
    "TurnMetricsEvent",
    "UserTranscriptObserver",
]
