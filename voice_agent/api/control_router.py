"""HTTP endpoint for injecting typed commands into the pipeline.

One route mounted at ``/api/control``:

* ``POST /api/control/text`` -- inject a typed command as a user turn.
  Body ``{text}``. The reply surfaces in the transcript panel and the action
  drives the shared simulator, exactly as a spoken command would.

Voice input is the browser-audio (WebRTC) path; this route is the dashboard
chatbox. The router takes a single ``inject_text`` callable instead of leaking
the :class:`pipecat.pipeline.task.PipelineTask` and :class:`LLMContext` into the
API layer. The callable is responsible for "append a user message to the shared
context, then trigger an LLM run" -- :func:`build_text_injector` in
:mod:`voice_agent.pipeline` constructs the real one; tests pass a list-append
stub.

A previous version pushed an ``LLMMessagesAppendFrame`` straight into the
pipeline. That broke under Pipecat 1.2: both the user and assistant
aggregators in :class:`LLMContextAggregatorPair` *each* handle that frame
and *each* call ``add_messages`` on the shared context, so every typed
command was appended twice. Using the context directly + ``LLMRunFrame``
sidesteps the duplication.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from voice_agent.api.events import EventBus, TranscriptEvent
from voice_agent.logging_setup import get_logger

# Async function signature for "inject a user-typed command as one turn".
# The implementation appends the message to the shared LLM context and
# triggers the LLM (via LLMRunFrame). The router never sees Pipecat types.
TextInjector = Callable[[str], Awaitable[None]]


# --- request models --------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextRequest(_Strict):
    text: str = Field(min_length=1, max_length=2000)


# --- router factory --------------------------------------------------------


def create_control_router(
    *,
    event_bus: EventBus,
    inject_text: TextInjector,
) -> APIRouter:
    """Build the ``/api/control`` router bound to a text injector."""
    router = APIRouter(prefix="/api/control", tags=["control"])
    log = get_logger("api.control")

    @router.post("/text")
    async def send_text(req: TextRequest) -> dict[str, Any]:
        """Inject a typed command as a user turn.

        Delegates to the ``inject_text`` callable which appends the user
        message to the shared :class:`LLMContext` and queues an
        ``LLMRunFrame`` to drive inference -- the rest of the pipeline
        (JsonActionProcessor → TTS → simulator) handles it identically to a
        transcribed voice command.

        We also publish a :class:`TranscriptEvent` so the browser's
        conversation panel surfaces the typed line; the
        :class:`~voice_agent.api.events.UserTranscriptObserver` only fires
        for STT-produced ``TranscriptionFrame`` s.
        """
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be blank.")

        # Surface the typed line in the conversation panel before the LLM
        # responds, so the operator sees their input land immediately.
        event_bus.publish(TranscriptEvent(text=text))

        await inject_text(text)
        log.info("text_command_queued", chars=len(text))
        return {
            "status": "queued",
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    return router
