"""HTTP endpoints for the control plane.

Three routes mounted at ``/api/control``:

* ``GET  /api/control/state``  -- snapshot used by the browser on first load.
* ``POST /api/control/mic``    -- toggle server-side mic; body ``{enabled}``.
  Broadcasts :class:`InputModeChangedEvent` to every WebSocket subscriber so
  any open tab updates immediately.
* ``POST /api/control/text``   -- inject a typed command into the pipeline.
  Body ``{text}``. Returns 409 if the mic is still enabled: the two input
  modes are mutually exclusive by design (the user explicitly disables the
  mic to type), so this is a programming error worth flagging loudly rather
  than letting the LLM see two overlapping turns. Returns 503 when no
  ``inject_text`` callable was supplied (browser-audio mode has no single
  local task to inject into -- voice is the input there), while the mic
  toggle keeps working: it gates the MicGate in every assembled pipeline.

The router takes a single ``inject_text`` callable instead of leaking the
:class:`pipecat.pipeline.task.PipelineTask` and :class:`LLMContext` into the
API layer. The callable is responsible for "append a user message to the
shared context, then trigger an LLM run" -- :func:`build_text_injector` in
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

from voice_agent.api.control import ControlState
from voice_agent.api.events import (
    EventBus,
    InputModeChangedEvent,
    TranscriptEvent,
)
from voice_agent.logging_setup import get_logger

# Async function signature for "inject a user-typed command as one turn".
# The implementation appends the message to the shared LLM context and
# triggers the LLM (via LLMRunFrame). The router never sees Pipecat types.
TextInjector = Callable[[str], Awaitable[None]]


# --- request / response models ---------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlStateResponse(_Strict):
    mic_enabled: bool


class MicRequest(_Strict):
    enabled: bool


class TextRequest(_Strict):
    text: str = Field(min_length=1, max_length=2000)


# --- router factory --------------------------------------------------------


def create_control_router(
    *,
    state: ControlState,
    event_bus: EventBus,
    inject_text: TextInjector | None,
) -> APIRouter:
    """Build the ``/api/control`` router bound to a live ``ControlState``."""
    router = APIRouter(prefix="/api/control", tags=["control"])
    log = get_logger("api.control")

    @router.get("/state")
    async def get_state() -> ControlStateResponse:
        """Current control-plane snapshot."""
        return ControlStateResponse(mic_enabled=state.mic_enabled)

    @router.post("/mic")
    async def set_mic(req: MicRequest) -> ControlStateResponse:
        """Enable / disable the server-side microphone input.

        The shared LLM context is wiped after every assistant turn by
        :class:`SingleTurnContextReset` in the pipeline tail, so toggling
        the mic has no special "clear history" effect -- history is always
        already empty at turn boundaries.
        """
        changed = state.mic_enabled != req.enabled
        state.mic_enabled = req.enabled
        if changed:
            log.info("mic_toggled", mic_enabled=state.mic_enabled)
            event_bus.publish(InputModeChangedEvent(mic_enabled=state.mic_enabled))
        return ControlStateResponse(mic_enabled=state.mic_enabled)

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
        if inject_text is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Text input is disabled in browser-audio mode -- there is "
                    "no single local pipeline task to inject into; use voice."
                ),
            )
        if state.mic_enabled:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Server mic is enabled; disable it via POST /api/control/mic "
                    "before sending typed commands."
                ),
            )

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
