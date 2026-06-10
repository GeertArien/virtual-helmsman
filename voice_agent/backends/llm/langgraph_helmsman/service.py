"""Pipecat frame processor that runs each turn through the LangGraph helmsman.

Drop-in peer of :class:`voice_agent.backends.llm.n8n.N8nLLMService`: it sits in
the same pipeline slot and emits the identical frame triple --
``LLMFullResponseStartFrame`` -> one ``LLMTextFrame`` carrying the internal
HelmsmanResponse JSON string -> ``LLMFullResponseEndFrame``. It is not a
streaming source; the graph returns synchronously and the whole reply lands in
one text frame.

Where the n8n adapter POSTs to a webhook, this one ``await``\\s the compiled
LangGraph runner built in :mod:`graph`. Every failure mode (graph exception,
empty context) is mapped to an ``error`` envelope so the downstream
:class:`JsonActionProcessor` never sees malformed JSON and the helmsman speaks
the graceful "Lost contact with the bridge" fallback.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import httpx
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.logging_setup import get_logger

from . import helpers

Runner = Callable[[str], Awaitable[dict[str, Any]]]


class LangGraphLLMService(FrameProcessor):
    """Proxy each LLM turn to an in-process LangGraph helmsman graph.

    Reacts to :class:`LLMContextFrame` the way Pipecat's OpenAI service does:
    push a Start frame, run the graph on the latest user message, emit the
    response text frame, push an End frame. All other frames pass through.

    ``runner`` is injected (the compiled graph's per-turn coroutine) so the
    network-touching graph stays out of unit tests -- tests pass a fake runner.
    ``client`` is the shared httpx client used by the graph's Qdrant/embedding
    calls; the service owns its lifecycle and closes it on cleanup.
    """

    def __init__(
        self,
        *,
        runner: Runner,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._runner = runner
        self._client = client
        self._log = get_logger("llm.langgraph")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        try:
            text = helpers.latest_user_text(list(frame.context.messages))
            if not text:
                self._log.warning("langgraph_no_user_message_in_context")
                envelope = helpers.error_envelope(
                    "No user message found in the LLM context.",
                    spoken="Standing by for your command, sir.",
                )
            else:
                envelope = await self._run(text)
            await self.push_frame(LLMTextFrame(json.dumps(envelope)))
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _run(self, chat_input: str) -> dict[str, Any]:
        """Run the graph; map any failure to a parseable ``error`` envelope."""
        try:
            envelope = await self._runner(chat_input)
        except Exception as exc:  # noqa: BLE001 -- never surface a raw crash to TTS
            self._log.error("langgraph_turn_failed", error=str(exc))
            return helpers.error_envelope(f"helmsman graph failed: {exc}")
        self._log.info(
            "langgraph_turn",
            action_type=(envelope.get("action") or {}).get("type"),
        )
        return envelope

    async def cleanup(self) -> None:
        """Pipecat lifecycle hook -- close the shared httpx client on shutdown."""
        await super().cleanup()
        if self._client is not None:
            await self._client.aclose()


def build_llm(config: Any) -> LangGraphLLMService:
    """Build the LangGraph adapter from the ``llm`` config block.

    Signature mirrors the other LLM builders so the factory dispatches on
    ``backend`` without per-builder kwargs. The compiled graph and its httpx
    client are created here; importing the heavy LangGraph stack happens inside
    :func:`graph.build_runner`, so a misconfigured install fails loudly here
    rather than at module import.
    """
    from . import graph

    client = httpx.AsyncClient(timeout=config.timeout_seconds)
    runner = graph.build_runner(config, client)
    return LangGraphLLMService(runner=runner, client=client)
