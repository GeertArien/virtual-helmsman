"""n8n helmsman webhook adapter, presented to the pipeline as an LLM service.

The n8n workflow described in ``API.md`` exposes a synchronous POST endpoint
(``/webhook/helmsman``) that takes a user message and returns one of two
envelope shapes depending on intent classification:

* command-intent: ``{intent: "command", output, action: {...}, source: null}``
  -- the action JSON inside follows the n8n action vocabulary, which (after
  Phase 1 of the modular-LLM work) matches our internal
  :class:`HelmsmanAction` exactly. We pass it through verbatim and let
  :class:`JsonActionProcessor` dispatch it to the simulator.
* question-intent: ``{intent: "question", output, action: null, source: {...}}``
  -- the output text already contains the citation suffix. We map this to
  the synthetic :class:`AnswerAction` so downstream parsing succeeds; no
  simulator dispatch happens.

The adapter sits in the same pipeline slot as :class:`OpenAILLMService` and
emits the same frame triple (``LLMFullResponseStartFrame`` -> one
``LLMTextFrame`` carrying the internal-JSON string ->
``LLMFullResponseEndFrame``). It is *not* a streaming source: n8n returns
synchronously, so the whole response lands in one text frame.

Failures (HTTP error, JSON parse fail, timeout, refused connection) emit
an ``error`` action so the downstream "Sorry sir" fallback path still
fires -- never let the pipeline see a malformed JSON string.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.actions.dispatch import BRIDGE_LOST
from voice_agent.logging_setup import get_logger


def _latest_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the most recent user-role message's content, or ``None``.

    The pipeline's user-aggregator appends one user message per turn, then
    pushes the context downstream -- so the last user-role entry is what
    n8n should answer. Skipping the system prompt and any prior assistant
    replies makes this robust to context-management changes elsewhere.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return None


def _error_envelope(reason: str, spoken: str = BRIDGE_LOST) -> str:
    """A HelmsmanResponse-shaped JSON encoding an ``error`` action.

    Used when the n8n call fails or returns garbage; JsonActionProcessor
    parses it cleanly and the helmsman speaks ``spoken`` ("Lost contact
    with the bridge, sir...") instead of crashing the pipeline.
    """
    return json.dumps(
        {
            "action": {
                "type": "error",
                "error_type": "bridge_error",
                "reason": reason,
                "suggestion": (
                    "Check that the n8n helmsman workflow is active and "
                    "reachable at the configured base_url."
                ),
            },
            "response": spoken,
        }
    )


def _translate_envelope(body: dict[str, Any]) -> str:
    """Map an n8n response body to our internal HelmsmanResponse JSON string.

    Branches on ``intent``:

    * ``command``: pass ``action`` through verbatim (the schemas align
      since Phase 1); ``response`` is n8n's ``output`` (the spoken
      acknowledgement).
    * ``error`` (iteration 12/13 of API.md): one of the four internal
      LLM calls inside the workflow failed (LM Studio unreachable,
      context overflow, model not loaded, etc.). n8n returns an
      ``action`` with ``type: "error"`` plus an extra ``http_status``
      field. We pass the action through -- the extra field is
      ignored by Pydantic since :class:`ErrorAction` doesn't declare
      ``extra="forbid"`` -- so the downstream
      :class:`JsonActionProcessor` publishes an :class:`ActionRefusedEvent`
      and TTS speaks the ``output`` message verbatim.
    * anything else (``question`` or missing): synthesise an ``answer``
      action so :class:`JsonActionProcessor` parses cleanly; the spoken
      response is n8n's ``output`` (citation already inline per API.md).
    """
    intent = body.get("intent")
    output = body.get("output", "")
    action = body.get("action")

    if intent in ("command", "error") and isinstance(action, dict):
        internal_action: dict[str, Any] = action
    else:
        # Question intent (or any unexpected shape) -> answer pseudo-action.
        internal_action = {"type": "answer"}

    return json.dumps({"action": internal_action, "response": output})


class N8nLLMService(FrameProcessor):
    """Pipecat frame processor that proxies LLM turns to an n8n webhook.

    Reacts to :class:`LLMContextFrame` exactly the way
    :class:`pipecat.services.openai.base_llm.BaseOpenAILLMService` does --
    pushes a Start frame, posts to n8n, emits the response text frame,
    pushes an End frame. Other frames pass through unchanged.

    The configured ``model`` is forwarded as the ``model`` field in the
    POST body. Per ``API.md`` the n8n workflow applies it uniformly to
    every internal LLM call (intent classify, command parse, rerank,
    RAG answer) and bubbles the upstream LM Studio "model not found"
    error if the identifier isn't loaded.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        webhook_path: str = "/webhook/helmsman",
        rerank: bool = True,
        timeout_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._url = base_url.rstrip("/") + webhook_path
        self._model = model
        self._rerank = rerank
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._log = get_logger("llm.n8n")
        self._log.info("n8n_llm_init", url=self._url, model=model, rerank=rerank)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        try:
            text = _latest_user_text(list(frame.context.messages))
            if not text:
                self._log.warning("n8n_no_user_message_in_context")
                payload = _error_envelope(
                    "No user message found in the LLM context.",
                    spoken="Standing by for your command, sir.",
                )
            else:
                payload = await self._call_n8n(text)
            await self.push_frame(LLMTextFrame(payload))
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _call_n8n(self, chat_input: str) -> str:
        """POST to the helmsman webhook; return an internal-JSON string.

        Maps every failure mode (timeout, network error, non-2xx, non-JSON
        body, missing fields) to an ``error`` action envelope so the
        downstream parser always sees a well-formed HelmsmanResponse.
        """
        try:
            res = await self._client.post(
                self._url,
                json={
                    "chatInput": chat_input,
                    "rerank": self._rerank,
                    "model": self._model,
                },
            )
        except httpx.RequestError as exc:
            self._log.error("n8n_unreachable", error=str(exc), url=self._url)
            return _error_envelope(f"n8n unreachable: {exc}")

        if res.status_code >= 400:
            self._log.error(
                "n8n_http_error",
                status=res.status_code,
                body=res.text[:500],
            )
            return _error_envelope(
                f"n8n returned HTTP {res.status_code}: {res.text[:200]}"
            )

        try:
            body = res.json()
        except ValueError as exc:
            self._log.error("n8n_non_json_body", error=str(exc), body=res.text[:500])
            return _error_envelope("n8n returned non-JSON.")

        if not isinstance(body, dict):
            self._log.error("n8n_non_object_body", body=str(body)[:300])
            return _error_envelope("n8n returned a non-object body.")

        self._log.info(
            "n8n_turn",
            intent=body.get("intent"),
            chars_out=len(body.get("output", "") or ""),
        )
        return _translate_envelope(body)

    async def cleanup(self) -> None:
        """Pipecat lifecycle hook -- close the httpx client on shutdown."""
        await super().cleanup()
        await self._client.aclose()


def build_llm(config: Any) -> N8nLLMService:
    """Build the n8n adapter from the ``llm`` config block.

    Signature mirrors :func:`voice_agent.backends.llm.openai_compatible.build_llm`
    so the factory can dispatch on backend without per-builder kwargs.
    """
    return N8nLLMService(
        base_url=config.base_url,
        model=config.model,
        webhook_path=config.webhook_path,
        rerank=config.rerank,
        timeout_seconds=config.timeout_seconds,
    )
