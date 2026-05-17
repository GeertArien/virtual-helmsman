"""Remote LLM client over an OpenAI-compatible HTTP API.

Wraps Pipecat's ``OpenAILLMService`` pointed at the remote ``$LLM_BASE_URL``.
The LLM runs on a separate machine; this module only consumes the ``/v1``
endpoint.

Note: ``timeout_seconds`` and ``max_retries`` from config are not yet forwarded
to the underlying OpenAI client — ``OpenAILLMService`` does not expose them
directly in this Pipecat version. See README "Remote LLM configuration".
"""

from __future__ import annotations

from typing import Any

from pipecat.services.openai.llm import OpenAILLMService

from voice_agent.logging_setup import get_logger


def build_llm(config: Any, *, extra: dict[str, Any] | None = None) -> OpenAILLMService:
    """Build the OpenAI-compatible LLM service from the ``llm`` config block.

    ``extra`` is merged verbatim into every chat-completion request — the caller
    uses it to pass server-specific params such as ``response_format``.
    """
    log = get_logger("llm")
    api_key = config.resolved_api_key()
    log.info(
        "llm_init",
        base_url=config.base_url,
        model=config.model,
        api_key_set=api_key is not None,
    )
    # Local OpenAI-compatible servers often need no key; the OpenAI client still
    # requires a non-empty string, so fall back to a placeholder.
    # The model goes through Settings — the bare `model=` kwarg is deprecated.
    return OpenAILLMService(
        api_key=api_key or "not-needed",
        base_url=config.base_url,
        settings=OpenAILLMService.Settings(model=config.model, extra=extra or {}),
    )
