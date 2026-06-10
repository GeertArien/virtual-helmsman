"""Remote LLM client over an OpenAI-compatible HTTP API.

Wraps Pipecat's ``OpenAILLMService`` pointed at the ``llm.base_url``. Typical
target is LM Studio's ``/v1`` endpoint -- one of two backends the helmsman
supports (the other is :mod:`voice_agent.backends.llm.langgraph_helmsman`,
which adds intent routing + RAG).

The :data:`~voice_agent.actions.schema.RESPONSE_FORMAT` JSON-schema constraint
is baked in here -- LM Studio honours ``response_format`` to grammar-constrain
decoding to the helmsman action envelope, which removes most malformed-JSON
failure modes at the inference layer.

Note: ``timeout_seconds`` and ``max_retries`` from config are not yet forwarded
to the underlying OpenAI client -- ``OpenAILLMService`` doesn't expose them
directly in this Pipecat version.
"""

from __future__ import annotations

from typing import Any

from pipecat.services.openai.llm import OpenAILLMService

from voice_agent.actions.schema import RESPONSE_FORMAT
from voice_agent.logging_setup import get_logger


def build_llm(config: Any) -> OpenAILLMService:
    """Build the OpenAI-compatible LLM service from the ``llm`` config block."""
    log = get_logger("llm.openai")
    api_key = config.resolved_api_key()
    log.info(
        "openai_llm_init",
        base_url=config.base_url,
        model=config.model,
        api_key_set=api_key is not None,
    )
    # Local OpenAI-compatible servers often need no key; the OpenAI client
    # still requires a non-empty string, so fall back to a placeholder.
    # The model goes through Settings -- the bare ``model=`` kwarg is
    # deprecated. ``response_format`` is the JSON-schema constraint for
    # the helmsman's action envelope; baking it in here keeps the
    # backend-specific knowledge out of the pipeline assembler.
    return OpenAILLMService(
        api_key=api_key or "not-needed",
        base_url=config.base_url,
        settings=OpenAILLMService.Settings(
            model=config.model,
            extra={"response_format": RESPONSE_FORMAT},
        ),
    )
