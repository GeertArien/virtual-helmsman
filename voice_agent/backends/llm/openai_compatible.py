"""Remote LLM client over an OpenAI-compatible HTTP API.

Wraps Pipecat's OpenAILLMService pointed at ``$LLM_BASE_URL``. The LLM runs on
a separate machine; this module only consumes the ``/v1`` endpoint.
"""

from __future__ import annotations


def build_llm(config):
    """Build an OpenAI-compatible LLM service from the ``llm`` config block."""
    raise NotImplementedError("openai_compatible.build_llm is a scaffold stub")
