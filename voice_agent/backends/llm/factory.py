"""LLM factory: selects an LLM backend from config.

Three backends are wired:

* ``openai_compatible`` -- :mod:`voice_agent.backends.llm.openai_compatible`,
  a Pipecat ``OpenAILLMService`` pointed at LM Studio (or any other
  ``/v1``-compatible server). Command parsing only.
* ``n8n`` -- :mod:`voice_agent.backends.llm.n8n`, a thin
  :class:`pipecat.processors.frame_processor.FrameProcessor` that proxies
  each LLM turn to the n8n helmsman webhook described in ``API.md``.
  Handles both command parsing and RAG question answering.
* ``langgraph`` -- :mod:`voice_agent.backends.llm.langgraph_helmsman`, the
  in-backend reimplementation of the n8n runtime path using LangGraph +
  LangChain + Langfuse (see ``docs/LANGGRAPH_BACKEND.md``). Same command +
  RAG behaviour as ``n8n`` with no external workflow engine. Requires the
  optional ``langgraph`` extra.

To add a backend: create a module under ``backends/llm/`` exposing
``build_llm(llm_config)`` and add one entry to ``_BUILDERS``. The signature
returns ``FrameProcessor`` so both pipecat ``LLMService`` subclasses and
plain frame processors slot into the same pipeline position.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.processors.frame_processor import FrameProcessor

from voice_agent.backends.llm import langgraph_helmsman, n8n, openai_compatible

_BUILDERS: dict[str, Callable[[Any], FrameProcessor]] = {
    "openai_compatible": openai_compatible.build_llm,
    "n8n": n8n.build_llm,
    "langgraph": langgraph_helmsman.build_llm,
}


def create_llm(config: Any) -> FrameProcessor:
    """Return a frame processor for ``config.backend`` (the ``llm`` block)."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown LLM backend: {config.backend!r}. "
            f"Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
