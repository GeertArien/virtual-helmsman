"""LLM factory: selects an LLM backend from config.

Two backends are wired:

* ``openai_compatible`` -- :mod:`voice_agent.backends.llm.openai_compatible`,
  a Pipecat ``OpenAILLMService`` pointed at LM Studio (or any other
  ``/v1``-compatible server). Command parsing only.
* ``langgraph`` -- :mod:`voice_agent.backends.llm.langgraph_helmsman`, the
  in-backend helmsman: intent routing -> command parsing or hybrid-RAG
  question answering, with LangGraph orchestration, LangChain chat models,
  optional Langfuse tracing, and per-turn runtime audit rows (see
  ``docs/LANGGRAPH_BACKEND.md``). Requires the optional ``langgraph`` extra.

To add a backend: create a module under ``backends/llm/`` exposing
``build_llm(llm_config)`` and add one entry to ``_BUILDERS``. The signature
returns ``FrameProcessor`` so both pipecat ``LLMService`` subclasses and
plain frame processors slot into the same pipeline position.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.processors.frame_processor import FrameProcessor

from voice_agent.backends.llm import langgraph_helmsman, openai_compatible

_BUILDERS: dict[str, Callable[[Any], FrameProcessor]] = {
    "openai_compatible": openai_compatible.build_llm,
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
