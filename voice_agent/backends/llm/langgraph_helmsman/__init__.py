"""In-backend LangGraph helmsman LLM backend.

Reimplements the runtime path of the n8n ``virtual_helmsman_unified`` workflow
(intent classification, command parsing, and hybrid-RAG question answering with
rerank + adjacent-chunk expansion) natively in Python, using LangGraph for
orchestration, LangChain (``ChatOpenAI``) for the LLM calls, and Langfuse for
optional tracing. See ``docs/LANGGRAPH_BACKEND.md``.

Only :func:`build_llm` is exported; importing this package is light (the heavy
LangGraph/LangChain imports are deferred until :func:`build_llm` runs).
"""

from __future__ import annotations

from .service import LangGraphLLMService, build_llm

__all__ = ["LangGraphLLMService", "build_llm"]
