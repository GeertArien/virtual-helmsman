"""LangGraph orchestration of the runtime helmsman turn.

This is the in-backend replacement for the n8n ``virtual_helmsman_unified``
workflow's runtime path. A single :class:`~langgraph.graph.StateGraph` routes
each turn:

    classify ─┬─ command ─────────────────────────────────▶ END
              └─ retrieve → select → expand → answer ─────▶ END

With ``llm.mode: commands_only`` the graph collapses to ``command ─▶ END``:
no classifier round-trip, no RAG branch, questions refused by the command
parser itself, and no qdrant/embedding dependency at runtime (issue #21).

* **classify** -- one-word intent classification (COMMAND / QUESTION).
* **command** -- the helmsman command parser (shared system prompt +
  JSON-schema ``response_format``), shaped into the internal HelmsmanResponse.
* **retrieve** -- embed the query (``bge-m3``) and run Qdrant hybrid RRF
  retrieval (dense + BM25).
* **select** -- LLM listwise rerank to top-3 (``rerank: true``) or RRF top-3.
* **expand** -- adjacent-chunk (``±1``) expansion (``expansion: true``) or
  passthrough.
* **answer** -- schema-constrained RAG answer + citation, shaped into the
  synthetic ``answer`` envelope.

LangChain (``ChatOpenAI``) makes the LLM calls; :mod:`retrieval` makes the
Qdrant/embedding calls; :mod:`helpers` does all the pure shaping. The heavy
imports (LangGraph, LangChain) are deferred to :func:`build_runner` so the
package imports cleanly even when those optional deps aren't installed.

The runner returned by :func:`build_runner` takes a user message and returns
the internal HelmsmanResponse dict (``{action, response}``); the Pipecat
service in :mod:`service` serialises that into one ``LLMTextFrame``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

import httpx

from voice_agent.actions.prompt import SYSTEM_PROMPT
from voice_agent.actions.schema import RESPONSE_FORMAT
from voice_agent.logging_setup import get_logger

from voice_agent import tracing

from . import helpers, retrieval

_log = get_logger("llm.langgraph")

# Type of the per-turn entry point the service awaits.
Runner = Callable[[str], Awaitable[dict[str, Any]]]
# Best-effort audit sink: (document_naam, actie, resultaat) -> None.
AuditWriter = Callable[[str, str, str], Awaitable[None]]


class _MissingDependency(ImportError):
    """Raised by :func:`build_runner` when the LangGraph stack isn't installed."""


class _TurnState(TypedDict, total=False):
    """LangGraph state schema for one helmsman turn.

    StateGraph derives its state channels from the schema's annotations; a
    bare ``dict`` has none, so the ``chat_input`` passed to ``ainvoke`` would
    be silently dropped and the first node would KeyError. Every key a node
    reads or returns must be declared here.
    """

    chat_input: str
    intent: str
    chunks: list[dict[str, Any]]
    result: dict[str, Any]


def _require_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import the optional LangGraph/LangChain stack or raise a clear error."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise _MissingDependency(
            "The 'langgraph' LLM backend requires the optional dependencies "
            "langgraph, langchain-openai, and langchain-core. Install them with "
            "`pip install -e \".[langgraph]\"`."
        ) from exc
    return StateGraph, START, END, ChatOpenAI, SystemMessage, HumanMessage


def build_runner(
    config: Any,
    client: httpx.AsyncClient,
    audit_writer: AuditWriter | None = None,
) -> Runner:
    """Compile the helmsman graph and return a per-turn runner coroutine.

    ``client`` is owned by the caller (the service) and reused for every
    Qdrant/embedding call; it is closed on service shutdown. ``audit_writer``,
    when supplied, receives one runtime audit row per command/question turn
    (best-effort -- a failed write is logged, never raised). Raises
    :class:`_MissingDependency` if the optional stack is absent.
    """
    StateGraph, START, END, ChatOpenAI, SystemMessage, HumanMessage = _require_stack()

    async def _audit(row: tuple[str, str, str]) -> None:
        if audit_writer is None:
            return
        try:
            await audit_writer(*row)
        except Exception as exc:  # noqa: BLE001 -- audit must never break a turn
            _log.warning("audit_write_failed", error=str(exc))

    api_key = config.resolved_api_key() or "not-needed"
    qdrant_headers = config.resolved_qdrant_headers()
    embed_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    callback_handler = tracing.build_callback_handler(config)

    # Four task-specific chat models against the same LM Studio /v1 server.
    classify_llm = ChatOpenAI(
        model=config.model, base_url=config.base_url, api_key=api_key,
        temperature=0, max_tokens=8,
    )
    command_llm = ChatOpenAI(
        model=config.model, base_url=config.base_url, api_key=api_key,
        temperature=0, max_tokens=512,
        model_kwargs={"response_format": RESPONSE_FORMAT},
    )
    rerank_llm = ChatOpenAI(
        model=config.model, base_url=config.base_url, api_key=api_key,
        temperature=0, max_tokens=64,
    )
    answer_llm = ChatOpenAI(
        model=config.model, base_url=config.base_url, api_key=api_key,
        temperature=0, max_tokens=800,
        model_kwargs={"response_format": helpers.RAG_RESPONSE_FORMAT},
    )

    async def classify_node(state: dict[str, Any]) -> dict[str, Any]:
        resp = await classify_llm.ainvoke(
            [SystemMessage(helpers.CLASSIFY_SYSTEM), HumanMessage(state["chat_input"])]
        )
        return {"intent": helpers.parse_intent(str(resp.content))}

    async def command_node(state: dict[str, Any]) -> dict[str, Any]:
        resp = await command_llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["chat_input"])]
        )
        envelope = helpers.command_envelope(str(resp.content))
        await _audit(helpers.command_audit_row(envelope))
        return {"result": envelope}

    async def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        if not config.qdrant_url:
            raise RuntimeError(
                "RAG question received but llm.qdrant_url is not configured."
            )
        embedding = await retrieval.embed_query(
            client,
            base_url=config.base_url,
            model=config.embedding_model,
            text=state["chat_input"],
            headers=embed_headers,
        )
        points = await retrieval.hybrid_query(
            client,
            qdrant_url=config.qdrant_url,
            collection=config.qdrant_collection,
            embedding=embedding,
            question=state["chat_input"],
            top_k=config.retrieval_top_k,
            embedding_vector_name=config.embedding_model,
            headers=qdrant_headers,
        )
        return {"chunks": helpers.map_qdrant_points(points)}

    async def select_node(state: dict[str, Any]) -> dict[str, Any]:
        chunks = state["chunks"]
        if not config.rerank or not chunks:
            return {"chunks": helpers.rrf_top3(chunks)}
        resp = await rerank_llm.ainvoke(
            [
                SystemMessage(helpers.RERANK_SYSTEM),
                HumanMessage(helpers.build_rerank_user(state["chat_input"], chunks)),
            ]
        )
        indices = helpers.parse_rerank_indices(str(resp.content), len(chunks))
        return {"chunks": helpers.apply_rerank(chunks, indices)}

    async def expand_node(state: dict[str, Any]) -> dict[str, Any]:
        winners = state["chunks"]
        if not config.expansion:
            return {"chunks": winners}
        groups = helpers.neighbour_ids(winners)
        points = await retrieval.scroll_neighbours(
            client,
            qdrant_url=config.qdrant_url,
            collection=config.qdrant_collection,
            groups=groups,
            headers=qdrant_headers,
        )
        return {"chunks": helpers.merge_neighbours(winners, points)}

    async def answer_node(state: dict[str, Any]) -> dict[str, Any]:
        chunks = state["chunks"]
        resp = await answer_llm.ainvoke(
            [
                SystemMessage(helpers.RAG_SYSTEM_PROMPT),
                HumanMessage(helpers.build_rag_user(state["chat_input"], chunks)),
            ]
        )
        parsed = helpers.parse_rag_answer(str(resp.content), chunks)
        output = helpers.format_question_output(parsed)
        await _audit(helpers.question_audit_row(parsed, output))
        return {"result": helpers.answer_envelope(output)}

    commands_only = getattr(config, "mode", "full") == "commands_only"

    graph = StateGraph(_TurnState)
    graph.add_node("command", command_node)
    if commands_only:
        # One LLM call per turn: no intent classifier, no RAG branch. The
        # command parser itself refuses questions as out-of-scope, and the
        # qdrant/embedding services are never contacted (issue #21).
        graph.add_edge(START, "command")
    else:
        graph.add_node("classify", classify_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("select", select_node)
        graph.add_node("expand", expand_node)
        graph.add_node("answer", answer_node)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: state["intent"],
            {"command": "command", "question": "retrieve"},
        )
        graph.add_edge("retrieve", "select")
        graph.add_edge("select", "expand")
        graph.add_edge("expand", "answer")
        graph.add_edge("answer", END)
    graph.add_edge("command", END)
    compiled = graph.compile()

    run_config: dict[str, Any] = (
        {"callbacks": [callback_handler]} if callback_handler else {}
    )

    async def runner(chat_input: str) -> dict[str, Any]:
        final = await compiled.ainvoke({"chat_input": chat_input}, config=run_config)
        return final["result"]

    _log.info(
        "langgraph_runner_built",
        model=config.model,
        mode="commands_only" if commands_only else "full",
        rerank=config.rerank,
        expansion=config.expansion,
        qdrant=bool(config.qdrant_url),
        traced=bool(callback_handler),
    )
    return runner
