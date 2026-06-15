"""Tests for the in-backend ``langgraph`` LLM backend.

Covers the three testable layers without standing up LangGraph/LangChain,
Qdrant, or LM Studio:

* the pure :mod:`helpers` ports of the n8n Code nodes (intent parsing, RRF
  top-3, rerank-index parsing, adjacent-chunk id math + merge, RAG-answer
  parsing + citation, reply shaping);
* the :mod:`retrieval` httpx calls, driven through a stub client that asserts
  the Qdrant/embedding request shapes; and
* :class:`LangGraphLLMService` driven with a fake runner -- the same
  Start/Text/End frame-triple contract the n8n adapter is held to, plus the
  failure -> ``error`` envelope guarantee.

The heavy graph wiring in :mod:`graph` is integration code (it needs the
optional ``langgraph`` extra + live services) and is not unit-tested here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from voice_agent.actions.dispatch import BRIDGE_LOST
from voice_agent.actions.schema import (
    AnswerAction,
    ErrorAction,
    RudderAction,
    parse_response,
)
from voice_agent.backends.llm.factory import create_llm
from voice_agent.backends.llm.langgraph_helmsman import helpers, retrieval
from voice_agent.backends.llm.langgraph_helmsman.service import (
    LangGraphLLMService,
    build_llm,
)


# ---------- intent + envelope shaping -------------------------------------


def test_parse_intent_question_and_default() -> None:
    assert helpers.parse_intent("QUESTION") == "question"
    assert helpers.parse_intent("  question  ") == "question"
    assert helpers.parse_intent("COMMAND") == "command"
    # Garbled/empty completions default to the safe command branch.
    assert helpers.parse_intent("") == "command"
    assert helpers.parse_intent("???") == "command"


def test_command_envelope_passes_valid_action_through() -> None:
    raw = json.dumps(
        {
            "action": {"type": "rudder", "direction": "starboard", "degrees": 20},
            "response": "Starboard twenty, aye.",
        }
    )
    env = helpers.command_envelope(raw)
    parsed = parse_response(json.dumps(env))
    assert isinstance(parsed.action, RudderAction)
    assert parsed.response == "Starboard twenty, aye."


def test_command_envelope_strips_code_fence() -> None:
    raw = '```json\n{"action": {"type": "status_query", "query": "heading"}, "response": "Aye."}\n```'
    env = helpers.command_envelope(raw)
    assert env["action"]["type"] == "status_query"
    assert env["response"] == "Aye."


def test_command_envelope_parse_failure_is_error_action() -> None:
    env = helpers.command_envelope("not json at all")
    parsed = parse_response(json.dumps(env))
    assert isinstance(parsed.action, ErrorAction)
    assert parsed.action.error_type == "parse_failure"


def test_answer_envelope_is_answer_action() -> None:
    env = helpers.answer_envelope("Rule 15 ...\n\nSource: COLREGS.pdf, page 14 (chunk_026)")
    parsed = parse_response(json.dumps(env))
    assert isinstance(parsed.action, AnswerAction)
    assert "Rule 15" in parsed.response


def test_error_envelope_is_parseable() -> None:
    parsed = parse_response(json.dumps(helpers.error_envelope("boom")))
    assert isinstance(parsed.action, ErrorAction)
    assert parsed.action.error_type == "bridge_error"
    assert parsed.response == BRIDGE_LOST


# ---------- latest_user_text ----------------------------------------------


def test_latest_user_text() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    assert helpers.latest_user_text(msgs) == "second"
    assert helpers.latest_user_text([{"role": "user", "content": ["x"]}]) is None
    assert helpers.latest_user_text([]) is None


# ---------- retrieval shaping helpers -------------------------------------


def _point(chunk_id: str, filename: str = "A.pdf", page: int = 1, score: float = 0.9) -> dict:
    return {
        "score": score,
        "payload": {
            "text": f"text {chunk_id}",
            "filename": filename,
            "page": page,
            "chunk_id": chunk_id,
            "document_type": "PDF",
            "document_summary": "summary",
        },
    }


def test_map_qdrant_points_assigns_rank() -> None:
    chunks = helpers.map_qdrant_points([_point("chunk_000"), _point("chunk_001")])
    assert [c["rank"] for c in chunks] == [1, 2]
    assert chunks[0]["text"] == "text chunk_000"
    assert chunks[0]["chunk_id"] == "chunk_000"


def test_rrf_top3_takes_three_and_flags_bypass() -> None:
    chunks = helpers.map_qdrant_points([_point(f"chunk_{i:03d}") for i in range(5)])
    top = helpers.rrf_top3(chunks)
    assert len(top) == 3
    assert all(c["rerank_bypassed"] for c in top)
    assert top[0]["original_rank"] == 1


def test_parse_rerank_indices_one_based_to_zero_based() -> None:
    assert helpers.parse_rerank_indices('{"top_3": [2, 1, 3]}', 5) == [1, 0, 2]


def test_parse_rerank_indices_drops_out_of_range_and_falls_back() -> None:
    # All out of range -> fallback to RRF top-3 positions.
    assert helpers.parse_rerank_indices('{"top_3": [99, 100]}', 4) == [0, 1, 2]
    # Garbage -> fallback.
    assert helpers.parse_rerank_indices("nonsense", 4) == [0, 1, 2]
    # Fallback respects a short candidate list.
    assert helpers.parse_rerank_indices("nonsense", 2) == [0, 1]


def test_apply_rerank_projects_selected_chunks() -> None:
    chunks = helpers.map_qdrant_points([_point(f"chunk_{i:03d}") for i in range(4)])
    top = helpers.apply_rerank(chunks, [2, 0])
    assert [c["chunk_id"] for c in top] == ["chunk_002", "chunk_000"]
    assert [c["rerank_rank"] for c in top] == [1, 2]


def test_neighbour_ids_pm1_and_chunk_zero_floor() -> None:
    winners = [
        {"chunk_id": "chunk_026", "filename": "A.pdf"},
        {"chunk_id": "chunk_000", "filename": "A.pdf"},
        {"chunk_id": "chunk_005", "filename": "B.pdf"},
        {"chunk_id": "not_a_chunk", "filename": "A.pdf"},  # ignored
        {"chunk_id": "chunk_010"},  # no filename -> ignored
    ]
    groups = helpers.neighbour_ids(winners)
    assert set(groups["A.pdf"]) == {"chunk_025", "chunk_027", "chunk_001"}
    assert "chunk_000" not in groups["A.pdf"]  # chunk_000 has no lower neighbour
    assert set(groups["B.pdf"]) == {"chunk_004", "chunk_006"}


def test_merge_neighbours_dedups_and_skips_sentinel() -> None:
    winners = helpers.map_qdrant_points([_point("chunk_001")])
    neighbour_points = [
        _point("chunk_000"),  # genuine neighbour
        _point("chunk_001"),  # duplicate of winner -> skipped
        {"payload": {"chunk_id": "chunk_x", "filename": "__no_winners__"}},  # sentinel
        {"payload": {"filename": "A.pdf"}},  # no chunk_id -> skipped
    ]
    merged = helpers.merge_neighbours(winners, neighbour_points)
    ids = [c["chunk_id"] for c in merged]
    assert ids == ["chunk_001", "chunk_000"]
    assert merged[1]["is_neighbour"] is True


def test_parse_rag_answer_reliable_citation() -> None:
    chunks = helpers.map_qdrant_points([_point("chunk_026", page=14)])
    raw = json.dumps({"answer": "Give way.", "source_chunk_id": "chunk_026"})
    parsed = helpers.parse_rag_answer(raw, chunks)
    assert parsed["answer"] == "Give way."
    assert parsed["citation_reliable"] is True
    assert parsed["citation"] == "A.pdf, page 14 (chunk_026)"
    out = helpers.format_question_output(parsed)
    assert out.endswith("Source: A.pdf, page 14 (chunk_026)")
    assert out.startswith("Give way.")


def test_parse_rag_answer_unreliable_when_cited_chunk_absent() -> None:
    chunks = helpers.map_qdrant_points([_point("chunk_000")])
    raw = json.dumps({"answer": "x", "source_chunk_id": "chunk_999"})
    parsed = helpers.parse_rag_answer(raw, chunks)
    # Falls back to the first chunk; flagged unreliable.
    assert parsed["citation_reliable"] is False
    assert parsed["source_chunk"]["chunk_id"] == "chunk_000"


def test_parse_rag_answer_parse_failure_uses_raw_text() -> None:
    chunks = helpers.map_qdrant_points([_point("chunk_000")])
    parsed = helpers.parse_rag_answer("totally not json", chunks)
    assert parsed["parse_failure"] is True
    assert parsed["answer"] == "totally not json"
    assert parsed["citation_reliable"] is False


# ---------- retrieval httpx calls -----------------------------------------


@dataclass
class _FakeResponse:
    status_code: int = 200
    body: Any = None

    def json(self) -> Any:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


@dataclass
class _StubClient:
    script: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def queue(self, *responses: Any) -> None:
        self.script.extend(responses)

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, "kwargs": kwargs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def test_embed_query_returns_vector_and_posts_model() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
    vec = await retrieval.embed_query(
        stub, base_url="http://lm:1234/v1", model="text-embedding-bge-m3", text="rule 15"
    )
    assert vec == [0.1, 0.2, 0.3]
    assert stub.calls[0]["url"] == "http://lm:1234/v1/embeddings"
    assert stub.calls[0]["kwargs"]["json"] == {
        "input": "rule 15",
        "model": "text-embedding-bge-m3",
    }


async def test_hybrid_query_builds_rrf_prefetch_body() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, {"result": {"points": [_point("chunk_000")]}}))
    points = await retrieval.hybrid_query(
        stub,
        qdrant_url="http://qd:6333",
        collection="maritime_hybrid",
        embedding=[0.1, 0.2],
        question="what does rule 15 say",
        top_k=20,
    )
    assert len(points) == 1
    body = stub.calls[0]["kwargs"]["json"]
    assert stub.calls[0]["url"] == "http://qd:6333/collections/maritime_hybrid/points/query"
    assert body["query"] == {"fusion": "rrf"}
    assert body["limit"] == 20
    assert body["prefetch"][0]["using"] == "text-embedding-bge-m3"
    assert body["prefetch"][0]["limit"] == 40  # top_k * 2
    assert body["prefetch"][1]["query"] == {
        "text": "what does rule 15 say",
        "model": "qdrant/bm25",
    }


async def test_scroll_neighbours_skips_failed_groups() -> None:
    stub = _StubClient()
    # First group ok, second group errors -> skipped, not fatal.
    stub.queue(
        _FakeResponse(200, {"result": {"points": [_point("chunk_000")]}}),
        _FakeResponse(500, None),
    )
    points = await retrieval.scroll_neighbours(
        stub,
        qdrant_url="http://qd:6333",
        collection="maritime_hybrid",
        groups={"A.pdf": ["chunk_000"], "B.pdf": ["chunk_004"]},
    )
    assert len(points) == 1
    assert points[0]["payload"]["chunk_id"] == "chunk_000"


# ---------- LangGraphLLMService -------------------------------------------


async def _drive(proc: LangGraphLLMService, frame: Frame) -> list[Frame]:
    pushed: list[Frame] = []

    async def fake_push(f: Frame, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(f)

    proc.push_frame = fake_push  # type: ignore[method-assign]
    await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
    return pushed


def _context_with_user(text: str) -> LLMContextFrame:
    ctx = LLMContext([{"role": "system", "content": "sys"}, {"role": "user", "content": text}])
    return LLMContextFrame(context=ctx)


async def test_service_command_emits_start_text_end_triple() -> None:
    async def runner(chat_input: str) -> dict:
        assert chat_input == "come to starboard twenty"
        return {
            "action": {"type": "rudder", "direction": "starboard", "degrees": 20},
            "response": "Starboard twenty, aye.",
        }

    proc = LangGraphLLMService(runner=runner)
    pushed = await _drive(proc, _context_with_user("come to starboard twenty"))

    assert len(pushed) == 3
    assert isinstance(pushed[0], LLMFullResponseStartFrame)
    assert isinstance(pushed[1], LLMTextFrame)
    assert isinstance(pushed[2], LLMFullResponseEndFrame)
    parsed = parse_response(pushed[1].text)
    assert isinstance(parsed.action, RudderAction)
    assert parsed.response == "Starboard twenty, aye."


async def test_service_question_emits_answer_action() -> None:
    async def runner(chat_input: str) -> dict:
        return helpers.answer_envelope("Rule 15 governs crossing.\n\nSource: COLREGS.pdf, page 14")

    proc = LangGraphLLMService(runner=runner)
    pushed = await _drive(proc, _context_with_user("what does rule 15 say"))
    parsed = parse_response(next(f for f in pushed if isinstance(f, LLMTextFrame)).text)
    assert isinstance(parsed.action, AnswerAction)
    assert "Rule 15" in parsed.response


async def test_service_runner_exception_emits_bridge_lost() -> None:
    async def runner(chat_input: str) -> dict:
        raise RuntimeError("qdrant down")

    proc = LangGraphLLMService(runner=runner)
    pushed = await _drive(proc, _context_with_user("come to 270"))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    parsed = parse_response(text)
    assert parsed.action.type == "error"
    assert parsed.response == BRIDGE_LOST
    assert "qdrant down" in parsed.action.reason  # type: ignore[union-attr]


async def test_service_empty_context_does_not_call_runner() -> None:
    called = False

    async def runner(chat_input: str) -> dict:
        nonlocal called
        called = True
        return {}

    proc = LangGraphLLMService(runner=runner)
    ctx = LLMContext([{"role": "system", "content": "sys"}])
    pushed = await _drive(proc, LLMContextFrame(context=ctx))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    parsed = parse_response(text)
    assert parsed.action.type == "error"
    assert "Standing by" in parsed.response
    assert called is False


async def test_service_non_context_frames_pass_through() -> None:
    async def runner(chat_input: str) -> dict:
        raise AssertionError("runner should not be called")

    proc = LangGraphLLMService(runner=runner)
    pushed = await _drive(proc, TextFrame("hi"))
    assert len(pushed) == 1
    assert isinstance(pushed[0], TextFrame)


async def test_service_cleanup_closes_client() -> None:
    class _Closeable:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = _Closeable()

    async def runner(chat_input: str) -> dict:
        return {}

    proc = LangGraphLLMService(runner=runner, client=client)  # type: ignore[arg-type]
    await proc.cleanup()
    assert client.closed is True


# ---------- factory + config ----------------------------------------------


def test_factory_dispatches_to_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    """The langgraph builder is selected without constructing the real graph."""
    from voice_agent.backends.llm import factory as factory_mod

    sentinel = object()

    def fake_build(config: Any) -> Any:
        assert config.backend == "langgraph"
        return sentinel

    monkeypatch.setitem(factory_mod._BUILDERS, "langgraph", fake_build)
    cfg = SimpleNamespace(backend="langgraph")
    assert create_llm(cfg) is sentinel


def test_build_llm_constructs_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_llm wires a runner from the graph and returns a service.

    The heavy graph build is stubbed so this runs without the langgraph extra.
    """
    from voice_agent.backends.llm.langgraph_helmsman import graph as graph_mod

    async def fake_runner(chat_input: str) -> dict:
        return {}

    monkeypatch.setattr(
        graph_mod,
        "build_runner",
        lambda config, client, audit_writer=None: fake_runner,
    )
    cfg = SimpleNamespace(timeout_seconds=30.0, audit_enabled=False)
    svc = build_llm(cfg)
    assert isinstance(svc, LangGraphLLMService)
    # A real httpx client was created and is wired for cleanup.
    assert svc._client is not None
    assert svc._audit_writer is None  # audit disabled


def test_llm_runtime_langgraph_fields_and_qdrant_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.config import parse_config

    rt = parse_config(
        {
            "stt": {"model": "m"},
            "tts": {"voice": "v"},
            "llm": {"backend": "langgraph", "model": "unsloth/gemma-4-e4b-it"},
            "lm_studio": {"base_url": "http://localhost:1234/v1"},
            "qdrant": {"url": "http://localhost:6333"},
        }
    ).llm_runtime()
    assert rt.backend == "langgraph"
    assert rt.qdrant_collection == "maritime_hybrid"
    assert rt.embedding_model == "text-embedding-bge-m3"
    assert rt.retrieval_top_k == 20
    assert rt.langfuse_enabled is False
    # No key set -> no header.
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    assert rt.resolved_qdrant_headers() == {}
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    assert rt.resolved_qdrant_headers() == {"api-key": "secret"}


# ---------- runtime audit rows ------------------------------------------------


def test_command_audit_row() -> None:
    envelope = {
        "action": {"type": "rudder", "direction": "starboard", "degrees": 20},
        "response": "Starboard twenty, aye.",
    }
    doc, actie, resultaat = helpers.command_audit_row(envelope)
    assert doc == "n.v.t. (command)"
    assert actie == "command_runtime"
    assert resultaat == "action_type=rudder | output=Starboard twenty, aye."


def test_question_audit_row() -> None:
    parsed = {
        "source_chunk": {"filename": "COLREGS.pdf"},
        "source_chunk_id": "chunk_026",
        "citation_reliable": True,
        "parse_failure": False,
    }
    doc, actie, resultaat = helpers.question_audit_row(parsed, "Give way.\n\nSource: …")
    assert doc == "COLREGS.pdf"
    assert actie == "question_runtime"
    assert "chunk=chunk_026" in resultaat
    assert "citation_reliable=true" in resultaat
    assert "parse_failure=false" in resultaat


def test_question_audit_row_no_source_uses_nvt() -> None:
    doc, _, _ = helpers.question_audit_row(
        {"source_chunk": None, "source_chunk_id": None}, "out"
    )
    assert doc == "n.v.t."


def test_error_audit_row() -> None:
    doc, actie, resultaat = helpers.error_audit_row("n8n unreachable: boom")
    assert actie == "llm_error_runtime"
    assert resultaat.startswith("error=n8n unreachable")


async def test_service_failure_writes_error_audit_row() -> None:
    rows: list[tuple[str, str, str]] = []

    async def writer(doc: str, actie: str, resultaat: str) -> None:
        rows.append((doc, actie, resultaat))

    async def runner(chat_input: str) -> dict:
        raise RuntimeError("qdrant down")

    proc = LangGraphLLMService(runner=runner, audit_writer=writer)
    await _drive(proc, _context_with_user("come to 270"))
    assert len(rows) == 1
    assert rows[0][1] == "llm_error_runtime"
    assert "qdrant down" in rows[0][2]


async def test_service_audit_write_failure_does_not_break_turn() -> None:
    async def writer(doc: str, actie: str, resultaat: str) -> None:
        raise RuntimeError("audit db locked")

    async def runner(chat_input: str) -> dict:
        raise RuntimeError("boom")

    proc = LangGraphLLMService(runner=runner, audit_writer=writer)
    # Even though both the turn and the audit write fail, a parseable error
    # envelope still reaches the pipeline.
    pushed = await _drive(proc, _context_with_user("come to 270"))
    parsed = parse_response(next(f for f in pushed if isinstance(f, LLMTextFrame)).text)
    assert parsed.action.type == "error"


def test_build_llm_wires_audit_writer(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """audit_enabled -> a writer backed by the shared store is created and
    passed to both the graph runner and the service."""
    from voice_agent.backends.llm.langgraph_helmsman import graph as graph_mod

    captured: dict[str, Any] = {}

    async def fake_runner(chat_input: str) -> dict:
        return {}

    def fake_build_runner(config: Any, client: Any, audit_writer: Any = None) -> Any:
        captured["audit_writer"] = audit_writer
        return fake_runner

    monkeypatch.setattr(graph_mod, "build_runner", fake_build_runner)
    cfg = SimpleNamespace(
        timeout_seconds=30.0,
        audit_enabled=True,
        audit_db_path=str(tmp_path / "audit.db"),
    )
    svc = build_llm(cfg)
    assert svc._audit_writer is not None
    assert captured["audit_writer"] is svc._audit_writer

    # The writer actually inserts into the shared audit store.
    import asyncio

    from voice_agent.ingestion.store import IngestionStore

    asyncio.run(svc._audit_writer("n.v.t. (command)", "command_runtime", "ok"))
    out = IngestionStore(cfg.audit_db_path).query_audit(actie="command_runtime")
    assert out["total_returned"] == 1


# ---------- compiled graph (requires the langgraph extra) -------------------


@pytest.mark.asyncio
async def test_build_runner_carries_chat_input_through_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the graph was compiled with ``StateGraph(dict)``, which has
    no annotated state channels -- LangGraph silently dropped the
    ``chat_input`` passed to ``ainvoke`` and the first node raised
    ``KeyError: 'chat_input'`` on every turn. Compile the real graph (with
    fake chat models) and drive a command turn end to end through it."""
    pytest.importorskip("langgraph")
    import langchain_openai

    from voice_agent.backends.llm.langgraph_helmsman import graph as graph_mod
    from voice_agent.config import parse_config

    seen: list[str] = []

    class _FakeResp:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            # The classify model is the only one built with max_tokens=8.
            self._is_classify = kwargs.get("max_tokens") == 8

        async def ainvoke(self, messages: Any) -> _FakeResp:
            seen.append(messages[-1].content)
            if self._is_classify:
                return _FakeResp("COMMAND")
            return _FakeResp(
                json.dumps(
                    {
                        "action": {
                            "type": "rudder",
                            "direction": "starboard",
                            "degrees": 20,
                        },
                        "response": "Starboard twenty, aye.",
                    }
                )
            )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChat)
    cfg = parse_config(
        {
            "stt": {"model": "m"},
            "tts": {"voice": "v"},
            "llm": {"backend": "langgraph", "model": "unsloth/gemma-4-e4b-it"},
            "lm_studio": {"base_url": "http://localhost:1234/v1"},
        }
    ).llm_runtime()
    async with httpx.AsyncClient() as client:
        runner = graph_mod.build_runner(cfg, client)
        result = await runner("turn starboard twenty degrees")

    # classify and command must both have received the user's text.
    assert seen == ["turn starboard twenty degrees"] * 2
    assert result["action"]["type"] == "rudder"
    assert result["response"] == "Starboard twenty, aye."
