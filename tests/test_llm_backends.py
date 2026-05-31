"""Tests for the LLM backend factory and the n8n adapter.

The factory is a simple dict dispatch -- we cover the dispatch plus the
``ValueError`` on an unknown backend. The n8n adapter is exercised by
stubbing :class:`httpx.AsyncClient` and running a fake
``LLMContextFrame`` through ``process_frame``, then asserting on the
``LLMTextFrame`` it emits (which downstream
:class:`JsonActionProcessor` would parse).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

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
    NavigationAction,
    RudderAction,
    parse_response,
)
from voice_agent.backends.llm import n8n
from voice_agent.backends.llm.factory import create_llm
from voice_agent.backends.llm.n8n import (
    N8nLLMService,
    _error_envelope,
    _latest_user_text,
    _translate_envelope,
)


# ---------- factory --------------------------------------------------------


def test_factory_dispatches_to_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The openai_compatible builder is selected and called with the config.

    The real OpenAILLMService construction touches httpx/openai-sdk
    internals we don't want to spin up; stub the builder to a sentinel.
    The factory captures function references in ``_BUILDERS`` at import
    time, so we patch the dict entry directly rather than the source
    attribute (the latter would not be seen by the existing dict).
    """
    from voice_agent.backends.llm import factory as factory_mod

    sentinel = object()

    def fake_build(config: Any) -> Any:
        assert config.backend == "openai_compatible"
        assert config.model == "nvidia/nemotron-3-nano-4b"
        return sentinel

    monkeypatch.setitem(factory_mod._BUILDERS, "openai_compatible", fake_build)

    cfg = SimpleNamespace(
        backend="openai_compatible",
        base_url="http://x:1234/v1",
        model="nvidia/nemotron-3-nano-4b",
        timeout_seconds=30.0,
        api_key_env="LLM_API_KEY",
        max_retries=1,
        webhook_path="/webhook/helmsman",
        rerank=True,
        resolved_api_key=lambda: None,
    )
    result = create_llm(cfg)
    assert result is sentinel


def test_factory_dispatches_to_n8n() -> None:
    """The n8n builder returns a real N8nLLMService instance."""
    cfg = SimpleNamespace(
        backend="n8n",
        base_url="http://localhost:5678",
        model="unsloth/gemma-4-e4b-it",
        webhook_path="/webhook/helmsman",
        rerank=True,
        timeout_seconds=60.0,
        resolved_n8n_headers=lambda: {},
    )
    result = create_llm(cfg)
    assert isinstance(result, N8nLLMService)


def test_factory_rejects_unknown_backend() -> None:
    cfg = SimpleNamespace(backend="anthropic_haiku")
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_llm(cfg)


# ---------- _latest_user_text ---------------------------------------------


def test_latest_user_text_returns_most_recent_user_message() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first command"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second command"},
    ]
    assert _latest_user_text(msgs) == "second command"


def test_latest_user_text_skips_non_string_content() -> None:
    msgs = [
        {"role": "user", "content": ["multi", "modal"]},  # ignored
        {"role": "user", "content": "text only"},
    ]
    assert _latest_user_text(msgs) == "text only"


def test_latest_user_text_returns_none_when_empty() -> None:
    assert _latest_user_text([]) is None
    assert _latest_user_text([{"role": "system", "content": "sys"}]) is None


# ---------- _translate_envelope -------------------------------------------


def test_translate_command_intent_passes_action_through() -> None:
    body = {
        "intent": "command",
        "output": "Starboard twenty, aye.",
        "action": {"type": "rudder", "direction": "starboard", "degrees": 20},
        "source": None,
    }
    internal = json.loads(_translate_envelope(body))
    assert internal == {
        "action": {"type": "rudder", "direction": "starboard", "degrees": 20},
        "response": "Starboard twenty, aye.",
    }
    # And the result is parseable as a HelmsmanResponse.
    parsed = parse_response(_translate_envelope(body))
    assert isinstance(parsed.action, RudderAction)


def test_translate_question_intent_synthesises_answer_action() -> None:
    body = {
        "intent": "question",
        "output": "Rule 15 requires the give-way vessel to keep out of the way.\n\nSource: COLREGS.pdf, page 14",
        "action": None,
        "source": {"chunk_id": "chunk_026", "filename": "COLREGS.pdf"},
    }
    internal = json.loads(_translate_envelope(body))
    assert internal["action"] == {"type": "answer"}
    assert "Rule 15" in internal["response"]
    # Parses cleanly as an AnswerAction.
    parsed = parse_response(_translate_envelope(body))
    assert isinstance(parsed.action, AnswerAction)


def test_translate_missing_intent_falls_back_to_answer() -> None:
    """Defensive: an n8n response without ``intent`` shouldn't crash --
    map it to an answer with whatever ``output`` exists."""
    body = {"output": "Standing by, sir."}
    parsed = parse_response(_translate_envelope(body))
    assert isinstance(parsed.action, AnswerAction)
    assert parsed.response == "Standing by, sir."


def test_translate_error_intent_passes_error_action_through() -> None:
    """Iteration 12/13 of API.md: when an LLM call inside the workflow
    fails, n8n returns intent='error' with a structured error action.
    We pass the action through verbatim so JsonActionProcessor publishes
    an ActionRefusedEvent rather than swallowing the failure as a Q&A
    answer. The extra ``http_status`` field in n8n's payload is dropped
    by Pydantic (ErrorAction doesn't ``extra='forbid'``)."""
    from voice_agent.actions.schema import ErrorAction

    body = {
        "intent": "error",
        "output": "LLM call failed: context length exceeded (HTTP 400)",
        "action": {
            "type": "error",
            "error_type": "llm_call_failure",
            "reason": "context length exceeded",
            "http_status": 400,
        },
        "source": None,
        "raw_model_output": "",
    }
    parsed = parse_response(_translate_envelope(body))
    assert isinstance(parsed.action, ErrorAction)
    assert parsed.action.error_type == "llm_call_failure"
    assert parsed.action.reason == "context length exceeded"
    # The spoken text is n8n's `output` -- TTS will say this aloud.
    assert "context length exceeded" in parsed.response
    # `http_status` is silently dropped (not on our ErrorAction schema).
    assert not hasattr(parsed.action, "http_status")


# ---------- N8nLLMService.process_frame -----------------------------------


@dataclass
class _FakeResponse:
    status_code: int
    body: Any
    text_body: str = ""

    def json(self) -> Any:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body

    @property
    def text(self) -> str:
        return self.text_body


class _StubClient:
    """httpx.AsyncClient stand-in. Pops scripted responses in FIFO order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list[Any] = []
        self.closed = False

    def queue(self, *responses: Any) -> None:
        self.script.extend(responses)

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, "kwargs": kwargs})
        if not self.script:
            raise AssertionError(f"Unexpected POST to {url}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


async def _drive(
    proc: N8nLLMService, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
) -> list[Frame]:
    """Run one frame through ``proc.process_frame`` and capture pushed frames."""
    pushed: list[Frame] = []

    async def fake_push(
        f: Frame, d: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        pushed.append(f)

    proc.push_frame = fake_push  # type: ignore[method-assign]
    await proc.process_frame(frame, direction)
    return pushed


def _make_n8n_service(
    stub: _StubClient, model: str = "unsloth/gemma-4-e4b-it"
) -> N8nLLMService:
    proc = N8nLLMService(
        base_url="http://n8n:5678",
        model=model,
        webhook_path="/webhook/helmsman",
        rerank=True,
        timeout_seconds=10.0,
    )
    # Swap in the stub client after construction so we don't open a real
    # httpx connection during tests.
    proc._client = stub  # type: ignore[assignment]
    return proc


def _context_with_user(text: str) -> LLMContextFrame:
    ctx = LLMContext(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": text},
        ]
    )
    return LLMContextFrame(context=ctx)


async def test_n8n_command_emits_start_text_end_triple() -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "intent": "command",
                "output": "Starboard twenty, aye.",
                "action": {
                    "type": "rudder",
                    "direction": "starboard",
                    "degrees": 20,
                },
                "source": None,
                "raw_model_output": "...",
            },
        )
    )
    proc = _make_n8n_service(stub)

    pushed = await _drive(proc, _context_with_user("come to starboard twenty"))

    # Exact triple: Start, one TextFrame, End.
    assert len(pushed) == 3
    assert isinstance(pushed[0], LLMFullResponseStartFrame)
    assert isinstance(pushed[1], LLMTextFrame)
    assert isinstance(pushed[2], LLMFullResponseEndFrame)
    # Text frame is a parseable rudder HelmsmanResponse.
    parsed = parse_response(pushed[1].text)
    assert isinstance(parsed.action, RudderAction)
    assert parsed.response == "Starboard twenty, aye."
    # n8n got the right shape -- chatInput + rerank + model per API.md.
    assert stub.calls[0]["url"] == "http://n8n:5678/webhook/helmsman"
    assert stub.calls[0]["kwargs"]["json"] == {
        "chatInput": "come to starboard twenty",
        "rerank": True,
        "model": "unsloth/gemma-4-e4b-it",
    }


async def test_n8n_uses_configured_model() -> None:
    """The model identifier flows from config through to the POST body."""
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "intent": "command",
                "output": "Aye.",
                "action": {"type": "status_query", "query": "heading"},
            },
        )
    )
    proc = _make_n8n_service(stub, model="nvidia/nemotron-3-nano-4b")
    await _drive(proc, _context_with_user("what's our heading"))
    assert stub.calls[0]["kwargs"]["json"]["model"] == "nvidia/nemotron-3-nano-4b"


async def test_n8n_question_emits_answer_action() -> None:
    stub = _StubClient()
    stub.queue(
        _FakeResponse(
            200,
            {
                "intent": "question",
                "output": "Rule 15 governs crossing situations.\n\nSource: COLREGS.pdf, page 14",
                "action": None,
                "source": {"chunk_id": "chunk_026"},
            },
        )
    )
    proc = _make_n8n_service(stub)

    pushed = await _drive(proc, _context_with_user("what does rule 15 say"))
    parsed = parse_response(
        next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    )
    assert isinstance(parsed.action, AnswerAction)
    assert "Rule 15" in parsed.response


async def test_n8n_non_context_frames_pass_through() -> None:
    """Frames other than LLMContextFrame must not trigger an n8n call."""
    stub = _StubClient()  # empty script -- a call would fail
    proc = _make_n8n_service(stub)
    pushed = await _drive(proc, TextFrame("hi"))
    assert len(pushed) == 1
    assert isinstance(pushed[0], TextFrame)
    assert stub.calls == []


async def test_n8n_http_error_emits_bridge_lost() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(500, {"error": "x"}, text_body="server died"))
    proc = _make_n8n_service(stub)
    pushed = await _drive(proc, _context_with_user("come to 270"))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    assert BRIDGE_LOST in text  # canonical pipeline error response
    # The error envelope is parseable as a HelmsmanResponse -- never let
    # JsonActionProcessor see a malformed JSON.
    parsed = parse_response(text)
    assert parsed.action.type == "error"


async def test_n8n_network_error_emits_bridge_lost() -> None:
    import httpx as _httpx

    stub = _StubClient()
    stub.queue(_httpx.ConnectError("refused"))
    proc = _make_n8n_service(stub)
    pushed = await _drive(proc, _context_with_user("come to 270"))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    parsed = parse_response(text)
    assert parsed.action.type == "error"
    assert "n8n unreachable" in parsed.action.reason  # type: ignore[union-attr]


async def test_n8n_non_json_body_emits_error() -> None:
    stub = _StubClient()
    stub.queue(_FakeResponse(200, ValueError("not json"), text_body="<html>"))
    proc = _make_n8n_service(stub)
    pushed = await _drive(proc, _context_with_user("come to 270"))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    parsed = parse_response(text)
    assert parsed.action.type == "error"


async def test_n8n_empty_context_emits_standing_by() -> None:
    """A context without any user message shouldn't even reach n8n."""
    stub = _StubClient()  # empty script
    proc = _make_n8n_service(stub)
    ctx = LLMContext([{"role": "system", "content": "sys"}])
    pushed = await _drive(proc, LLMContextFrame(context=ctx))
    text = next(f for f in pushed if isinstance(f, LLMTextFrame)).text
    parsed = parse_response(text)
    assert parsed.action.type == "error"
    assert "Standing by" in parsed.response
    assert stub.calls == []


async def test_n8n_cleanup_closes_client() -> None:
    stub = _StubClient()
    proc = _make_n8n_service(stub)
    await proc.cleanup()
    assert stub.closed is True


# ---------- _error_envelope (the safety net) -------------------------------


def test_error_envelope_is_parseable_helmsman_response() -> None:
    """Every failure mode must emit a JSON the downstream parser accepts."""
    payload = _error_envelope("simulated failure")
    parsed = parse_response(payload)
    assert parsed.action.type == "error"
    assert parsed.action.error_type == "bridge_error"  # type: ignore[union-attr]
    assert parsed.response == BRIDGE_LOST


# ---------- N8nLlmConfig validation ---------------------------------------


def test_llm_config_requires_model_for_both_backends() -> None:
    """``model`` is forwarded to n8n's POST body too, so it's required
    regardless of backend (n8n's API.md applies it to every LLM call)."""
    from pydantic import ValidationError

    from voice_agent.config import LlmConfig

    with pytest.raises(ValidationError):
        LlmConfig(
            backend="openai_compatible", base_url="http://localhost:1234/v1"
        )  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        LlmConfig(backend="n8n", base_url="http://localhost:5678")  # type: ignore[call-arg]


def test_llm_config_n8n_accepts_same_model_enum() -> None:
    """Both supported model strings work for both backends."""
    from voice_agent.config import LlmConfig

    cfg = LlmConfig(
        backend="n8n",
        base_url="http://localhost:5678",
        model="unsloth/gemma-4-e4b-it",
    )
    assert cfg.model == "unsloth/gemma-4-e4b-it"
    assert cfg.webhook_path == "/webhook/helmsman"
    assert cfg.rerank is True


def test_llm_config_rejects_unknown_backend() -> None:
    from pydantic import ValidationError

    from voice_agent.config import LlmConfig

    with pytest.raises(ValidationError):
        LlmConfig(backend="anthropic", base_url="http://x", model="unsloth/gemma-4-e4b-it")  # type: ignore[arg-type]
