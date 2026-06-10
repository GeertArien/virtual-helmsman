"""Tests for the LLM backend factory.

Two backends remain after the n8n migration: ``openai_compatible`` and
``langgraph``. The langgraph adapter has its own dedicated suite
(``test_langgraph_llm.py``); here we cover the factory dispatch and the
shared ``LlmConfig`` model requirement.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from voice_agent.backends.llm.factory import create_llm


def test_factory_dispatches_to_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The openai_compatible builder is selected and called with the config.

    The real OpenAILLMService construction touches httpx/openai-sdk internals
    we don't want to spin up; stub the builder to a sentinel. The factory
    captures function references in ``_BUILDERS`` at import time, so we patch
    the dict entry directly.
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
        resolved_api_key=lambda: None,
    )
    assert create_llm(cfg) is sentinel


def test_factory_dispatches_to_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.backends.llm import factory as factory_mod

    sentinel = object()
    monkeypatch.setitem(factory_mod._BUILDERS, "langgraph", lambda config: sentinel)
    assert create_llm(SimpleNamespace(backend="langgraph")) is sentinel


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_llm(SimpleNamespace(backend="n8n"))  # removed backend
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_llm(SimpleNamespace(backend="anthropic_haiku"))


def test_llm_config_requires_model() -> None:
    from pydantic import ValidationError

    from voice_agent.config import LlmConfig

    with pytest.raises(ValidationError):
        LlmConfig(backend="openai_compatible", base_url="http://localhost:1234/v1")  # type: ignore[call-arg]


def test_llm_config_accepts_known_model() -> None:
    from voice_agent.config import LlmConfig

    cfg = LlmConfig(
        backend="openai_compatible",
        base_url="http://localhost:1234/v1",
        model="unsloth/gemma-4-e4b-it",
    )
    assert cfg.model == "unsloth/gemma-4-e4b-it"
    assert cfg.rerank is True
    assert cfg.expansion is True


def test_llm_config_rejects_unknown_backend() -> None:
    from pydantic import ValidationError

    from voice_agent.config import LlmConfig

    with pytest.raises(ValidationError):
        LlmConfig(backend="n8n", base_url="http://x", model="unsloth/gemma-4-e4b-it")  # type: ignore[arg-type]
