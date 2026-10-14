"""Optional Langfuse tracing, shared infrastructure for both process halves.

Used by the LangGraph helmsman backend (voice side) and the ingestion
doc-summary call (knowledge-base side) -- which is why it lives at the top
level rather than inside either package (issue #12 §6).

Langfuse is wired as a LangChain callback handler, passed into every graph
invocation so each LLM / retrieval step is traced (prompt, completion, token
usage, latency). It is entirely optional: when disabled, or when the SDK or
its credentials are missing, :func:`build_callback_handler` returns ``None``
and the graph runs untraced. Tracing must never take the pipeline down.

The Langfuse SDK moved the LangChain handler's import path between v2
(``langfuse.callback``) and v3 (``langfuse.langchain``); both are tried.
Credentials are read from the environment (the keys named in config), matching
the project's "secrets by env-var name" convention.
"""

from __future__ import annotations

import os
from typing import Any

from voice_agent.logging_setup import get_logger

_log = get_logger("llm.langgraph.tracing")


def _import_callback_handler() -> type | None:
    """Return the Langfuse LangChain ``CallbackHandler`` class, or ``None``."""
    try:  # Langfuse v3
        from langfuse.langchain import CallbackHandler  # type: ignore

        return CallbackHandler
    except Exception:  # noqa: BLE001 -- fall through to the v2 path
        pass
    try:  # Langfuse v2
        from langfuse.callback import CallbackHandler  # type: ignore

        return CallbackHandler
    except Exception:  # noqa: BLE001
        return None


def build_callback_handler(config: Any) -> Any | None:
    """Construct a Langfuse callback handler from the ``llm`` config, or ``None``.

    Returns ``None`` (untraced) when tracing is disabled, the SDK is absent, the
    public/secret keys are unset, or construction fails for any reason -- all
    logged at info/warning, never raised.
    """
    if not getattr(config, "langfuse_enabled", False):
        return None

    public_key = os.environ.get(config.langfuse_public_key_env)
    secret_key = os.environ.get(config.langfuse_secret_key_env)
    if not public_key or not secret_key:
        _log.warning(
            "langfuse_keys_missing",
            public_key_env=config.langfuse_public_key_env,
            secret_key_env=config.langfuse_secret_key_env,
        )
        return None

    handler_cls = _import_callback_handler()
    if handler_cls is None:
        _log.warning("langfuse_sdk_missing")
        return None

    # The SDK reads keys/host from the environment; set them so both v2 and v3
    # handler signatures pick them up without per-version constructor kwargs.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    if config.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", config.langfuse_host)

    try:
        handler = handler_cls()
    except Exception as exc:  # noqa: BLE001 -- never let tracing break the run
        _log.warning("langfuse_handler_init_failed", error=str(exc))
        return None

    _log.info("langfuse_enabled", host=config.langfuse_host or "default")
    return handler
