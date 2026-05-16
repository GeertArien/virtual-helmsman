"""Structured logging setup.

Configures ``structlog`` so every log line carries: ``timestamp`` (ISO 8601,
UTC), ``level``, ``message``, ``session_id``, plus any structured fields. Each
component binds its own ``component`` field (``stt``, ``tts``, ``llm``, ``vad``,
``turn``, ``tools``, ``simulator``, ``pipeline``) via :func:`get_logger`.

The ``session_id`` is bound once into structlog's contextvars and then appears
on every line for the lifetime of the process run.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import structlog


def new_session_id() -> str:
    """Return a fresh UUID4 session id for this process run."""
    return str(uuid.uuid4())


def get_logger(component: str) -> Any:
    """Return a structlog logger bound to ``component``."""
    return structlog.get_logger().bind(component=component)


def configure_logging(config: Any, session_id: str) -> None:
    """Configure structlog from the ``logging`` config block.

    ``config`` is a ``LoggingConfig`` (``level``, ``format``). ``session_id`` is
    bound into contextvars so it tags every subsequent log line.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.EventRenamer("message"),
    ]
    if config.format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(event_key="message")

    structlog.configure(
        processors=shared + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(session_id=session_id)
