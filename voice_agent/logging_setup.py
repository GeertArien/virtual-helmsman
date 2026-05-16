"""Structured logging setup (JSON or console formatter).

Every log line carries timestamp, level, component, session_id, message.
"""

from __future__ import annotations


def configure_logging(config) -> None:
    """Configure structlog/stdlib logging from the ``logging`` config block."""
    raise NotImplementedError(
        "voice_agent.logging_setup.configure_logging is a scaffold stub"
    )
