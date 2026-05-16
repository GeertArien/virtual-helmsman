"""Whisper STT backend (Pipecat first-party ``WhisperSTTService``)."""

from __future__ import annotations

from typing import Any

from pipecat.services.whisper.stt import WhisperSTTService

from voice_agent.backends.stt.base import to_language


def build_stt(config: Any) -> WhisperSTTService:
    """Build the Whisper STT service from the ``stt`` config block."""
    return WhisperSTTService(
        model=config.model,
        device=config.device,
        language=to_language(config.language),
    )
