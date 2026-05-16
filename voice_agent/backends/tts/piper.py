"""Piper TTS backend (Pipecat first-party local ``PiperTTSService``).

Local Piper inference (not the HTTP ``PiperHttpTTSService``). The voice model
auto-downloads on first use.
"""

from __future__ import annotations

from typing import Any

from pipecat.services.piper.tts import PiperTTSService


def build_tts(config: Any) -> PiperTTSService:
    """Build the Piper TTS service from the ``tts`` config block."""
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.voice),
        use_cuda=config.device.lower().startswith("cuda"),
    )
