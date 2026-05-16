"""Kokoro-82M TTS backend (default; Pipecat first-party ``KokoroTTSService``).

Kokoro runs locally via ONNX Runtime; model files auto-download to
``~/.cache/kokoro-onnx/`` on first use.
"""

from __future__ import annotations

from typing import Any

from pipecat.services.kokoro.tts import KokoroTTSService


def build_tts(config: Any) -> KokoroTTSService:
    """Build the Kokoro TTS service from the ``tts`` config block."""
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=config.voice),
    )
