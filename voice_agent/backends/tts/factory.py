"""TTS factory: selects a TTS backend from config."""

from __future__ import annotations


def create_tts(config):
    """Return a TTSService for ``config.tts.backend``."""
    raise NotImplementedError("tts.factory.create_tts is a scaffold stub")
