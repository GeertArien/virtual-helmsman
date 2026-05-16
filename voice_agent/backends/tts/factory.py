"""TTS factory: selects a TTS backend from config.

To add a backend: create a module under ``backends/tts/`` exposing
``build_tts(tts_config)`` and add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.services.tts_service import TTSService

from voice_agent.backends.tts import kokoro, piper

_BUILDERS: dict[str, Callable[[Any], TTSService]] = {
    "kokoro": kokoro.build_tts,
    "piper": piper.build_tts,
}


def create_tts(config: Any) -> TTSService:
    """Return a ``TTSService`` for ``config.backend`` (the ``tts`` block)."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown TTS backend: {config.backend!r}. Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
