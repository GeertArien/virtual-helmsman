"""STT factory: selects an STT backend from config.

To add a backend: create a module under ``backends/stt/`` exposing
``build_stt(stt_config)`` and add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.services.stt_service import STTService

from voice_agent.backends.stt import parakeet_nemo, parakeet_onnx, whisper

_BUILDERS: dict[str, Callable[[Any], STTService]] = {
    "parakeet_onnx": parakeet_onnx.build_stt,
    "parakeet_nemo": parakeet_nemo.build_stt,
    "whisper": whisper.build_stt,
}


def create_stt(config: Any) -> STTService:
    """Return an ``STTService`` for ``config.backend`` (the ``stt`` block)."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown STT backend: {config.backend!r}. Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
