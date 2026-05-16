"""VAD factory: selects a VAD backend from config.

To add a backend: create a module under ``backends/vad/`` exposing
``build_vad(vad_config)`` and add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.audio.vad.vad_analyzer import VADAnalyzer

from voice_agent.backends.vad import silero

_BUILDERS: dict[str, Callable[[Any], VADAnalyzer]] = {
    "silero": silero.build_vad,
}


def create_vad(config: Any) -> VADAnalyzer:
    """Return a VAD analyzer for ``config.backend`` (the ``vad`` block)."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown VAD backend: {config.backend!r}. Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
