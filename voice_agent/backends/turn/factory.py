"""Turn-detection factory: selects a turn backend from config.

Returns a Pipecat turn analyzer, or ``None`` for the ``vad_only`` backend
(no semantic model — Pipecat falls back to VAD silence timing).

To add a backend: create a module under ``backends/turn/`` exposing
``build_turn(turn_config)`` and add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from voice_agent.backends.turn import smart_turn_v3, vad_only

_BUILDERS: dict[str, Callable[[Any], Any]] = {
    "smart_turn_v3": smart_turn_v3.build_turn,
    "vad_only": vad_only.build_turn,
}


def create_turn(config: Any) -> Any:
    """Return a turn analyzer (or ``None``) for ``config.backend``."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown turn backend: {config.backend!r}. Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
