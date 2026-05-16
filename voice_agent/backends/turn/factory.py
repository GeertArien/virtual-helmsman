"""Turn-detection factory: selects a turn backend from config.

Returns a Pipecat ``BaseUserTurnStopStrategy`` for ``pipeline.py`` to place in
``UserTurnStrategies(stop=[...])`` on the user context aggregator.

To add a backend: create a module under ``backends/turn/`` exposing
``build_turn(turn_config)`` and add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from pipecat.turns.user_stop import BaseUserTurnStopStrategy

from voice_agent.backends.turn import smart_turn_v3, vad_only

_BUILDERS: dict[str, Callable[[Any], BaseUserTurnStopStrategy]] = {
    "smart_turn_v3": smart_turn_v3.build_turn,
    "vad_only": vad_only.build_turn,
}


def create_turn(config: Any) -> BaseUserTurnStopStrategy:
    """Return a user-turn stop strategy for ``config.backend``."""
    try:
        build = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown turn backend: {config.backend!r}. Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(config)
