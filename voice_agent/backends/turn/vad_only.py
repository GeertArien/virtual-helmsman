"""VAD-only turn detection: no semantic model (benchmarking fallback).

There is no analyzer object — when no turn analyzer is attached to the
transport, Pipecat falls back to ending the turn on VAD silence timing
(``VADParams.stop_secs``). The builder therefore returns ``None``, and
``pipeline.py`` simply omits the ``turn_analyzer`` argument in that case.
"""

from __future__ import annotations

from typing import Any


def build_turn(config: Any) -> None:
    """Return ``None``: VAD-only turn-taking needs no analyzer object."""
    return None
