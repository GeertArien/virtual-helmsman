"""VAD-only turn detection: no semantic model (benchmarking baseline).

The user turn ends purely on a VAD silence timeout — Pipecat's
``SpeechTimeoutUserTurnStopStrategy``. This is the cheaper baseline to compare
against ``smart_turn_v3`` when measuring what the semantic model costs in
latency versus what it buys in fewer mid-sentence cut-offs.
"""

from __future__ import annotations

from typing import Any

from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)


def build_turn(config: Any) -> SpeechTimeoutUserTurnStopStrategy:
    """Build the VAD-silence-timeout stop strategy."""
    return SpeechTimeoutUserTurnStopStrategy()
