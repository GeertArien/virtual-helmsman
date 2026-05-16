"""Smart Turn v3 turn-detection backend (default).

Wraps Pipecat's first-party ``LocalSmartTurnAnalyzerV3`` (a local ONNX semantic
end-of-turn model, shipped with Pipecat) in a ``TurnAnalyzerUserTurnStopStrategy``.
"""

from __future__ import annotations

from typing import Any

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_turn_strategies import TurnAnalyzerUserTurnStopStrategy


def build_turn(config: Any) -> TurnAnalyzerUserTurnStopStrategy:
    """Build the Smart Turn v3 stop strategy from the ``turn_detection`` block.

    The Smart Turn v3 model is a CPU ONNX model; ``config.device`` is accepted
    for schema symmetry but not needed here.
    """
    return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
