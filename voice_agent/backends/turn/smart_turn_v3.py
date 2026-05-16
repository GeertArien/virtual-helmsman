"""Smart Turn v3 turn-detection backend (default).

Pipecat first-party ``LocalSmartTurnAnalyzerV3``: a local ONNX semantic
end-of-turn model. The model auto-downloads on first use.
"""

from __future__ import annotations

from typing import Any

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3


def build_turn(config: Any) -> LocalSmartTurnAnalyzerV3:
    """Build the Smart Turn v3 analyzer from the ``turn_detection`` config block."""
    return LocalSmartTurnAnalyzerV3()
