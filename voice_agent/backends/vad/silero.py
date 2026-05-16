"""Silero VAD backend (Pipecat first-party ``SileroVADAnalyzer``)."""

from __future__ import annotations

from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


def build_vad(config: Any) -> SileroVADAnalyzer:
    """Build the Silero VAD analyzer from the ``vad`` config block.

    The config ``threshold`` maps to ``VADParams.confidence`` (minimum voice
    detection confidence). Timing fields keep Pipecat defaults.
    """
    return SileroVADAnalyzer(params=VADParams(confidence=config.threshold))
