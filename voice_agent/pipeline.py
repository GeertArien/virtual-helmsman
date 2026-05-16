"""Builds the Pipecat pipeline from a validated config object.

Wires: transport -> VAD/turn -> STT -> context aggregator -> LLM (+tools)
-> TTS -> transport, plus the LatencyTracker processor.
"""

from __future__ import annotations


def build_pipeline(config):
    """Construct the Pipecat pipeline and supporting objects from config."""
    raise NotImplementedError("voice_agent.pipeline.build_pipeline is a scaffold stub")
