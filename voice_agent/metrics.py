"""Latency tracking and metrics output.

``LatencyTracker`` is a Pipecat ``FrameProcessor`` that stamps per-turn
timestamps; writers emit per-turn JSONL and an end-of-session summary.
"""

from __future__ import annotations


class LatencyTracker:
    """Pipecat FrameProcessor that stamps per-turn latency timestamps."""

    pass
