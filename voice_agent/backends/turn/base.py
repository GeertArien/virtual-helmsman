"""Turn-detection backend contract.

Each turn backend exposes a builder returning a Pipecat turn analyzer.
Backends: ``smart_turn_v3`` (default), ``vad_only`` (benchmarking fallback).
"""

from __future__ import annotations
