"""Turn-detection backend contract.

Each turn backend module exposes ``build_turn(turn_config)`` returning a
Pipecat ``BaseUserTurnStopStrategy`` — the strategy that decides when the user
has finished speaking. ``pipeline.py`` places it in
``UserTurnStrategies(stop=[...])`` on the user context aggregator.

Backends:

* ``smart_turn_v3`` — semantic end-of-turn model (default).
* ``vad_only`` — end the turn on a VAD silence timeout; benchmarking baseline.
"""

from __future__ import annotations
