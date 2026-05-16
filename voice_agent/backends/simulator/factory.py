"""Simulator factory: selects ``real`` or ``mock`` from config.

To add a backend: create a module under ``backends/simulator/`` exposing a
``build_simulator(sub_config)`` function and add one entry to ``_BUILDERS``.
No other code changes.
"""

from __future__ import annotations

from typing import Any, Callable

from voice_agent.backends.simulator import mock, real
from voice_agent.backends.simulator.base import SimulatorClient

# backend name -> (builder, selector picking that backend's sub-config block).
_BUILDERS: dict[str, tuple[Callable[[Any], SimulatorClient], Callable[[Any], Any]]] = {
    "mock": (mock.build_simulator, lambda cfg: cfg.mock),
    "real": (real.build_simulator, lambda cfg: cfg.real),
}


def create_simulator(config: Any) -> SimulatorClient:
    """Return a ``SimulatorClient`` for ``config.backend`` (the ``simulator`` block)."""
    try:
        build, select_sub_config = _BUILDERS[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown simulator backend: {config.backend!r}. "
            f"Valid: {sorted(_BUILDERS)}"
        ) from None
    return build(select_sub_config(config))
