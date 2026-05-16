"""Real simulator backend: adapter around the vendored pythonnet wrappers.

The in-house wrapper ``.py`` files and the managed .NET DLL are dropped by the
user into ``voice_agent/backends/simulator/vendor/``. This adapter dispatches
``SimulatorClient`` protocol calls to the wrapper. If the wrapper API is
synchronous and blocks, calls must be offloaded via ``asyncio.to_thread()``.

Windows-only: the .NET DLL is loaded via ``pythonnet``.
"""

from __future__ import annotations


def build_simulator(config):
    """Build the real ``SimulatorClient`` from the ``simulator.real`` config."""
    raise NotImplementedError("real.build_simulator is a scaffold stub")
