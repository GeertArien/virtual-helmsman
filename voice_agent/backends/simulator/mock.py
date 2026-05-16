"""Mock simulator backend: in-memory ``SimulatorClient`` (default for dev/tests).

Holds heading, engine order, and a derived speed. Records every command to
``command_history`` for test assertions.
"""

from __future__ import annotations


def build_simulator(config):
    """Build the mock ``SimulatorClient`` from the ``simulator.mock`` config."""
    raise NotImplementedError("mock.build_simulator is a scaffold stub")
