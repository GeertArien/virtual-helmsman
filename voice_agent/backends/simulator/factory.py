"""Simulator factory: selects ``real`` or ``mock`` from config."""

from __future__ import annotations


def create_simulator(config):
    """Return a ``SimulatorClient`` for ``config.simulator.backend``."""
    raise NotImplementedError("simulator.factory.create_simulator is a scaffold stub")
