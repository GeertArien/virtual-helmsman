"""Turn-detection factory: selects a turn backend from config."""

from __future__ import annotations


def create_turn(config):
    """Return a turn analyzer for ``config.turn_detection.backend``."""
    raise NotImplementedError("turn.factory.create_turn is a scaffold stub")
