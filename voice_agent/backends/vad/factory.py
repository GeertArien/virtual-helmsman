"""VAD factory: selects a VAD backend from config."""

from __future__ import annotations


def create_vad(config):
    """Return a VAD analyzer for ``config.vad.backend``."""
    raise NotImplementedError("vad.factory.create_vad is a scaffold stub")
