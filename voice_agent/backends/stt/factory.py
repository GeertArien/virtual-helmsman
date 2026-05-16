"""STT factory: selects an STT backend from config."""

from __future__ import annotations


def create_stt(config):
    """Return an STTService for ``config.stt.backend``."""
    raise NotImplementedError("stt.factory.create_stt is a scaffold stub")
