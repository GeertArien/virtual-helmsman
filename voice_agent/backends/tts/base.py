"""TTS backend contract.

Each TTS backend module exposes a builder returning a Pipecat ``TTSService``
(or subclass). Backends: ``kokoro``, ``piper``.
"""

from __future__ import annotations
