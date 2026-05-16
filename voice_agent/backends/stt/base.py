"""STT backend contract.

Each STT backend module exposes a builder returning a Pipecat ``STTService``
(or subclass). Backends: ``parakeet_onnx``, ``parakeet_nemo``, ``whisper``.
"""

from __future__ import annotations
