"""STT backend contract.

Each STT backend module exposes ``build_stt(stt_config)`` returning a Pipecat
``STTService`` (or subclass). Backends: ``parakeet_onnx`` (default),
``parakeet_nemo``, ``whisper``.

Pipecat's first-party "Parakeet" support targets the NVIDIA Riva *server*, not
an in-process model, so ``parakeet_onnx`` and ``parakeet_nemo`` are hand-wrapped
as custom ``STTService`` subclasses. ``whisper`` uses Pipecat's first-party
``WhisperSTTService`` directly.
"""

from __future__ import annotations

from pipecat.transcriptions.language import Language


def to_language(code: str) -> Language | None:
    """Best-effort map an ISO language code (e.g. ``en``) to a Pipecat ``Language``."""
    try:
        return Language(code)
    except ValueError:
        return None
