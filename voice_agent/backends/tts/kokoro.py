"""Kokoro-82M TTS backend (default; Pipecat first-party ``KokoroTTSService``).

Kokoro runs locally via ONNX Runtime; model files auto-download to
``~/.cache/kokoro-onnx/`` on first use.
"""

from __future__ import annotations

import os
from typing import Any

from pipecat.services.kokoro.tts import KokoroTTSService

from voice_agent.logging_setup import get_logger


def build_tts(config: Any) -> KokoroTTSService:
    """Build the Kokoro TTS service from the ``tts`` config block.

    ``KokoroTTSService`` exposes no execution-provider hook, and ``kokoro-onnx``
    only enables the GPU when ``importlib.util.find_spec("onnxruntime-gpu")``
    succeeds — but ``onnxruntime-gpu`` is a PyPI *package* name; the import name
    is ``onnxruntime``, so that probe always fails and Kokoro silently runs on
    CPU. ``kokoro-onnx`` also honours the ``ONNX_PROVIDER`` env var, so set it
    explicitly when the config asks for CUDA. It is read once, when the first
    ``Kokoro`` session is constructed inside ``KokoroTTSService.__init__``.
    """
    if str(config.device).lower().startswith("cuda"):
        os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
        get_logger("tts").info(
            "tts_provider_forced", backend="kokoro", provider="CUDAExecutionProvider"
        )
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=config.voice),
    )
