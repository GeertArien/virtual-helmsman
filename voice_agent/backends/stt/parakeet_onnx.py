"""Parakeet-TDT STT via ONNX Runtime CUDA EP (default STT backend).

Wraps an ``onnx-asr`` Parakeet model in a custom Pipecat ``STTService``. Pipecat
ships no in-process Parakeet service (its first-party Parakeet support targets
the NVIDIA Riva server), so this backend is hand-wrapped per the project brief.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from voice_agent.backends.stt.base import to_language
from voice_agent.logging_setup import get_logger


def _providers(device: str) -> list[str]:
    """ONNX Runtime execution providers for ``device`` (pure-CUDA client)."""
    if device.lower().startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ParakeetOnnxSTTService(STTService):
    """Parakeet-TDT speech-to-text via ``onnx-asr`` on ONNX Runtime."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        language: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._log = get_logger("stt")
        self._language = to_language(language)
        self._model_name = model
        self._asr = self._load_model(model, device)

    def _load_model(self, model: str, device: str) -> Any:
        # Imported lazily so onnx-asr need not be importable to load this module.
        import onnx_asr

        self._log.info(
            "stt_model_loading", backend="parakeet_onnx", model=model, device=device
        )
        asr = onnx_asr.load_model(model, providers=_providers(device))
        self._log.info("stt_model_loaded", backend="parakeet_onnx", model=model)
        return asr

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Incoming audio is 16-bit mono PCM at the transport sample rate (16 kHz).
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            text = self._asr.recognize(samples)
        except Exception as exc:
            self._log.error("stt_failed", backend="parakeet_onnx", error=str(exc))
            yield ErrorFrame(f"Parakeet-ONNX STT failed: {exc}")
            return
        text = (text or "").strip()
        if text:
            self._log.info("stt_transcript", backend="parakeet_onnx", text=text)
            yield TranscriptionFrame(text, "", time_now_iso8601(), self._language)


def build_stt(config: Any) -> ParakeetOnnxSTTService:
    """Build the Parakeet-ONNX STT service from the ``stt`` config block."""
    return ParakeetOnnxSTTService(
        model=config.model,
        device=config.device,
        language=config.language,
    )
