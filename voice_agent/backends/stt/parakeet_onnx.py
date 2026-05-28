"""Parakeet-TDT STT via ONNX Runtime CUDA EP (default STT backend).

Wraps an ``onnx-asr`` Parakeet model in a custom Pipecat
``SegmentedSTTService``. Pipecat ships no in-process Parakeet service (its
first-party Parakeet support targets the NVIDIA Riva server), so this backend
is hand-wrapped per the project brief.

``SegmentedSTTService`` is used (not the plain continuous ``STTService``) so
Parakeet runs once per *utterance*: the base class buffers audio, and on the
``VADUserStoppedSpeakingFrame`` (pushed upstream by the user aggregator's VAD
controller) it hands :meth:`run_stt` one complete WAV-format segment. A
continuous ``STTService`` would instead transcribe every ~20 ms audio chunk,
including silence — which yields a stream of garbage transcripts.
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from voice_agent.backends.stt.base import to_language
from voice_agent.logging_setup import get_logger


def _providers(device: str) -> list[str]:
    """ONNX Runtime execution providers for ``device`` (pure-CUDA client)."""
    if device.lower().startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ParakeetOnnxSTTService(SegmentedSTTService):
    """Parakeet-TDT speech-to-text via ``onnx-asr`` on ONNX Runtime."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        language: str = "en",
        quantization: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Provide a complete settings store so the base class does not warn
        # about NOT_GIVEN model/language fields.
        super().__init__(
            settings=STTSettings(model=model, language=to_language(language)),
            **kwargs,
        )
        self._log = get_logger("stt")
        self._language = to_language(language)
        self._model_name = model
        self._asr = self._load_model(model, device, quantization)

    def _load_model(
        self,
        model: str,
        device: str,
        quantization: str | None,
    ) -> Any:
        # Imported lazily so onnx-asr need not be importable to load this module.
        import onnx_asr

        self._log.info(
            "stt_model_loading",
            backend="parakeet_onnx",
            model=model,
            device=device,
            quantization=quantization,
        )
        # ``quantization`` is forwarded to onnx-asr's loader, which resolves
        # the matching variant (e.g. ``encoder-model.int8.onnx``) from the
        # Hugging Face repo. ``None`` keeps the FP32 default.
        load_kwargs: dict[str, Any] = {"providers": _providers(device)}
        if quantization is not None:
            load_kwargs["quantization"] = quantization
        asr = onnx_asr.load_model(model, **load_kwargs)
        self._log.info(
            "stt_model_loaded",
            backend="parakeet_onnx",
            model=model,
            quantization=quantization,
        )
        return asr

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # SegmentedSTTService hands us one complete utterance as a WAV-format
        # buffer (header + 16-bit mono PCM at the transport sample rate).
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                pcm = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError) as exc:
            self._log.error("stt_decode_failed", backend="parakeet_onnx", error=str(exc))
            yield ErrorFrame(f"Parakeet-ONNX STT failed to decode audio: {exc}")
            return

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return

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
        # SttConfig defines this for parakeet_onnx; the other STT backends
        # ignore the field (they don't pull it from config here).
        quantization=getattr(config, "quantization", None),
    )
