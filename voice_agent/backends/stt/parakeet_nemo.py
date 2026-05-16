"""Parakeet-TDT STT via NVIDIA NeMo (alternative STT backend).

Optional: requires the ``nemo`` extra (``pip install 'virtual-helmsman[nemo]'``).
NeMo's ``transcribe`` call is synchronous and GPU-bound, so it is offloaded with
``asyncio.to_thread`` to keep the Pipecat event loop responsive.

The NeMo transcription API varies across NeMo versions; ``_transcribe_sync``
normalises the common return shapes (plain strings vs. ``Hypothesis`` objects).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from voice_agent.backends.stt.base import to_language
from voice_agent.logging_setup import get_logger


class ParakeetNemoSTTService(STTService):
    """Parakeet-TDT speech-to-text via the NVIDIA NeMo toolkit."""

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
        self._asr = self._load_model(model, device)

    def _load_model(self, model: str, device: str) -> Any:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "The parakeet_nemo STT backend requires the optional 'nemo' "
                "extra: pip install 'virtual-helmsman[nemo]'."
            ) from exc

        self._log.info(
            "stt_model_loading", backend="parakeet_nemo", model=model, device=device
        )
        asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
        if device.lower().startswith("cuda"):
            asr = asr.cuda()
        asr.eval()
        self._log.info("stt_model_loaded", backend="parakeet_nemo", model=model)
        return asr

    def _transcribe_sync(self, samples: np.ndarray) -> str:
        results = self._asr.transcribe([samples], batch_size=1, verbose=False)
        first: Any = results[0] if results else ""
        # Some NeMo versions nest results by output type (e.g. [hyps, ...]).
        if isinstance(first, (list, tuple)):
            first = first[0] if first else ""
        return (getattr(first, "text", first) or "").strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            text = await asyncio.to_thread(self._transcribe_sync, samples)
        except Exception as exc:
            self._log.error("stt_failed", backend="parakeet_nemo", error=str(exc))
            yield ErrorFrame(f"Parakeet-NeMo STT failed: {exc}")
            return
        if text:
            self._log.info("stt_transcript", backend="parakeet_nemo", text=text)
            yield TranscriptionFrame(text, "", time_now_iso8601(), self._language)


def build_stt(config: Any) -> ParakeetNemoSTTService:
    """Build the Parakeet-NeMo STT service from the ``stt`` config block."""
    return ParakeetNemoSTTService(
        model=config.model,
        device=config.device,
        language=config.language,
    )
