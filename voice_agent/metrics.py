"""Latency tracking, metrics output, and the conversation log.

Two monitors, both attached to the pipeline:

* :class:`LatencyTracker` is a Pipecat *observer*. It watches frames as they
  pass between every pair of processors, stamps per-turn timestamps, writes one
  JSONL record per turn to ``logs/metrics/<session_id>.jsonl``, and on session
  end appends a p50/p95/p99 summary. It must be an observer, not a pipeline
  processor: a processor only sees frames that reach its position, but the
  ``TranscriptionFrame`` is consumed by the user aggregator and the real
  streaming ``LLMTextFrame`` s are consumed by ``JsonActionProcessor`` — an
  observer sees them all, at their source.
* :class:`ConversationLogger` is a Pipecat ``FrameProcessor`` that writes one
  JSONL object per conversation event (user transcript, assistant reply) to
  ``logs/conversations/<session_id>.jsonl``.

Latency math uses ``time.monotonic()`` (immune to wall-clock jumps); the wall
clock is recorded only as a human-readable ``ts`` field.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.api.events import EventBus, TurnMetricsEvent
from voice_agent.logging_setup import get_logger

# Derived metric -> (end timestamp field, start timestamp field) on TurnMetrics.
# Note v2v = stt_latency + llm_ttft + tts_ttfa (the three are a clean partition).
_DERIVED: dict[str, tuple[str, str]] = {
    "stt_latency_ms": ("stt_final_ts", "vad_speech_end_ts"),
    "llm_ttft_ms": ("llm_first_token_ts", "stt_final_ts"),
    "llm_total_ms": ("llm_last_token_ts", "llm_first_token_ts"),
    "tts_ttfa_ms": ("tts_first_audio_ts", "llm_first_token_ts"),
    "voice_to_voice_ms": ("tts_first_audio_ts", "vad_speech_end_ts"),
}


def _iso_now() -> str:
    """Wall-clock timestamp, ISO 8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


def percentiles(values: list[float]) -> dict[str, float]:
    """Return p50/p95/p99 (linear interpolation) and count for ``values``."""
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        k = (len(ordered) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(ordered) - 1)
        return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "count": len(ordered)}


@dataclass
class TurnMetrics:
    """Timestamps stamped during a single conversational turn (monotonic seconds)."""

    turn_index: int
    vad_speech_end_ts: float | None = None
    stt_first_partial_ts: float | None = None
    stt_final_ts: float | None = None
    llm_first_token_ts: float | None = None
    llm_last_token_ts: float | None = None
    tts_first_audio_ts: float | None = None
    tts_last_audio_ts: float | None = None

    def derived_ms(self) -> dict[str, float]:
        """Compute the derived ``*_ms`` metrics from stamped timestamps."""
        out: dict[str, float] = {}
        for name, (end_field, start_field) in _DERIVED.items():
            end = getattr(self, end_field)
            start = getattr(self, start_field)
            if end is not None and start is not None:
                out[name] = round((end - start) * 1000, 1)
        return out


class _JsonlWriter:
    """Append-only JSONL writer for one session file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


class LatencyTracker(BaseObserver):
    """Pipecat observer that stamps per-turn latency timestamps.

    A turn opens on ``VADUserStoppedSpeakingFrame`` (the user finished an
    utterance) and closes on ``TTSStoppedFrame``; the headline metric is
    ``voice_to_voice_ms``. Each timestamp is stamped on its *first* occurrence
    in the turn, so the real streaming LLM frames win over the rebuilt frames
    that ``JsonActionProcessor`` emits downstream.
    """

    def __init__(
        self,
        *,
        session_id: str,
        metrics_dir: Path,
        event_bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._log = get_logger("metrics")
        self._session_id = session_id
        self._writer = _JsonlWriter(Path(metrics_dir) / f"{session_id}.jsonl")
        self._turn_index = 0
        self._turn: TurnMetrics | None = None
        self._completed: list[TurnMetrics] = []
        self._summary_written = False
        self._event_bus = event_bus
        # A frame crosses many processor boundaries; on_push_frame fires once
        # per crossing. Dedupe so each frame is counted once (bounded history).
        self._seen_order: deque[int] = deque(maxlen=512)
        self._seen: set[int] = set()

    def _is_new(self, frame: Frame) -> bool:
        """True the first time this frame id is seen; remembers it (bounded)."""
        fid = frame.id
        if fid in self._seen:
            return False
        if len(self._seen_order) == self._seen_order.maxlen:
            self._seen.discard(self._seen_order[0])
        self._seen_order.append(fid)
        self._seen.add(fid)
        return True

    async def on_push_frame(self, data: FramePushed) -> None:
        """Stamp per-turn timestamps as frames flow through the pipeline."""
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame = data.frame
        if not self._is_new(frame):
            return
        now = time.monotonic()

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # A new utterance ended: open a fresh turn. Any prior turn that
            # never reached TTSStoppedFrame is dropped (incomplete).
            self._turn = TurnMetrics(turn_index=self._turn_index)
            self._turn.vad_speech_end_ts = now
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._finalize_session()
            return

        turn = self._turn
        if turn is None:
            return

        if isinstance(frame, InterimTranscriptionFrame):
            if turn.stt_first_partial_ts is None:
                turn.stt_first_partial_ts = now
        elif isinstance(frame, TranscriptionFrame):
            if turn.stt_final_ts is None:
                turn.stt_final_ts = now
        elif isinstance(frame, LLMTextFrame):
            if turn.llm_first_token_ts is None:
                turn.llm_first_token_ts = now
        elif isinstance(frame, LLMFullResponseEndFrame):
            if turn.llm_last_token_ts is None:
                turn.llm_last_token_ts = now
        elif isinstance(frame, TTSAudioRawFrame):
            if turn.tts_first_audio_ts is None:
                turn.tts_first_audio_ts = now
        elif isinstance(frame, TTSStoppedFrame):
            turn.tts_last_audio_ts = now
            self._finalize_turn()

    def _finalize_turn(self) -> None:
        if self._turn is None:
            return
        turn = self._turn
        self._completed.append(turn)
        self._turn = None
        self._turn_index += 1
        derived = turn.derived_ms()
        self._writer.write(
            {
                "ts": _iso_now(),
                "session_id": self._session_id,
                "turn_index": turn.turn_index,
                "metrics_ms": derived,
            }
        )
        self._log.info(
            "turn_metrics",
            turn_index=turn.turn_index,
            stt_latency_ms=derived.get("stt_latency_ms"),
            llm_ttft_ms=derived.get("llm_ttft_ms"),
            llm_total_ms=derived.get("llm_total_ms"),
            tts_ttfa_ms=derived.get("tts_ttfa_ms"),
            voice_to_voice_ms=derived.get("voice_to_voice_ms"),
        )
        if self._event_bus is not None:
            self._event_bus.publish(
                TurnMetricsEvent(turn_index=turn.turn_index, metrics_ms=derived)
            )

    def _finalize_session(self) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        by_metric: dict[str, list[float]] = {name: [] for name in _DERIVED}
        for turn in self._completed:
            for name, value in turn.derived_ms().items():
                by_metric[name].append(value)

        summary = {name: percentiles(values) for name, values in by_metric.items()}
        self._writer.write(
            {
                "ts": _iso_now(),
                "session_id": self._session_id,
                "type": "session_summary",
                "turns": len(self._completed),
                "summary": summary,
            }
        )
        self._log.info(
            "session_summary", turns=len(self._completed), summary=summary
        )


class ConversationLogger(FrameProcessor):
    """Pipecat FrameProcessor that writes the per-session conversation JSONL."""

    def __init__(
        self, *, session_id: str, conversation_dir: Path, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._writer = _JsonlWriter(Path(conversation_dir) / f"{session_id}.jsonl")
        self._assistant_parts: list[str] = []

    def _emit(self, record: dict[str, Any]) -> None:
        self._writer.write({"ts": _iso_now(), "session_id": self._session_id, **record})

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._emit({"role": "user", "transcript": frame.text})
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._assistant_parts = []
        elif isinstance(frame, LLMTextFrame):
            self._assistant_parts.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._assistant_parts).strip()
            self._assistant_parts = []
            if text:
                self._emit({"role": "assistant", "text": text})

        await self.push_frame(frame, direction)
