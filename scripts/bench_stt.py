"""Benchmark the configured STT backend on sample input; report latency.

Builds the STT backend selected in ``config.yaml``, runs it on a synthetic
audio buffer for N iterations, and prints p50/p95/p99 transcription latency.

    python scripts/bench_stt.py [--config config.yaml] [--runs 20] [--seconds 3]

Requires the configured STT model/weights to be available (downloads on first
run) and, for the CUDA backends, an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from voice_agent.backends.stt.factory import create_stt
from voice_agent.config import load_config
from voice_agent.logging_setup import configure_logging, new_session_id
from voice_agent.metrics import percentiles


def _sample_audio(seconds: float, sample_rate: int) -> bytes:
    """Synthetic 16-bit mono PCM: low-amplitude noise (avoids pure silence)."""
    count = int(seconds * sample_rate)
    rng = np.random.default_rng(seed=0)
    samples = (rng.standard_normal(count) * 600).astype(np.int16)
    return samples.tobytes()


async def _bench(config_path: str, runs: int, seconds: float) -> None:
    config = load_config(config_path)
    configure_logging(config.logging, new_session_id())

    service = create_stt(config.stt)
    audio = _sample_audio(seconds, config.audio.sample_rate)

    async def one_run() -> float:
        start = time.monotonic()
        async for _frame in service.run_stt(audio):
            pass
        return (time.monotonic() - start) * 1000.0

    await one_run()  # warmup (model load / CUDA graph capture)

    timings = [await one_run() for _ in range(runs)]
    stats = percentiles(timings)
    print(f"STT backend  : {config.stt.backend} ({config.stt.model})")
    print(f"Input        : {seconds:.1f}s @ {config.audio.sample_rate} Hz, {runs} runs")
    print(
        f"Latency (ms) : p50={stats['p50']}  p95={stats['p95']}  p99={stats['p99']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench_stt", description=__doc__)
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    asyncio.run(_bench(args.config, args.runs, args.seconds))


if __name__ == "__main__":
    main()
