"""Benchmark the configured TTS backend on sample input; report latency.

Builds the TTS backend selected in ``config.yaml``, synthesizes a fixed phrase
for N iterations, and prints p50/p95/p99 for time-to-first-audio and total
synthesis time.

    python scripts/bench_tts.py [--config config.yaml] [--runs 20]

Requires the configured TTS model/voice to be available (downloads on first
run).
"""

from __future__ import annotations

import argparse
import asyncio
import time

from pipecat.frames.frames import TTSAudioRawFrame

from voice_agent.backends.tts.factory import create_tts
from voice_agent.config import load_config
from voice_agent.logging_setup import configure_logging, new_session_id
from voice_agent.metrics import percentiles

_PHRASE = "Coming to heading two seven zero. Engines half ahead, aye."


async def _bench(config_path: str, runs: int) -> None:
    config = load_config(config_path)
    configure_logging(config.logging, new_session_id())

    service = create_tts(config.tts)

    async def one_run() -> tuple[float, float]:
        start = time.monotonic()
        ttfa: float | None = None
        async for frame in service.run_tts(_PHRASE):
            if ttfa is None and isinstance(frame, TTSAudioRawFrame):
                ttfa = (time.monotonic() - start) * 1000.0
        total = (time.monotonic() - start) * 1000.0
        return (ttfa if ttfa is not None else total), total

    await one_run()  # warmup (model load)

    ttfas: list[float] = []
    totals: list[float] = []
    for _ in range(runs):
        ttfa, total = await one_run()
        ttfas.append(ttfa)
        totals.append(total)

    ttfa_stats = percentiles(ttfas)
    total_stats = percentiles(totals)
    print(f"TTS backend       : {config.tts.backend} (voice {config.tts.voice})")
    print(f"Phrase            : {_PHRASE!r}")
    print(f"Runs              : {runs}")
    print(
        f"Time-to-first (ms): p50={ttfa_stats['p50']}  "
        f"p95={ttfa_stats['p95']}  p99={ttfa_stats['p99']}"
    )
    print(
        f"Total (ms)        : p50={total_stats['p50']}  "
        f"p95={total_stats['p95']}  p99={total_stats['p99']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench_tts", description=__doc__)
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_bench(args.config, args.runs))


if __name__ == "__main__":
    main()
