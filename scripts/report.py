"""Summarize a metrics JSONL file: p50/p95/p99 per metric, as a table.

Usage:
    python scripts/report.py logs/metrics/<session_id>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voice_agent.metrics import percentiles

# Headline latency target from the brief (voice-to-voice, p95).
_V2V_TARGET_MS = 800.0


def _load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="report", description="Summarize a metrics JSONL file."
    )
    parser.add_argument("metrics_file", help="Path to logs/metrics/<session_id>.jsonl")
    args = parser.parse_args()

    path = Path(args.metrics_file)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        raise SystemExit(1)

    records = _load(path)
    per_turn = [r for r in records if "metrics_ms" in r]

    # Recompute from per-turn records so the report works even if the session
    # ended without writing its summary line.
    by_metric: dict[str, list[float]] = {}
    tool_latencies: list[float] = []
    for rec in per_turn:
        for name, value in rec["metrics_ms"].items():
            by_metric.setdefault(name, []).append(value)
        for call in rec.get("tool_calls", []):
            tool_latencies.append(call["tool_latency_ms"])
    if tool_latencies:
        by_metric["tool_latency_ms"] = tool_latencies

    print(f"Metrics file : {path}")
    print(f"Turns        : {len(per_turn)}")
    print()
    print(f"{'metric':<22}{'p50':>10}{'p95':>10}{'p99':>10}{'count':>8}")
    print("-" * 60)
    for name in sorted(by_metric):
        stats = percentiles(by_metric[name])
        if not stats:
            continue
        print(
            f"{name:<22}{stats['p50']:>10.1f}{stats['p95']:>10.1f}"
            f"{stats['p99']:>10.1f}{stats['count']:>8}"
        )

    v2v = percentiles(by_metric.get("voice_to_voice_ms", []))
    if v2v:
        verdict = "PASS" if v2v["p95"] <= _V2V_TARGET_MS else "OVER TARGET"
        print()
        print(
            f"voice-to-voice p95 = {v2v['p95']:.1f} ms "
            f"(target <= {_V2V_TARGET_MS:.0f} ms) -> {verdict}"
        )


if __name__ == "__main__":
    main()
