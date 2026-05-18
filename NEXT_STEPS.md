# Next steps

> **Session name: `virtual-helmsman-build`**
> Resume with `claude --resume` in `D:\dev\virtual-helmsman` (pick the session
> for this work) or `claude -c` to continue the most recent. This file is the
> entry point — read it first.

Status as of 2026-05-17: the agent has been brought up on the target NVIDIA
client and reworked from LLM tool calls to JSON structured output. Phase 1 (a
live spoken run) is **done** — the full mic→speakers pipeline works against the
mock simulator. Latency (Phase 2) is the main thing left.

## Verified on the client

- **Environment** — `.venv` on Python 3.13.13; `pip install -e ".[dev,cuda]"`.
  59 unit tests pass, `ruff` clean.
- **CUDA** — `onnxruntime-gpu` 1.26 loads the CUDA execution provider. The CUDA
  12.x runtime + cuDNN 9 ship as venv-local `nvidia-*-cu12` wheels (the `cuda`
  extra); `voice_agent/_cuda.py` puts them on the DLL search path at import.
- **Pipeline build** — `build_pipeline()` loads Parakeet-0.6b STT, Kokoro TTS,
  Silero VAD, and smart-turn v3 onto CUDA; models download on first run.
- **LLM** — local LM Studio serving NVIDIA Nemotron 3 Nano 4B at
  `http://localhost:1234/v1`. Reasoning/thinking is disabled in the LM Studio
  per-model settings (no API toggle worked; the setting persists across loads).
- **JSON action path** — `scripts/smoke.py` passes end to end
  (LLM → JSON action → mock simulator). A 49/49 hit-rate benchmark of the
  prompt + `response_format` schema constraint was clean across all action
  types, English and Dutch.
- **Live spoken run** — verified 2026-05-17 against the mock simulator: 10
  spoken commands → 10 correct actions, no phantom turns. Three live-run bugs
  were found and fixed (see Phase 1 below).

## Not yet verified

- **All latency numbers.** Measured `voice_to_voice_ms` is ~2000 ms (range
  1500–2986) — well over the 800 ms budget. No per-component p50/p95 yet. The
  local LLM with partial RAM offload is the prime suspect.

---

## Phase 1 — Live spoken run — DONE (2026-05-17)

`python -m voice_agent.main --config config.yaml` runs end to end, mic →
speakers, against the mock simulator. Three bugs were found and fixed during
the first live runs:

1. **STT was not VAD-segmented.** `ParakeetOnnxSTTService` extended the
   continuous `STTService`, which transcribes every ~20 ms audio chunk — so it
   hallucinated garbage (a stream of `"M"`) on idle audio. It now extends
   `SegmentedSTTService`; `run_stt` decodes the WAV-format segment the base
   class hands it once per utterance.
2. **VAD `stop_secs` was 0.2 s** (Pipecat default) — short enough to split a
   command at the natural pauses between words. It is now a `vad.stop_secs`
   config field, default 0.8.
3. **The agent transcribed its own TTS.** The local mic/speaker setup has no
   echo cancellation, so spoken replies came back in as phantom commands. Fixed
   with `AlwaysUserMuteStrategy` in `LLMUserAggregatorParams` — it mutes STT
   input while the bot speaks (this also disables barge-in, acceptable here).

Pre-req that bit us first: the Windows default mic had been disabled in
Settings (it captured pure silence). Re-enable the mic device before a run.

## Phase 2 — Latency

1. `python scripts/bench_stt.py` and `python scripts/bench_tts.py` per backend.
2. Run a representative spoken session, then
   `python scripts/report.py logs/metrics/<session_id>.jsonl`.
3. Fill in the **latency table in `README.md`** with measured p50/p95.
4. If `voice_to_voice_ms` p95 ≥ 800 ms, in rough order of impact: confirm the
   LLM is the bottleneck (more GPU offload / a smaller quant), then try
   `turn_detection.backend: vad_only` or `tts.backend: piper`.

## Phase 3 — Known gaps

- **Dutch replies sound like gibberish.** The LLM sometimes replies in Dutch
  (even to an English command), but the Kokoro voice `af_bella` is American
  English and Kokoro-82M ships no Dutch voice at all (en/es/fr/hi/it/ja/pt/zh
  only) — Dutch text comes out as mangled phonemes. Deferred by decision on
  2026-05-17. Fix later by either forcing English-only replies (tighten the
  prompt — it still *understands* Dutch orders) or switching `tts.backend` to
  `piper`, which has Dutch `nl_*` voices, and picking the voice by reply
  language.
- **Spoken response can contradict the action.** Observed once: "two seven
  zero" set heading 270 correctly, but the spoken line parroted an earlier
  "zero nine zero" from the context history. The action JSON is
  schema-constrained and reliable; the free-text `response` is not. A
  small-model artifact — revisit with prompt tweaks if it recurs.
- **ConversationLogger writes nothing.** `logs/conversations/` stayed empty
  after a full spoken session (per-turn metrics logging works). The observer is
  wired into the pipeline but produces no files — needs debugging.
- **Un-awaited STT coroutine warning.** Segmented STT logs a RuntimeWarning
  ("coroutine 'STTService._ttfb_timeout_handler' was never awaited") — a
  Pipecat internal interaction with `SegmentedSTTService`; non-fatal.
- **Audio device selection** — `audio.input_device` / `output_device` are
  accepted but not mapped to device indices; the OS default is used. Map them
  via `LocalAudioTransportParams(input_device_index=...)`. File: `pipeline.py`.
- **LLM timeout / max_retries** — in the config schema but not forwarded;
  `OpenAILLMService` exposes no hook in Pipecat 1.2.1.
- **onnxruntime Memcpy warning** — Parakeet on CUDA logs "2 Memcpy nodes added";
  a minor perf note (a couple of ops fall back to CPU). Revisit only if STT
  latency is a problem in Phase 2.

## Phase 4 — Real simulator integration (Windows)

1. Drop the in-house wrapper `.py` files and the .NET DLL into
   `voice_agent/backends/simulator/vendor/`.
2. `pip install -e ".[real-sim]"` (adds `pythonnet`; needs Python ≤ 3.13).
3. Fill the `TODO(integration)` stubs in
   `voice_agent/backends/simulator/real.py`: `_connect`, `_to_ship_state`, and
   the four `_*_sync` helpers.
4. Run with `config.examples/config.real_sim.yaml` (or `SIMULATOR_BACKEND=real`).
5. Add real-simulator tests mirroring `tests/test_mock_simulator.py`.

## Open questions

- The benchmark prompt's richer command set (rudder, throttle, autopilot,
  anchor, multi-step) was scoped out: the agent keeps the three actions the
  `SimulatorClient` supports. Expanding it means extending the protocol *and*
  the real simulator's capabilities.
- LM Studio loads Nemotron with a 4096-token context. Fine for short helmsman
  turns; bump it (`lms load -c ...` or the GUI) if a session's history grows.
- After the first good install, lock the transitive dependency set
  (`pip freeze`) for reproducible deploys.
