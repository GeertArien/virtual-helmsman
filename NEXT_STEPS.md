# Next steps

> **Session name: `virtual-helmsman-build`**
> Resume with `claude --resume` in `D:\dev\virtual-helmsman` (pick the session
> for this work) or `claude -c` to continue the most recent. This file is the
> entry point — read it first.

Status as of 2026-05-17: the agent has been brought up on the target NVIDIA
client and reworked from LLM tool calls to JSON structured output. The build
and the no-audio path are verified; a live spoken run is the main thing left.

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

## Not yet verified

- **A live spoken run.** `python -m voice_agent.main` has never been driven with
  a real microphone — STT on live audio, VAD/turn endpointing, TTS playback, and
  audio-device selection are unexercised.
- **All latency numbers.** No `voice_to_voice_ms` has been measured. The local
  LLM with partial RAM offload is the prime suspect for blowing the 800 ms
  budget.

---

## Phase 1 — Live spoken run

Goal: `python -m voice_agent.main --config config.yaml` runs end to end,
mic → speakers, against the mock simulator.

1. Confirm LM Studio is up with Nemotron loaded and thinking disabled.
2. Run it; speak a few orders ("steer course two seven zero", "all ahead full",
   "what is our heading"). Confirm the helmsman executes and replies in speech.
3. Watch for: the OS default mic/speakers being the intended devices (see the
   device-mapping gap below); STT accuracy on spoken digits; turn endpointing
   feeling responsive.

## Phase 2 — Latency

1. `python scripts/bench_stt.py` and `python scripts/bench_tts.py` per backend.
2. Run a representative spoken session, then
   `python scripts/report.py logs/metrics/<session_id>.jsonl`.
3. Fill in the **latency table in `README.md`** with measured p50/p95.
4. If `voice_to_voice_ms` p95 ≥ 800 ms, in rough order of impact: confirm the
   LLM is the bottleneck (more GPU offload / a smaller quant), then try
   `turn_detection.backend: vad_only` or `tts.backend: piper`.

## Phase 3 — Known gaps

- **Latency-frame semantics** — `JsonActionProcessor` buffers the LLM's real
  response and emits a fresh `LLMText`/`...End` frame triple after parsing, so
  `LatencyTracker` stamps `llm_first_token_ts` at *that* point. `llm_ttft_ms` is
  therefore ≈ `llm_total_ms`; `voice_to_voice_ms` stays correct. Revisit if a
  true time-to-first-token is needed.
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
