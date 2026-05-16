# Next steps

Status at end of scaffold/implementation session (2026-05-16): all code, tests,
scripts, and docs are written and committed; 49 unit tests pass; `ruff` is
clean; every Pipecat 1.2.1 import/constructor used is verified against the
installed package. **Nothing has yet run on real hardware** — no GPU, no remote
LLM, and no audio device were available this session. Work the phases below in
order; each phase gates the next.

## Verified vs. unverified

**Verified this session:** `pip install -e .[dev]` on Python 3.13; all imports
resolve; 49 unit tests pass; `ruff` clean; `report.py` runs on synthetic data;
constructor signatures for STT/TTS/VAD/turn/LLM/transport/pipeline classes.

**Not yet verified (needs the target client):** full pipeline run, model
downloads, CUDA execution, audio I/O, the remote LLM path, and all latency
numbers. The bench scripts and `smoke.py` have never executed.

---

## Phase 1 — Validate the pipeline on the CUDA client (mock simulator)

Goal: `python -m voice_agent.main` runs end to end with the default config
(mock simulator), mic → speakers.

1. Install on the client: `py -3.13 -m venv .venv` then `pip install -e ".[dev]"`.
2. Confirm the GPU runtime: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
   must list `CUDAExecutionProvider`. Both `onnxruntime` (CPU) and
   `onnxruntime-gpu` install — make sure the GPU one wins (README "CUDA setup").
3. `pytest` on the client — should stay 49/49.
4. `python -m voice_agent.main --config config.yaml`. First run downloads the
   Parakeet, Kokoro, Silero, and smart-turn models.
5. Watch specifically for (assumptions not yet exercised):
   - **onnx-asr model id** — `config.yaml` `stt.model: nvidia/parakeet-tdt-1.1b`
     is passed straight to `onnx_asr.load_model()`. onnx-asr may expect a
     different identifier (e.g. its own model name, or the
     `istupakov/parakeet-tdt-1.1b-onnx` repo). Adjust `stt.model` if it fails.
     File: `voice_agent/backends/stt/parakeet_onnx.py`.
   - **TTS `Settings`** — `KokoroTTSService.Settings(voice=...)` /
     `PiperTTSService.Settings(voice=...)`: the nested classes exist, but
     construction with `voice=` was not exercised. Files:
     `voice_agent/backends/tts/{kokoro,piper}.py`.
   - **LocalAudioTransport** opens the OS default audio device — confirm the
     right mic/speakers are selected (see Phase 3, device mapping).

## Phase 2 — End-to-end smoke and latency

1. `python scripts/smoke.py` against a reachable `$LLM_BASE_URL` — confirms the
   LLM → tool → simulator path. Expect `PASS`.
2. `python scripts/bench_stt.py` and `python scripts/bench_tts.py` per backend.
3. Run a representative spoken session, then
   `python scripts/report.py logs/metrics/<session_id>.jsonl`.
4. Fill in the **latency table in `README.md`** with measured p50/p95.
5. If `voice_to_voice_ms` p95 ≥ 800 ms: try `turn_detection.backend: vad_only`,
   `stt.model: nvidia/parakeet-tdt-0.6b`, or `tts.backend: piper`, and re-measure.

## Phase 3 — Close the known gaps

- **LatencyTracker turn boundaries** — confirm `UserStoppedSpeakingFrame` and
  `TTSStoppedFrame` actually flow to the last processor in a live run, so
  per-turn metrics get stamped. If the universal turn machinery emits different
  frames, adjust `voice_agent/metrics.py`.
- **LLM timeout / max_retries** — `timeout_seconds` and `max_retries` are in the
  config schema but not forwarded; `OpenAILLMService` exposes no hook in 1.2.1.
  Revisit when a clean path exists. File:
  `voice_agent/backends/llm/openai_compatible.py`.
- **Audio device selection** — `audio.input_device` / `output_device` names are
  accepted but not mapped to device indices. Map them via
  `LocalAudioTransportParams(input_device_index=..., output_device_index=...)`.
  File: `voice_agent/pipeline.py` (see the `TODO` there).

## Phase 4 — Real simulator integration (Windows)

1. Drop the in-house wrapper `.py` files and the .NET DLL into
   `voice_agent/backends/simulator/vendor/`.
2. `pip install -e ".[real-sim]"` (adds `pythonnet`; needs Python ≤ 3.13).
3. Fill the `TODO(integration)` stubs in
   `voice_agent/backends/simulator/real.py`: `_connect`, `_to_ship_state`, and
   the four `_*_sync` helpers.
4. Run with `config.examples/config.real_sim.yaml` (or `SIMULATOR_BACKEND=real`).
5. Add real-simulator tests mirroring `tests/test_mock_simulator.py` where
   feasible.

## Deferred / open questions

- `smoke.py` seeds the LLM context directly rather than injecting a literal
  `TranscriptionFrame` (the 1.2.x universal aggregator buffers a bare
  transcription into a pending turn that never completes without audio).
  Revisit if a frame-level smoke test becomes important.
- Pinned dependency versions in `pyproject.toml` are May-2026 picks; after the
  first good install on the client, lock the transitive set (`pip freeze`).
