# Virtual Helmsman

A modular local voice agent for a ship simulator. It listens to spoken English,
replies in speech, and executes ship-simulator actions through remote-LLM tool
calls — targeting **< 800 ms voice-to-voice latency** on a single NVIDIA client.

STT, TTS, VAD, and turn detection run **locally on the GPU**; the LLM is a
**remote** OpenAI-compatible endpoint. Every model and the simulator client is a
swappable backend selected from `config.yaml` — no code edits to switch.

See [`TASK.md`](TASK.md) for the full project brief.

## Architecture

```
mic → VAD → STT → smart-turn → LLM (remote, with tools) → TTS → speakers
                                      │
                                      ▼
                                SimulatorClient
                                (real | mock)
```

Built on [Pipecat](https://docs.pipecat.ai). Tool handlers call into a
`SimulatorClient` abstraction with two interchangeable implementations: `real`
(wraps the in-house UDP-syncing library) and `mock` (in-memory, the dev default).

## Requirements

- **Python 3.11–3.13.** *Not 3.14* — `kokoro-onnx` (the default TTS) and
  `pythonnet` have no 3.14 wheels yet. Develop on 3.13.
- **NVIDIA GPU**, pure CUDA. ≥ 8 GB VRAM comfortable; ≥ 4 GB floor with
  Parakeet-0.6B + Kokoro. No DirectML, no ROCm, no Vulkan.
- A reachable **remote LLM** exposing an OpenAI-compatible `/v1` API.
- **Windows** is required only for the `real` simulator backend (it loads a
  managed .NET DLL via `pythonnet`). The `mock` backend is platform-agnostic.

## CUDA setup

1. Install a recent NVIDIA driver (CUDA 12.x capable). The CUDA runtime and
   cuDNN themselves are **not** installed system-wide — they come from the
   `cuda` extra (see Installation) as venv-local pip wheels, and
   `voice_agent/_cuda.py` adds them to the DLL search path at import time.
2. **onnxruntime conflict:** Pipecat's `silero` extra pulls CPU `onnxruntime`,
   while this project also installs `onnxruntime-gpu`. They share the
   `onnxruntime` import name and clobber each other. On the CUDA client, after
   installing, force the GPU build to win:
   ```
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install onnxruntime-gpu==1.26.0
   ```
3. Verify the CUDA execution provider actually loads (a session, not just the
   provider list — the list shows `CUDAExecutionProvider` even when its DLLs
   are missing):
   ```
   python -c "import voice_agent, onnxruntime as ort, numpy as np; \
   from onnx import helper, TensorProto; \
   g=helper.make_graph([helper.make_node('Add',['X','X'],['Y'])],'g', \
   [helper.make_tensor_value_info('X',TensorProto.FLOAT,[2])], \
   [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[2])]); \
   m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]); m.ir_version=10; \
   s=ort.InferenceSession(m.SerializeToString(),providers=['CUDAExecutionProvider']); \
   print(s.get_providers())"
   ```
   `CUDAExecutionProvider` must appear in the printed list.

## Installation

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev,cuda]"      # core + dev tools + CUDA runtime wheels
```

On the NVIDIA client always include the `cuda` extra; then resolve the
onnxruntime conflict as described in **CUDA setup** above.

Optional extras:

| Extra        | Adds                                  | When                                   |
|--------------|---------------------------------------|----------------------------------------|
| `cuda`       | `nvidia-*-cu12` (CUDA 12.x + cuDNN 9) | Running the ONNX models on the GPU      |
| `real-sim`   | `pythonnet`                           | Integrating the real simulator backend |
| `nemo`       | `nemo-toolkit[asr]`                   | Using the `parakeet_nemo` STT backend  |

```bash
pip install -e ".[dev,cuda,real-sim]"  # for real-simulator integration (Windows)
```

Versions are pinned in `pyproject.toml` as of May 2026. After the first
successful install on the target client, lock the full transitive set
(`pip freeze`) for reproducible deploys.

## Configuration & backend switching

Everything is driven by a single YAML file (default `./config.yaml`; override
with `--config path/to/file.yaml`). Switching a backend is a **config change
only** — e.g. set `tts.backend: piper`, or `stt.model: nvidia/parakeet-tdt-0.6b`,
or `simulator.backend: real`.

Backends shipped in v1:

| Type     | Backends                                                |
|----------|---------------------------------------------------------|
| STT      | `parakeet_onnx` (default), `parakeet_nemo`, `whisper`   |
| TTS      | `kokoro` (default), `piper`                             |
| VAD      | `silero`                                                |
| Turn     | `smart_turn_v3` (default), `vad_only` (benchmark baseline) |
| Simulator| `real`, `mock` (default)                                |

Ready-made variants are in [`config.examples/`](config.examples/):
`config.real_sim.yaml`, `config.parakeet_06b.yaml`, `config.whisper.yaml`,
`config.piper.yaml`.

**Environment overrides** (applied over the file): `LLM_BASE_URL`,
`SIMULATOR_BACKEND`. The LLM API key is always read from the env var named by
`llm.api_key_env` (default `LLM_API_KEY`).

## Running the agent

```bash
python -m voice_agent.main --config config.yaml
# or, via the installed console script:
virtual-helmsman --config config.yaml
```

The default config uses the **mock** simulator, so the full STT→LLM→TTS
pipeline runs without a real simulator attached.

## Remote LLM configuration

The LLM is never hosted by this client — it is consumed over HTTP. Point
`llm.base_url` at the remote OpenAI-compatible `/v1` endpoint (or set
`LLM_BASE_URL`). Set the key via the `LLM_API_KEY` env var; local servers that
need no key still work (a placeholder key is sent).

```yaml
llm:
  base_url: http://llm-server:8000/v1
  model: qwen3-30b-a3b-instruct
  api_key_env: LLM_API_KEY
  timeout_seconds: 30
  max_retries: 1
```

> **Known gap:** `timeout_seconds` / `max_retries` are validated but not yet
> forwarded to the underlying OpenAI client — `OpenAILLMService` does not
> expose them in Pipecat 1.2.1. They are kept in the schema for when a clean
> hook exists. See `voice_agent/backends/llm/openai_compatible.py`.

## Logs and metrics

```
logs/
  conversations/<session_id>.jsonl   # one object per turn (user / tool / assistant)
  metrics/<session_id>.jsonl         # one object per turn + a session summary
```

Structured logs (JSON or console, per `logging.format`) carry `timestamp`,
`level`, `component`, `session_id`, and `message`. Per-turn metrics include
`stt_latency_ms`, `llm_ttft_ms`, `llm_total_ms`, `tts_ttfa_ms`, and the headline
`voice_to_voice_ms`; the session summary adds p50/p95/p99 for each.

Summarize a run:

```bash
python scripts/report.py logs/metrics/<session_id>.jsonl
```

## Latency

Target: **`voice_to_voice_ms` p95 < 800 ms** over a representative session.

Achieved numbers depend on the GPU and backend combination and must be measured
on the target client — they are **not yet filled in** (no GPU/LLM was available
at scaffold time). To populate this table:

1. `python scripts/bench_stt.py` and `python scripts/bench_tts.py` for
   component latency per backend.
2. Run a representative session, then `python scripts/report.py` on its metrics
   file for end-to-end `voice_to_voice_ms`.

| Backend combo (STT / TTS / turn)        | v2v p50 | v2v p95 |
|-----------------------------------------|---------|---------|
| parakeet_onnx / kokoro / smart_turn_v3  | _TBD_   | _TBD_   |
| parakeet_onnx / kokoro / vad_only       | _TBD_   | _TBD_   |
| parakeet-0.6b / piper / smart_turn_v3   | _TBD_   | _TBD_   |

## Testing

```bash
pytest
```

`tests/` covers tool handlers against the mock simulator, mock-simulator command
sequences, config validation + env overrides, and factory dispatch. No tests
make network calls or load GPU models.

`scripts/smoke.py` exercises the end-to-end LLM→tool→simulator path (no audio,
no real sim) and **requires a reachable remote LLM**.

## Integrating the real simulator

The in-house simulator integration — Python wrapper classes plus a managed .NET
DLL — is **not** distributed via pip. Vendor it by hand:

1. Drop the wrapper `.py` files and the .NET DLL into
   [`voice_agent/backends/simulator/vendor/`](voice_agent/backends/simulator/vendor/).
   That directory is intentionally tracked by git (see its `README.md`).
2. `pip install -e ".[real-sim]"` to add `pythonnet` (Windows).
3. Fill in the `TODO(integration)` stubs in
   [`voice_agent/backends/simulator/real.py`](voice_agent/backends/simulator/real.py):
   - `_connect()` — import the wrapper class and construct it with `host`/`port`.
   - `_to_ship_state()` — map the wrapper's fields onto `ShipState`.
   - `_set_heading_sync()`, `_set_engine_telegraph_sync()`, `_get_state_sync()`,
     `_close_sync()` — call the actual wrapper methods.
   The adapter already offloads these synchronous calls via `asyncio.to_thread()`
   so blocking UDP I/O never stalls the Pipecat event loop.
4. Set `simulator.backend: real` in config (or `SIMULATOR_BACKEND=real`), or use
   `config.examples/config.real_sim.yaml`.

**Platform:** the `real` backend is Windows-only because the .NET DLL is loaded
via `pythonnet`. Do STT/TTS/pipeline development with the `mock` backend on any
OS; only real-simulator integration is Windows-pinned.

## Project layout

```
voice_agent/        package: config, pipeline, metrics, logging, backends, tools
  backends/{stt,tts,vad,turn,llm,simulator}/   swappable backends + factories
  tools/            ship tool schemas and handlers
scripts/            smoke, report, bench_stt, bench_tts
tests/              unit tests (no network, no GPU)
config.yaml         default config
config.examples/    backend-variant configs
```

## Non-goals (v1)

English-only; no multi-user/WebRTC; no persona/voice cloning; no cross-run
memory; this client does not host the LLM, implement the UDP protocol, or model
ship dynamics in the mock.
