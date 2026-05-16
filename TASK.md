# Task: Modular local voice agent for ship simulator (NVIDIA client, remote LLM)

## Goal

Build a Python voice agent that:

- Listens to spoken English via microphone, replies via spoken English.
- Executes ship-simulator actions through LLM tool calls during the conversation.
- Runs **STT, TTS, VAD, and turn detection locally** on an NVIDIA GPU client.
- Talks to a **remote LLM** over an OpenAI-compatible HTTP API (LLM is owned by another machine; this client only consumes it).
- Talks to the **simulator via a Python library that syncs over UDP**, with a **mock backend** available as a drop-in replacement for development and tests.
- Targets <800 ms voice-to-voice latency on a single client.
- Is **modular and configurable** so models and backends can be swapped without code changes.

This is a from-scratch project. External dependencies the agent does not build or install:
- The simulator integration: a set of in-house Python wrapper classes that load a managed .NET DLL via `pythonnet`. Distributed as plain `.py` files plus a DLL, **not** packaged for pip or as a wheel. The user vendors them into the project tree.
- The remote LLM endpoint (assume `$LLM_BASE_URL`, OpenAI-compatible `/v1`).

## Architecture

Cascade pipeline:

```
mic → VAD → STT → smart-turn → LLM (remote, with tools) → TTS → speakers
                                       │
                                       ▼
                                 SimulatorClient
                                 (real | mock)
```

The LLM is reached over the network. Every other model runs locally on the NVIDIA GPU. Tool handlers execute locally and call into a **`SimulatorClient` abstraction**, which has two interchangeable implementations: a real client wrapping the UDP-syncing Python library, and a mock that holds state in memory.

Use Pipecat as the orchestration framework. Standard Pipecat context aggregator wraps the LLM so user transcripts and assistant replies enter context. Tools declared via `FunctionSchema` + `ToolsSchema`, registered with `llm.register_function(name, handler)`, handlers reply via `params.result_callback(...)`.

## Modularity requirements

Each model type (STT, TTS, VAD, turn detection, LLM client) **and the simulator client** is a swappable component selected at runtime via a config file. Switching backends — Kokoro to Piper, real simulator to mock, Parakeet-1.1B to Parakeet-0.6B — must require **only a config change, no code edits**.

### Configuration

A single YAML config file (`config.yaml`) drives everything. Example shape:

```yaml
stt:
  backend: parakeet_onnx          # parakeet_onnx | parakeet_nemo | whisper
  model: nvidia/parakeet-tdt-1.1b
  device: cuda
  language: en

tts:
  backend: kokoro                  # kokoro | piper
  voice: af_bella
  device: cuda

vad:
  backend: silero
  threshold: 0.5

turn_detection:
  backend: smart_turn_v3           # smart_turn_v3 | vad_only
  device: cpu

llm:
  base_url: http://llm-server:8000/v1
  model: qwen3-30b-a3b-instruct
  api_key_env: LLM_API_KEY         # optional, read from env
  timeout_seconds: 30
  max_retries: 1

simulator:
  backend: real                    # real | mock
  real:
    host: 127.0.0.1                # passed to the vendored wrapper
    port: 9100
    connect_timeout_seconds: 2
  mock:
    initial_heading: 0
    initial_engine_order: stop
    log_commands: true             # log every command for test inspection

audio:
  input_device: default
  output_device: default
  sample_rate: 16000

logging:
  level: info
  format: json                     # json | console
  conversation_log_path: ./logs/conversations
  metrics_log_path: ./logs/metrics
```

Env vars override config values where present (`LLM_BASE_URL`, `LLM_API_KEY`, `SIMULATOR_BACKEND`). CLI flag `--config path/to/config.yaml` selects the file; default `./config.yaml`.

### Factory pattern

Each component type has:
- A protocol/ABC defining the interface.
- A factory function that takes the config block and returns a concrete instance.
- Concrete implementations in their own modules under `voice_agent/backends/{stt,tts,vad,turn,simulator}/`.

Backends to ship in v1:
- **STT:** `parakeet_onnx` (default), `parakeet_nemo`, `whisper`.
- **TTS:** `kokoro` (default), `piper`.
- **VAD:** `silero` (only).
- **Turn detection:** `smart_turn_v3` (default), `vad_only` (benchmarking fallback).
- **Simulator:** `real` (wraps the Python UDP library), `mock` (in-memory).

Adding a new backend later = drop a new file under `backends/<type>/`, register in the factory, no other code changes.

## Stack (default backends)

- **Orchestration:** Pipecat (`pipecat-ai`).
- **STT:** Parakeet-TDT-1.1B via ONNX Runtime CUDA EP, using `onnx-asr` and `istupakov/parakeet-tdt-1.1b-onnx` weights (or current equivalent). Wrap as a custom Pipecat `STTService` subclass.
- **TTS:** Kokoro-82M via Pipecat's `KokoroTTSService` if current; otherwise wrap the local Kokoro ONNX path.
- **VAD:** Silero via `SileroVADAnalyzer`.
- **Turn detection:** `LocalSmartTurnAnalyzerV3`.
- **LLM client:** Pipecat's `OpenAILLMService` (or equivalent) pointed at `$LLM_BASE_URL`.
- **Transport:** `LocalAudioTransport` for mic + speakers.

If a Pipecat plugin you expect to exist is missing or broken, **stop and surface it** rather than silently substituting.

## Simulator client interface

Define `SimulatorClient` as a protocol/ABC with three async methods. The tool handlers depend on this interface, not on either concrete implementation.

```python
class SimulatorClient(Protocol):
    async def set_heading(self, degrees: float) -> ShipState: ...
    async def set_engine_telegraph(self, order: EngineOrder) -> ShipState: ...
    async def get_state(self) -> ShipState: ...
    async def close(self) -> None: ...
```

`ShipState` is a typed dataclass/pydantic model with at minimum: `heading_deg: float`, `speed_kn: float`, `engine_order: EngineOrder`, `timestamp: datetime`. Add fields as the real library exposes them.

`EngineOrder` is an Enum with the nine valid telegraph positions.

### Real backend (`backends/simulator/real.py`)

Thin adapter around the in-house simulator wrappers. The wrappers are plain Python files that load a managed .NET DLL via `pythonnet` (`import clr; clr.AddReference(...)`). They are **not** distributed via pip or as a wheel — the user drops them into the project tree at a known vendored location (`voice_agent/backends/simulator/vendor/`) along with the DLL.

The real backend:
- Adds `pythonnet` to project dependencies.
- Imports the wrapper class(es) from the vendored location.
- Constructs the client with `host`/`port` from config.
- Dispatches `SimulatorClient` protocol methods to wrapper calls.
- Handles connection errors and timeouts gracefully — log them and surface a usable error to the tool handler, which speaks a short failure phrase ("Lost contact with bridge").

**Async/sync bridge.** The exact concurrency model of the wrapper is not yet fixed (a threaded and a non-threaded version of the underlying library exist). If the wrapper's Python API is synchronous and any method does blocking I/O, the real backend must dispatch those calls via `asyncio.to_thread()` (or equivalent thread-pool offload) to avoid blocking the Pipecat event loop. The mock backend is naturally async and needs no special handling. Resource cleanup (closing UDP sockets, joining threads if any) belongs in `SimulatorClient.close()`.

**Note for the agent:** the wrapper API is not yet documented here. Build the adapter against the protocol above using clearly-marked TODO stubs where the wrapper calls go, and assume the wrapper files live at `voice_agent/backends/simulator/vendor/` (create the directory with a `.gitkeep` or `README.md` placeholder explaining what goes there). The user will drop in the real wrappers and DLL during integration and fill in the actual method names. Keep the adapter tiny.

### Mock backend (`backends/simulator/mock.py`)

In-memory implementation of `SimulatorClient`. Keeps current `heading_deg`, `engine_order`, and a derived `speed_kn` (simple mapping from engine order is fine, e.g. `full_ahead → 20`, `half_ahead → 12`, `slow_ahead → 6`, `stop → 0`, astern orders negative). Every method updates state and returns the new `ShipState`.

Optional but encouraged: on each command, append a record to an in-memory log accessible via `mock.command_history` for test assertions. With `log_commands: true` in config, also log each command at INFO level.

The mock is **the default for development** so the agent can run the full pipeline without a real simulator. Switch to `backend: real` only when integrating against the actual library.

## Tool surface

Three Pipecat tools, each a small handler that delegates to the injected `SimulatorClient`:

1. `set_heading(degrees: number)` → `SimulatorClient.set_heading(degrees % 360)` → returns new `ShipState` to the LLM.
2. `set_engine_telegraph(order: enum)` → `SimulatorClient.set_engine_telegraph(order)` → returns new `ShipState`. Valid orders: `full_astern, half_astern, slow_astern, dead_slow_astern, stop, dead_slow_ahead, slow_ahead, half_ahead, full_ahead`.
3. `get_ship_state()` → `SimulatorClient.get_state()` → returns current `ShipState`.

Handlers should be free of business logic beyond input validation. The `SimulatorClient` instance is constructed once at pipeline startup and shared by all three handlers.

## System prompt

Short, English, first person, terse. Includes domain vocabulary so the LLM corrects obvious mishearings implicitly (no separate post-correction stage in v1):

- Role: virtual helmsman on a ship simulator. User is the captain.
- Behavior: acknowledge each command in one short sentence, execute via the appropriate tool, confirm. Never change heading or engine order without an explicit command. On ambiguity, ask — don't act.
- Domain vocabulary inline: engine telegraph orders (full list above), spoken-digit headings ("two seven zero" = 270), common phrases ("steer course", "come to", "hold this heading", "rudder amidships").
- Style: short replies, no filler.

## Logging and metrics

### Structured logging

Use `structlog` or stdlib `logging` with a JSON formatter. Every log line carries:
- `timestamp` (ISO 8601, UTC)
- `level`
- `component` (`stt`, `tts`, `llm`, `vad`, `turn`, `tools`, `simulator`, `pipeline`)
- `session_id` (uuid per process run)
- `message`
- structured fields

Log levels:
- **DEBUG:** raw frames, model loading, UDP packets to/from simulator
- **INFO:** session start/end, transcripts, tool calls, tool results, simulator commands, model swaps
- **WARNING:** retries, slow responses, fallbacks, UDP timeouts that auto-recover
- **ERROR:** unrecoverable errors, simulator connection loss, surfaced via TTS

### Conversation log

Per-session JSONL at `logs/conversations/<session_id>.jsonl`, one object per turn:

```json
{"ts": "...", "session_id": "...", "role": "user", "transcript": "...", "latency_ms": {...}}
{"ts": "...", "session_id": "...", "role": "tool_call", "name": "set_heading", "arguments": {...}}
{"ts": "...", "session_id": "...", "role": "tool_result", "name": "set_heading", "result": {...}}
{"ts": "...", "session_id": "...", "role": "assistant", "text": "...", "latency_ms": {...}}
```

### Performance metrics

`LatencyTracker` `FrameProcessor` stamps these per turn:

- `vad_speech_end_ts`
- `stt_first_partial_ts`
- `stt_final_ts`
- `llm_first_token_ts`
- `llm_last_token_ts`
- `tts_first_audio_ts`
- `tts_last_audio_ts`
- per tool call: `tool_call_start_ts`, `tool_call_end_ts`

Derived metrics:
- `stt_latency_ms = stt_final_ts - vad_speech_end_ts`
- `llm_ttft_ms = llm_first_token_ts - stt_final_ts`
- `llm_total_ms = llm_last_token_ts - llm_first_token_ts`
- `tts_ttfa_ms = tts_first_audio_ts - llm_first_token_ts`
- `voice_to_voice_ms = tts_first_audio_ts - vad_speech_end_ts`  ← headline number
- per-tool: `tool_latency_ms` (separate values for real vs mock to make swap impact visible)

Write per-turn metrics as JSONL to `logs/metrics/<session_id>.jsonl`. On session end, write a summary with p50/p95/p99 of each metric.

`scripts/report.py` reads a metrics file and prints a summary table.

## Latency target

End-to-end voice-to-voice: **< 800 ms**, measured as `voice_to_voice_ms`, p95 over a representative session. Document achieved p50/p95 in the README, per backend combination if you swap any during development.

## Module structure

```
voice_agent/
  __init__.py
  main.py
  config.py                  # pydantic schema + env overrides
  pipeline.py                # builds the Pipecat pipeline from config
  metrics.py                 # LatencyTracker FrameProcessor + writers
  logging_setup.py
  backends/
    stt/
      base.py
      parakeet_onnx.py
      parakeet_nemo.py
      whisper.py
      factory.py
    tts/
      base.py
      kokoro.py
      piper.py
      factory.py
    vad/
      base.py
      silero.py
      factory.py
    turn/
      base.py
      smart_turn_v3.py
      vad_only.py
      factory.py
    llm/
      openai_compatible.py
    simulator/
      base.py                # SimulatorClient protocol, ShipState, EngineOrder
      real.py                # adapter around the vendored pythonnet wrappers
      mock.py                # in-memory implementation
      factory.py
      vendor/                # user drops wrapper .py files + .NET DLL here
        README.md            # explain what goes here
  tools/
    ship.py                  # three tool handlers, depend on SimulatorClient
    schemas.py               # FunctionSchema definitions
scripts/
  smoke.py                   # full LLM-to-tool path against mock simulator
  report.py
  bench_stt.py
  bench_tts.py
tests/
  test_tools.py              # tool handlers against MockSimulatorClient
  test_mock_simulator.py     # mock behaves sensibly under command sequences
  test_config.py
  test_factories.py
config.yaml                  # default: Parakeet-1.1B / Kokoro / Silero / smart_turn_v3 / mock simulator
config.examples/
  config.real_sim.yaml
  config.parakeet_06b.yaml
  config.whisper.yaml
  config.piper.yaml
```

## Non-goals (explicit v1 exclusions)

- Multilingual support. English-only.
- Multi-user / WebRTC rooms.
- Persona / voice cloning.
- Persistent conversation memory across runs. In-memory only.
- Hosting/running an LLM server.
- Implementing the UDP protocol or the simulator wrappers themselves — both are external. The agent does not vendor, install, or fetch the wrapper files or the DLL.
- Simulating ship dynamics in the mock beyond simple state storage. No turning physics, no inertia, no rudder modeling.
- LLM post-correction as a separate pipeline stage. Domain vocabulary lives in the main system prompt.

## Deliverables

1. `voice_agent/` package as outlined.
2. `pyproject.toml` with pinned versions.
3. `README.md` covering: CUDA setup, dependency install (including `pythonnet`), remote LLM configuration, backend switching via `config.yaml`, log/metric layout, achieved latency numbers, and a dedicated **"Integrating the real simulator"** section explaining that the in-house wrappers and DLL are dropped into `backends/simulator/vendor/`, what `backends/simulator/real.py`'s TODOs expect, and the platform requirement noted under "Hardware target".
4. `tests/test_tools.py` — tool handlers tested against `MockSimulatorClient`. No network calls.
5. `tests/test_mock_simulator.py` — sequences of commands produce expected state.
6. `tests/test_config.py` — config validation, env overrides.
7. `tests/test_factories.py` — each factory returns the right concrete type for each backend value.
8. `scripts/smoke.py` — injects fake `TranscriptionFrame("steer course two seven zero")`, asserts LLM emits `set_heading` with `degrees ≈ 270`, and that `MockSimulatorClient.command_history` records it. End-to-end LLM-to-tool-to-simulator path, no audio, no real sim.
9. `scripts/bench_stt.py` and `scripts/bench_tts.py` — run configured backend on sample input, report latency.
10. `scripts/report.py` — summarize a metrics JSONL file.

## Hardware target

- Client: single NVIDIA GPU machine (≥8 GB VRAM comfortable; ≥4 GB floor with Parakeet-0.6B + Kokoro).
- LLM server: external, reached over LAN.
- No DirectML, no ROCm, no Vulkan. Pure CUDA.
- **OS:** Windows is required when running the `real` simulator backend (managed .NET DLL via `pythonnet`). The `mock` simulator backend is platform-agnostic, so STT/TTS/pipeline development can happen on Linux or Windows; only integration against the real simulator is Windows-pinned.

## First step

Read this brief end-to-end. Confirm against current Pipecat docs which of the default backends are first-party and which need wrapping. Confirm the `SimulatorClient` protocol shape makes sense given typical UDP-sync ship libraries (synchronous read of last-known state vs request-response semantics). Sketch the implementation plan in one paragraph — particularly: how the factory pattern integrates with Pipecat service construction, and how the `SimulatorClient` instance gets injected into tool handlers — before writing code.
