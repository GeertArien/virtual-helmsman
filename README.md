# Virtual Helmsman

[![CI](https://github.com/GeertArien/virtual-helmsman/actions/workflows/ci.yml/badge.svg)](https://github.com/GeertArien/virtual-helmsman/actions/workflows/ci.yml)

A modular local voice agent for a ship simulator. It listens to spoken
English, replies in speech, and either executes ship-simulator actions or
answers maritime questions grounded in a RAG corpus — targeting
**< 800 ms voice-to-voice latency** on a single NVIDIA client.

You talk to it from the **web dashboard**: the browser captures your mic and
plays the reply back over WebRTC (there is no local-hardware audio path). STT,
TTS, VAD, and turn detection run **locally on the GPU**. The LLM runs over
HTTP; two backends ship:

- **`langgraph`** *(shipped default)* — runs the helmsman **in this
  backend**: a LangGraph graph classifies each turn, parses commands, and
  answers questions via hybrid RAG over qdrant, calling LM Studio (or any
  OpenAI-compatible `/v1` server) through LangChain, with optional Langfuse
  tracing. No external workflow engine. Full design in
  [`docs/LANGGRAPH_BACKEND.md`](docs/LANGGRAPH_BACKEND.md).
- **`openai_compatible`** — direct chat completion against any
  OpenAI-shaped `/v1` server (e.g. LM Studio). Command parsing only, no
  retrieval.

> **n8n removed.** Earlier versions proxied the helmsman and the document
> ingestion to an external n8n instance. Both now run in-backend — the
> `langgraph` LLM backend and the local HITL ingestion pipeline
> ([`docs/LOCAL_INGESTION.md`](docs/LOCAL_INGESTION.md)).

Every model and the simulator client is a swappable backend selected from
`config.yaml` — no code edits to switch.

## Architecture

```
browser mic ─(WebRTC)─▶ VAD → STT → smart-turn → LLM ─┬─▶ command action ──▶ SimulatorClient
                                                      │                       (real | mock)
                                                      └─▶ answer (hybrid RAG)
                                                                  │
              browser ◀─(WebRTC)─ TTS ◀── spoken response or answer ┘
```

Built on [Pipecat](https://docs.pipecat.ai). The LLM answers each turn
with one JSON object — `action` plus a spoken `response` (or an `answer`
for the RAG branch) — rather than native tool calls, which small local
models emit far less reliably. A processor parses that object, dispatches
the action to a `SimulatorClient` abstraction, and forwards only the spoken
text to TTS.

`SimulatorClient` has two interchangeable implementations: `real` (drives
the hand-dropped in-house integration) and `mock` (in-memory, the dev
default).

The backend additionally handles document ingestion with human-in-the-loop
chunk review — uploaded PDFs are chunked, summarized, paused for reviewer
approval, then embedded with `bge-m3` and upserted into qdrant. The webapp's
document/audit/review routes drive that flow (see
[`docs/LOCAL_INGESTION.md`](docs/LOCAL_INGESTION.md)).

## Requirements

- **Python 3.11–3.13.** *Not 3.14* — `kokoro-onnx` (the default TTS) has no
  3.14 wheels yet. Develop on 3.13.
- **NVIDIA GPU**, pure CUDA. ≥ 8 GB VRAM comfortable; ≥ 4 GB floor with
  Parakeet-0.6B + Kokoro. No DirectML, no ROCm, no Vulkan.
- A reachable **OpenAI-compatible `/v1` LLM endpoint** (e.g. LM Studio).
  The `langgraph` backend additionally needs **qdrant** for the RAG branch.
- The `real` simulator backend needs the hand-dropped vendor integration and
  inherits its platform requirements (see *Integrating the real simulator*).
  The `mock` backend is platform-agnostic.

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
| `nemo`       | `nemo-toolkit[asr]`                   | Using the `parakeet_nemo` STT backend  |

Versions are pinned in `pyproject.toml` as of May 2026. After the first
successful install on the target client, lock the full transitive set
(`pip freeze`) for reproducible deploys.

## Quickstart

The agent and the dashboard are two processes. Run each in its own terminal:

```bash
# Terminal 1 — voice agent + control-plane API (loads the GPU models and serves
# the API + browser-audio signalling on http://127.0.0.1:8765)
python -m voice_agent.main --config config.yaml

# Terminal 2 — SvelteKit dashboard (http://localhost:5173)
cd frontend && npm install && npm run dev
```

Then open <http://localhost:5173>. The dashboard talks to the backend at
`http://127.0.0.1:8765` by default (override with `?api=http://host:port`), so
the API must be up for the live transcript, ship state, and chat box to work.

The default `config.yaml` uses the `mock` simulator and has `api.enabled: true`,
so this runs the full STT→LLM→TTS pipeline with no real simulator attached.
**Speak** via the dashboard's browser-audio control (needs the `webrtc` extra;
see [`docs/BROWSER_AUDIO.md`](docs/BROWSER_AUDIO.md)) or **type** commands in the
chat box. The default `langgraph` LLM and the document/review pages additionally
need LM Studio and qdrant reachable; everything degrades gracefully when they
aren't.

Details for each side live in [Running the agent](#running-the-agent) and
[Frontend](#frontend) below.

## Configuration & backend switching

Everything is driven by a single YAML file (default `./config.yaml`; override
with `--config path/to/file.yaml`). Switching a backend is a **config change
only** — e.g. set `tts.backend: piper`, or `stt.model: nvidia/parakeet-tdt-0.6b`,
or `simulator.backend: real`.

Backends shipped in v1:

| Type      | Backends                                                                                  |
|-----------|-------------------------------------------------------------------------------------------|
| STT       | `parakeet_onnx` (default), `parakeet_nemo`, `whisper`                                     |
| TTS       | `kokoro` (default), `piper`                                                               |
| VAD       | `silero`                                                                                  |
| Turn      | `smart_turn_v3` (default), `vad_only` (benchmark baseline)                                |
| LLM       | `langgraph` (default; in-backend command parsing + RAG), `openai_compatible` (command parsing only) |
| Simulator | `real`, `mock` (default)                                                                  |

Ready-made variants are in [`config.examples/`](config.examples/):
`config.real_sim.yaml`, `config.parakeet_06b.yaml`, `config.whisper.yaml`,
`config.piper.yaml`, `config.langgraph.yaml`.

**Environment overrides** (applied over the file): `LLM_BASE_URL`,
`SIMULATOR_BACKEND`.

### Secrets / API keys

Keys never live in `config.yaml` — the YAML only references the **name** of an
environment variable, and the app reads the secret from the environment at
runtime. On startup the agent loads a local **`.env`** (gitignored) via
`python-dotenv`; real environment variables already set take precedence.

```bash
cp .env.example .env   # then fill in the keys you need
```

| Env var | Used by | Config field | How it's sent |
|---------|---------|--------------|----------------|
| `LLM_API_KEY` | LM Studio `/v1` — chat (both LLM backends) + bge-m3 embeddings (`langgraph` RAG + ingestion) | `lm_studio.api_key_env` | OpenAI `Authorization: Bearer` (local LM Studio usually needs none) |
| `QDRANT_API_KEY` | Documents page proxy, `langgraph` RAG retrieval, and the ingestion upserts | `qdrant.api_key_env` | `api-key` header |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional Langfuse tracing (LLM turns + doc-summary) | `langfuse.public_key_env` / `langfuse.secret_key_env` | Langfuse SDK (self-host or cloud) |

Leave a key blank to send no credential (fine for unauthenticated local
services).

## Running the agent

```bash
python -m voice_agent.main --config config.yaml
# or, via the installed console script:
virtual-helmsman --config config.yaml
```

The default config uses the **mock** simulator, so the full STT→LLM→TTS
pipeline runs without a real simulator attached. Voice input/output is the
browser (WebRTC); the agent needs `api.enabled: true` and the `webrtc` extra
to serve it.

## LLM configuration

Both backends are HTTP-only and typically reachable on the same host as
the agent (LM Studio at `:1234`).

### `langgraph` (default, in-backend RAG)

**LangGraph** orchestrates the turn (intent classify → command parse
or hybrid-RAG with rerank + adjacent-chunk expansion), **LangChain**
(`ChatOpenAI`) makes the LLM calls against LM Studio, and **Langfuse**
optionally traces every step. No external workflow engine. Requires the
optional extra:

```bash
pip install -e ".[langgraph]"
```

The connection settings live in the shared `lm_studio` / `qdrant` / `langfuse`
/ `database` blocks (consumed by RAG, the Documents page, and ingestion alike);
`llm` keeps only the backend tuning:

```yaml
llm:
  backend: langgraph
  model: unsloth/gemma-4-e4b-it        # any id the lm_studio server serves
  rerank: true
  expansion: true
  retrieval_top_k: 20
  audit_enabled: true                  # write command_runtime / question_runtime rows to the audit log
lm_studio:
  base_url: http://localhost:1234/v1   # /v1 server: chat + bge-m3 embeddings
  embedding_model: text-embedding-bge-m3
qdrant:
  url: http://localhost:6333           # omit to run command-only
  collection: maritime_hybrid
langfuse:
  enabled: false                       # true + LANGFUSE_* keys to trace
  host:                                # blank = Langfuse Cloud; or http://localhost:3000
database:
  path: ./data/ingestion.db            # the Audit page shows runtime + ingestion rows from here
```

`rerank: false` skips the LLM-as-reranker step in the RAG branch (faster,
lower-quality on long contexts); `expansion: false` skips the adjacent-chunk
Qdrant scroll (chunk_id ±1) that stitches answers split across a chunk
boundary — independent of `rerank`, so any combination is valid. With
`audit_enabled`, each turn writes a `command_runtime` / `question_runtime`
row to the shared audit log (the Audit page then shows live helmsman activity
alongside ingestion events).

Langfuse is open-source and free to self-host (Docker/Helm). Point
`langfuse.host` at your instance and set the `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` env vars (names overridable via
`langfuse.public_key_env` / `langfuse.secret_key_env`); leave `langfuse.host`
blank to use Langfuse Cloud.

Qdrant and the embedding endpoint are reached over plain HTTP, so no
`qdrant-client` is added. Full design + node-by-node parity with the n8n
runtime workflow: [`docs/LANGGRAPH_BACKEND.md`](docs/LANGGRAPH_BACKEND.md).
The companion `review:` block moves the document-ingestion + HITL review side
in-backend too ([`docs/LOCAL_INGESTION.md`](docs/LOCAL_INGESTION.md)); together
they remove the n8n dependency entirely
(`config.examples/config.langgraph.yaml` enables both).

### `openai_compatible`

Direct chat completion against any OpenAI-shaped `/v1` server. Command
parsing only; no retrieval. The agent sends a JSON-schema `response_format`
so the server constrains output to the helmsman action object — the model
needs structured-output support, but **not** tool calling. Disable any
reasoning/thinking mode (its output never reaches the `content` field the
agent parses).

```yaml
llm:
  backend: openai_compatible
  model: unsloth/gemma-4-e4b-it
  timeout_seconds: 30
  max_retries: 1
lm_studio:
  base_url: http://localhost:1234/v1   # e.g. LM Studio's local server
  api_key_env: LLM_API_KEY
```

Set the key via the env var named by `lm_studio.api_key_env` (default
`LLM_API_KEY`); servers that need no key still work (a placeholder key
is sent).

> **Known gap (openai_compatible only):** `timeout_seconds` /
> `max_retries` are validated but not forwarded to the underlying client
> — `OpenAILLMService` exposes no hook in Pipecat 1.2.1. The `langgraph`
> backend honours `timeout_seconds`. See
> `voice_agent/backends/llm/openai_compatible.py`.

## Logs and metrics

```
logs/
  conversations/<session_id>.jsonl   # one object per turn (user / assistant)
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

Early measurement on the NVIDIA dev rig shows v2v in the ~1500–3000 ms
range — well over budget. The local LLM call (with partial RAM offload)
is the prime suspect; per-component p50/p95 has not been collected yet.
To populate the table below:

1. `python scripts/bench_stt.py` and `python scripts/bench_tts.py` for
   component latency per backend.
2. Run a representative session, then `python scripts/report.py` on its
   metrics file for end-to-end `voice_to_voice_ms`.

| Backend combo (STT / TTS / turn)        | v2v p50 | v2v p95 |
|-----------------------------------------|---------|---------|
| parakeet_onnx / kokoro / smart_turn_v3  | _TBD_   | _TBD_   |
| parakeet_onnx / kokoro / vad_only       | _TBD_   | _TBD_   |
| parakeet-0.6b / piper / smart_turn_v3   | _TBD_   | _TBD_   |

## Testing

```bash
pytest
```

`tests/` covers:

- Action parsing, dispatch, and the JSON action processor against the
  mock simulator (`test_actions.py`, `test_mock_simulator.py`).
- Config validation + env overrides (`test_config.py`) and per-type
  factory dispatch (`test_factories.py`).
- The LLM backend factory and the `langgraph` helmsman — pure node ports,
  retrieval request shapes, runtime audit rows, and the frame contract
  (`test_llm_backends.py`, `test_langgraph_llm.py`).
- The local ingestion pipeline — chunking/metadata/decision ports, the
  SQLite store, qdrant request shapes, and the full upload→review→upsert
  loop (`test_ingestion_pure.py`, `test_ingestion_store.py`).
- The FastAPI control plane: `/api/config` (`test_api_config.py`),
  control text-injection (`test_api_control.py`), document
  list/delete/upload (`test_api_documents.py`), the WebSocket event
  stream (`test_api_events.py`), and the in-backend HITL review pipeline
  (`test_api_review.py`).

No tests make network calls or load GPU models.

`scripts/smoke.py` exercises the end-to-end LLM→JSON action→simulator path
(no audio, no real sim) and **requires a reachable LLM**.

## Integrating the real simulator

The in-house simulator integration is **not** distributed via pip and **not**
published in this repo — what it consists of, how to build it, and its runtime
prerequisites are documented in the **private** simulator repository. To
integrate:

1. Obtain the vendor integration files and drop them into
   [`voice_agent/backends/simulator/vendor/`](voice_agent/backends/simulator/vendor/).
   That directory is **git-ignored**: none of it may be published here.
2. Set `simulator.backend: real` in config (or `SIMULATOR_BACKEND=real`), start
   from `config.examples/config.real_sim.yaml`, and fill in the `simulator.real`
   values from the vendor integration notes **in a local copy** — the working
   endpoint values are deliberately not recorded in this repository.

No code changes are needed: `real.py` drives the integration through the
vendor-neutral `SimulatorWrapper` protocol in
[`wrapper_api.py`](voice_agent/backends/simulator/wrapper_api.py), and loads it
lazily, so its absence is a clear runtime message rather than an import error.

**Lifecycle.** The link is connectionless and its loss is silent, so the backend
connects lazily, judges health by whether frames are still arriving, and rebuilds
the session whenever it goes quiet:

- Starting the agent **without** a running simulator is fine — it comes up on a
  quiet bridge, stays useful for questions, and connects when the simulator
  appears.
- A simulator that stops, crashes, or ends its exercise is noticed within a
  configurable number of missed frames, and reconnected automatically with
  backoff.
- Orders issued while the link is down **fail fast** ("Lost contact with the
  bridge") rather than being queued — a helm order must not execute minutes late.

**Platform:** the `real` backend inherits the vendor integration's platform
requirements. Do STT/TTS/pipeline development with the `mock` backend on any
OS. The lifecycle above is tested against a fake wrapper, so it runs in CI on
Linux too.

## Project layout

```
voice_agent/        package: config, pipeline, metrics, logging, backends, actions
  backends/{stt,tts,vad,turn,llm,simulator}/   swappable backends + factories
  actions/          JSON action schema, parser, dispatch, processor, prompt
  api/              FastAPI + WebSocket control plane for the frontend
  kb/               knowledge-base half: documents + review routers, HITL ingestion
scripts/            smoke, report, bench_stt, bench_tts
tests/              unit tests (no network, no GPU)
frontend/           SvelteKit dashboard (see frontend/README.md)
docs/               LANGGRAPH_BACKEND.md (runtime helmsman),
                    LOCAL_INGESTION.md (HITL ingestion)
config.yaml         default config
config.examples/    backend-variant configs
```

## Frontend

A live dashboard that subscribes to the voice agent over WebSocket lives
in [`frontend/`](frontend/) (SvelteKit + TypeScript).

On every page load an **AI Act Art. 50 transparency gate** blocks the UI
until the user acknowledges they are interacting with an AI system (Dutch
modal; no persistence — it re-prompts each session). On acknowledge it
best-effort logs an `art50_acknowledged` row to the audit trail (via
`POST /api/review/audit-event`), never blocking the user if the backend is
down. Full declaration text: [`frontend/static/documentation/transparantieverklaring.md`](frontend/static/documentation/transparantieverklaring.md).

Four pages:

- **Monitor** (`/`) — live transcript, ship state, per-turn latency,
  plus the two inputs: a **browser-audio** control that captures the mic and
  plays the reply in the browser over WebRTC (see
  [`docs/BROWSER_AUDIO.md`](docs/BROWSER_AUDIO.md)), and a text-command chatbox.
- **Documents** (`/documents`) — upload a PDF to the in-backend HITL
  ingestion pipeline, list and delete document chunks in qdrant, and drill
  into a pending review batch at `/documents/<batch_id>` to approve / edit /
  reject individual chunks. All ingestion and qdrant traffic is proxied
  through `/api/documents/*` and `/api/review/*` so API keys never reach the
  browser.
- **Audit** (`/audit`) — recent entries from the audit log,
  rendered per `actie` (ingestion success, all-rejected failure,
  LLM-error rows, runtime command/question turns, transparency
  acknowledgements, etc.), with an `actie` filter.
- **Config** (`/config`) — view, edit, and reload `config.yaml`
  in-place.

Enable the control plane and point the shared blocks at your services in
`config.yaml` — the Documents page and the in-backend HITL ingestion both read
`qdrant` + `lm_studio`, so there's nothing per-page to configure:

```yaml
api:
  enabled: true
qdrant:
  url: http://127.0.0.1:6333
  collection: maritime_hybrid
lm_studio:
  base_url: http://localhost:1234/v1
database:
  path: ./data/ingestion.db              # HITL pending batches + audit log
```

The review pipeline runs in this backend (LangChain doc-summary, local
SQLite for pending batches + audit log, direct Qdrant upserts — requires the
`langgraph` extra). See [`docs/LOCAL_INGESTION.md`](docs/LOCAL_INGESTION.md).

`qdrant.url` is optional — the Documents and write-ingestion endpoints return
HTTP 503 with a "configure `qdrant.url`" message until it's set, so the
frontend boots before all integrations are wired. Then
`cd frontend && npm install && npm run dev`. See
[`frontend/README.md`](frontend/README.md) for details.

## Non-goals (v1)

English-only; single local user (browser audio is intended for one browser
at a time); no persona/voice cloning; no cross-run memory; this client does
not host the LLM, implement the simulator's transport, or model ship dynamics
in the mock.
