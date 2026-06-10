# `langgraph` LLM backend — in-backend helmsman

The `langgraph` LLM backend reimplements the **runtime** path of the n8n
helmsman workflow (`POST /webhook/helmsman`, documented in
[`API.md`](API.md)) natively in this project, with no external workflow
engine. It is a drop-in peer of the `n8n` and `openai_compatible` backends:
same pipeline slot, same internal action contract, selected by config alone.

It is the first piece of the n8n → backend migration tracked in the project's
GitHub issue. The HITL document-ingestion + audit path
([`REVIEW_API.md`](REVIEW_API.md)) is **not** part of this backend yet — it
remains on n8n until a follow-up.

## Stack

| Concern | Tool |
|---|---|
| Orchestration | **LangGraph** `StateGraph` (intent routing + RAG subgraph) |
| LLM calls | **LangChain** `ChatOpenAI` against LM Studio's `/v1` |
| Query embedding | LM Studio `/v1/embeddings` (`bge-m3`) over `httpx` |
| Retrieval | **Qdrant** hybrid query (dense + BM25, RRF) over `httpx` |
| Tracing (optional) | **Langfuse** LangChain callback handler |

Qdrant and the embedding endpoint are reached over plain HTTP (`httpx`, a core
dependency) using the exact REST shapes the n8n workflow used, so no
`qdrant-client` is pulled in. Install the stack with:

```bash
pip install -e ".[langgraph]"
```

## The graph

```
classify ─┬─ command ─────────────────────────────────▶ END
          └─ retrieve → select → expand → answer ─────▶ END
```

| Node | n8n analogue | What it does |
|---|---|---|
| `classify` | Classify Intent / Parse Intent | One-word COMMAND/QUESTION classification. Anything that isn't QUESTION routes to the command branch (the safe path that never touches Qdrant). |
| `command` | Command Parser / Format Command Reply | Helmsman command parse with the shared system prompt (`voice_agent/actions/prompt.py`) and the JSON-schema `response_format` (`actions/schema.py`). |
| `retrieve` | Embed Query / Query Points | Embed the query (`bge-m3`), then Qdrant hybrid RRF over a dense and a BM25 prefetch (`top_k * 2` each, fused to `top_k`). |
| `select` | Rerank Enabled? / Rerank Chunks / Apply Rerank / RRF Top-3 | `rerank: true` → LLM listwise rerank to top-3; `rerank: false` → RRF top-3. |
| `expand` | Expansion Enabled? / Build Neighbour Requests / Fetch Neighbours / Merge Neighbours | `expansion: true` → scroll adjacent chunks (`chunk_id ±1`) per file and merge-dedup; `expansion: false` → passthrough. |
| `answer` | Build Prompt / RAG Answer / Parse RAG Response / Format Question Reply | Schema-constrained `{answer, source_chunk_id}` RAG answer, resolved to a citation line. |

All pure shaping (intent parsing, RRF top-3, rerank-index parsing,
neighbour-id math, merge-dedup, RAG-answer parsing, citation) lives in
`helpers.py` and is unit-tested without any external service. The graph wiring
is in `graph.py`; the Qdrant/embedding HTTP calls are in `retrieval.py`.

## Output contract

Like the n8n adapter, the backend emits one `LLMTextFrame` carrying an internal
`HelmsmanResponse` JSON (`{action, response}`) that
`JsonActionProcessor` parses:

- **command** → the parsed action verbatim + the spoken acknowledgement.
- **question** → the synthetic `answer` action; `response` is the RAG answer
  with its `Source: <file>, page <n> (<chunk_id>)` line appended.
- **any failure** (LLM/Qdrant unreachable, parse failure, empty context) → an
  `error` action so TTS speaks the graceful "Lost contact with the bridge"
  fallback. The pipeline never sees malformed JSON.

## Configuration

```yaml
llm:
  backend: langgraph
  base_url: http://localhost:1234/v1   # LM Studio /v1 (chat + embeddings)
  model: unsloth/gemma-4-e4b-it
  api_key_env: LLM_API_KEY
  timeout_seconds: 30
  rerank: true
  expansion: true
  qdrant_url: http://localhost:6333    # omit to run command-only
  qdrant_collection: maritime_hybrid
  qdrant_api_key_env: QDRANT_API_KEY
  embedding_model: text-embedding-bge-m3
  retrieval_top_k: 20
  langfuse_enabled: false
  langfuse_host:                       # blank = Langfuse Cloud
  langfuse_public_key_env: LANGFUSE_PUBLIC_KEY
  langfuse_secret_key_env: LANGFUSE_SECRET_KEY
```

Ready-made: [`config.examples/config.langgraph.yaml`](../config.examples/config.langgraph.yaml).

- **`qdrant_url` unset** → the command branch still works; a question turn
  returns a graceful error envelope instead of attempting retrieval.
- **`embedding_model`** must match the collection's dense named vector
  (`bge-m3`, 1024-dim) — swapping it is a re-ingestion event, same caveat as
  the n8n pipeline.
- **Langfuse** is best-effort: disabled by default, and a missing SDK or
  missing keys silently disables tracing rather than failing the run.

## Parity vs. n8n

Behaviourally equivalent to the n8n runtime workflow: same prompts (classifier,
command, rerank, RAG), same hybrid retrieval, same RRF/rerank/expansion
toggles, same citation format. Differences:

- **Audit log:** the n8n workflow writes a runtime audit row per turn. This
  backend emits Langfuse traces instead; the user-facing audit trail is part of
  the follow-up ingestion/audit migration.
- **`model` per request:** n8n accepted a per-call `model`; here the model is
  fixed from config (the pipeline issues no per-turn override), matching how the
  n8n adapter forwarded `llm.model`.
