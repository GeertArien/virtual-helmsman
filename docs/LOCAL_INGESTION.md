# In-backend HITL ingestion

The review pipeline reimplements the original n8n **document-ingestion +
human-in-the-loop review + audit** workflows (`ingestion_with_hitl`,
`webapp_api`) natively in this project. Together with the
[`langgraph` LLM backend](LANGGRAPH_BACKEND.md) it completes the n8n → backend
migration: **n8n is no longer used at all.**

It serves the **same five `/api/review/*` routes with the same shapes** the
frontend's Documents / Review / Audit pages already call, so those pages work
unchanged.

```bash
pip install -e ".[langgraph]"   # LangChain + Langfuse + pypdf
```

## The pipeline

```
POST /upload ──202──▶ extract (pypdf) → clean → LLM doc-summary → chunk
                      → metadata → pending_review_chunks (SQLite)
                                          │
                       GET /pending ◀─────┤   ← the HITL pause
                                          │
POST /{batch_id}/resume ─▶ apply decisions → all-rejected? ─▶ audit row, stop
                              │
                              └▶ ensure collection → BM25 avg_len
                                 → bge-m3 embeddings → Qdrant upsert
                                 → audit row (Succes — HITL batch …)
```

### The store-backed pause

n8n paused on a **Wait node** with a per-execution `resume_url`, persisting
chunks to a datatable. Here the **`pending_review_chunks` SQLite table *is*
the pause state**: phase one ends by writing it, phase two starts by reading
it. Batches therefore survive backend restarts by construction, there is no
separate workflow-state copy to drift, and resuming is addressed by
`batch_id` against our own API (one-shot: a second resume of the same batch
returns 404, matching the dead-resume-URL semantics).

The audit log lives in the same database (`audit_log` table) and keeps the
n8n column names and Dutch result strings (`Succes — HITL batch …`,
`Fout — PDF extractie mislukt`, `actie: llm_error_ingestion`, …) so the Audit
page renders old n8n rows and new local rows identically.

### Faithful ports

The text processing is a line-for-line port of the n8n Code nodes, pinned by
unit tests:

| n8n node | Port |
|---|---|
| Clean Text | `ingestion/chunking.py::clean_pdf_text` (header/footer frequency filter, page-number strip) |
| Chunk: Paragraph Aware / Fixed Size | `chunk_paragraph_aware` / `chunk_fixed_size` (800/75/725/400 constants; same strategy tags) |
| Chunk + Complete Metadata | `ingestion/metadata.py::complete_metadata` (chunk ids, page estimate, char offsets, `point_id`) |
| Apply Decisions | `apply_decisions` (default-approve, 50-char edit floor, all-rejected sentinel) |
| Calculate avg_len | `compute_avg_len` |
| Message a model | `SUMMARY_SYSTEM` + LangChain `ChatOpenAI` (Langfuse-traced) |
| Create Collection / Indexes / Upsert | `ingestion/qdrant.py` (same REST bodies: named `bge-m3` dense vector + `bm25` IDF sparse, server-side BM25 inference with `avg_len`) |

Upserted points carry the identical payload (including `hitl_reviewed: true`
and `hitl_batch_id`), so locally-ingested chunks are indistinguishable from
n8n-ingested ones to the runtime RAG branch and the Documents page.

The only intentional difference: PDF text extraction uses `pypdf` instead of
n8n's extractor, so raw text can differ slightly before cleaning normalises
it.

## Configuration

```yaml
review:
  db_path: ./data/ingestion.db         # pending batches + audit log
  llm_base_url: http://localhost:1234/v1   # LM Studio (summary + embeddings)
  llm_api_key_env: LLM_API_KEY
  qdrant_url: http://localhost:6333
  qdrant_api_key_env: QDRANT_API_KEY
  embedding_model: text-embedding-bge-m3
  default_document_type: PDF
  default_collection_name: maritime_hybrid
  default_categories: algemeen
  default_chunking_strategy: paragraph_aware
  # optional Langfuse tracing of the doc-summary call:
  langfuse_enabled: false
  langfuse_host:                       # blank = Langfuse Cloud
```

Write endpoints return 503 with a "configure `review.<field>`" message until
`qdrant_url` and `llm_base_url` are set; the read endpoints (pending list,
audit log) work immediately.

Set `db_path` to the same file as `llm.audit_db_path` so the runtime helmsman's
`command_runtime` / `question_runtime` rows and the ingestion rows share one
audit log — the Audit page then shows both.

## Differences from the n8n contract (intentional)

- **No `resume_url`** anywhere: the n8n proxy already stripped it from
  `/pending` responses, so the frontend never saw it; local mode simply has
  none. Resume stays `POST /api/review/{batch_id}/resume`.
- **Resume response** is a clean
  `{"status": "ingested"|"rejected", approved, edited, rejected, indexed}`
  envelope instead of n8n's raw last-node output. The frontend only checks
  `response.ok`.
- **Audit-log `id`s** restart from 1 in the local table; the n8n datatable's
  history is not migrated.
