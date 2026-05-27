# Chunk Review API — HTTP contract for the Helmsman HITL webapp

This document is the **contract between the n8n ingestion pipeline and the webapp
that drives the human-in-the-loop chunk review**. The webapp is built in a
separate Claude Code session and has no knowledge of the n8n internals — only
what is documented here.

> **Companion document:** `API.md` describes the *runtime* helmsman endpoint
> (`POST /webhook/helmsman`) for the chat/command webapp. That endpoint and
> the one in this file are independent — different routes, different shapes.
> A single webapp may serve both UIs (two routes), but it doesn't have to.

## Overview

There are **four HTTP interactions** between the webapp and n8n:

| # | Direction | Endpoint | Purpose |
|---|---|---|---|
| 1 | webapp → n8n | `POST /webhook/review/upload` (multipart) | Upload a PDF + its metadata; starts the ingestion pipeline asynchronously |
| 2 | webapp → n8n | `GET  /webhook/review/pending` | List all batches of chunks currently awaiting human review |
| 3 | webapp → n8n | `POST <resume_url>` (unique per batch) | Submit the human's decisions for one batch; this resumes the paused ingestion run |
| 4 | webapp → n8n | `GET  /webhook/audit-log` | Read recent audit log entries (ingestion outcomes, rejected batches, future per-chunk events) |

There is **no callback to the webapp**. The webapp polls (or refreshes on user
action) — n8n never pushes.

The full lifecycle:

```
1. Operator selects a PDF + fills metadata in the webapp form.
2. Webapp POSTs the file (multipart/form-data) to /webhook/review/upload.
   → n8n returns 202 Accepted immediately.
   → ingestion_with_hitl.json then runs asynchronously:
        PDF → clean → summary → chunk → write to pending_review_chunks
3. Workflow pauses on the Wait node with a per-execution resume_url.

4. Webapp polls GET /webhook/review/pending
   → eventually the new batch appears (chunking takes a few seconds).
5. Operator reviews chunks in the webapp UI (approve / edit / reject).
6. Webapp POSTs decisions to the batch's resume_url.

7. Ingestion resumes, applies the decisions, recalculates avg_len for BM25,
   embeds with bge-m3, upserts to Qdrant, logs to audit-log-maritime.
8. The next GET pending call no longer returns this batch.
```

---

## Endpoint 1 — `POST /webhook/review/upload` (multipart)

### Request

```
POST http://<n8n-host>:5678/webhook/review/upload
Content-Type: multipart/form-data; boundary=…
```

The body is a `multipart/form-data` payload with **one file part** and **three
text parts**:

| Part | Type | Required | Notes |
|---|---|---|---|
| *(any name)* | file | yes | The PDF. The field name can be anything — n8n takes the first binary part. Use something descriptive (`pdf`, `file`, `pdf_upload`) for clarity in dev tools. |
| `Document_Type` | text | yes | Typically `"PDF"`. Stored on every chunk's payload in Qdrant — useful as a filter for later analytics. |
| `Collection_Name` | text | yes | Qdrant collection to upsert into. **Default: `maritime_hybrid`** — the same collection the runtime helmsman queries, so reviewed chunks become live knowledge immediately. The webapp can offer a "destination collection" dropdown if you ever want to target a sandbox collection. |
| `Categories` | text | no | Comma-separated tags (e.g. `"colregs, rules"`). Stored as an array on every chunk's payload. Defaults to `"algemeen"` if omitted. |
| `Chunking_Strategy` | text | no | One of `paragraph_aware` (default) or `fixed_size`. See *Chunking strategies* below. Unknown values are silently coerced to the default. |

### Chunking strategies

Two strategies in v1, toggled by the multipart `Chunking_Strategy` field:

| Strategy | What it does | Overlap | Cost |
|---|---|---|---|
| `paragraph_aware` *(default)* | Recursive split on paragraph → line → sentence boundaries, merged up to ~725 chars, sentence-boundary trim on the last 15% of each chunk, tail-merged below 400 chars. | 75 chars | Free — pure JS |
| `fixed_size` | Naive char-window: slice every 800 chars, 75-char overlap. No structural awareness — chunks routinely start mid-word and end mid-sentence. The "what a 5-line tutorial would give you" baseline, kept for in-demo contrast. | 75 chars | Free — pure JS |

Every chunk in Qdrant is tagged with the strategy that produced it
(`chunking_strategy` payload field). To A/B the same PDF, upload it twice with
different `Chunking_Strategy` values into the same collection — retrieved
chunks become filterable on the strategy tag.

A third option — `llm_semantic`, using Gemma to group paragraphs by topic —
was prototyped and removed before v1 ship. See
`documentation/future_improvements.md` for what was tried, what broke, and
what a future revival would need to fix.

### Response — `202 Accepted`

```json
{
  "status": "queued",
  "message": "PDF received. Poll /webhook/review/pending for the chunk-review batch."
}
```

The response is **immediate** — n8n's webhook is configured with
`responseMode: onReceived`, so the HTTP request returns before any chunking
runs. The chunking + write-to-datatable steps happen asynchronously; expect
the batch to appear in the pending list within a few seconds (depends on PDF
size and the doc-summary LLM call).

### Error responses

| Code | Cause | Webapp behaviour |
|---|---|---|
| `4xx` from `fetch()` itself | Network / wrong URL | Show "Could not reach the ingestion service." |
| 202 returned but batch never appears in pending list | PDF extraction failed (image-only PDF, corrupted file, password-protected). The workflow's `Log Error` node fired and the run terminated quietly. | Surface a timeout after ~30 seconds of polling without seeing the new batch. Suggest checking the n8n executions list. |

### Example — JavaScript `fetch`

```javascript
async function uploadPdf(file, { collectionName = 'maritime_hybrid', categories = 'algemeen' } = {}) {
  const fd = new FormData();
  fd.append('pdf', file, file.name);
  fd.append('Document_Type', 'PDF');
  fd.append('Collection_Name', collectionName);
  fd.append('Categories', categories);

  const res = await fetch('http://localhost:5678/webhook/review/upload', {
    method: 'POST',
    body: fd  // browser sets Content-Type with boundary automatically
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json(); // { status: "queued", ... }
}
```

### Example — `curl`

```bash
curl -X POST http://localhost:5678/webhook/review/upload \
  -F "pdf=@./Albertkanaal_Pleasure_Boating.pdf" \
  -F "Document_Type=PDF" \
  -F "Collection_Name=maritime_hybrid" \
  -F "Categories=binnenvaart, regelgeving"
```

---

## Endpoint 2 — `GET /webhook/review/pending`

### Request

```
GET http://<n8n-host>:5678/webhook/review/pending
```

No body, no query parameters, no auth header.

### Response — `200 OK`

```json
{
  "total_pending_batches": 2,
  "batches": [
    {
      "batch_id": "batch_1716210000123_a7f3qx",
      "filename": "COLREGS_Parts_AB_Steering_Sailing_Rules.pdf",
      "collection_name": "maritime_hybrid",
      "resume_url": "http://localhost:5678/webhook-waiting/8a3f.../review",
      "created_at": "2026-05-20T13:00:00.000Z",
      "pending_chunk_count": 28,
      "chunks": [
        {
          "chunk_id": "chunk_000",
          "text": "PART A — GENERAL\n\nRule 1 — Application\n(a) These Rules shall apply to all vessels…",
          "metadata": {
            "idx": 0,
            "filename": "COLREGS_Parts_AB_Steering_Sailing_Rules.pdf",
            "page": 1,
            "chunk_id": "chunk_000",
            "total_chunks": 28,
            "start_char": 0,
            "end_char": 712,
            "chunk_length": 712,
            "words_in_text": 124,
            "document_summary": "Part A and B of the COLREGS …",
            "section_title": "",
            "document_type": "PDF",
            "upload_timestamp": "2026-05-20T13:00:00.000Z",
            "categories": ["colregs", "rules"],
            "chunking_strategy": "paragraph_aware_sentence_boundary",
            "chunk_overlap": 75
          }
        }
        // ... 27 more chunks ...
      ]
    }
    // ... 1 more batch ...
  ]
}
```

### Field semantics

| Field | Notes |
|---|---|
| `batch_id` | Opaque string. Use as the React key / list key. Treat as immutable. |
| `filename` | Original PDF filename. Show as the batch header. |
| `collection_name` | Which Qdrant collection these chunks will land in. Useful as a sanity check ("am I reviewing the right pipeline?"). |
| `resume_url` | **The exact URL to POST decisions to for *this* batch.** Different per batch. Do not hardcode — always read from the response. |
| `created_at` | ISO timestamp the batch entered the queue. Show as "5 minutes ago" or similar. |
| `pending_chunk_count` | Equals `chunks.length`. Provided for convenience. |
| `chunks[].chunk_id` | `chunk_NNN` zero-padded, contiguous within a batch. Use as the per-chunk decision key. |
| `chunks[].text` | The chunk's content. May be up to ~880 characters. May contain newlines. |
| `chunks[].metadata` | The full chunk metadata payload n8n built. The webapp does **not** need to display all of this — `page`, `chunk_length`, `words_in_text`, and `document_summary` are the most useful for review UI. |

### Empty queue

If there are no pending batches:

```json
{ "total_pending_batches": 0, "batches": [] }
```

Render an empty state — *"No chunks waiting for review."* Do not error.

### Error responses

The endpoint should not 4xx/5xx under normal operation. Network failure to n8n
itself surfaces as a fetch error in the browser — handle that as
*"Could not reach the review service. Is n8n running?"*

---

## Endpoint 3 — `POST <resume_url>`

The URL is taken **verbatim** from the batch object's `resume_url` field
returned by Endpoint 1. It is a one-shot URL — once posted to, it stops
responding to subsequent calls for that batch.

### Request

```
POST <resume_url>
Content-Type: application/json
```

```json
{
  "batch_id": "batch_1716210000123_a7f3qx",
  "decisions": [
    { "chunk_id": "chunk_000", "action": "approve" },
    { "chunk_id": "chunk_001", "action": "reject", "reason": "marketing fluff" },
    { "chunk_id": "chunk_002", "action": "edit",   "edited_text": "Cleaned up version of the chunk…" },
    { "chunk_id": "chunk_003", "action": "approve" }
  ]
}
```

### Field semantics

| Field | Required | Notes |
|---|---|---|
| `batch_id` | yes | Echoed back into the audit log. Useful for traceability. |
| `decisions` | yes | Array of per-chunk decisions. **A chunk omitted from this array is treated as *approve*** — the workflow is conservative about silent drops. |
| `decisions[].chunk_id` | yes | Must match a `chunk_id` returned by Endpoint 1 for this batch. Unknown chunk_ids are ignored. |
| `decisions[].action` | yes | One of `approve`, `reject`, `edit` (case-insensitive). Unknown actions default to `approve`. |
| `decisions[].edited_text` | required if `action="edit"` | Replacement text. Must be ≥ 50 characters after trim, otherwise the chunk is silently rejected (don't send tiny edits). |
| `decisions[].reason` | optional | Free-text reason for a reject/edit. Not persisted in v1 — reserved for future audit logging. |

### Edge cases the webapp should handle

| Situation | Webapp behaviour |
|---|---|
| User rejects every chunk in a batch | Submit anyway. n8n will throw `"all chunks were rejected"` and the batch will fail visibly — fine for the demo, but warn the user before submitting. |
| User edits a chunk to fewer than 50 chars | Block on the client (disable submit, show inline error). |
| User leaves a chunk unmarked | Treated as `approve`. UX-wise, either gray out the chunk with an "approve (default)" label, or force a click before submit. The first option is simpler. |
| User submits the same batch twice | The second POST hits a dead resume URL → n8n returns 404 or 410. Catch and refresh the pending list. |

### Response — `200 OK`

n8n's Wait node returns the resumed-execution's last-node output. For v1 this
is the Qdrant upsert response, which is verbose. The webapp does not need to
display it — just check `response.ok` and refresh the pending list.

A future tightening is to add a *Format Resume Reply* node so the response is
a clean `{ "status": "ingested", "approved": N, "rejected": M, "edited": K }`.

---

## Endpoint 4 — `GET /webhook/audit-log`

Returns recent entries from the `audit-log-maritime` datatable. The same
datatable that the ingestion pipeline writes to via *Log Success* and
*Log All Rejected*. Lets the webapp surface a "recent activity" feed —
useful both for the demo (showing the audit trail right after a HITL
submission) and for the rubric §1 *Foutafhandeling en logging* story.

### Request

```
GET http://<n8n-host>:5678/webhook/audit-log?limit=50&actie=ingestie_hitl&since=2026-05-20T00:00:00Z
```

All query parameters are **optional**:

| Param | Type | Default | Notes |
|---|---|---|---|
| `limit` | int | 50 | Capped server-side at 500. Negative / non-numeric values fall back to the default. |
| `actie` | string | *(no filter)* | Exact-match filter on the `actie` column. Common values: `ingestie_hitl` (successful HITL ingestion), `vraag` (runtime question — once the runtime workflow logs), `verwijdering` (chunk deletion — future). |
| `since` | ISO-8601 | *(no filter)* | Return only entries with `createdAt >= since`. Invalid timestamps are silently ignored. |

No body, no auth header.

### Response — `200 OK`

```json
{
  "total_in_log": 142,
  "total_returned": 3,
  "applied_filters": {
    "limit": 50,
    "actie": "ingestie_hitl",
    "since": null
  },
  "entries": [
    {
      "id": 142,
      "createdAt": "2026-05-27T09:14:22.118Z",
      "document_naam": "Albertkanaal_Pleasure_Boating.pdf",
      "actie": "ingestie_hitl",
      "resultaat": "Succes — HITL batch batch_1716210000123_a7f3qx → approved=8 / edited=1 / rejected=1 — indexed 9 chunks"
    },
    {
      "id": 141,
      "createdAt": "2026-05-27T09:11:05.002Z",
      "document_naam": "COLREGS_Parts_AB_Steering_Sailing_Rules.pdf",
      "actie": "ingestie_hitl",
      "resultaat": "Fout — alle chunks afgewezen door reviewer (batch batch_1716209800001_kx9p3a)"
    }
  ]
}
```

### Field semantics

| Field | Notes |
|---|---|
| `total_in_log` | Total row count in `audit-log-maritime` **before** filtering. Useful as a sanity check that the table is growing. |
| `total_returned` | Length of `entries` after filter + limit. |
| `applied_filters` | Echo of the parsed query params after defaults/clamps. Helps debug filter typos. |
| `entries[].id` | n8n's row id (monotonic int). Use as React key. |
| `entries[].createdAt` | ISO timestamp, set automatically by n8n on insert. |
| `entries[].document_naam` | Filename of the affected document. May be `"onbekend"` for error rows where the filename wasn't yet known. |
| `entries[].actie` | Event category. See the `actie` filter table above. |
| `entries[].resultaat` | Free-text summary in Dutch (legacy from Module 2). v1 ingestion writes one of three patterns: `"Succes — HITL batch …"` / `"Fout — alle chunks afgewezen …"` / `"Fout — PDF extractie mislukt"`. |

Ordering is always **newest-first** (by `createdAt`). No pagination — use
`limit` + `since` to scope a tail.

### Empty result

```json
{ "total_in_log": 0, "total_returned": 0, "applied_filters": { "limit": 50, "actie": null, "since": null }, "entries": [] }
```

Render an empty state — *"No audit entries yet."*

### Error responses

Should not 4xx/5xx under normal operation. Bad query-param values are
coerced to defaults rather than erroring. Network failure to n8n itself
surfaces as a fetch error in the browser.

### Demo move

After submitting a HITL batch, the webapp polls `/webhook/audit-log?limit=5`
and renders the most recent row inline as confirmation. The audience sees
the audit row appear *because* they clicked submit — visible cause/effect
for the rubric §1 logging story.

### Example — JavaScript `fetch`

```javascript
async function fetchRecentAudit({ limit = 50, actie, since } = {}) {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (actie) params.set('actie', actie);
  if (since) params.set('since', since);
  const res = await fetch(`http://localhost:5678/webhook/audit-log?${params}`);
  if (!res.ok) throw new Error(`Audit-log fetch failed: ${res.status}`);
  return res.json();
}
```

### Example — `curl`

```bash
# 50 most-recent rows
curl http://localhost:5678/webhook/audit-log | jq .

# only successful ingestions in the last hour
SINCE=$(date -u -d '1 hour ago' +'%Y-%m-%dT%H:%M:%SZ')
curl "http://localhost:5678/webhook/audit-log?actie=ingestie_hitl&since=$SINCE" | jq .
```

---

## Sample webapp flow (pseudocode)

```javascript
// 1. Upload — webapp form posts the file + metadata
async function onUploadFormSubmit(file, fields) {
  const fd = new FormData();
  fd.append('pdf', file, file.name);
  fd.append('Document_Type', fields.documentType);
  fd.append('Collection_Name', fields.collectionName);
  fd.append('Categories', fields.categories);
  await fetch('http://localhost:5678/webhook/review/upload', { method: 'POST', body: fd });
  // The 202 returns immediately — start polling for the new batch.
  startPollingPending();
}

// 2. List pending batches (poll every ~3 seconds while in the review view)
async function refreshPendingList() {
  const { batches } = await fetch('http://localhost:5678/webhook/review/pending')
    .then(r => r.json());
  renderBatches(batches);
}

// 3. Submit decisions for one batch
async function submitDecisions(batch, decisions) {
  const res = await fetch(batch.resume_url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ batch_id: batch.batch_id, decisions })
  });
  if (!res.ok) throw new Error(`Submit failed: ${res.status}`);
  await refreshPendingList();   // batch should be gone now
  await refreshAuditLog();      // new audit row should be on top
}

// 4. Pull the recent-activity feed
async function refreshAuditLog() {
  const { entries } = await fetch('http://localhost:5678/webhook/audit-log?limit=20')
    .then(r => r.json());
  renderAuditFeed(entries);
}
```

---

## CORS

Same caveat as `API.md`:

- If the webapp is served from the same origin as n8n (e.g. fronted by nginx),
  no CORS handling is needed.
- If the webapp runs on a separate origin during development, set
  `N8N_PUBLIC_API_DISABLE_CORS=false` (or the equivalent for the running n8n
  version) so the webhook nodes return the right headers.
- A common dev pattern is to proxy `/webhook/*` from the webapp's dev server
  to `http://localhost:5678/webhook/*` — avoids touching n8n config and
  sidesteps CORS entirely.

## Auth

None in v1. Closed-local-demo posture. For a production deployment the
endpoints would sit behind a reverse proxy with header-based auth, and the
`resume_url` would need to either include a per-batch HMAC or be issued only
to authenticated reviewers.

---

## Demo flow (live, Les 22)

1. Show Docker Desktop — n8n + Qdrant containers running.
2. Open the webapp `/review` route. Upload the demo PDF via the form (file +
   metadata).
   - Use the *Albertkanaal pleasure-boating* PDF (small, ~10 chunks — keeps the
     demo short).
   - Collection name: `maritime_hybrid` — same collection the runtime
     helmsman queries, so the reviewed chunks become live knowledge as soon as
     the operator hits *Submit*.
3. The webapp shows a "queued" toast (the 202 response). After a few seconds
   the new batch appears on the same page (polling).
4. Walk through the chunks. Approve most, reject one (e.g. a boilerplate front
   page), edit one (e.g. clean up an OCR glitch).
5. Submit.
6. Switch to the n8n execution view — the paused execution resumed and
   finished. Open the *Log Success* node to show the per-batch audit summary
   (`approved=8 / edited=1 / rejected=1 — indexed 9 chunks`).
7. Switch to the runtime helmsman webapp and ask a question that hits one of
   the reviewed chunks. Show that the answer comes back grounded in the
   reviewed material.

The end-to-end story is **~3 minutes** and is the strongest single
rubric-aligned moment of the demo (§1 *foutafhandeling getoond* 4pt, §3
*compliance & ethiek* — humans validating LLM inputs).

---

## Quick test with curl (no webapp needed)

```bash
# 1. Upload a PDF
curl -X POST http://localhost:5678/webhook/review/upload \
  -F "pdf=@./test.pdf" \
  -F "Document_Type=PDF" \
  -F "Collection_Name=maritime_hybrid" \
  -F "Categories=binnenvaart"
# → {"status":"queued",...}

# 2. Wait ~5 seconds for chunking, then read pending batches
sleep 5
curl http://localhost:5678/webhook/review/pending | jq .

# 3. Extract the first batch's resume_url
RESUME_URL=$(curl -s http://localhost:5678/webhook/review/pending | jq -r '.batches[0].resume_url')
BATCH_ID=$(curl -s http://localhost:5678/webhook/review/pending | jq -r '.batches[0].batch_id')

# 4. Approve everything (absence-of-decision means approve)
curl -X POST "$RESUME_URL" \
  -H "Content-Type: application/json" \
  -d "{\"batch_id\": \"$BATCH_ID\", \"decisions\": []}"

# 5. Confirm an audit row landed for this ingestion
curl 'http://localhost:5678/webhook/audit-log?limit=1&actie=ingestie_hitl' | jq .
```

This is useful for sanity-checking the n8n side before the webapp exists.

---

*Document captured during v1 development of the Eindproject HITL ingestion.*
