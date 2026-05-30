# Virtual Helmsman — Frontend

SvelteKit + TypeScript dashboard for the voice agent. Subscribes to the
backend's WebSocket for live conversation, ship state, and per-turn
latency, and drives the n8n HITL ingestion + qdrant document-management
routes through the Python backend's proxy.

Four pages:

- **`/`** — *Monitor.* Live transcript, ship state, per-turn latency,
  plus a text-command chatbox and a mic on/off toggle.
- **`/documents`** — upload PDFs to the n8n ingestion pipeline, list
  documents in qdrant, and review pending chunk batches.
  - **`/documents/<batch_id>`** — per-batch chunk review: approve, edit,
    or reject each chunk, then submit decisions back to n8n.
- **`/audit`** — the n8n audit log, rendered per-`actie` (ingestion
  success, all-rejected failure, LLM-error rows, etc.).
- **`/config`** — view, edit, and reload `config.yaml` in-place.

## Requirements

- Node.js ≥ 20.
- The backend running with the API enabled. In `config.yaml`:
  ```yaml
  api:
    enabled: true
    host: 127.0.0.1
    port: 8765
    cors_allow_origins:
      - "http://localhost:5173"   # tighten from "*" once the UI is up
  ```
- Start the agent normally: `python -m voice_agent.main --config config.yaml`.

The Documents / Audit / Config pages additionally require the
`documents.*` and `review.*` blocks in `config.yaml` — see the main
[`../README.md`](../README.md#frontend). Endpoints whose dependencies
aren't configured return HTTP 503 with a "configure
`<field>`" message and the corresponding panel renders a clear
empty-state, so the dashboard boots before all integrations are wired.

## Develop

```bash
cd frontend
npm install
npm run dev
```

Vite serves the SPA at <http://localhost:5173>. The dashboard connects
to `ws://127.0.0.1:8765/ws/events` and `http://127.0.0.1:8765/api/*` by
default; point it at a different backend by appending
`?api=http://other-host:8765` to the URL.

## Build

```bash
npm run build
```

Produces a static bundle under `frontend/build/` via
`@sveltejs/adapter-static`. Serve it with any static server, or copy it
next to the FastAPI app and mount it via
`fastapi.staticfiles.StaticFiles` (not wired yet — the dev workflow uses
Vite).

## Type-check

```bash
npm run check
```

Runs `svelte-kit sync` then `svelte-check`.

## Layout

```
src/
  routes/
    +layout.svelte         # opens the WS stream once, renders <Header>
    +layout.ts             # SSR off (SPA-only)
    +page.svelte           # Monitor page — conversation, ship state, metrics, chat, mic
    documents/
      +page.svelte         # upload, qdrant list/delete, pending-batch index
      [batch_id]/
        +page.svelte       # per-batch chunk review
    audit/
      +page.svelte         # n8n audit log
    config/
      +page.svelte         # view/edit config.yaml
  lib/
    api.ts                 # typed event union + reconnecting WebSocket + REST clients
    liveState.svelte.ts    # global `$state` proxy, owned by +layout.svelte
    components/
      Header.svelte             # session id, WS connection indicator, tab nav
      ConversationPanel.svelte  # transcript / assistant replies
      ChatPanel.svelte          # text-command box on the monitor page
      ShipStatePanel.svelte     # heading / speed / engine order
      MetricsPanel.svelte       # per-turn latency, rolling p50/p95
      UploadDialog.svelte       # PDF + metadata multipart upload
      DeleteDocumentPanel.svelte
      PendingReviewPanel.svelte
      AuditLogPanel.svelte
      SchemaField.svelte        # generic edit widget used by /config
      conversation.ts           # Entry types shared by the conversation views
```

`+layout.svelte` opens the WebSocket once and owns the global `live`
proxy from `liveState.svelte.ts`; routes and panels read directly from
it rather than duplicating state. Route changes do **not** drop the
WebSocket — navigating between `/`, `/documents`, `/audit`, `/config`
stays connected.

On WS (re)connect, `liveState` re-fetches the REST snapshots (current
mic state, session info) so the UI never shows a stale toggle after a
backend restart.

## Toolchain

- Svelte 5 (`$state` / `$props` runes)
- SvelteKit 2 with `@sveltejs/adapter-static` (SPA, no SSR)
- Vite 6
- TypeScript 5
- `svelte-check` for type-checking
