# Virtual Helmsman — Frontend

SvelteKit + TypeScript dashboard for the voice agent. Subscribes to the
backend's WebSocket and renders live conversation, ship state, and per-turn
latency.

Monitor-only at this stage — no controls, no n8n integration, no mic-in-browser
yet. Those land in later iterations.

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

## Develop

```bash
cd frontend
npm install
npm run dev
```

Vite serves the SPA at <http://localhost:5173>. The dashboard connects to
`ws://127.0.0.1:8765/ws/events` by default; point it elsewhere by appending
`?api=http://other-host:8765` to the URL.

## Build

```bash
npm run build
```

Produces a static bundle under `frontend/build/`. Serve it with any static
server, or copy it next to the FastAPI app and mount it via
`fastapi.staticfiles.StaticFiles` (not wired yet — the dev workflow uses Vite).

## Type-check

```bash
npm run check
```

## Layout

```
src/
  routes/
    +layout.svelte       # base CSS + slot
    +layout.ts           # SSR off (SPA-only)
    +page.svelte         # the dashboard — owns all live state
  lib/
    api.ts               # typed event union + reconnecting WebSocket client
    components/
      Header.svelte
      ConversationPanel.svelte
      ShipStatePanel.svelte
      MetricsPanel.svelte
```

The page is the single owner of live state (Svelte 5 `$state` runes); panels
are stateless and take props. That keeps the WebSocket-to-DOM data flow easy to
trace.
