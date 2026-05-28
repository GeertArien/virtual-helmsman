# Virtual Helmsman — External API

The unified workflow exposes two HTTP entry points:

1. **`Webhook (API)`** — a standard `n8n-nodes-base.webhook` node. **This is the
   recommended entry point for webapps, scripts, and any non-chat-UI caller.** It uses
   n8n's documented webhook protocol, accepts arbitrary JSON bodies, and returns
   synchronous JSON.
2. **`When chat message received`** (langchain chatTrigger) — primarily for n8n's
   internal chat panel. The chatTrigger is built around n8n's chat-UI protocol; it can be
   called externally, but the request/response shape is less predictable across n8n
   versions. Use the dedicated webhook instead.

Both entry points feed into the same `Normalize Input` node and from there into the
unified downstream flow, so command parsing, RAG, rerank toggle, and adjacent-chunk
expansion all behave identically.

## Endpoint (recommended for webapps)

```
POST  http://<n8n-host>:5678/webhook/helmsman
Content-Type: application/json
```

- `<n8n-host>` = `localhost` if the webapp runs on the same machine as n8n; whatever n8n
  is reachable as otherwise.
- The path is `/webhook/helmsman` because the Webhook (API) node's *Path* parameter is
  `helmsman`. To change it, open the node and edit the path.
- The workflow must be **Active** for production calls. While inactive, use the *test*
  webhook URL at `/webhook-test/helmsman` (only responds while a manual execution is
  running in n8n).

## Request body

```json
{
  "chatInput": "What does COLREGS Rule 15 require?",
  "rerank": true,
  "model": "unsloth/gemma-4-e4b-it",
  "sessionId": "optional-uuid-for-conversation-continuity"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `chatInput` | string | yes | — | The user's message. Routed by the intent classifier into either the command parser or the RAG branch. |
| `rerank` | boolean | no | `true` | Toggles the LLM-as-a-reranker step in the RAG branch. `false` bypasses it and feeds RRF top-3 directly into adjacent-chunk expansion. Has no effect on the command branch. |
| `model` | string | no | `unsloth/gemma-4-e4b-it` | LM Studio model identifier used for all four LLM calls in this workflow (intent-classify, command-parse, LLM-rerank, RAG-answer). Must be a model currently loaded in your LM Studio Local Server. Trailing whitespace is stripped; empty string and missing field both fall back to the default. The embedding model (`bge-m3`) is **not** parameterisable — it is tied to the dense-vector dimension of the Qdrant collection. |
| `sessionId` | string | no | n8n-generated | Identifies a chat session. Affects nothing structural in the current workflow (we deliberately don't carry conversation memory in v1) but is forwarded to the chatTrigger. |

Any additional fields are accepted and accessible inside the workflow via
`$('When chat message received').item.json.<field>` — useful for future extensions
(e.g. a `top_k` override).

## Response shape

Synchronous JSON. **Both branches return the same five top-level fields** so the
webapp can switch on `intent` alone and read the rest from a stable schema.

Enforcement is asymmetric per branch:

- **Question branch** — *hard schema enforcement* via LM Studio's
  `response_format: {"type": "json_schema", "json_schema": {…}}`. The model is
  grammar-constrained to emit exactly `{answer: string, source_chunk_id: string}`
  before sampling. Failure modes essentially impossible at the inference layer.
- **Command branch** — *soft enforcement* via the Command Parser's system prompt,
  with a `JSON.parse` + fallback path in *Format Command Reply*. Measured ~95-99%
  reliable on Gemma 4 E4B in Module 3. The langchain Agent's LLM-subnode wrapper
  doesn't expose a schema payload field, so hard enforcement here would require
  either bypassing the Agent (Item 15 in `future_improvements.md` — Structured
  Output Parser via chat-template fix) or replacing it with a direct httpRequest.
  Deferred.

Either way, the downstream Code nodes (*Format Command Reply* / *Parse RAG Response*
/ *Format Question Reply*) are defensive: they emit a parse-failure shape rather
than crashing if the LLM somehow produces invalid output. The webapp can rely on
the canonical shape regardless of which enforcement layer is active.

### Audit-log step (iteration 9)

Each request also writes one row to the `audit-log-maritime` n8n datatable on its
way out:

| Field | Command branch value | Question branch value |
|---|---|---|
| `createdAt` | auto | auto |
| `document_naam` | `"n.v.t. (command)"` | `source.filename` or `"n.v.t."` |
| `actie` | `"command_runtime"` | `"question_runtime"` |
| `resultaat` | `action_type=<type> \| output=<first 120 chars>` | `chunk=<id> \| citation_reliable=<bool> \| parse_failure=<bool> \| output=<first 120 chars>` |

The audit-log nodes (*Log Runtime Command* / *Log Runtime Question*) are
side-effects; their inserted-row output does **not** reach the caller. A
terminal *Re-emit Reply* Code node restores the canonical 5-field response
shape before the webhook returns. This means: the API contract is unchanged —
the audit-log writes are transparent to the caller.

### Canonical schema

```json
{
  "intent": "command" | "question",
  "output": "<user-facing text to display>",
  "action": { /* object */ } | null,
  "source": { /* object */ } | null,
  "raw_model_output": "<full LLM string, kept for diagnostics>"
}
```

| Field | Type | Populated when | Notes |
|---|---|---|---|
| `intent` | enum | always | `"command"` or `"question"` — the only discriminator the webapp needs |
| `output` | string | always | What to display in the chat UI. For command intents this is the helmsman's spoken acknowledgement; for question intents it's the RAG answer with a `Source: …` line appended |
| `action` | object \| null | `intent === "command"` (else `null`) | Structured action the webapp may execute. Schema below. |
| `source` | object \| null | `intent === "question"` (else `null`) | Citation metadata for the RAG answer. Schema below. |
| `raw_model_output` | string | always | The full LLM response before parsing. Kept for diagnostics — never displayed to end users |

### Webapp routing (pseudocode)

```javascript
const res = await fetch('/webhook/helmsman', { method: 'POST', body: ... }).then(r => r.json());
ui.display(res.output);
if (res.intent === 'command' && res.action && res.action.type !== 'error') {
  executor.run(res.action);
}
if (res.intent === 'question' && res.source) {
  ui.showCitation(res.source);
}
```

### Command branch — `action` schema

```json
{
  "intent": "command",
  "output": "Starboard twenty degrees, aye sir! Helm is coming to starboard twenty.",
  "action": {
    "type": "rudder",
    "direction": "starboard",
    "degrees": 20
  },
  "source": null,
  "raw_model_output": "{\"action\":{\"type\":\"rudder\",\"direction\":\"starboard\",\"degrees\":20},\"response\":\"Starboard twenty degrees, aye sir!…\"}"
}
```

The `action` object follows the schema from
`LLM_Module_1/.../simulator_command_schema.json`. See the helmsman system prompt for the
full enum of `action.type` values: `rudder`, `throttle`, `navigation`, `autopilot`,
`anchor`, `status_query`, `multi_step`, `error`.

If LM Studio's JSON mode somehow fails or the model emits schema-valid JSON with a
wrong shape, `action` becomes
`{ "type": "error", "error_type": "parse_failure", "reason": "Model output was not valid JSON" }`
and `output` carries the cleaned model text for the operator to see.

### Question branch — `source` schema

```json
{
  "intent": "question",
  "output": "When two power-driven vessels are crossing so as to involve risk of collision, the vessel which has the other on her own starboard side shall keep out of the way…\n\nSource: COLREGS_Parts_AB_Steering_Sailing_Rules.pdf, page 14 (chunk_026)",
  "action": null,
  "source": {
    "chunk_id": "chunk_026",
    "filename": "COLREGS_Parts_AB_Steering_Sailing_Rules.pdf",
    "page": 14,
    "document_summary": "These rules establish international regulations for navigation on high seas and connected waters…",
    "citation_reliable": true,
    "parse_failure": false
  },
  "raw_model_output": "{\"answer\":\"When two power-driven vessels…\",\"source_chunk_id\":\"chunk_026\"}"
}
```

| `source` sub-field | Meaning |
|---|---|
| `chunk_id` | The chunk the model named as primary source |
| `filename` / `page` / `document_summary` | Looked up from the retrieved-chunks array by `chunk_id`. Useful for the webapp to render a "from this PDF, page N" badge |
| `citation_reliable` | `true` iff the model emitted a valid `source_chunk_id` AND that id matched an actually-retrieved chunk. `false` means the citation is a best-effort fallback to the top-ranked chunk |
| `parse_failure` | `true` iff JSON.parse of the model output failed entirely. The `output` field then contains the raw model text. Shouldn't happen in normal operation thanks to JSON mode — flagged for diagnostics |

## Examples

### curl

```bash
# Default model (unsloth/gemma-4-e4b-it)
curl -X POST http://localhost:5678/webhook/helmsman \
  -H "Content-Type: application/json" \
  -d '{"chatInput": "hard to starboard 20 degrees"}'

# Specific model — pass the LM Studio identifier exactly as it appears in
# LM Studio → Developer tab → Local Server. The model must be loaded.
curl -X POST http://localhost:5678/webhook/helmsman \
  -H "Content-Type: application/json" \
  -d '{
    "chatInput": "What does COLREGS Rule 15 require?",
    "model": "unsloth/gemma-4-e4b-it"
  }'
```

### Per-call model selection — caveats

- The embedding model (`bge-m3`) is fixed. Swapping the chat model does *not* re-embed
  any chunks; retrieval quality is independent of the `model` field.
- Different chat models have different reliability characteristics on the RAG branch's
  `response_format: json_schema` mode. Gemma 4 E4B is verified working; Qwen-class
  models with native function-calling work; very small (< 3B) models may degrade
  schema-compliance. If you see `parse_failure: true` in the response after switching
  models, the model probably ignored the schema constraint.
- If the supplied model identifier is not loaded in LM Studio, the call returns the
  upstream LM Studio error (typically HTTP 400 with *"model not found"*) bubbled
  through n8n. There is no allowlist enforced workflow-side.
- The model identifier is applied uniformly to all four LLM calls in the request:
  intent-classify, command-parse (langchain Agent), LLM-rerank (if `rerank: true`),
  and RAG-answer. There is no separate per-step override in v1.

### JavaScript (fetch)

```javascript
async function askHelmsman(message, { rerank = true } = {}) {
  const response = await fetch(
    'http://localhost:5678/webhook/helmsman',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chatInput: message, rerank })
    }
  );
  if (!response.ok) {
    throw new Error(`Helmsman returned ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

// Demo A/B comparison: same question, with and without LLM rerank
const withRerank    = await askHelmsman('What does COLREGS Rule 15 require?', { rerank: true });
const withoutRerank = await askHelmsman('What does COLREGS Rule 15 require?', { rerank: false });
```

## Caveats

### CORS (when webapp is on a different origin)

The n8n webhook by default may not return CORS headers, which blocks browser-based
webapps served from a different origin. Two ways to handle this:

- Run the webapp on the same host as n8n (`http://localhost:<webapp-port>` calling
  `http://localhost:5678`) — same-origin from n8n's perspective if a reverse proxy
  fronts both.
- Set `N8N_PUBLIC_API_DISABLE_CORS=false` (or equivalent CORS config — see the n8n
  environment-variables docs for the current version) in the n8n container's
  environment to enable permissive CORS for development.
- For production, front n8n with nginx / traefik that adds the right CORS headers
  for your webapp's origin.

### Timeouts

The RAG branch with all the bells (classifier + rerank + expand + answer) can take
~10-20 seconds on a single-stream Gemma 4 E4B-it. Set your client timeout accordingly
(60s is a reasonable default).

### Authentication

The chatTrigger webhook is unauthenticated by default. For production:
- Use n8n's "Authentication" option on the chatTrigger (Header Auth, Basic Auth, JWT).
- Or front it with a reverse proxy that handles auth before forwarding.
- The included v1 workflow has no auth — fine for a closed local demo, not for anything
  reachable from the public internet.

### A/B rerank toggle — what to expect

| `rerank` | Behaviour | Typical query latency | When to use |
|---|---|---|---|
| `true` (default) | LLM rerank on (RankGPT listwise call), then adjacent expansion | longer | Production / default UX |
| `false` | Skip rerank, use RRF top-3 directly, then adjacent expansion | shorter | Demo A/B, or fallback if the rerank call ever times out |

The classifier, intent routing, command parser, and adjacent-chunk expansion all run
identically in both modes. Only the chunk-selection step changes.
