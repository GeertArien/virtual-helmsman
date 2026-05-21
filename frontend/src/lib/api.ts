/**
 * Backend API client: typed event union mirroring `voice_agent.api.events`,
 * plus a small WebSocket helper with auto-reconnect, plus the typed REST
 * client for the qdrant document-management endpoints (proxied through the
 * Python backend at /api/documents/*).
 *
 * The discriminated union on `kind` lets components narrow with a single
 * `switch` rather than instanceof / structural checks.
 */

export interface BaseEvent {
  ts: string;
}

export interface TranscriptEvent extends BaseEvent {
  kind: 'transcript';
  text: string;
}

export interface AssistantReplyEvent extends BaseEvent {
  kind: 'assistant_reply';
  text: string;
}

export interface ActionDispatchedEvent extends BaseEvent {
  kind: 'action_dispatched';
  action: 'set_heading' | 'set_engine_telegraph' | 'get_ship_state';
  details: Record<string, unknown>;
}

export interface ActionRefusedEvent extends BaseEvent {
  kind: 'action_refused';
  error_type: string;
  reason: string;
  suggestion: string;
}

export interface ShipStateEvent extends BaseEvent {
  kind: 'ship_state';
  heading_deg: number;
  speed_kn: number;
  engine_order: string;
}

export interface TurnMetricsEvent extends BaseEvent {
  kind: 'turn_metrics';
  turn_index: number;
  metrics_ms: Partial<{
    stt_latency_ms: number;
    llm_ttft_ms: number;
    llm_total_ms: number;
    tts_ttfa_ms: number;
    voice_to_voice_ms: number;
  }>;
}

export type AgentEvent =
  | TranscriptEvent
  | AssistantReplyEvent
  | ActionDispatchedEvent
  | ActionRefusedEvent
  | ShipStateEvent
  | TurnMetricsEvent;

export interface SessionInfo {
  session_id: string;
  started_at: string;
  stt_backend: string;
  tts_backend: string;
  vad_backend: string;
  turn_backend: string;
  simulator_backend: string;
  llm_model: string;
  subscribers: number;
  events_dropped: number;
}

/** Where the Python control plane is reachable. Override via the URL query
 *  (?api=http://host:port) for quick swaps without rebuilding. */
export function backendUrl(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8765';
  const fromQuery = new URLSearchParams(window.location.search).get('api');
  return fromQuery ?? 'http://127.0.0.1:8765';
}

/** Equivalent for the WebSocket; derived from `backendUrl` so a single override
 *  configures both. */
export function wsUrl(): string {
  const http = backendUrl();
  return http.replace(/^http/i, 'ws') + '/ws/events';
}

export async function fetchSession(): Promise<SessionInfo> {
  const res = await fetch(`${backendUrl()}/api/session`);
  if (!res.ok) throw new Error(`/api/session: HTTP ${res.status}`);
  return (await res.json()) as SessionInfo;
}

// --- Documents (qdrant management) ------------------------------------------

/** One document known to qdrant. `chunk_count` is the number of vector points
 *  that share this document_id (i.e., what gets deleted). */
export interface DocumentInfo {
  document_id: string;
  title: string | null;
  source: string | null;
  chunk_count: number;
  /** Best-effort upload timestamp (ISO 8601). Null when payload didn't carry one. */
  uploaded_at: string | null;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
}

/** Returned by POST /api/documents/upload. `status` is the workflow's
 *  acknowledgement; with human-in-the-loop n8n flows the actual ingestion
 *  happens asynchronously after a reviewer approves. */
export interface UploadResponse {
  status: 'accepted' | 'queued' | 'completed';
  document_id?: string;
  message?: string;
}

export interface DeleteResponse {
  status: 'deleted';
  document_id: string;
  deleted_chunks: number;
}

/** Friendly Error subclass so UI can show server messages without leaking
 *  raw fetch internals. The backend's 4xx/5xx body (when JSON) is preserved
 *  on `.detail` for components that want to render it. */
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function readError(res: Response): Promise<ApiError> {
  let detail: unknown = null;
  let message = `HTTP ${res.status}`;
  try {
    const ct = res.headers.get('content-type') ?? '';
    if (ct.includes('application/json')) {
      detail = await res.json();
      // FastAPI default error shape: { detail: "..." } or { detail: [...] }
      if (detail && typeof detail === 'object' && 'detail' in detail) {
        const d = (detail as { detail: unknown }).detail;
        if (typeof d === 'string') message = d;
      }
    } else {
      const text = await res.text();
      if (text) message = text;
    }
  } catch {
    // ignore -- the bare HTTP code message is fine
  }
  return new ApiError(res.status, message, detail);
}

/** GET /api/documents -- list distinct documents currently in the collection. */
export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${backendUrl()}/api/documents`);
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as DocumentListResponse;
  return body.documents;
}

/** POST /api/documents/upload -- forward a file to the n8n ingestion workflow.
 *  Sent as multipart/form-data with the file under `file` and the optional
 *  title under `title`. */
export async function uploadDocument(file: File, title?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file, file.name);
  if (title) form.append('title', title);
  const res = await fetch(`${backendUrl()}/api/documents/upload`, {
    method: 'POST',
    body: form
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as UploadResponse;
}

/** DELETE /api/documents/{document_id} -- remove all chunks for one document. */
export async function deleteDocument(documentId: string): Promise<DeleteResponse> {
  const res = await fetch(
    `${backendUrl()}/api/documents/${encodeURIComponent(documentId)}`,
    { method: 'DELETE' }
  );
  if (!res.ok) throw await readError(res);
  return (await res.json()) as DeleteResponse;
}

// --- WebSocket --------------------------------------------------------------

export type ConnectionState = 'connecting' | 'open' | 'closed';

/**
 * Minimal reconnecting WebSocket. Reconnect is on a fixed 1 s backoff -- the
 * agent and the browser are usually on the same machine, so anything fancier
 * is wasted complexity.
 */
export class EventStream {
  private ws: WebSocket | null = null;
  private closedByCaller = false;
  private retryHandle: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly onEvent: (ev: AgentEvent) => void,
    private readonly onState: (state: ConnectionState) => void = () => {}
  ) {}

  connect(): void {
    this.closedByCaller = false;
    this.open();
  }

  close(): void {
    this.closedByCaller = true;
    if (this.retryHandle) clearTimeout(this.retryHandle);
    this.retryHandle = null;
    this.ws?.close();
    this.ws = null;
  }

  private open(): void {
    this.onState('connecting');
    const ws = new WebSocket(wsUrl());
    this.ws = ws;
    ws.addEventListener('open', () => this.onState('open'));
    ws.addEventListener('message', (msg) => {
      try {
        const ev = JSON.parse(msg.data) as AgentEvent;
        this.onEvent(ev);
      } catch (err) {
        // A malformed payload is a backend bug; surface but keep the stream alive.
        console.error('Bad event payload', err, msg.data);
      }
    });
    ws.addEventListener('close', () => {
      this.onState('closed');
      if (!this.closedByCaller) {
        this.retryHandle = setTimeout(() => this.open(), 1000);
      }
    });
    ws.addEventListener('error', () => {
      // Browsers fire 'error' before 'close'; closure handles the reconnect.
    });
  }
}
