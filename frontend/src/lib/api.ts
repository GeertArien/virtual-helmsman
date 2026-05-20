/**
 * Backend API client: typed event union mirroring `voice_agent.api.events`,
 * plus a small WebSocket helper with auto-reconnect.
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
