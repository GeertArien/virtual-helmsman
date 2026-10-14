/**
 * The typed event union mirroring `voice_agent.api.events`, plus the
 * reconnecting WebSocket that delivers it.
 *
 * The discriminated union on `kind` lets components narrow with a single
 * `switch` rather than instanceof / structural checks.
 */

import { wsUrl } from './http';

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

/** The helmsman action vocabulary. `details` shape is action-specific (see
 *  `JsonActionProcessor._publish_turn_events` on the backend). */
export interface ActionDispatchedEvent extends BaseEvent {
  kind: 'action_dispatched';
  action: 'rudder' | 'throttle' | 'navigation' | 'autopilot' | 'anchor' | 'status_query';
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
  /** Actual rudder angle: negative port, positive starboard. Lags an order --
   *  a real rudder slews only a few degrees per second. */
  rudder_angle_deg: number;
  /** Exercise clock in seconds; null on backends without one (the mock). */
  sim_time_s: number | null;
  /** GPS position in signed decimal degrees; null when unavailable. */
  lat_deg: number | null;
  lon_deg: number | null;
}

/** Health of the link to the simulator.
 *  - `connecting`: trying, but no data yet (e.g. the simulator is not running)
 *  - `stale`: the link was live and has gone quiet; reconnecting
 *  Orders are only carried out while `connected`. */
export type SimulatorConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'stale';

export interface ConnectionStateEvent extends BaseEvent {
  kind: 'connection_state';
  state: SimulatorConnectionState;
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
  | TurnMetricsEvent
  | ConnectionStateEvent;

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
