/**
 * App-wide live agent state, owned at the layout level so it survives route
 * changes (no WS reconnect when navigating between /, /documents, etc.).
 *
 * The single $state object is mutated by the WebSocket handler; components
 * read its fields directly. Treat the returned `live` proxy as the single
 * source of truth for live data -- do not duplicate fields into local state.
 */

import { EventStream, fetchControlState, fetchSession } from './api';
import type {
  AgentEvent,
  ConnectionState,
  SessionInfo,
  ShipStateEvent,
  TurnMetricsEvent
} from './api';
import type { Entry } from './components/conversation';

/** Bounded conversation log so a long-running session doesn't grow forever. */
const MAX_ENTRIES = 500;
/** Bounded metrics history (~200 turns ≈ many minutes of conversation). */
const MAX_TURNS = 200;

export const live = $state({
  connection: 'connecting' as ConnectionState,
  session: null as SessionInfo | null,
  entries: [] as Entry[],
  shipState: null as ShipStateEvent | null,
  turnMetrics: [] as TurnMetricsEvent[],
  /** ``null`` while the initial /api/control/state fetch is in flight.
   *  Components should treat that as "unknown" and disable both inputs
   *  rather than guessing. */
  micEnabled: null as boolean | null
});

let stream: EventStream | null = null;

function append(entry: Entry) {
  const next = live.entries.concat(entry);
  live.entries = next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
}

function actionLabel(action: string, details: Record<string, unknown>): string {
  if (action === 'set_heading' && 'degrees' in details) return `set_heading ${details.degrees}°`;
  if (action === 'set_engine_telegraph' && 'order' in details) return `engine ${details.order}`;
  return action;
}

function onEvent(ev: AgentEvent) {
  switch (ev.kind) {
    case 'transcript':
      append({ kind: 'user', ts: ev.ts, text: ev.text });
      break;
    case 'assistant_reply':
      append({ kind: 'assistant', ts: ev.ts, text: ev.text });
      break;
    case 'action_dispatched':
      append({
        kind: 'action',
        ts: ev.ts,
        label: actionLabel(ev.action, ev.details),
        ok: true
      });
      break;
    case 'action_refused':
      append({ kind: 'refused', ts: ev.ts, reason: ev.reason });
      break;
    case 'ship_state':
      live.shipState = ev;
      break;
    case 'turn_metrics':
      live.turnMetrics = live.turnMetrics.concat(ev).slice(-MAX_TURNS);
      break;
    case 'input_mode_changed':
      live.micEnabled = ev.mic_enabled;
      break;
  }
}

/** Refresh every REST snapshot the UI shows in the header / chat toggle.
 *
 *  These fields don't have a corresponding pipeline event so they can't
 *  auto-update from the WebSocket -- the only way to keep them in sync is
 *  to re-fetch on connect. Called below whenever the WS transitions to
 *  ``open``, so a backend reload (the WS drops, auto-reconnects, then
 *  fires ``open``) refreshes them naturally.
 */
function refreshSnapshots(): void {
  fetchSession()
    .then((info) => (live.session = info))
    .catch((err) => console.warn('GET /api/session failed', err));
  fetchControlState()
    .then((s) => (live.micEnabled = s.mic_enabled))
    .catch((err) => console.warn('GET /api/control/state failed', err));
}

/**
 * Idempotently open the event stream and start refreshing session info.
 * Safe to call from +layout.svelte's onMount; repeated calls are no-ops.
 * Returns a cleanup function the caller can wire into onMount's teardown.
 */
export function startLiveStream(): () => void {
  if (stream) return () => {};
  stream = new EventStream(onEvent, (next) => {
    // Refresh REST snapshots on every transition into ``open`` -- covers
    // first connect, reconnect after a network blip, and reconnect after
    // a backend reload (post-/api/config/reload). The previous-state
    // guard keeps duplicate transitions (open -> open) from re-firing.
    if (next === 'open' && live.connection !== 'open') {
      refreshSnapshots();
    }
    live.connection = next;
  });
  stream.connect();
  return () => {
    stream?.close();
    stream = null;
  };
}
