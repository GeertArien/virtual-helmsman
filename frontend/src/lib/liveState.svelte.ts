/**
 * App-wide live agent state, owned at the layout level so it survives route
 * changes (no WS reconnect when navigating between /, /documents, etc.).
 *
 * The single $state object is mutated by the WebSocket handler; components
 * read its fields directly. Treat the returned `live` proxy as the single
 * source of truth for live data -- do not duplicate fields into local state.
 */

import { EventStream, fetchSession, fetchSimulatorState } from './api';
import type {
  AgentEvent,
  ConnectionState,
  SessionInfo,
  ShipStateEvent,
  SimulatorConnectionState,
  TurnMetricsEvent
} from './api';
import type { Entry } from './components/conversation';

/** Bounded conversation log so a long-running session doesn't grow forever. */
const MAX_ENTRIES = 500;
/** Bounded metrics history (~200 turns ≈ many minutes of conversation). */
const MAX_TURNS = 200;

export const live = $state({
  /** Browser -> backend WebSocket. Not the same thing as `simulator` below:
   *  this dashboard can be perfectly connected to a backend that has lost the
   *  ship. */
  connection: 'connecting' as ConnectionState,
  /** Backend -> simulator link. `null` until first known (the backend may not
   *  expose the link routes at all, e.g. no simulator wired in). */
  simulator: null as SimulatorConnectionState | null,
  session: null as SessionInfo | null,
  entries: [] as Entry[],
  shipState: null as ShipStateEvent | null,
  turnMetrics: [] as TurnMetricsEvent[]
});

let stream: EventStream | null = null;

function append(entry: Entry) {
  const next = live.entries.concat(entry);
  live.entries = next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
}

function actionLabel(action: string, details: Record<string, unknown>): string {
  // Helmsman action vocabulary -- map each type to a one-line readable summary
  // for the conversation panel. Keep the prefix matching the action.type
  // so operators can grep for it in the audit log.
  if (action === 'rudder' && 'direction' in details && 'degrees' in details) {
    // A rudder action is a helm order: degrees is the rudder angle held, not a
    // heading change. Zero is midships.
    return details.degrees === 0
      ? 'rudder midships'
      : `rudder ${details.direction} ${details.degrees}°`;
  }
  if (action === 'throttle') {
    // Either form may be present: a telegraph position, or knots.
    if ('order' in details) return `telegraph ${String(details.order).replace(/_/g, ' ')}`;
    if ('speed' in details) return `throttle ${details.speed} kn`;
  }
  if (action === 'navigation' && 'course' in details) {
    return `navigation ${details.course}°`;
  }
  if (action === 'autopilot' && 'state' in details) {
    return `autopilot ${details.state}`;
  }
  if (action === 'anchor' && 'operation' in details) {
    const chain =
      'chain_length' in details ? ` (${details.chain_length}m)` : '';
    return `anchor ${details.operation}${chain}`;
  }
  if (action === 'status_query' && 'query' in details) {
    return `status_query ${details.query}`;
  }
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
    case 'connection_state':
      live.simulator = ev.state;
      break;
  }
}

/** Refresh every REST snapshot the UI shows in the header.
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
  // The link state is pushed on every *change*, so without an initial read the
  // panel would sit blank until the next transition -- which on a healthy link
  // never comes.
  fetchSimulatorState()
    .then((res) => (live.simulator = res.state))
    .catch((err) => console.warn('GET /api/control/simulator failed', err));
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
