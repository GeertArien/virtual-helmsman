<script lang="ts">
  import { onMount } from 'svelte';
  import { EventStream, fetchSession } from '$lib/api';
  import type {
    AgentEvent,
    ConnectionState,
    ShipStateEvent,
    SessionInfo,
    TurnMetricsEvent
  } from '$lib/api';
  import Header from '$lib/components/Header.svelte';
  import ConversationPanel from '$lib/components/ConversationPanel.svelte';
  import type { Entry } from '$lib/components/conversation';
  import ShipStatePanel from '$lib/components/ShipStatePanel.svelte';
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';

  // Single page owns all live state. Components stay pure.
  let connection = $state<ConnectionState>('connecting');
  let session = $state<SessionInfo | null>(null);
  let entries = $state<Entry[]>([]);
  let shipState = $state<ShipStateEvent | null>(null);
  let turnMetrics = $state<TurnMetricsEvent[]>([]);

  /** Bounded log so a long-running session doesn't grow the DOM forever. */
  const MAX_ENTRIES = 500;

  function append(entry: Entry) {
    const next = entries.concat(entry);
    entries = next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
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
        shipState = ev;
        break;
      case 'turn_metrics':
        turnMetrics = turnMetrics.concat(ev).slice(-200);
        break;
    }
  }

  onMount(() => {
    fetchSession()
      .then((info) => (session = info))
      .catch((err) => console.warn('GET /api/session failed', err));

    const stream = new EventStream(onEvent, (s) => (connection = s));
    stream.connect();
    return () => stream.close();
  });
</script>

<Header {session} state={connection} />

<main>
  <section class="left">
    <ConversationPanel {entries} />
  </section>
  <section class="right">
    <ShipStatePanel state={shipState} />
    <MetricsPanel turns={turnMetrics} />
  </section>
</main>

<style>
  main {
    display: grid;
    grid-template-columns: 1fr 22rem;
    gap: 0.75rem;
    padding: 0.75rem;
    height: calc(100vh - 3.5rem); /* viewport minus header */
    min-height: 0;
  }
  .left, .right {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-height: 0;
  }
  .left { min-height: 0; }
  @media (max-width: 800px) {
    main {
      grid-template-columns: 1fr;
      height: auto;
    }
  }
</style>
