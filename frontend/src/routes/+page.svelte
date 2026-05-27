<script lang="ts">
  import { live } from '$lib/liveState.svelte';
  import ChatPanel from '$lib/components/ChatPanel.svelte';
  import ConversationPanel from '$lib/components/ConversationPanel.svelte';
  import ShipStatePanel from '$lib/components/ShipStatePanel.svelte';
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';

  // Live state is owned by +layout.svelte (via liveState.svelte.ts) so it
  // persists across route navigation. This page just renders the dashboard.
</script>

<main>
  <section class="left">
    <ConversationPanel entries={live.entries} />
    <ChatPanel />
  </section>
  <section class="right">
    <ShipStatePanel state={live.shipState} />
    <MetricsPanel turns={live.turnMetrics} />
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
  /* The conversation panel grows to fill, the chat panel pins to the bottom
     of the column with its own intrinsic height. */
  .left :global(:first-child) { flex: 1; min-height: 0; }
  @media (max-width: 800px) {
    main {
      grid-template-columns: 1fr;
      height: auto;
    }
  }
</style>
