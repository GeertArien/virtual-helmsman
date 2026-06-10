<script lang="ts">
  import { live } from '$lib/liveState.svelte';
  import ChatPanel from '$lib/components/ChatPanel.svelte';
  import BrowserAudioPanel from '$lib/components/BrowserAudioPanel.svelte';
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
    {#if live.session?.browser_audio}
      <BrowserAudioPanel />
    {/if}
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
  /* The conversation panel (the first *direct* child of .left) grows to
     fill; the chat panel pins to the bottom with its own intrinsic height.
     The `>` is load-bearing: without it the selector matches every
     :first-child anywhere in the subtree -- including the h2 inside the
     panel and the first row inside .log -- which gave them flex: 1 and
     made them fight the real grow-target for space. */
  .left > :global(:first-child) { flex: 1; min-height: 0; }
  @media (max-width: 800px) {
    main {
      grid-template-columns: 1fr;
      height: auto;
    }
  }
</style>
