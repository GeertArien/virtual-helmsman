<script lang="ts">
  import type { ConnectionState, SessionInfo } from '$lib/api';

  let { session, state }: { session: SessionInfo | null; state: ConnectionState } = $props();

  const stateLabel = $derived(
    state === 'open' ? 'connected' : state === 'connecting' ? 'connecting…' : 'disconnected'
  );
</script>

<header>
  <div class="brand">
    <span class="badge" data-state={state} title={stateLabel}></span>
    <h1>Virtual Helmsman</h1>
    <span class="state mono">{stateLabel}</span>
  </div>
  <dl class="meta">
    {#if session}
      <div><dt>session</dt><dd class="mono">{session.session_id.slice(0, 8)}</dd></div>
      <div><dt>llm</dt><dd class="mono">{session.llm_model}</dd></div>
      <div><dt>stt</dt><dd class="mono">{session.stt_backend}</dd></div>
      <div><dt>tts</dt><dd class="mono">{session.tts_backend}</dd></div>
      <div><dt>sim</dt><dd class="mono">{session.simulator_backend}</dd></div>
    {:else}
      <div><dt>session</dt><dd class="mono">—</dd></div>
    {/if}
  </dl>
</header>

<style>
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.75rem 1.25rem;
    background: var(--bg-elev);
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 0.75rem; }
  h1 { font-size: 1rem; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
  .state { color: var(--fg-muted); font-size: 0.8rem; }
  .badge {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--bad);
  }
  .badge[data-state='open'] { background: var(--good); box-shadow: 0 0 8px var(--good); }
  .badge[data-state='connecting'] { background: var(--warn); }
  .meta { display: flex; flex-wrap: wrap; gap: 0.5rem 1.25rem; margin: 0; }
  .meta div { display: flex; align-items: baseline; gap: 0.4rem; }
  dt { color: var(--fg-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
  dd { margin: 0; font-size: 0.85rem; }
</style>
