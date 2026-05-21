<script lang="ts">
  import type { Entry } from './conversation';

  let { entries }: { entries: Entry[] } = $props();

  function fmtTime(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }
</script>

<section class="panel">
  <h2>Conversation</h2>
  <div class="log" role="log" aria-live="polite">
    {#each entries as e (e.ts + e.kind + ('text' in e ? e.text : 'label' in e ? e.label : e.reason))}
      <div class="row" data-kind={e.kind}>
        <span class="ts mono">{fmtTime(e.ts)}</span>
        {#if e.kind === 'user'}
          <span class="who">captain</span>
          <span class="text">{e.text}</span>
        {:else if e.kind === 'assistant'}
          <span class="who">helmsman</span>
          <span class="text">{e.text}</span>
        {:else if e.kind === 'action'}
          <span class="who">action</span>
          <span class="text mono" data-ok={e.ok}>{e.label}</span>
        {:else}
          <span class="who">refused</span>
          <span class="text refused">{e.reason}</span>
        {/if}
      </div>
    {/each}
    {#if entries.length === 0}
      <div class="empty">Waiting for the captain's orders…</div>
    {/if}
  </div>
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.5rem 0;
    font-weight: 600;
  }
  .log {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .empty { color: var(--fg-muted); font-style: italic; padding: 1rem 0; }
  .row { display: grid; grid-template-columns: 5rem 5.5rem 1fr; gap: 0.5rem; align-items: baseline; padding: 0.15rem 0; }
  .ts { color: var(--fg-muted); font-size: 0.75rem; }
  .who {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
  }
  .row[data-kind='user'] .who { color: var(--accent); }
  .row[data-kind='assistant'] .who { color: var(--good); }
  .row[data-kind='action'] .who { color: var(--warn); }
  .row[data-kind='refused'] .who { color: var(--bad); }
  .text { color: var(--fg); font-size: 0.9rem; }
  .refused { color: var(--fg-muted); }
</style>
