<script lang="ts">
  import { tick } from 'svelte';
  import type { Entry } from './conversation';

  let { entries }: { entries: Entry[] } = $props();

  /** The scrollable log element; bound below so the auto-scroll effect can
   *  drive it. ``undefined`` until first mount. */
  let log: HTMLDivElement | undefined = $state();

  /** ``true`` when the user is at (or within a few pixels of) the bottom
   *  of the log. We only auto-scroll when this is true so a user reading
   *  history isn't yanked back to the bottom by every new event. The
   *  initial value is ``true`` because an empty log is "at the bottom"
   *  by definition. */
  let stuckToBottom = $state(true);

  /** Reasonable slack so a sub-pixel rounding error doesn't unstick us. */
  const STICKY_THRESHOLD_PX = 16;

  function onScroll() {
    if (!log) return;
    const distanceFromBottom = log.scrollHeight - log.scrollTop - log.clientHeight;
    stuckToBottom = distanceFromBottom <= STICKY_THRESHOLD_PX;
  }

  /** Whenever the entries list grows, re-scroll to the bottom -- but only
   *  if the user was already there. The ``tick()`` await lets the DOM
   *  apply the new row before we measure scrollHeight. */
  $effect(() => {
    void entries.length; // tracked dep
    if (!log || !stuckToBottom) return;
    void tick().then(() => {
      if (log && stuckToBottom) log.scrollTop = log.scrollHeight;
    });
  });

  function fmtTime(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }
</script>

<section class="panel">
  <h2>Conversation</h2>
  <div
    class="log"
    role="log"
    aria-live="polite"
    bind:this={log}
    onscroll={onScroll}
  >
    {#if entries.length === 0}
      <div class="empty">Waiting for the captain's orders…</div>
    {:else}
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
    /* The panel's own flex container; .log below gets flex: 1 so it owns
       the scroll. h2 stays an intrinsic-height header. */
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.5rem 0;
    font-weight: 600;
  }

  /* Plain block layout inside the scroller -- *not* flex. Flex with one
     child used to produce a weird "single message centered" artifact on
     first render in some layouts (gap interacting with flex-shrink on
     a row that doesn't fill the cross-axis). A normal block column
     stacks rows from the top, end of story, and the auto-scroll effect
     pins us to the bottom whenever the user is already there. */
  .log {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    /* Scroll-anchoring keeps the viewport position stable when content
       is appended *above* the visible area -- a defensive default for
       cases where the user has scrolled up. */
    overflow-anchor: auto;
  }

  .row + .row { margin-top: 0.3rem; }

  .empty {
    color: var(--fg-muted);
    font-style: italic;
    padding: 0.5rem 0;
  }

  .row {
    display: grid;
    grid-template-columns: 5rem 5.5rem 1fr;
    gap: 0.5rem;
    align-items: baseline;
    padding: 0.15rem 0;
  }
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
