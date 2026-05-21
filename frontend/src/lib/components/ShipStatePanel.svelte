<script lang="ts">
  import type { ShipStateEvent } from '$lib/api';

  let { state }: { state: ShipStateEvent | null } = $props();

  function prettyOrder(o: string): string {
    return o.replace(/_/g, ' ');
  }
</script>

<section class="panel">
  <h2>Ship state</h2>
  {#if state}
    <div class="grid">
      <div class="cell">
        <div class="label">Heading</div>
        <div class="value mono">{Math.round(state.heading_deg)}°</div>
      </div>
      <div class="cell">
        <div class="label">Speed</div>
        <div class="value mono">{state.speed_kn.toFixed(1)} <small>kn</small></div>
      </div>
      <div class="cell wide">
        <div class="label">Engine</div>
        <div class="value mono">{prettyOrder(state.engine_order)}</div>
      </div>
    </div>
  {:else}
    <div class="empty">No state reported yet.</div>
  {/if}
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.75rem 0;
    font-weight: 600;
  }
  .empty { color: var(--fg-muted); font-style: italic; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  .cell {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
  }
  .cell.wide { grid-column: 1 / -1; }
  .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin-bottom: 0.25rem;
  }
  .value { font-size: 1.5rem; font-weight: 600; }
  small { color: var(--fg-muted); font-size: 0.85rem; font-weight: 400; }
</style>
