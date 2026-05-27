<script lang="ts">
  import { onMount } from 'svelte';
  import { ApiError, fetchAuditLog, type AuditEntry } from '$lib/api';

  type LoadState =
    | { kind: 'loading' }
    | { kind: 'ready'; entries: AuditEntry[]; totalInLog: number }
    | { kind: 'error'; message: string };

  /** Default page size. The contract caps at 500 server-side; 50 is the
   *  sensible recent-activity tail. */
  const DEFAULT_LIMIT = 50;

  /** The filter set surfaced in the UI. The contract lists three known
   *  values; we also offer "all" (no filter). New `actie` values will
   *  still render — they just won't be selectable here until added. */
  const ACTIE_FILTERS: { value: string; label: string }[] = [
    { value: '', label: 'All' },
    { value: 'ingestie_hitl', label: 'HITL ingestion' },
    { value: 'verwijdering', label: 'Deletion' },
    { value: 'vraag', label: 'Question' }
  ];

  let { refreshKey = 0 }: { refreshKey?: number } = $props();

  let load = $state<LoadState>({ kind: 'loading' });
  let lastUpdated = $state<Date | null>(null);
  let filterActie = $state<string>('');

  async function refresh() {
    try {
      const body = await fetchAuditLog({
        limit: DEFAULT_LIMIT,
        actie: filterActie || undefined
      });
      load = { kind: 'ready', entries: body.entries, totalInLog: body.total_in_log };
      lastUpdated = new Date();
    } catch (err) {
      load = {
        kind: 'error',
        message:
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : err instanceof Error
              ? err.message
              : 'Could not load audit log'
      };
    }
  }

  /** Re-fetch whenever the parent bumps `refreshKey` (e.g. after a successful
   *  submit upstream) or when the user toggles the filter. */
  $effect(() => {
    void refreshKey;
    void filterActie;
    void refresh();
  });

  /** Classify the entry by the leading word of `resultaat`. The legacy Dutch
   *  patterns are "Succes — …" and "Fout — …"; everything else renders neutral. */
  function outcomeOf(entry: AuditEntry): 'ok' | 'fail' | 'neutral' {
    const r = entry.resultaat?.trim().toLowerCase() ?? '';
    if (r.startsWith('succes')) return 'ok';
    if (r.startsWith('fout')) return 'fail';
    return 'neutral';
  }

  function fmtDate(iso: string): string {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { hour12: false });
  }

  function fmtAge(iso: string): string {
    const t = new Date(iso).getTime();
    if (Number.isNaN(t)) return iso;
    const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
    return fmtDate(iso);
  }

  onMount(refresh);
</script>

<section class="panel">
  <div class="header">
    <h2>Recent activity</h2>
    <div class="header-meta">
      <label class="filter">
        <span class="filter-label">filter</span>
        <select bind:value={filterActie}>
          {#each ACTIE_FILTERS as opt (opt.value)}
            <option value={opt.value}>{opt.label}</option>
          {/each}
        </select>
      </label>
      {#if load.kind === 'ready'}
        <span class="count mono">
          {load.entries.length} of {load.totalInLog} row{load.totalInLog === 1 ? '' : 's'}
        </span>
      {/if}
      {#if lastUpdated}
        <span class="updated mono">updated {fmtAge(lastUpdated.toISOString())}</span>
      {/if}
      <button type="button" class="ghost" onclick={refresh}>Refresh</button>
    </div>
  </div>
  <p class="hint">
    Outcomes of the ingestion pipeline as logged by n8n
    (<code>audit-log-maritime</code>). Newest first; capped at {DEFAULT_LIMIT} rows.
  </p>

  {#if load.kind === 'loading'}
    <div class="empty">Loading audit log…</div>
  {:else if load.kind === 'error'}
    <div class="status err" role="alert">
      <strong>Could not load audit log.</strong> {load.message}
    </div>
  {:else if load.entries.length === 0}
    <div class="empty">No audit entries yet.</div>
  {:else}
    <ol class="entries">
      {#each load.entries as entry (entry.id)}
        {@const outcome = outcomeOf(entry)}
        <li class="entry" data-outcome={outcome}>
          <div class="entry-head">
            <span class="dot" aria-hidden="true"></span>
            <span class="document">{entry.document_naam}</span>
            <span class="actie mono">{entry.actie}</span>
            <span class="when mono" title={fmtDate(entry.createdAt)}>{fmtAge(entry.createdAt)}</span>
          </div>
          <div class="resultaat">{entry.resultaat}</div>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    min-height: 0;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0;
    font-weight: 600;
  }
  .header-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .filter { display: inline-flex; align-items: center; gap: 0.35rem; }
  .filter-label { font-size: 0.7rem; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .filter select {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    color: var(--fg);
    font-size: 0.8rem;
    outline: none;
  }
  .filter select:focus { border-color: var(--accent); }
  .count { color: var(--fg-muted); font-size: 0.8rem; }
  .updated { color: var(--fg-muted); font-size: 0.72rem; }
  .hint { margin: 0; color: var(--fg-muted); font-size: 0.85rem; }
  .hint code {
    background: var(--bg-elev-2);
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.85em;
  }

  button {
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-size: 0.8rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
    color: inherit;
  }
  button.ghost { color: var(--fg-muted); }
  button.ghost:hover { color: var(--fg); }

  .empty {
    padding: 0.75rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .status { padding: 0.6rem 0.85rem; border-radius: 4px; border: 1px solid var(--border); }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }

  .entries {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    /* The audit log can grow long; cap height with internal scroll so it
       doesn't push other panels off-screen. */
    max-height: 22rem;
    overflow-y: auto;
  }
  .entry {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-left-width: 3px;
    border-radius: 4px;
    padding: 0.45rem 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .entry[data-outcome='ok']      { border-left-color: var(--good); }
  .entry[data-outcome='fail']    { border-left-color: var(--bad); }
  .entry[data-outcome='neutral'] { border-left-color: var(--fg-muted); }

  .entry-head {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--fg-muted);
    flex-shrink: 0;
    align-self: center;
  }
  .entry[data-outcome='ok'] .dot   { background: var(--good); }
  .entry[data-outcome='fail'] .dot { background: var(--bad); }
  .document {
    font-size: 0.9rem;
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 28ch;
  }
  .actie {
    font-size: 0.7rem;
    color: var(--fg-muted);
    padding: 0.05rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--bg-elev);
  }
  .when {
    margin-left: auto;
    color: var(--fg-muted);
    font-size: 0.72rem;
  }
  .resultaat {
    color: var(--fg);
    font-size: 0.82rem;
    line-height: 1.4;
    word-break: break-word;
  }
</style>
