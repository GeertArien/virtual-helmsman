<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ApiError, fetchPending, type PendingBatch } from '$lib/api';
  import UploadDialog from '$lib/components/UploadDialog.svelte';

  type LoadState =
    | { kind: 'loading' }
    | { kind: 'ready'; batches: PendingBatch[] }
    | { kind: 'error'; message: string };

  /** Poll cadence while sitting on the list. Matches the contract example. */
  const POLL_MS = 3000;

  let load = $state<LoadState>({ kind: 'loading' });
  let lastUpdated = $state<Date | null>(null);
  let pollHandle: ReturnType<typeof setTimeout> | null = null;
  let uploadOpen = $state(false);

  async function refresh() {
    try {
      const body = await fetchPending();
      load = { kind: 'ready', batches: body.batches };
      lastUpdated = new Date();
    } catch (err) {
      load = {
        kind: 'error',
        message:
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : err instanceof Error
              ? err.message
              : 'Could not load pending batches'
      };
    }
  }

  function startPolling() {
    const tick = async () => {
      await refresh();
      pollHandle = setTimeout(tick, POLL_MS);
    };
    void tick();
  }

  function stopPolling() {
    if (pollHandle) clearTimeout(pollHandle);
    pollHandle = null;
  }

  function fmtAge(iso: string): string {
    const t = new Date(iso).getTime();
    if (Number.isNaN(t)) return iso;
    const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
    return new Date(iso).toLocaleString(undefined, { hour12: false });
  }

  onMount(startPolling);
  onDestroy(stopPolling);
</script>

<main>
  <header class="toolbar">
    <div class="title">
      <h1>Pending review</h1>
      {#if load.kind === 'ready'}
        <span class="count">{load.batches.length} batch{load.batches.length === 1 ? '' : 'es'}</span>
      {/if}
      {#if lastUpdated}
        <span class="updated mono">updated {fmtAge(lastUpdated.toISOString())}</span>
      {/if}
    </div>
    <div class="actions">
      <button type="button" class="ghost" onclick={refresh}>Refresh</button>
      <button type="button" class="primary" onclick={() => (uploadOpen = true)}>+ New upload</button>
    </div>
  </header>

  {#if load.kind === 'loading'}
    <div class="empty">Loading pending batches…</div>
  {:else if load.kind === 'error'}
    <div class="status err" role="alert">
      <strong>Could not load pending batches.</strong> {load.message}
    </div>
  {:else if load.batches.length === 0}
    <div class="empty">No chunks waiting for review.</div>
  {:else}
    <ul class="batches">
      {#each load.batches as batch (batch.batch_id)}
        <li>
          <a class="batch" href={`/review/${encodeURIComponent(batch.batch_id)}`}>
            <div class="batch-main">
              <div class="filename">{batch.filename}</div>
              <div class="meta mono">
                <span>{batch.pending_chunk_count} chunk{batch.pending_chunk_count === 1 ? '' : 's'}</span>
                <span>·</span>
                <span>{fmtAge(batch.created_at)}</span>
                <span>·</span>
                <span title="qdrant collection">{batch.collection_name}</span>
              </div>
            </div>
            <div class="arrow" aria-hidden="true">→</div>
          </a>
        </li>
      {/each}
    </ul>
  {/if}
</main>

<UploadDialog bind:open={uploadOpen} onUploaded={refresh} />

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    height: calc(100vh - 3.5rem);
    min-height: 0;
    max-width: 60rem;
    margin: 0 auto;
    width: 100%;
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
  }
  .title { display: flex; align-items: baseline; gap: 0.75rem; }
  h1 { margin: 0; font-size: 1.05rem; font-weight: 600; }
  .count { color: var(--fg-muted); font-size: 0.85rem; }
  .updated { color: var(--fg-muted); font-size: 0.75rem; }
  .actions { display: flex; gap: 0.4rem; }

  button {
    border-radius: 4px;
    padding: 0.4rem 0.85rem;
    font-size: 0.85rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
    color: inherit;
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #08111e; font-weight: 600; }
  button.primary:hover { filter: brightness(1.1); }
  button.ghost { color: var(--fg-muted); }
  button.ghost:hover { color: var(--fg); }

  .empty {
    padding: 1rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .status {
    padding: 0.6rem 0.85rem;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }

  .batches { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.4rem; overflow-y: auto; min-height: 0; }
  .batch {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: inherit;
    text-decoration: none;
    transition: background 80ms ease, border-color 80ms ease;
  }
  .batch:hover { background: var(--bg-elev-2); border-color: var(--accent); }
  .batch-main { min-width: 0; flex: 1; }
  .filename { font-size: 0.95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { color: var(--fg-muted); font-size: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.2rem; }
  .arrow { color: var(--fg-muted); font-size: 1rem; }
</style>
