<script lang="ts">
  import { onMount } from 'svelte';
  import {
    ApiError,
    deleteDocument,
    listDocuments,
    type DocumentInfo
  } from '$lib/api';

  type LoadState =
    | { kind: 'loading' }
    | { kind: 'ready'; documents: DocumentInfo[] }
    | { kind: 'error'; message: string };

  /** Per-row delete lifecycle, keyed by document_id. */
  type RowState =
    | { kind: 'idle' }
    | { kind: 'confirm' }
    | { kind: 'deleting' }
    | { kind: 'error'; message: string };

  let load = $state<LoadState>({ kind: 'loading' });
  let rows = $state<Record<string, RowState>>({});

  async function refresh() {
    load = { kind: 'loading' };
    try {
      const documents = await listDocuments();
      load = { kind: 'ready', documents };
      rows = {};
    } catch (err) {
      load = { kind: 'error', message: errMessage(err) };
    }
  }

  function errMessage(err: unknown): string {
    if (err instanceof ApiError) return `${err.message} (HTTP ${err.status})`;
    if (err instanceof Error) return err.message;
    return 'Request failed';
  }

  function setRow(id: string, state: RowState) {
    rows = { ...rows, [id]: state };
  }

  function rowOf(id: string): RowState {
    return rows[id] ?? { kind: 'idle' };
  }

  function askConfirm(id: string) {
    setRow(id, { kind: 'confirm' });
  }

  function cancelConfirm(id: string) {
    setRow(id, { kind: 'idle' });
  }

  async function confirmDelete(doc: DocumentInfo) {
    setRow(doc.document_id, { kind: 'deleting' });
    try {
      const res = await deleteDocument(doc.document_id);
      // Optimistically drop the row from the list -- avoids a full refetch.
      if (load.kind === 'ready') {
        load = {
          kind: 'ready',
          documents: load.documents.filter((d) => d.document_id !== doc.document_id)
        };
      }
      // Stash a short toast via the row state -- but since the row is gone,
      // surface it as a top-level notice instead.
      lastDeleted = { name: doc.title ?? doc.document_id, chunks: res.deleted_chunks };
    } catch (err) {
      setRow(doc.document_id, { kind: 'error', message: errMessage(err) });
    }
  }

  let lastDeleted = $state<{ name: string; chunks: number } | null>(null);

  function dismissDeletedNotice() {
    lastDeleted = null;
  }

  function fmtDate(iso: string | null): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { hour12: false });
  }

  onMount(() => {
    refresh();
  });
</script>

<section class="panel">
  <div class="header">
    <h2>Delete document</h2>
    <button type="button" class="ghost" onclick={refresh} disabled={load.kind === 'loading'}>
      {load.kind === 'loading' ? 'Refreshing…' : 'Refresh'}
    </button>
  </div>
  <p class="hint">
    Removes every vector chunk that shares a <code>document_id</code> from qdrant.
    This is a direct delete — there's no review step.
  </p>

  {#if lastDeleted}
    <div class="status ok" role="status">
      Deleted <strong>{lastDeleted.name}</strong> · {lastDeleted.chunks} chunk{lastDeleted.chunks === 1 ? '' : 's'} removed.
      <button type="button" class="dismiss" onclick={dismissDeletedNotice} aria-label="Dismiss">×</button>
    </div>
  {/if}

  {#if load.kind === 'loading'}
    <div class="empty">Loading documents…</div>
  {:else if load.kind === 'error'}
    <div class="status err" role="alert">
      <strong>Could not load documents.</strong> {load.message}
    </div>
  {:else if load.documents.length === 0}
    <div class="empty">No documents in the collection.</div>
  {:else}
    <ul class="docs">
      {#each load.documents as doc (doc.document_id)}
        {@const row = rowOf(doc.document_id)}
        <li class="doc">
          <div class="doc-main">
            <div class="title">{doc.title ?? doc.document_id}</div>
            <div class="meta mono">
              <span title="qdrant document_id">{doc.document_id}</span>
              <span>·</span>
              <span>{doc.chunk_count} chunk{doc.chunk_count === 1 ? '' : 's'}</span>
              {#if doc.source}<span>·</span><span class="source" title={doc.source}>{doc.source}</span>{/if}
              {#if doc.uploaded_at}<span>·</span><span>{fmtDate(doc.uploaded_at)}</span>{/if}
            </div>
          </div>
          <div class="doc-actions">
            {#if row.kind === 'idle'}
              <button type="button" class="danger" onclick={() => askConfirm(doc.document_id)}>
                Delete
              </button>
            {:else if row.kind === 'confirm'}
              <span class="confirm-text">Delete all chunks?</span>
              <button type="button" class="danger" onclick={() => confirmDelete(doc)}>Yes, delete</button>
              <button type="button" class="ghost" onclick={() => cancelConfirm(doc.document_id)}>Cancel</button>
            {:else if row.kind === 'deleting'}
              <span class="busy">Deleting…</span>
            {:else if row.kind === 'error'}
              <span class="err-text" title={row.message}>{row.message}</span>
              <button type="button" class="ghost" onclick={() => setRow(doc.document_id, { kind: 'idle' })}>
                Dismiss
              </button>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
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
    gap: 0.75rem;
    min-height: 0;
  }
  .header { display: flex; justify-content: space-between; align-items: center; }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0;
    font-weight: 600;
  }
  .hint { margin: 0; color: var(--fg-muted); font-size: 0.85rem; }
  .hint code {
    background: var(--bg-elev-2);
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.85em;
  }
  .empty { color: var(--fg-muted); font-style: italic; padding: 0.5rem 0; }

  .docs {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    overflow-y: auto;
    min-height: 0;
  }
  .doc {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.75rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .doc-main { min-width: 0; flex: 1; }
  .title {
    font-size: 0.95rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    color: var(--fg-muted);
    font-size: 0.75rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.2rem;
  }
  .source { max-width: 30ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-actions { display: flex; align-items: center; gap: 0.4rem; }

  button {
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-size: 0.8rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
    color: inherit;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.danger { color: var(--bad); border-color: var(--bad); background: transparent; }
  button.danger:hover:not(:disabled) { background: rgba(248, 81, 73, 0.1); }
  button.ghost { color: var(--fg-muted); }
  button.ghost:hover:not(:disabled) { color: var(--fg); }

  .confirm-text { color: var(--warn); font-size: 0.8rem; }
  .busy { color: var(--fg-muted); font-size: 0.8rem; font-style: italic; }
  .err-text {
    color: var(--bad);
    font-size: 0.8rem;
    max-width: 22ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .status {
    position: relative;
    padding: 0.5rem 2rem 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.85rem;
    border: 1px solid var(--border);
  }
  .status.ok { border-color: var(--good); background: rgba(63, 185, 80, 0.08); }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }
  .dismiss {
    position: absolute;
    right: 0.25rem;
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: var(--fg-muted);
    font-size: 1rem;
    line-height: 1;
    padding: 0.1rem 0.4rem;
  }
  .dismiss:hover { color: var(--fg); }
</style>
