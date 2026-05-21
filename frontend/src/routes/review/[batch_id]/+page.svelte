<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import {
    ApiError,
    fetchPending,
    submitDecisions,
    type ChunkDecision,
    type PendingBatch,
    type ReviewChunk
  } from '$lib/api';

  /** n8n silently rejects edits below this threshold -- enforce client-side. */
  const MIN_EDIT_CHARS = 50;

  type LoadState =
    | { kind: 'loading' }
    | { kind: 'ready'; batch: PendingBatch }
    | { kind: 'missing' }   // batch is no longer in the pending list
    | { kind: 'error'; message: string };

  type Action = 'approve' | 'reject' | 'edit';
  type Decision = { action: Action; edited_text: string; reason: string };

  /** Per-chunk UI state keyed by chunk_id. Default is approve. */
  let decisionsByChunk = $state<Record<string, Decision>>({});
  let load = $state<LoadState>({ kind: 'loading' });
  let submitting = $state(false);
  let submitError = $state<string | null>(null);

  const batchId = $derived(page.params.batch_id ?? '');

  async function refresh() {
    load = { kind: 'loading' };
    try {
      const body = await fetchPending();
      const found = body.batches.find((b) => b.batch_id === batchId);
      if (!found) {
        load = { kind: 'missing' };
        return;
      }
      // Initialise per-chunk decisions on first load. Preserve any in-flight
      // edits across refreshes so a poll doesn't wipe the operator's work.
      const next: Record<string, Decision> = { ...decisionsByChunk };
      for (const ch of found.chunks) {
        if (!next[ch.chunk_id]) {
          next[ch.chunk_id] = { action: 'approve', edited_text: ch.text, reason: '' };
        }
      }
      decisionsByChunk = next;
      load = { kind: 'ready', batch: found };
    } catch (err) {
      load = {
        kind: 'error',
        message:
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : err instanceof Error
              ? err.message
              : 'Could not load this batch'
      };
    }
  }

  function setAction(chunkId: string, action: Action, originalText: string) {
    const cur = decisionsByChunk[chunkId] ?? {
      action: 'approve',
      edited_text: originalText,
      reason: ''
    };
    decisionsByChunk = { ...decisionsByChunk, [chunkId]: { ...cur, action } };
  }

  function setEditedText(chunkId: string, text: string) {
    const cur = decisionsByChunk[chunkId];
    if (!cur) return;
    decisionsByChunk = { ...decisionsByChunk, [chunkId]: { ...cur, edited_text: text } };
  }

  function setReason(chunkId: string, reason: string) {
    const cur = decisionsByChunk[chunkId];
    if (!cur) return;
    decisionsByChunk = { ...decisionsByChunk, [chunkId]: { ...cur, reason } };
  }

  function decisionFor(chunk: ReviewChunk): Decision {
    return (
      decisionsByChunk[chunk.chunk_id] ?? {
        action: 'approve',
        edited_text: chunk.text,
        reason: ''
      }
    );
  }

  function isEditValid(d: Decision): boolean {
    return d.action !== 'edit' || d.edited_text.trim().length >= MIN_EDIT_CHARS;
  }

  const summary = $derived(() => {
    if (load.kind !== 'ready') return { approve: 0, reject: 0, edit: 0 };
    let approve = 0, reject = 0, edit = 0;
    for (const ch of load.batch.chunks) {
      const d = decisionFor(ch);
      if (d.action === 'approve') approve++;
      else if (d.action === 'reject') reject++;
      else edit++;
    }
    return { approve, reject, edit };
  });

  const editsBelowMinimum = $derived(() => {
    if (load.kind !== 'ready') return 0;
    return load.batch.chunks.filter((ch) => !isEditValid(decisionFor(ch))).length;
  });

  const canSubmit = $derived(
    load.kind === 'ready' && !submitting && editsBelowMinimum() === 0
  );

  async function submit() {
    if (load.kind !== 'ready' || submitting) return;
    submitError = null;

    const totals = summary();
    if (totals.reject === load.batch.chunks.length) {
      const ok = window.confirm(
        'You are rejecting every chunk in this batch. n8n will mark the batch ' +
          'as failed. Submit anyway?'
      );
      if (!ok) return;
    }

    submitting = true;
    try {
      const payload: ChunkDecision[] = load.batch.chunks.map((ch) => {
        const d = decisionFor(ch);
        const base: ChunkDecision = { chunk_id: ch.chunk_id, action: d.action };
        if (d.action === 'edit') base.edited_text = d.edited_text.trim();
        if (d.reason.trim()) base.reason = d.reason.trim();
        return base;
      });
      await submitDecisions(batchId, payload);
      // Done -- back to the list.
      await goto('/review');
    } catch (err) {
      submitError =
        err instanceof ApiError
          ? `${err.message} (HTTP ${err.status})`
          : err instanceof Error
            ? err.message
            : 'Submit failed';
      if (err instanceof ApiError && err.status === 404) {
        // The batch is gone (already resumed or stale URL). Bounce to the list.
        await goto('/review');
      }
    } finally {
      submitting = false;
    }
  }

  function fmtMeta(meta: Record<string, unknown>): string[] {
    const pairs: string[] = [];
    if (typeof meta.page === 'number') pairs.push(`page ${meta.page}`);
    if (typeof meta.chunk_length === 'number') pairs.push(`${meta.chunk_length} chars`);
    if (typeof meta.words_in_text === 'number') pairs.push(`${meta.words_in_text} words`);
    return pairs;
  }

  onMount(refresh);
</script>

<main>
  <header class="toolbar">
    <a href="/review" class="back">← Back</a>
    {#if load.kind === 'ready'}
      <div class="title">
        <div class="filename">{load.batch.filename}</div>
        <div class="meta mono">
          {load.batch.pending_chunk_count} chunk{load.batch.pending_chunk_count === 1 ? '' : 's'} ·
          collection {load.batch.collection_name} ·
          batch <span title={load.batch.batch_id}>{load.batch.batch_id.slice(0, 12)}…</span>
        </div>
      </div>
    {/if}
    <button type="button" class="ghost" onclick={refresh} disabled={load.kind === 'loading'}>
      Refresh
    </button>
  </header>

  {#if load.kind === 'loading'}
    <div class="empty">Loading batch…</div>
  {:else if load.kind === 'missing'}
    <div class="status warn" role="alert">
      <strong>Batch not pending anymore.</strong>
      Someone may have already submitted it. <a href="/review">Back to the queue</a>.
    </div>
  {:else if load.kind === 'error'}
    <div class="status err" role="alert">
      <strong>Could not load this batch.</strong> {load.message}
    </div>
  {:else}
    {#if typeof load.batch.chunks[0]?.metadata?.document_summary === 'string'}
      <div class="summary">
        <span class="summary-label">summary</span>
        <span>{load.batch.chunks[0].metadata.document_summary}</span>
      </div>
    {/if}

    <section class="chunks">
      {#each load.batch.chunks as chunk (chunk.chunk_id)}
        {@const d = decisionFor(chunk)}
        {@const editTooShort = d.action === 'edit' && !isEditValid(d)}
        {@const metaPairs = fmtMeta(chunk.metadata)}
        <article class="chunk" data-action={d.action}>
          <header class="chunk-header">
            <div class="chunk-id mono">{chunk.chunk_id}</div>
            <div class="chunk-meta mono">{metaPairs.join(' · ')}</div>
            <div class="chunk-actions" role="radiogroup" aria-label="Decision">
              <button
                type="button"
                class="pill approve"
                aria-pressed={d.action === 'approve'}
                onclick={() => setAction(chunk.chunk_id, 'approve', chunk.text)}
              >approve</button>
              <button
                type="button"
                class="pill edit"
                aria-pressed={d.action === 'edit'}
                onclick={() => setAction(chunk.chunk_id, 'edit', chunk.text)}
              >edit</button>
              <button
                type="button"
                class="pill reject"
                aria-pressed={d.action === 'reject'}
                onclick={() => setAction(chunk.chunk_id, 'reject', chunk.text)}
              >reject</button>
            </div>
          </header>

          {#if d.action === 'edit'}
            <textarea
              class="text editable"
              rows="6"
              value={d.edited_text}
              oninput={(e) => setEditedText(chunk.chunk_id, (e.currentTarget as HTMLTextAreaElement).value)}
            ></textarea>
            <div class="edit-meta mono" class:warn={editTooShort}>
              {d.edited_text.trim().length} / {MIN_EDIT_CHARS} chars minimum
              {#if editTooShort}— too short, edit will be dropped{/if}
            </div>
          {:else}
            <pre class="text">{chunk.text}</pre>
          {/if}

          {#if d.action !== 'approve'}
            <label class="reason">
              <span class="reason-label">Reason (optional)</span>
              <input
                type="text"
                value={d.reason}
                oninput={(e) => setReason(chunk.chunk_id, (e.currentTarget as HTMLInputElement).value)}
                placeholder={d.action === 'reject' ? 'why this chunk should not be ingested' : 'what you changed and why'}
              />
            </label>
          {/if}
        </article>
      {/each}
    </section>

    <footer class="submitbar">
      <div class="summary-line mono">
        <span class="tag approve">{summary().approve} approve</span>
        <span class="tag edit">{summary().edit} edit</span>
        <span class="tag reject">{summary().reject} reject</span>
        {#if editsBelowMinimum() > 0}
          <span class="tag warn">{editsBelowMinimum()} edit{editsBelowMinimum() === 1 ? '' : 's'} below {MIN_EDIT_CHARS} chars</span>
        {/if}
      </div>
      {#if submitError}
        <div class="status err inline">{submitError}</div>
      {/if}
      <div class="submit-actions">
        <a href="/review" class="ghost-link">Cancel</a>
        <button type="button" class="primary" onclick={submit} disabled={!canSubmit}>
          {submitting ? 'Submitting…' : 'Submit decisions'}
        </button>
      </div>
    </footer>
  {/if}
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 0.75rem;
    min-height: 0;
    max-width: 60rem;
    margin: 0 auto;
    width: 100%;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .back {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .back:hover { background: var(--bg-elev-2); }
  .title { flex: 1; min-width: 0; }
  .filename { font-size: 1rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { color: var(--fg-muted); font-size: 0.75rem; }
  .ghost { background: var(--bg-elev-2); border: 1px solid var(--border); color: var(--fg-muted); border-radius: 4px; padding: 0.35rem 0.7rem; cursor: pointer; }
  .ghost:hover:not(:disabled) { color: var(--fg); }
  .ghost:disabled { opacity: 0.5; cursor: not-allowed; }

  .empty { padding: 1rem; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 4px; color: var(--fg-muted); font-style: italic; }
  .status { padding: 0.6rem 0.85rem; border-radius: 4px; border: 1px solid var(--border); font-size: 0.85rem; }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }
  .status.warn { border-color: var(--warn); background: rgba(210, 153, 34, 0.08); }
  .status.inline { font-size: 0.8rem; padding: 0.4rem 0.6rem; }

  .summary {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    color: var(--fg-muted);
    font-size: 0.85rem;
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
  }
  .summary-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }

  .chunks {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    overflow-y: auto;
    min-height: 0;
  }
  .chunk {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-left-width: 3px;
    border-radius: 4px;
    padding: 0.6rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .chunk[data-action='approve'] { border-left-color: var(--good); }
  .chunk[data-action='edit']    { border-left-color: var(--warn); }
  .chunk[data-action='reject']  { border-left-color: var(--bad); }

  .chunk-header { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
  .chunk-id { color: var(--accent); font-size: 0.8rem; font-weight: 600; }
  .chunk-meta { color: var(--fg-muted); font-size: 0.75rem; flex: 1; min-width: 0; }
  .chunk-actions { display: flex; gap: 0.25rem; }
  .pill {
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    cursor: pointer;
    color: var(--fg-muted);
    text-transform: lowercase;
  }
  .pill[aria-pressed='true'] { color: var(--fg); }
  .pill.approve[aria-pressed='true'] { background: var(--good); border-color: var(--good); color: #082014; }
  .pill.edit[aria-pressed='true']    { background: var(--warn); border-color: var(--warn); color: #1f1605; }
  .pill.reject[aria-pressed='true']  { background: var(--bad);  border-color: var(--bad);  color: #200707; }

  .text {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    color: var(--fg);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.6rem 0.75rem;
    line-height: 1.45;
    max-height: 14rem;
    overflow-y: auto;
  }
  textarea.editable {
    resize: vertical;
    min-height: 8rem;
    max-height: none;
    outline: none;
  }
  textarea.editable:focus { border-color: var(--accent); }
  .edit-meta { font-size: 0.72rem; color: var(--fg-muted); }
  .edit-meta.warn { color: var(--bad); }

  .reason { display: flex; flex-direction: column; gap: 0.25rem; }
  .reason-label { font-size: 0.7rem; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .reason input {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.35rem 0.55rem;
    color: var(--fg);
    outline: none;
  }
  .reason input:focus { border-color: var(--accent); }

  .submitbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    position: sticky;
    bottom: 0;
    flex-wrap: wrap;
  }
  .summary-line { display: flex; gap: 0.4rem; flex-wrap: wrap; font-size: 0.75rem; }
  .tag {
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    color: var(--fg-muted);
  }
  .tag.approve { color: var(--good); border-color: var(--good); }
  .tag.edit    { color: var(--warn); border-color: var(--warn); }
  .tag.reject  { color: var(--bad);  border-color: var(--bad); }
  .tag.warn    { color: var(--warn); border-color: var(--warn); background: rgba(210, 153, 34, 0.1); }
  .submit-actions { display: flex; gap: 0.4rem; align-items: center; }
  .ghost-link {
    color: var(--fg-muted);
    text-decoration: none;
    font-size: 0.85rem;
    padding: 0.4rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-elev-2);
  }
  .ghost-link:hover { color: var(--fg); }
  .primary {
    background: var(--accent);
    color: #08111e;
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 0.4rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  .primary:hover:not(:disabled) { filter: brightness(1.1); }
  .primary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
