<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ApiError, fetchAuditLog, type AuditEntry } from '$lib/api';

  type LoadState =
    | { kind: 'loading' }
    | { kind: 'ready'; entries: AuditEntry[]; totalInLog: number }
    | { kind: 'error'; message: string };

  /** Default page size. n8n caps server-side at 500; 50 is a sensible tail. */
  const DEFAULT_LIMIT = 50;

  /** Auto-refresh cadence. Runtime entries (command/question) land in real
   *  time so the panel needs to keep pulling; 5s is roughly the period
   *  between turns in the demo without being chatty. */
  const POLL_MS = 5_000;

  /** The known `actie` filter values surfaced as dropdown options.
   *  Unknown future actie values still render -- they just don't have a
   *  dedicated filter button until added here. */
  const ACTIE_FILTERS: { value: string; label: string }[] = [
    { value: '', label: 'All' },
    { value: 'command_runtime', label: 'Command (runtime)' },
    { value: 'question_runtime', label: 'Question (runtime)' },
    { value: 'ingestie_hitl', label: 'HITL ingestion' },
    { value: 'llm_error_runtime', label: 'Runtime LLM error' },
    { value: 'llm_error_ingestion', label: 'Ingestion error' },
    { value: 'art50_acknowledged', label: 'AI transparency (Art. 50)' }
  ];

  let load = $state<LoadState>({ kind: 'loading' });
  let lastUpdated = $state<Date | null>(null);
  let filterActie = $state<string>('');
  let pollHandle: ReturnType<typeof setTimeout> | null = null;

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

  /** When the filter changes, restart polling immediately with the new
   *  filter so the user doesn't wait POLL_MS to see the result. */
  $effect(() => {
    void filterActie;
    stopPolling();
    startPolling();
  });

  /** Parse a `key1=value1 | key2=value2 | output=...` resultaat string into
   *  an object. The `output=` value is captured greedily so newlines and
   *  pipe characters inside the spoken answer don't break parsing. */
  function parseResultaat(resultaat: string | undefined): Record<string, string> {
    const fields: Record<string, string> = {};
    if (!resultaat) return fields;
    // Greedy capture of everything after `output=` -- the spoken text /
    // RAG answer can contain newlines and pipes.
    const outMatch = /output=([\s\S]*)$/.exec(resultaat);
    let head = resultaat;
    if (outMatch) {
      fields.output = outMatch[1].trim();
      head = resultaat.slice(0, outMatch.index);
    }
    for (const part of head.split('|')) {
      const eq = part.indexOf('=');
      if (eq === -1) continue;
      const key = part.slice(0, eq).trim();
      const val = part.slice(eq + 1).trim();
      if (key) fields[key] = val;
    }
    return fields;
  }

  type Outcome = 'ok' | 'fail' | 'neutral';

  /** Map an entry to an outcome flavour that drives the left-border colour
   *  and the LED dot. Per-actie rules:
   *  - any `llm_error_*` actie: always a fail. The prefix is broad on
   *    purpose so a future `llm_error_helmsman` / `llm_error_classify`
   *    type lands red without a code change.
   *  - `ingestie_hitl`: classified by the leading keyword of `resultaat`.
   *    "Succes" is ok; "Fout" (Dutch) or "All rejected" (English variant
   *    emitted by *Log All Rejected*) are fails. Anything else is neutral.
   *  - `command_runtime`: `action_type=error` means the LLM refused.
   *  - `question_runtime`: `parse_failure=true` is the upstream-LLM-output
   *    fallback path -- we render it as a fail. `citation_reliable=false`
   *    is *not* a fail; the answer still exists.
   */
  function outcomeOf(entry: AuditEntry): Outcome {
    const a = entry.actie ?? '';
    const r = (entry.resultaat ?? '').trim();
    if (a.startsWith('llm_error_')) return 'fail';
    if (a === 'ingestie_hitl') {
      const lower = r.toLowerCase();
      if (lower.startsWith('succes')) return 'ok';
      if (lower.startsWith('fout')) return 'fail';
      if (lower.startsWith('all rejected')) return 'fail';
      return 'neutral';
    }
    if (a === 'command_runtime') {
      return /action_type=error\b/.test(r) ? 'fail' : 'ok';
    }
    if (a === 'question_runtime') {
      return /parse_failure=true\b/.test(r) ? 'fail' : 'ok';
    }
    return 'neutral';
  }

  /** Human-readable label for the document column. `n.v.t. (command)` is
   *  noise; render commands as a dash. */
  function documentLabel(entry: AuditEntry): string {
    const d = (entry.document_naam ?? '').trim();
    if (!d || d.toLowerCase().startsWith('n.v.t.')) return '—';
    return d;
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

  onMount(startPolling);
  onDestroy(stopPolling);
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
    Outcomes from the n8n <code>audit-log-maritime</code> datatable: HITL
    ingestion runs, runtime command parses, and runtime question (RAG)
    answers. Newest first; capped at {DEFAULT_LIMIT} rows. Auto-refresh every
    {Math.round(POLL_MS / 1000)}s.
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
        {@const parsed = parseResultaat(entry.resultaat)}
        {@const doc = documentLabel(entry)}
        <li class="entry" data-outcome={outcome} data-actie={entry.actie}>
          <div class="entry-head">
            <span class="dot" aria-hidden="true"></span>
            <span class="document" title={entry.document_naam}>{doc}</span>
            <span class="actie mono">{entry.actie}</span>
            <span class="when mono" title={fmtDate(entry.createdAt)}>{fmtAge(entry.createdAt)}</span>
          </div>

          {#if entry.actie === 'command_runtime'}
            <div class="meta-row">
              {#if parsed.action_type}
                <span class="chip" data-flavor={parsed.action_type === 'error' ? 'bad' : 'accent'}>
                  action: {parsed.action_type}
                </span>
              {/if}
            </div>
            {#if parsed.output}
              <div class="quote">{parsed.output}</div>
            {/if}
          {:else if entry.actie === 'question_runtime'}
            <div class="meta-row">
              {#if parsed.chunk}
                <span class="chip" data-flavor="accent">chunk: {parsed.chunk}</span>
              {/if}
              {#if parsed.citation_reliable === 'true'}
                <span class="chip" data-flavor="good">cited</span>
              {:else if parsed.citation_reliable === 'false'}
                <span class="chip" data-flavor="warn">cite uncertain</span>
              {/if}
              {#if parsed.parse_failure === 'true'}
                <span class="chip" data-flavor="bad">parse failure</span>
              {/if}
            </div>
            {#if parsed.output}
              <div class="quote">{parsed.output}</div>
            {/if}
          {:else if entry.actie?.startsWith('llm_error_')}
            <!-- Structured llm_error_* rows: error=<msg> | http=<code> | input_chars=<n>.
                 The error string itself can be long, so it goes in the quote;
                 http / input_chars are short -> chips. -->
            <div class="meta-row">
              {#if parsed.http && parsed.http !== 'n.v.t.'}
                <span class="chip" data-flavor="bad">http: {parsed.http}</span>
              {/if}
              {#if parsed.input_chars}
                <span class="chip" data-flavor="accent">{parsed.input_chars} chars in</span>
              {/if}
            </div>
            {#if parsed.error}
              <div class="quote err">{parsed.error}</div>
            {:else}
              <div class="resultaat">{entry.resultaat}</div>
            {/if}
          {:else}
            <!-- ingestie_hitl and any future / unknown actie -->
            <div class="resultaat">{entry.resultaat}</div>
          {/if}
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
    gap: 0.4rem;
    /* No internal max-height -- the page is scrollable. Past versions
       capped this at 22rem when it sat inside another panel; on its
       dedicated /audit route it gets the full viewport. */
  }
  .entry {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-left-width: 3px;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
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
    max-width: 36ch;
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

  .meta-row {
    display: flex;
    gap: 0.3rem;
    flex-wrap: wrap;
  }
  .chip {
    font-size: 0.72rem;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--fg-muted);
    line-height: 1.4;
  }
  .chip[data-flavor='accent'] { color: var(--accent); border-color: var(--accent); }
  .chip[data-flavor='good']   { color: var(--good);   border-color: var(--good);   }
  .chip[data-flavor='warn']   { color: var(--warn);   border-color: var(--warn);   }
  .chip[data-flavor='bad']    { color: var(--bad);    border-color: var(--bad);    }

  .quote {
    color: var(--fg);
    font-size: 0.82rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
    /* A subtle quote marker so the spoken text reads as a quotation. */
    padding-left: 0.55rem;
    border-left: 2px solid var(--border);
    color: var(--fg-muted);
  }
  /* err-flavoured quotes pull the same border colour as the failed-row
     left border. Used to display llm_error_* messages prominently. */
  .quote.err {
    border-left-color: var(--bad);
    color: var(--fg);
  }

  .resultaat {
    color: var(--fg);
    font-size: 0.82rem;
    line-height: 1.4;
    word-break: break-word;
  }
</style>
