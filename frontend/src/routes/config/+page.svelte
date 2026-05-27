<script lang="ts">
  import { onMount } from 'svelte';
  import {
    ApiError,
    fetchConfig,
    fetchConfigSchema,
    reloadBackend,
    saveConfig,
    waitForBackend,
    type ConfigDict,
    type JsonSchema
  } from '$lib/api';
  import SchemaField from '$lib/components/SchemaField.svelte';

  /** Loaded once on mount. */
  let schema = $state<JsonSchema | null>(null);
  /** Editable copy of the config. */
  let draft = $state<ConfigDict | null>(null);
  /** Original snapshot for the "dirty" indicator + Reset button. */
  let original = $state<string>('');

  type Phase =
    | { kind: 'loading' }
    | { kind: 'ready' }
    | { kind: 'saving' }
    | { kind: 'saved' }
    | { kind: 'reloading'; message: string }
    | { kind: 'reloaded' }
    | { kind: 'error'; message: string };

  let phase = $state<Phase>({ kind: 'loading' });

  /** Pydantic error list from the most recent PUT (if any), as parsed
   *  from ApiError.detail. Each entry is `{loc: ["stt", "backend"], msg: "..."}`. */
  type PydanticError = { loc: (string | number)[]; msg: string; type?: string };
  let pydanticErrors = $state<PydanticError[]>([]);

  async function load() {
    phase = { kind: 'loading' };
    try {
      const [s, c] = await Promise.all([fetchConfigSchema(), fetchConfig()]);
      schema = s;
      draft = c;
      original = JSON.stringify(c);
      phase = { kind: 'ready' };
    } catch (err) {
      phase = { kind: 'error', message: errMsg(err) };
    }
  }

  function errMsg(err: unknown): string {
    if (err instanceof ApiError) {
      if (Array.isArray(err.detail)) {
        // Pydantic error list -- summarise the count for the top banner.
        return `${err.message} (HTTP ${err.status}) -- ${err.detail.length} field error${err.detail.length === 1 ? '' : 's'}`;
      }
      return `${err.message} (HTTP ${err.status})`;
    }
    if (err instanceof Error) return err.message;
    return 'Request failed';
  }

  const dirty = $derived(draft !== null && JSON.stringify(draft) !== original);

  /** Look up a field error in `pydanticErrors` by exact path match. */
  function fieldError(loc: (string | number)[]): string | null {
    for (const e of pydanticErrors) {
      // Pydantic loc starts at the top of the model (e.g. ["stt", "backend"]).
      // We pass paths matching that exact shape.
      if (
        e.loc.length === loc.length &&
        e.loc.every((part, i) => part === loc[i])
      ) {
        return e.msg;
      }
    }
    return null;
  }

  function reset() {
    draft = JSON.parse(original);
    pydanticErrors = [];
    phase = { kind: 'ready' };
  }

  async function save() {
    if (!draft) return;
    phase = { kind: 'saving' };
    pydanticErrors = [];
    try {
      await saveConfig(draft);
      original = JSON.stringify(draft);
      phase = { kind: 'saved' };
    } catch (err) {
      if (err instanceof ApiError && Array.isArray(err.detail)) {
        pydanticErrors = err.detail as PydanticError[];
      }
      phase = { kind: 'error', message: errMsg(err) };
    }
  }

  async function reload() {
    phase = { kind: 'reloading', message: 'Asking backend to restart...' };
    try {
      await reloadBackend();
    } catch (err) {
      phase = { kind: 'error', message: errMsg(err) };
      return;
    }
    phase = { kind: 'reloading', message: 'Waiting for backend to come back...' };
    try {
      // Wait until /api/health responds; the backend usually needs ~5-10s
      // to reload model weights so 60s is a generous cap.
      await waitForBackend(60_000, 500);
      phase = { kind: 'reloaded' };
      // Refresh the config view from the new process -- defensive in case
      // the restart picked up any env-var changes.
      await load();
    } catch (err) {
      phase = { kind: 'error', message: errMsg(err) };
    }
  }

  /** Update the top-level section in the draft (e.g. draft.stt = {...}). */
  function updateSection(key: string, value: unknown) {
    if (!draft) return;
    draft = { ...draft, [key]: value };
  }

  onMount(load);
</script>

<main>
  <header class="toolbar">
    <div class="title">
      <h1>Backend config</h1>
      {#if dirty}
        <span class="dirty mono" title="Unsaved edits">● modified</span>
      {/if}
    </div>
    <div class="actions">
      <button type="button" class="ghost" onclick={reset} disabled={!dirty || phase.kind === 'saving' || phase.kind === 'reloading'}>
        Reset
      </button>
      <button type="button" class="primary" onclick={save} disabled={!dirty || phase.kind === 'saving' || phase.kind === 'reloading'}>
        {phase.kind === 'saving' ? 'Saving…' : 'Save'}
      </button>
      <button type="button" class="danger" onclick={reload} disabled={dirty || phase.kind === 'reloading' || phase.kind === 'saving'}>
        {phase.kind === 'reloading' ? 'Reloading…' : 'Reload backend'}
      </button>
    </div>
  </header>

  <p class="hint">
    Edits are written to <code>config.yaml</code> on disk when you click <strong>Save</strong>.
    The running agent only picks them up on <strong>Reload backend</strong>, which
    restarts the Python process. Comments in <code>config.yaml</code> are lost on save.
  </p>

  {#if phase.kind === 'saved'}
    <div class="status ok">Config saved to disk. Click <strong>Reload backend</strong> to apply.</div>
  {:else if phase.kind === 'reloading'}
    <div class="status warn">{phase.message}</div>
  {:else if phase.kind === 'reloaded'}
    <div class="status ok">Backend restarted with the new config.</div>
  {:else if phase.kind === 'error'}
    <div class="status err">{phase.message}</div>
  {/if}

  {#if pydanticErrors.length > 0}
    <details class="errors" open>
      <summary>{pydanticErrors.length} validation error{pydanticErrors.length === 1 ? '' : 's'}</summary>
      <ul>
        {#each pydanticErrors as e (e.loc.join('.'))}
          <li><code>{e.loc.join('.')}</code> — {e.msg}</li>
        {/each}
      </ul>
    </details>
  {/if}

  {#if phase.kind === 'loading' && draft === null}
    <div class="empty">Loading config…</div>
  {:else if schema && draft && schema.properties}
    {#each Object.entries(schema.properties) as [key, sub] (key)}
      <section class="section">
        <h2>{key}</h2>
        <SchemaField
          name={key}
          schema={sub}
          value={draft[key]}
          rootSchema={schema}
          path={[key]}
          {fieldError}
          onChange={(v) => updateSection(key, v)}
        />
      </section>
    {/each}
  {/if}
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    max-width: 60rem;
    margin: 0 auto;
    width: 100%;
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    background: var(--bg);
    padding: 0.4rem 0;
    z-index: 10;
  }
  .title { display: flex; align-items: baseline; gap: 0.75rem; }
  h1 { margin: 0; font-size: 1.05rem; font-weight: 600; }
  .dirty { color: var(--warn); font-size: 0.78rem; }
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
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #08111e; font-weight: 600; }
  button.primary:hover:not(:disabled) { filter: brightness(1.1); }
  button.danger { color: var(--bad); border-color: var(--bad); background: transparent; }
  button.danger:hover:not(:disabled) { background: rgba(248, 81, 73, 0.1); }
  button.ghost { color: var(--fg-muted); }
  button.ghost:hover:not(:disabled) { color: var(--fg); }

  .hint {
    margin: 0;
    padding: 0.6rem 0.85rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-muted);
    font-size: 0.85rem;
  }
  .hint code {
    background: var(--bg-elev-2);
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.85em;
  }

  .status {
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    font-size: 0.85rem;
  }
  .status.ok   { border-color: var(--good); background: rgba(63, 185, 80, 0.08); }
  .status.warn { border-color: var(--warn); background: rgba(210, 153, 34, 0.08); }
  .status.err  { border-color: var(--bad);  background: rgba(248, 81, 73, 0.08); }

  .errors {
    background: rgba(248, 81, 73, 0.06);
    border: 1px solid var(--bad);
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
  }
  .errors summary { cursor: pointer; color: var(--bad); }
  .errors ul { margin: 0.4rem 0 0 1.2rem; padding: 0; }
  .errors li { margin: 0.15rem 0; }
  .errors code { background: var(--bg-elev-2); padding: 0.05rem 0.3rem; border-radius: 3px; }

  .empty {
    padding: 1rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-muted);
    font-style: italic;
  }

  .section {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  .section h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.5rem 0;
    font-weight: 600;
  }
</style>
