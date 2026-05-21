<script lang="ts">
  import { ApiError, uploadForReview, type UploadFields } from '$lib/api';

  let {
    open = $bindable(false),
    onUploaded = () => {}
  }: { open?: boolean; onUploaded?: () => void } = $props();

  /** Default values mirror the backend's ReviewConfig pre-fills. The
   *  webhook treats Document_Type / Collection_Name as required; Categories
   *  and Chunking_Strategy fall back to webhook defaults when blank. */
  const DEFAULTS: Required<UploadFields> = {
    document_type: 'PDF',
    collection_name: 'maritime_hybrid',
    categories: 'algemeen',
    chunking_strategy: 'paragraph_aware'
  };

  type Phase =
    | { kind: 'idle' }
    | { kind: 'uploading' }
    | { kind: 'success'; message: string }
    | { kind: 'error'; message: string };

  let file = $state<File | null>(null);
  let documentType = $state(DEFAULTS.document_type);
  let collectionName = $state(DEFAULTS.collection_name);
  let categories = $state(DEFAULTS.categories);
  let chunkingStrategy = $state<UploadFields['chunking_strategy']>(DEFAULTS.chunking_strategy);
  let isDragOver = $state(false);
  let phase = $state<Phase>({ kind: 'idle' });
  let fileInput: HTMLInputElement | undefined = $state();

  const canSubmit = $derived(
    file !== null &&
      documentType.trim().length > 0 &&
      collectionName.trim().length > 0 &&
      phase.kind !== 'uploading'
  );

  function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function pick(picked: File | null | undefined) {
    if (!picked) return;
    file = picked;
    if (phase.kind === 'success' || phase.kind === 'error') phase = { kind: 'idle' };
  }

  function reset() {
    file = null;
    documentType = DEFAULTS.document_type;
    collectionName = DEFAULTS.collection_name;
    categories = DEFAULTS.categories;
    chunkingStrategy = DEFAULTS.chunking_strategy;
    phase = { kind: 'idle' };
    if (fileInput) fileInput.value = '';
  }

  function close() {
    if (phase.kind === 'uploading') return;
    open = false;
    // Defer reset slightly so the closing transition doesn't visibly clear fields.
    setTimeout(reset, 200);
  }

  async function submit(e: Event) {
    e.preventDefault();
    if (!canSubmit || !file) return;
    phase = { kind: 'uploading' };
    try {
      const res = await uploadForReview(file, {
        document_type: documentType.trim(),
        collection_name: collectionName.trim(),
        categories: categories.trim(),
        chunking_strategy: chunkingStrategy
      });
      phase = {
        kind: 'success',
        message: res.message || 'Submitted. The batch will appear in the list shortly.'
      };
      onUploaded();
    } catch (err) {
      phase = {
        kind: 'error',
        message:
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : err instanceof Error
              ? err.message
              : 'Upload failed'
      };
    }
  }
</script>

{#if open}
  <div
    class="backdrop"
    onclick={close}
    role="presentation"
  ></div>
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="upload-title">
    <header>
      <h2 id="upload-title">Upload PDF for review</h2>
      <button type="button" class="close" onclick={close} aria-label="Close" disabled={phase.kind === 'uploading'}>×</button>
    </header>

    <form onsubmit={submit}>
      <div
        class="dropzone"
        class:over={isDragOver}
        class:has-file={file !== null}
        ondragover={(e) => { e.preventDefault(); isDragOver = true; }}
        ondragleave={() => (isDragOver = false)}
        ondrop={(e) => { e.preventDefault(); isDragOver = false; pick(e.dataTransfer?.files?.[0]); }}
        role="button"
        tabindex="0"
        onclick={() => fileInput?.click()}
        onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInput?.click()}
      >
        <div class="drop-title">{file ? 'File ready' : 'Drop a PDF here'}</div>
        <div class="drop-sub mono">
          {file ? `${file.name} · ${formatBytes(file.size)}` : 'or click to browse'}
        </div>
        <input
          type="file"
          bind:this={fileInput}
          onchange={(e) => pick((e.currentTarget as HTMLInputElement).files?.[0])}
          accept=".pdf,application/pdf"
          hidden
        />
      </div>

      <div class="grid">
        <label class="field">
          <span class="label">Document_Type *</span>
          <input type="text" bind:value={documentType} required />
        </label>
        <label class="field">
          <span class="label">Collection_Name *</span>
          <input type="text" bind:value={collectionName} required />
        </label>
        <label class="field wide">
          <span class="label">Categories <span class="optional">(comma-separated)</span></span>
          <input type="text" bind:value={categories} />
        </label>
        <label class="field wide">
          <span class="label">Chunking_Strategy</span>
          <select bind:value={chunkingStrategy}>
            <option value="paragraph_aware">paragraph_aware (default)</option>
            <option value="fixed_size">fixed_size</option>
          </select>
        </label>
      </div>

      {#if phase.kind === 'success'}
        <div class="status ok">{phase.message}</div>
      {:else if phase.kind === 'error'}
        <div class="status err">{phase.message}</div>
      {/if}

      <div class="actions">
        <button type="button" class="ghost" onclick={close} disabled={phase.kind === 'uploading'}>Close</button>
        <button type="submit" class="primary" disabled={!canSubmit}>
          {phase.kind === 'uploading' ? 'Uploading…' : 'Upload → n8n'}
        </button>
      </div>
    </form>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 50;
  }
  .dialog {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    z-index: 51;
    width: min(560px, 92vw);
    max-height: 90vh;
    overflow-y: auto;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0;
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  h2 { margin: 0; font-size: 0.95rem; font-weight: 600; }
  .close {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    font-size: 1.4rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.4rem;
  }
  .close:hover:not(:disabled) { color: var(--fg); }
  form { display: flex; flex-direction: column; gap: 0.75rem; padding: 1rem; }

  .dropzone {
    border: 1.5px dashed var(--border);
    border-radius: 6px;
    padding: 1rem;
    background: var(--bg-elev-2);
    text-align: center;
    cursor: pointer;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .dropzone:hover { border-color: var(--accent); }
  .dropzone.over { border-color: var(--accent); background: rgba(88, 166, 255, 0.08); }
  .dropzone.has-file { border-style: solid; border-color: var(--good); }
  .drop-title { font-size: 0.95rem; margin-bottom: 0.25rem; }
  .drop-sub { color: var(--fg-muted); font-size: 0.85rem; word-break: break-all; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
  }
  .field { display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
  .field.wide { grid-column: 1 / -1; }
  .label { font-size: 0.72rem; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .optional { text-transform: none; letter-spacing: 0; font-style: italic; }
  input[type='text'], select {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    color: var(--fg);
    outline: none;
  }
  input[type='text']:focus, select:focus { border-color: var(--accent); }

  .actions { display: flex; gap: 0.5rem; justify-content: flex-end; }
  button {
    border-radius: 4px;
    padding: 0.45rem 0.9rem;
    font-size: 0.85rem;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #08111e; font-weight: 600; }
  button.primary:hover:not(:disabled) { filter: brightness(1.1); }
  button.ghost { color: var(--fg-muted); }
  button.ghost:hover:not(:disabled) { color: var(--fg); }

  .status {
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.85rem;
    border: 1px solid var(--border);
  }
  .status.ok { border-color: var(--good); background: rgba(63, 185, 80, 0.08); }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }
</style>
