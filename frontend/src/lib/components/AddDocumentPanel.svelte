<script lang="ts">
  import { ApiError, uploadDocument, type UploadResponse } from '$lib/api';

  type Phase =
    | { kind: 'idle' }
    | { kind: 'uploading'; fileName: string }
    | { kind: 'success'; response: UploadResponse; fileName: string }
    | { kind: 'error'; message: string };

  let file = $state<File | null>(null);
  let title = $state('');
  let isDragOver = $state(false);
  let phase = $state<Phase>({ kind: 'idle' });
  let fileInput: HTMLInputElement | undefined = $state();

  const fileLabel = $derived(
    file ? `${file.name} · ${formatBytes(file.size)}` : 'No file selected'
  );

  function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function chooseFile(picked: File | null | undefined) {
    if (!picked) return;
    file = picked;
    phase = { kind: 'idle' };
  }

  function onPickerChange(e: Event) {
    const target = e.currentTarget as HTMLInputElement;
    chooseFile(target.files?.[0] ?? null);
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    isDragOver = true;
  }

  function onDragLeave() {
    isDragOver = false;
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    isDragOver = false;
    chooseFile(e.dataTransfer?.files?.[0] ?? null);
  }

  async function onSubmit(e: Event) {
    e.preventDefault();
    if (!file || phase.kind === 'uploading') return;
    const fileName = file.name;
    phase = { kind: 'uploading', fileName };
    try {
      const response = await uploadDocument(file, title.trim() || undefined);
      phase = { kind: 'success', response, fileName };
      // Clear the form so a subsequent upload doesn't accidentally re-send.
      file = null;
      title = '';
      if (fileInput) fileInput.value = '';
    } catch (err) {
      const message =
        err instanceof ApiError
          ? `${err.message} (HTTP ${err.status})`
          : err instanceof Error
            ? err.message
            : 'Upload failed';
      phase = { kind: 'error', message };
    }
  }

  function reset() {
    file = null;
    title = '';
    phase = { kind: 'idle' };
    if (fileInput) fileInput.value = '';
  }
</script>

<section class="panel">
  <h2>Add document</h2>
  <p class="hint">
    Files are routed through an n8n workflow with a human-in-the-loop review step
    before they're ingested into qdrant.
  </p>

  <form onsubmit={onSubmit}>
    <div
      class="dropzone"
      class:over={isDragOver}
      class:has-file={file !== null}
      ondragover={onDragOver}
      ondragleave={onDragLeave}
      ondrop={onDrop}
      role="button"
      tabindex="0"
      onclick={() => fileInput?.click()}
      onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInput?.click()}
    >
      <div class="drop-inner">
        <div class="drop-title">
          {file ? 'File ready' : 'Drop a file here'}
        </div>
        <div class="drop-sub mono">{fileLabel}</div>
        <div class="drop-action">
          <span class="link">{file ? 'choose a different file' : 'or click to browse'}</span>
        </div>
      </div>
      <input
        type="file"
        bind:this={fileInput}
        onchange={onPickerChange}
        hidden
        accept=".pdf,.md,.txt,.html,.htm,.docx,.json,.csv,application/pdf,text/plain,text/markdown,text/html"
      />
    </div>

    <label class="field">
      <span class="label">Title <span class="optional">(optional)</span></span>
      <input
        type="text"
        bind:value={title}
        placeholder="Defaults to the file name"
        autocomplete="off"
      />
    </label>

    <div class="actions">
      <button type="submit" class="primary" disabled={!file || phase.kind === 'uploading'}>
        {phase.kind === 'uploading' ? 'Uploading…' : 'Upload → n8n'}
      </button>
      <button type="button" class="ghost" onclick={reset} disabled={phase.kind === 'uploading' || (!file && !title && phase.kind === 'idle')}>
        Reset
      </button>
    </div>

    {#if phase.kind === 'success'}
      <div class="status ok" role="status">
        <strong>{phase.response.status}</strong>
        — {phase.fileName}{phase.response.document_id ? ` · id ${phase.response.document_id}` : ''}
        {#if phase.response.message}<div class="status-msg">{phase.response.message}</div>{/if}
      </div>
    {:else if phase.kind === 'error'}
      <div class="status err" role="alert">
        <strong>Upload failed.</strong> {phase.message}
      </div>
    {/if}
  </form>
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
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0;
    font-weight: 600;
  }
  .hint { margin: 0; color: var(--fg-muted); font-size: 0.85rem; }
  form { display: flex; flex-direction: column; gap: 0.75rem; }
  .dropzone {
    border: 1.5px dashed var(--border);
    border-radius: 6px;
    padding: 1.25rem 1rem;
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
  .drop-action { margin-top: 0.5rem; font-size: 0.85rem; }
  .link { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }

  .field { display: flex; flex-direction: column; gap: 0.3rem; }
  .label { font-size: 0.75rem; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .optional { text-transform: none; letter-spacing: 0; font-style: italic; }
  input[type='text'] {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    color: var(--fg);
    outline: none;
  }
  input[type='text']:focus { border-color: var(--accent); }

  .actions { display: flex; gap: 0.5rem; }
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
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); color: var(--fg); }
  .status-msg { margin-top: 0.25rem; color: var(--fg-muted); }
</style>
