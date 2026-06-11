<script lang="ts">
  import { ApiError, sendTextCommand } from '$lib/api';

  /** Typed-command chatbox. Voice input is handled entirely by the
   *  browser-audio control (BrowserAudioPanel) -- there is no separate
   *  mic toggle; this panel only sends text. */
  let draft = $state('');
  let sending = $state(false);
  let sendError = $state<string | null>(null);
  let textarea: HTMLTextAreaElement | undefined = $state();

  const canSend = $derived(!sending && draft.trim().length > 0);

  function errText(err: unknown): string {
    if (err instanceof ApiError) return `${err.message} (HTTP ${err.status})`;
    if (err instanceof Error) return err.message;
    return 'Request failed';
  }

  async function submit(e: Event) {
    e.preventDefault();
    if (!canSend) return;
    sendError = null;
    sending = true;
    const payload = draft.trim();
    try {
      await sendTextCommand(payload);
      draft = '';
    } catch (err) {
      sendError = errText(err);
    } finally {
      sending = false;
      queueMicrotask(() => textarea?.focus());
    }
  }

  /** Enter sends; Shift+Enter inserts a newline. */
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submit(e);
    }
  }
</script>

<section class="panel">
  <form class="input-row" onsubmit={submit}>
    <textarea
      bind:this={textarea}
      bind:value={draft}
      onkeydown={onKey}
      placeholder={'Type a helm command, e.g. "come to two seven zero"'}
      rows="2"
      aria-label="Text command"
      data-primary-input
    ></textarea>
    <button type="submit" class="send" disabled={!canSend}>
      {sending ? 'Sending…' : 'Send'}
    </button>
  </form>
  {#if sendError}
    <div class="status err">{sendError}</div>
  {/if}
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .input-row {
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
  }
  textarea {
    flex: 1;
    min-height: 2.4rem;
    max-height: 8rem;
    resize: vertical;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.45rem 0.6rem;
    color: var(--fg);
    font-family: inherit;
    font-size: 0.9rem;
    line-height: 1.35;
    outline: none;
  }
  textarea:focus { border-color: var(--accent); }
  textarea:disabled { color: var(--fg-muted); background: var(--bg-elev); cursor: not-allowed; }

  .send {
    background: var(--accent);
    color: #08111e;
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 0.45rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    align-self: stretch;
  }
  .send:hover:not(:disabled) { filter: brightness(1.1); }
  .send:disabled { opacity: 0.5; cursor: not-allowed; }

  .status {
    padding: 0.35rem 0.55rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    font-size: 0.78rem;
  }
  .status.err { border-color: var(--bad); background: rgba(248, 81, 73, 0.08); }
</style>
