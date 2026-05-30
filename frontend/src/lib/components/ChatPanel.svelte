<script lang="ts">
  import { ApiError, sendTextCommand, setMicEnabled } from '$lib/api';
  import { live } from '$lib/liveState.svelte';

  /** Local UI state. The mic-enabled flag lives on $live so every tab
   *  stays in sync via the WS event stream; this component just renders
   *  it and offers a toggle button. */
  let draft = $state('');
  let sending = $state(false);
  let togglePending = $state(false);
  let sendError = $state<string | null>(null);
  let toggleError = $state<string | null>(null);
  let textarea: HTMLTextAreaElement | undefined = $state();

  /** Until /api/control/state returns, micEnabled is null -- treat that
   *  as "loading" and disable both inputs to avoid racing the snapshot. */
  const micEnabled = $derived(live.micEnabled);
  const micUnknown = $derived(micEnabled === null);
  const chatLocked = $derived(micUnknown || micEnabled === true);

  const canSend = $derived(
    !chatLocked && !sending && draft.trim().length > 0
  );

  function errText(err: unknown): string {
    if (err instanceof ApiError) return `${err.message} (HTTP ${err.status})`;
    if (err instanceof Error) return err.message;
    return 'Request failed';
  }

  async function toggleMic() {
    if (togglePending || micUnknown) return;
    toggleError = null;
    togglePending = true;
    try {
      // Optimistic feel: send the inverse of what we're showing. The WS
      // event will overwrite micEnabled regardless, so we don't pre-mutate.
      const next = !micEnabled;
      await setMicEnabled(next);
      // The backend broadcasts the change; the WS handler updates live.
      // If the focus was on the textarea and we just unlocked it, refocus.
      if (next === false) {
        // Give Svelte a tick to render the enabled state.
        queueMicrotask(() => textarea?.focus());
      }
    } catch (err) {
      toggleError = errText(err);
    } finally {
      togglePending = false;
    }
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
  <header class="hd">
    <div class="mic-row">
      <button
        type="button"
        class="toggle"
        data-on={micEnabled === true}
        onclick={toggleMic}
        disabled={togglePending || micUnknown}
        aria-pressed={micEnabled === true}
        title={micEnabled === true ? 'Click to disable the server mic' : 'Click to enable the server mic'}
      >
        <span class="led" aria-hidden="true"></span>
        <span class="lbl">
          {#if micUnknown}
            mic: unknown
          {:else if micEnabled}
            mic: recording
          {:else}
            mic: paused
          {/if}
        </span>
      </button>
      <span class="hint">
        {#if chatLocked && !micUnknown}
          Disable the mic to type commands.
        {:else if !chatLocked}
          Mic is paused — type commands below.
        {/if}
      </span>
    </div>
    {#if toggleError}
      <div class="status err inline">{toggleError}</div>
    {/if}
  </header>

  <form class="input-row" onsubmit={submit}>
    <textarea
      bind:this={textarea}
      bind:value={draft}
      onkeydown={onKey}
      placeholder={chatLocked
        ? 'Mic is recording — type is disabled.'
        : 'Type a helm command, e.g. "come to two seven zero"'}
      rows="2"
      disabled={chatLocked}
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
  .hd {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .mic-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .toggle {
    /* Flex-grow inside the .mic-row flex container so the toggle pill
       stretches across the available row width; the hint text wraps to
       the next line. (This used to come from a stray over-broad
       :global(:first-child) selector on the page; now expressed where
       it actually belongs.) */
    flex: 1;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
    border-radius: 999px;
    padding: 0.25rem 0.7rem 0.25rem 0.55rem;
    font-size: 0.78rem;
    color: var(--fg);
    cursor: pointer;
    line-height: 1;
  }
  .toggle:disabled { opacity: 0.55; cursor: not-allowed; }
  .toggle[data-on='true']  { border-color: var(--bad);  background: rgba(248, 81, 73, 0.1); }
  .toggle[data-on='false'] { border-color: var(--good); background: rgba(63, 185, 80, 0.08); }
  .led {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--fg-muted);
  }
  .toggle[data-on='true'] .led {
    background: var(--bad);
    box-shadow: 0 0 6px var(--bad);
    /* gentle pulse so "recording" reads at a glance */
    animation: blink 1.2s ease-in-out infinite;
  }
  .toggle[data-on='false'] .led { background: var(--good); }
  @keyframes blink {
    50% { opacity: 0.35; }
  }
  .lbl { font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace; }

  .hint { color: var(--fg-muted); font-size: 0.78rem; }

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
  .status.inline { padding: 0.25rem 0.5rem; font-size: 0.75rem; }
</style>
