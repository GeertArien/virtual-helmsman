<script lang="ts">
  import { WebRTCAudio, type WebRTCStatus } from '$lib/audio/webrtcAudio';
  import { onDestroy } from 'svelte';

  // Browser audio over WebRTC (issue #7): captures the mic and connects to the
  // helmsman's per-connection pipeline, so you can speak to the agent and hear
  // its reply in the browser. Shown only when the backend reports
  // `browser_audio` (audio.browser_enabled). Separate from the server-mic
  // toggle, which gates the local-hardware pipeline.
  let status = $state<WebRTCStatus>('idle');
  let detail = $state<string | null>(null);

  let audio: WebRTCAudio | null = null;
  const connected = $derived(status === 'live' || status === 'connecting');

  async function toggle() {
    if (connected) {
      await audio?.stop();
      audio = null;
      return;
    }
    detail = null;
    audio = new WebRTCAudio({
      onStatus: (s, d) => {
        status = s;
        detail = d ?? null;
      }
    });
    await audio.start();
  }

  onDestroy(() => {
    void audio?.stop();
  });
</script>

<section class="panel">
  <header class="hd">
    <button
      type="button"
      class="toggle"
      data-on={connected}
      onclick={toggle}
      disabled={status === 'connecting'}
      aria-pressed={connected}
      title={connected ? 'Disconnect browser audio' : 'Talk to the helmsman from your browser'}
    >
      <span class="led" aria-hidden="true"></span>
      <span class="lbl">
        {#if status === 'connecting'}
          browser audio: connecting…
        {:else if status === 'live'}
          browser audio: live
        {:else if status === 'error'}
          browser audio: error
        {:else}
          browser audio: off
        {/if}
      </span>
    </button>
  </header>
  <p class="hint">
    Speak to the helmsman from your browser — your mic streams over WebRTC and
    the agent's reply plays back here.
  </p>
  {#if detail && status === 'error'}
    <div class="status err">{detail}</div>
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
    gap: 0.4rem;
  }
  .hd {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .toggle {
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
  .toggle[data-on='true'] { border-color: var(--accent); background: rgba(56, 139, 253, 0.1); }
  .led {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--fg-muted);
  }
  .toggle[data-on='true'] .led {
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: blink 1.2s ease-in-out infinite;
  }
  @keyframes blink {
    50% { opacity: 0.4; }
  }
  .lbl { font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace; }
  .hint { color: var(--fg-muted); font-size: 0.76rem; margin: 0; }
  .status {
    padding: 0.35rem 0.55rem;
    border-radius: 4px;
    border: 1px solid var(--bad);
    background: rgba(248, 81, 73, 0.08);
    font-size: 0.78rem;
  }
</style>
