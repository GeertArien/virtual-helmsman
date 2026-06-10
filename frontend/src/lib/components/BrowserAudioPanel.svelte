<script lang="ts">
  import { BrowserAudio, type AudioStatus } from '$lib/audio/browserAudio';
  import { onDestroy } from 'svelte';

  // Browser-audio preview (issue #7, phase one). Captures the mic in the
  // browser, streams raw PCM to /ws/audio, and plays the looped-back audio —
  // proving the capture → stream → playback path. It does NOT yet talk to the
  // helmsman pipeline (that's phase two); kept separate from the server-mic
  // toggle in ChatPanel to avoid conflating the two.
  let status = $state<AudioStatus>('idle');
  let detail = $state<string | null>(null);
  let level = $state(0);

  let audio: BrowserAudio | null = null;
  const live = $derived(status === 'live' || status === 'connecting');

  async function toggle() {
    if (live) {
      await audio?.stop();
      audio = null;
      return;
    }
    detail = null;
    audio = new BrowserAudio({
      onStatus: (s, d) => {
        status = s;
        detail = d ?? null;
      },
      onLevel: (p) => (level = p)
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
      data-on={live}
      onclick={toggle}
      disabled={status === 'connecting'}
      aria-pressed={live}
      title={live ? 'Stop browser audio' : 'Start browser audio (mic → backend → playback)'}
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
    <div class="meter" aria-hidden="true">
      <div class="bar" style:width={`${Math.min(100, Math.round(level * 140))}%`}></div>
    </div>
  </header>
  <p class="hint">
    Preview: your mic is streamed to the backend and looped back to your
    speakers. Talking to the helmsman over browser audio is the next step.
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
  }
  .lbl { font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace; }

  .meter {
    flex: 1;
    height: 6px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
  }
  .bar {
    height: 100%;
    background: var(--good);
    transition: width 80ms linear;
  }
  .hint { color: var(--fg-muted); font-size: 0.76rem; margin: 0; }
  .status {
    padding: 0.35rem 0.55rem;
    border-radius: 4px;
    border: 1px solid var(--bad);
    background: rgba(248, 81, 73, 0.08);
    font-size: 0.78rem;
  }
</style>
