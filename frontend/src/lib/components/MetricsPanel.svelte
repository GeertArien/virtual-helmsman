<script lang="ts">
  import type { TurnMetricsEvent } from '$lib/api';

  let { turns }: { turns: TurnMetricsEvent[] } = $props();

  const V2V_TARGET_MS = 800;

  /** Most-recent turn's voice-to-voice latency, big and easy to read. */
  const latest = $derived(turns.at(-1));

  /** Trailing p50 over the most recent 20 turns; outliers (cold start, stalls)
   *  are typical for this stack so a median is more informative than an average. */
  const recentP50 = $derived(p50(turns.slice(-20).map((t) => t.metrics_ms.voice_to_voice_ms ?? 0).filter(Boolean)));

  function p50(values: number[]): number | null {
    if (values.length === 0) return null;
    const s = [...values].sort((a, b) => a - b);
    const i = Math.floor((s.length - 1) * 0.5);
    return s[i];
  }

  function statusFor(ms: number): string {
    if (ms <= V2V_TARGET_MS) return 'good';
    if (ms <= V2V_TARGET_MS * 1.5) return 'warn';
    return 'bad';
  }

  function fmt(ms: number | undefined): string {
    return ms == null ? '—' : `${Math.round(ms)} ms`;
  }
</script>

<section class="panel">
  <h2>Latency</h2>
  <div class="headline">
    <div class="big">
      <span class="value mono" data-status={latest ? statusFor(latest.metrics_ms.voice_to_voice_ms ?? 0) : 'neutral'}>
        {fmt(latest?.metrics_ms.voice_to_voice_ms)}
      </span>
      <span class="label">voice → voice (last)</span>
    </div>
    <div class="med">
      <span class="value mono">{fmt(recentP50 ?? undefined)}</span>
      <span class="label">p50 (last {Math.min(turns.length, 20)})</span>
    </div>
    <div class="med">
      <span class="value mono">{V2V_TARGET_MS} ms</span>
      <span class="label">target</span>
    </div>
  </div>

  {#if latest}
    <table class="breakdown">
      <thead>
        <tr>
          <th>stage</th><th>last turn</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>stt</td><td class="mono">{fmt(latest.metrics_ms.stt_latency_ms)}</td></tr>
        <tr><td>llm ttft</td><td class="mono">{fmt(latest.metrics_ms.llm_ttft_ms)}</td></tr>
        <tr><td>llm total</td><td class="mono">{fmt(latest.metrics_ms.llm_total_ms)}</td></tr>
        <tr><td>tts ttfa</td><td class="mono">{fmt(latest.metrics_ms.tts_ttfa_ms)}</td></tr>
      </tbody>
    </table>
    <div class="footnote">turn #{latest.turn_index} · {turns.length} total</div>
  {:else}
    <div class="empty">No turns measured yet.</div>
  {/if}
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0 0 0.75rem 0;
    font-weight: 600;
  }
  .empty { color: var(--fg-muted); font-style: italic; }
  .headline {
    display: flex;
    gap: 1rem;
    align-items: flex-end;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
  }
  .big .value { font-size: 1.75rem; font-weight: 700; }
  .med .value { font-size: 1rem; font-weight: 500; }
  .value[data-status='good'] { color: var(--good); }
  .value[data-status='warn'] { color: var(--warn); }
  .value[data-status='bad'] { color: var(--bad); }
  .label {
    display: block;
    color: var(--fg-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.15rem;
  }
  .breakdown {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  .breakdown th, .breakdown td {
    padding: 0.25rem 0.4rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .breakdown th {
    color: var(--fg-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.72rem;
  }
  .breakdown tr:last-child td { border-bottom: none; }
  .footnote { color: var(--fg-muted); font-size: 0.75rem; margin-top: 0.5rem; }
</style>
