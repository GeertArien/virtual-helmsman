<script lang="ts">
  import { connectSimulator, disconnectSimulator } from '$lib/api';
  import type { ShipStateEvent, SimulatorConnectionState } from '$lib/api';

  // Not named `state`: a local binding by that name makes Svelte read the
  // `$state` rune below as a store subscription on it.
  let {
    ship,
    link
  }: {
    ship: ShipStateEvent | null;
    link: SimulatorConnectionState | null;
  } = $props();

  let busy = $state(false);
  let error: string | null = $state(null);

  function prettyOrder(o: string): string {
    return o.replace(/_/g, ' ');
  }

  /** Helm convention: negative is port, positive is starboard, zero midships.
   *  Spoken the way it would be read back on the bridge, rather than as a
   *  signed number an operator has to decode. */
  function prettyRudder(deg: number): string {
    const rounded = Math.round(deg);
    if (rounded === 0) return 'midships';
    return `${Math.abs(rounded)}° ${rounded < 0 ? 'port' : 'stbd'}`;
  }

  /** Exercise clock as h:mm:ss. */
  function prettySimTime(s: number): string {
    const total = Math.floor(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${h}:${pad(m)}:${pad(sec)}`;
  }

  /** Chart-style position: degrees + decimal minutes, hemisphere letter.
   *  Thousandths of a minute (GPS-display convention) resolve ~1.85 m, so the
   *  2 Hz telemetry visibly ticks while the ship has any way on. */
  function prettyCoord(deg: number, positive: string, negative: string): string {
    const hemi = deg < 0 ? negative : positive;
    const abs = Math.abs(deg);
    const whole = Math.floor(abs);
    const minutes = (abs - whole) * 60;
    return `${whole}° ${minutes.toFixed(3)}′ ${hemi}`;
  }

  const linkLabel = $derived(
    link === 'connected'
      ? 'connected'
      : link === 'connecting'
        ? 'connecting…'
        : link === 'stale'
          ? 'no data'
          : link === 'disconnected'
            ? 'disconnected'
            : 'unknown'
  );

  /** Only a live link carries orders to the ship; everything else means the
   *  helm will refuse. Say so plainly rather than colouring four states. */
  const linkHint = $derived(
    link === 'connected'
      ? 'Orders reach the ship.'
      : link === 'connecting'
        ? 'Waiting for the simulator. Orders are refused until it answers.'
        : link === 'stale'
          ? 'The simulator stopped sending data. Reconnecting; orders are refused.'
          : link === 'disconnected'
            ? 'Link closed. Orders are refused.'
            : ''
  );

  // Everything except a closed link offers Disconnect -- including
  // 'connecting': on a box with no simulator that is the *usual* state, and
  // it is exactly when an operator wants to stop the reconnect loop (e.g. to
  // release the console). Offering only Connect there is a dead button: the
  // backend treats connect-while-supervising as a retry request, not a stop.
  const canDisconnect = $derived(link !== null && link !== 'disconnected');

  async function toggle() {
    busy = true;
    error = null;
    try {
      if (canDisconnect) {
        await disconnectSimulator();
      } else {
        await connectSimulator();
      }
      // The resulting state arrives as a connection_state event; no need to
      // apply the response here and risk fighting the stream.
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel">
  <div class="head">
    <h2>Ship state</h2>
    {#if link !== null}
      <span class="link" data-state={link} title={linkHint}>
        <span class="badge"></span>
        <span class="mono">{linkLabel}</span>
      </span>
      <button onclick={toggle} disabled={busy} title={linkHint}>
        {canDisconnect ? 'Disconnect' : 'Connect'}
      </button>
    {/if}
  </div>

  {#if link !== null && link !== 'connected'}
    <p class="hint">{linkHint}</p>
  {/if}
  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if ship}
    <div class="grid">
      <div class="cell">
        <div class="label">Heading</div>
        <div class="value mono">{Math.round(ship.heading_deg)}°</div>
      </div>
      <div class="cell">
        <div class="label">Speed</div>
        <div class="value mono">{ship.speed_kn.toFixed(1)} <small>kn</small></div>
      </div>
      <div class="cell">
        <div class="label">Rudder</div>
        <div class="value mono">{prettyRudder(ship.rudder_angle_deg)}</div>
      </div>
      <div class="cell">
        <div class="label">Engine</div>
        <div class="value mono">{prettyOrder(ship.engine_order)}</div>
      </div>
      <div class="cell">
        <div class="label">Sim time</div>
        <div class="value mono">
          {ship.sim_time_s != null ? prettySimTime(ship.sim_time_s) : '—'}
        </div>
      </div>
      <div class="cell">
        <div class="label">Position</div>
        {#if ship.lat_deg != null && ship.lon_deg != null}
          <div class="value mono coord">
            {prettyCoord(ship.lat_deg, 'N', 'S')}<br />
            {prettyCoord(ship.lon_deg, 'E', 'W')}
          </div>
        {:else}
          <div class="value mono">—</div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="empty">No state reported yet.</div>
  {/if}
</section>

<style>
  .panel {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }
  h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin: 0;
    font-weight: 600;
    flex: 1;
  }
  .link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    color: var(--fg-muted);
  }
  .badge {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--fg-muted);
    flex: none;
  }
  .link[data-state='connected'] .badge {
    background: var(--good);
    box-shadow: 0 0 8px var(--good);
  }
  .link[data-state='connecting'] .badge { background: var(--warn); }
  .link[data-state='stale'] .badge { background: var(--warn); }
  .link[data-state='disconnected'] .badge { background: var(--fg-muted); }
  button {
    background: var(--bg-elev-2);
    color: inherit;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.2rem 0.55rem;
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: var(--fg-muted); }
  button:disabled { opacity: 0.5; cursor: default; }
  .hint,
  .error {
    margin: 0 0 0.6rem 0;
    font-size: 0.75rem;
    color: var(--fg-muted);
  }
  .error { color: var(--bad); }
  .empty { color: var(--fg-muted); font-style: italic; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  .cell {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
  }
  .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-muted);
    margin-bottom: 0.25rem;
  }
  .value { font-size: 1.5rem; font-weight: 600; }
  .coord { font-size: 1.05rem; line-height: 1.35; }
  small { color: var(--fg-muted); font-size: 0.85rem; font-weight: 400; }
</style>
