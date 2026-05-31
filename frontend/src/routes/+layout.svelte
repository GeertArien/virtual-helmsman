<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import Header from '$lib/components/Header.svelte';
  import TransparencyModal from '$lib/components/TransparencyModal.svelte';
  import { live, startLiveStream } from '$lib/liveState.svelte';
  import { logAuditEvent } from '$lib/api';

  let { children } = $props();

  // Open the event stream once for the whole app so route changes don't
  // reconnect the WebSocket. SPA mode means this runs purely client-side.
  onMount(() => startLiveStream());

  /** Version of the transparency declaration the user is acknowledging. Bump
   *  this when documentation/transparantieverklaring.md changes materially so
   *  the audit trail records which text was accepted. */
  const DISCLAIMER_VERSION = '1.0';

  /** Log the Art. 50 acknowledgement to the n8n audit trail. Best-effort and
   *  non-blocking: the gate has already closed by the time this runs, and a
   *  failure (n8n down, etc.) must never affect the user — so we fire and
   *  swallow. No PII/identity is sent, per the modal's privacy promise. */
  function onTransparencyAck(): void {
    void logAuditEvent({
      document_naam: `transparantieverklaring_v${DISCLAIMER_VERSION}`,
      actie: 'art50_acknowledged',
      resultaat: 'OK'
    }).catch((err) => console.warn('Art. 50 audit-event log failed (non-blocking):', err));
  }
</script>

<!-- AI Act Art. 50 gate: rendered first so it overlays everything and blocks
     interaction until the cursist acknowledges. Remounts on every page load. -->
<TransparencyModal onAcknowledge={onTransparencyAck} />

<Header session={live.session} state={live.connection} active={page.url.pathname} />
{@render children?.()}
