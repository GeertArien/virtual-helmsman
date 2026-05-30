<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import Header from '$lib/components/Header.svelte';
  import TransparencyModal from '$lib/components/TransparencyModal.svelte';
  import { live, startLiveStream } from '$lib/liveState.svelte';

  let { children } = $props();

  // Open the event stream once for the whole app so route changes don't
  // reconnect the WebSocket. SPA mode means this runs purely client-side.
  onMount(() => startLiveStream());
</script>

<!-- AI Act Art. 50 gate: rendered first so it overlays everything and blocks
     interaction until the cursist acknowledges. Remounts on every page load. -->
<TransparencyModal />

<Header session={live.session} state={live.connection} active={page.url.pathname} />
{@render children?.()}
