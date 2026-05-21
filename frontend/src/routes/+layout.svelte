<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import Header from '$lib/components/Header.svelte';
  import { live, startLiveStream } from '$lib/liveState.svelte';

  let { children } = $props();

  // Open the event stream once for the whole app so route changes don't
  // reconnect the WebSocket. SPA mode means this runs purely client-side.
  onMount(() => startLiveStream());
</script>

<Header session={live.session} state={live.connection} active={page.url.pathname} />
{@render children?.()}
