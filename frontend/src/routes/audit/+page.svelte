<script lang="ts">
  import AuditLogPanel from '$lib/components/AuditLogPanel.svelte';
</script>

<main>
  <AuditLogPanel />
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    max-width: 60rem;
    margin: 0 auto;
    width: 100%;
  }
</style>
