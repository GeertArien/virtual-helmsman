<script lang="ts">
  import AuditLogPanel from '$lib/components/AuditLogPanel.svelte';
  import DeleteDocumentPanel from '$lib/components/DeleteDocumentPanel.svelte';
  import PendingReviewPanel from '$lib/components/PendingReviewPanel.svelte';

  /** Bumped each time the pending panel reports a successful upload or
   *  whenever the per-batch review submit comes back -- the audit panel
   *  reacts to the bump and re-fetches. Keeps the two panels decoupled
   *  without a shared store. */
  let auditRefreshKey = $state(0);

  function bumpAudit() {
    auditRefreshKey += 1;
  }
</script>

<main>
  <PendingReviewPanel onUploaded={bumpAudit} />
  <AuditLogPanel refreshKey={auditRefreshKey} />
  <DeleteDocumentPanel />
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    /* No fixed height -- the page scrolls naturally as panels stack. */
    max-width: 60rem;
    margin: 0 auto;
    width: 100%;
  }
</style>
