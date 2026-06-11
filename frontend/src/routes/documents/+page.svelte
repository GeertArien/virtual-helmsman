<script lang="ts">
  import DeleteDocumentPanel from '$lib/components/DeleteDocumentPanel.svelte';
  import PendingReviewPanel from '$lib/components/PendingReviewPanel.svelte';
</script>

<main>
  <PendingReviewPanel />
  <DeleteDocumentPanel />
  <p class="hint">
    Ingestion outcomes, runtime command parses, and RAG question answers all
    flow into the audit log. See <a href="/audit">Audit</a> for the live feed.
  </p>
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
  .hint {
    margin: 0;
    padding: 0.6rem 0.85rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-muted);
    font-size: 0.85rem;
  }
  .hint a { color: var(--accent); }
</style>
