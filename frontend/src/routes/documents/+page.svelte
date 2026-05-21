<script lang="ts">
  import DeleteDocumentPanel from '$lib/components/DeleteDocumentPanel.svelte';
</script>

<main>
  <p class="hint">
    This page manages documents already ingested into qdrant. To add a new
    document, head to <a href="/review">Review</a> — uploads go through an n8n
    workflow with a human-in-the-loop chunk review step.
  </p>
  <DeleteDocumentPanel />
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    height: calc(100vh - 3.5rem);
    min-height: 0;
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
