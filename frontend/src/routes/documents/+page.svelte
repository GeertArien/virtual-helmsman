<script lang="ts">
  import AddDocumentPanel from '$lib/components/AddDocumentPanel.svelte';
  import DeleteDocumentPanel from '$lib/components/DeleteDocumentPanel.svelte';
</script>

<main>
  <section class="col">
    <AddDocumentPanel />
  </section>
  <section class="col">
    <DeleteDocumentPanel />
  </section>
</main>

<style>
  main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.75rem;
    padding: 0.75rem;
    height: calc(100vh - 3.5rem);
    min-height: 0;
  }
  .col {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-height: 0;
  }
  @media (max-width: 900px) {
    main {
      grid-template-columns: 1fr;
      height: auto;
    }
  }
</style>
