"""The local HITL ingestion engine: upload -> review pause -> finalize.

In-backend replacement for the n8n ``ingestion_with_hitl`` workflow. Two
phases bridged by the ``pending_review_chunks`` table (the store-backed
analogue of n8n's datatable + Wait node):

* **ingest** (fired by ``POST /api/review/upload``, runs as a background
  task after the route's immediate 202): extract PDF text -> clean ->
  LLM document summary -> chunk (strategy from the form) -> complete
  metadata -> write the pending batch. Failures terminate the run quietly
  with an audit row (``Fout — PDF extractie mislukt`` or
  ``llm_error_ingestion``), exactly like the workflow's error branches.
* **finalize** (fired by ``POST /api/review/{batch_id}/resume``): apply the
  reviewer's decisions -> delete the pending rows -> all-rejected short
  circuit, or ensure the Qdrant collection, recompute BM25 ``avg_len``,
  embed with ``bge-m3``, upsert, and write the success audit row.

The document summary is the pipeline's one LLM call; it goes through
LangChain's ``ChatOpenAI`` with the optional Langfuse callback handler, both
imported lazily (the ``langgraph`` extra). ``summarize_fn`` / ``extract_fn``
are injectable so tests drive the engine without the extra, the network, or
real PDFs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx

from voice_agent.kb.ingestion import chunking, metadata, qdrant
from voice_agent.kb.ingestion.pdf import PdfExtractionError, extract_pdf_text
from voice_agent.kb.ingestion.store import IngestionStore
from voice_agent.logging_setup import get_logger

SummarizeFn = Callable[[str, str], Awaitable[str]]
ExtractFn = Callable[[bytes], str]


class IngestionEngine:
    """Owns the store, the HTTP client, and the two pipeline phases."""

    def __init__(
        self,
        cfg: Any,
        *,
        default_model: str | None = None,
        client: httpx.AsyncClient | None = None,
        summarize_fn: SummarizeFn | None = None,
        extract_fn: ExtractFn | None = None,
    ) -> None:
        self._cfg = cfg
        self._default_model = default_model
        self._client = client or httpx.AsyncClient(
            timeout=cfg.request_timeout_seconds
        )
        self._summarize_fn = summarize_fn
        self._extract_fn = extract_fn or extract_pdf_text
        self._store = IngestionStore(cfg.db_path)
        self._log = get_logger("ingestion")
        # Strong refs to in-flight background ingests so they aren't GC'd.
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def store(self) -> IngestionStore:
        return self._store

    # ---- phase 1: upload -> pending batch --------------------------------

    def start_ingest(
        self,
        *,
        content: bytes,
        filename: str,
        document_type: str,
        collection_name: str,
        categories: str,
        chunking_strategy: str,
        model: str | None,
    ) -> None:
        """Kick off the async ingest; returns immediately (the 202 contract)."""
        task = asyncio.create_task(
            self._run_ingest(
                content=content,
                filename=filename,
                document_type=document_type,
                collection_name=collection_name,
                categories=categories,
                chunking_strategy=chunking_strategy,
                model=model or self._default_model or "",
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_ingest(
        self,
        *,
        content: bytes,
        filename: str,
        document_type: str,
        collection_name: str,
        categories: str,
        chunking_strategy: str,
        model: str,
    ) -> None:
        try:
            text = await asyncio.to_thread(self._extract_fn, content)
            clean = chunking.clean_pdf_text(text)
        except (PdfExtractionError, ValueError) as exc:
            self._log.warning("ingest_pdf_failed", filename=filename, error=str(exc))
            await asyncio.to_thread(
                self._store.insert_audit,
                filename or "onbekend",
                metadata.ACTIE_INGESTIE,
                metadata.audit_pdf_failed(),
            )
            return

        try:
            summary = await self._summarize(clean, model)
        except Exception as exc:  # noqa: BLE001 -- any LLM failure ends the run
            status = getattr(exc, "status_code", None)
            self._log.error("ingest_summary_failed", filename=filename, error=str(exc))
            await asyncio.to_thread(
                self._store.insert_audit,
                filename or "onbekend",
                metadata.ACTIE_LLM_ERROR,
                metadata.audit_llm_error(str(exc), status, len(clean)),
            )
            return

        chunks, strategy_tag = chunking.chunk_text(clean, chunking_strategy)
        enriched = metadata.complete_metadata(
            chunks,
            clean_text=clean,
            filename=filename,
            doc_summary=summary,
            document_type=document_type,
            collection_name=collection_name,
            categories=categories,
            strategy_tag=strategy_tag,
        )
        batch_id = metadata.new_batch_id()
        await asyncio.to_thread(self._store.write_pending_batch, batch_id, enriched)
        self._log.info(
            "ingest_batch_pending",
            batch_id=batch_id,
            filename=filename,
            chunks=len(enriched),
            strategy=strategy_tag,
        )

    async def _summarize(self, text: str, model: str) -> str:
        """Doc-summary LLM call (LangChain + optional Langfuse), or the stub."""
        if self._summarize_fn is not None:
            return await self._summarize_fn(text, model)

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from voice_agent import tracing

        handler = tracing.build_callback_handler(self._cfg)
        llm = ChatOpenAI(
            model=model,
            base_url=self._cfg.llm_base_url,
            api_key=self._cfg.resolved_llm_api_key() or "not-needed",
            temperature=0,
            max_tokens=300,
        )
        resp = await llm.ainvoke(
            [SystemMessage(metadata.SUMMARY_SYSTEM), HumanMessage(text)],
            config={"callbacks": [handler]} if handler else None,
        )
        return str(resp.content).strip()

    # ---- phase 2: decisions -> qdrant -------------------------------------

    async def resume(
        self, batch_id: str, decisions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Apply review decisions and finalize the batch.

        Returns the outcome summary, or ``None`` when the batch is unknown /
        already resumed (the router maps that to 404, preserving the one-shot
        semantics of n8n's resume URL).
        """
        chunks = await asyncio.to_thread(self._store.get_batch_chunks, batch_id)
        if not chunks:
            return None

        result = metadata.apply_decisions(chunks, decisions)
        filename = chunks[0].get("filename", "unknown.pdf")
        collection = chunks[0].get("Collection_Name") or self._cfg.default_collection_name
        # Pending rows are cleared regardless of outcome (n8n's parallel
        # Clear Reviewed Rows branch) -- a failed finalize should surface in
        # the audit log, not leave a zombie batch in the review queue.
        await asyncio.to_thread(self._store.delete_batch, batch_id)

        if result.all_rejected:
            await asyncio.to_thread(
                self._store.insert_audit,
                filename,
                metadata.ACTIE_INGESTIE,
                metadata.audit_all_rejected(batch_id, result),
            )
            self._log.info("resume_all_rejected", batch_id=batch_id)
            return {
                "status": "rejected",
                "approved": 0,
                "edited": 0,
                "rejected": result.rejected,
                "indexed": 0,
            }

        qdrant_headers = self._cfg.resolved_qdrant_headers() or None
        api_key = self._cfg.resolved_llm_api_key()
        embed_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

        await qdrant.ensure_collection(
            self._client,
            qdrant_url=self._cfg.qdrant_url,
            collection=collection,
            headers=qdrant_headers,
        )
        avg_len = metadata.compute_avg_len(result.kept)
        embeddings = await qdrant.embed_texts(
            self._client,
            base_url=self._cfg.llm_base_url,
            model=self._cfg.embedding_model,
            texts=[c["text"] for c in result.kept],
            headers=embed_headers,
        )
        points = qdrant.build_points(result.kept, embeddings, avg_len, batch_id)
        await qdrant.upsert_points(
            self._client,
            qdrant_url=self._cfg.qdrant_url,
            collection=collection,
            points=points,
            headers=qdrant_headers,
        )
        await asyncio.to_thread(
            self._store.insert_audit,
            filename,
            metadata.ACTIE_INGESTIE,
            metadata.audit_success(batch_id, result, len(result.kept)),
        )
        self._log.info(
            "resume_ingested",
            batch_id=batch_id,
            indexed=len(result.kept),
            rejected=result.rejected,
            edited=result.edited,
        )
        return {
            "status": "ingested",
            "approved": result.approved + result.default_approved,
            "edited": result.edited,
            "rejected": result.rejected,
            "indexed": len(result.kept),
        }

    # ---- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        """Close the HTTP client; lets in-flight ingest tasks finish naturally."""
        await self._client.aclose()
