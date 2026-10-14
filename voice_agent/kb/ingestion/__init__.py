"""In-backend HITL document ingestion (the local replacement for n8n's
``ingestion_with_hitl`` + ``webapp_api`` workflows).

Selected via ``review.backend: local``; see ``docs/LOCAL_INGESTION.md``.
The package import is light -- pypdf / LangChain / Langfuse are deferred
until an upload actually needs them.
"""
