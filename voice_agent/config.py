"""Pydantic config schema, YAML loader, and environment-variable overrides.

The schema mirrors ``config.yaml`` one-to-one. ``extra="forbid"`` makes typos
in the config file fail validation rather than being silently ignored.

Environment overrides (applied before validation):

* ``LLM_BASE_URL``      -> ``llm.base_url``
* ``SIMULATOR_BACKEND`` -> ``simulator.backend``

``LLM_API_KEY`` is not a config field: the key is always read at runtime from
the environment variable named by ``llm.api_key_env`` (default ``LLM_API_KEY``)
via :meth:`LlmConfig.resolved_api_key`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from voice_agent.backends.simulator.base import EngineOrder


class _Base(BaseModel):
    """Base model: reject unknown keys so config typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class SttConfig(_Base):
    backend: Literal["parakeet_onnx", "parakeet_nemo", "whisper"] = "parakeet_onnx"
    model: str
    device: str = "cuda"
    language: str = "en"
    # Pre-quantized weights variant published alongside the FP32 baseline.
    # Default ``"int8"`` -> INT8 encoder (~700 MB), about a 4x VRAM reduction
    # over the FP32 baseline (~2.4 GB) on the dominant cost. The 1-2%
    # relative WER hit is negligible for the helmsman command vocabulary,
    # and a fresh install fits comfortably in a 2 GB GPU. Set to ``None``
    # to fetch the FP32 weights instead. Forwarded to
    # ``onnx_asr.load_model(quantization=...)``; only applies to the
    # parakeet_onnx backend (and only when the upstream repo publishes
    # the matching files -- istupakov/parakeet-tdt-0.6b-v2-onnx and v3-
    # onnx both ship int8 variants).
    quantization: Literal["int8"] | None = "int8"


class TtsConfig(_Base):
    backend: Literal["kokoro", "piper"] = "kokoro"
    voice: str
    device: str = "cuda"


class VadConfig(_Base):
    backend: Literal["silero"] = "silero"
    threshold: float = 0.5
    # Silence (seconds) before speech is considered ended. Pipecat's default of
    # 0.2 is aggressive enough to split a command at the pauses between words;
    # a longer value keeps a paused-but-continuing utterance as one segment.
    stop_secs: float = 0.8


class TurnConfig(_Base):
    backend: Literal["smart_turn_v3", "vad_only"] = "smart_turn_v3"
    device: str = "cpu"


class LlmConfig(_Base):
    """LLM backend configuration.

    Two backends are supported:

    * ``openai_compatible`` -- direct chat-completion call against an
      OpenAI-shaped HTTP server (e.g. LM Studio). Command parsing only;
      no RAG. Uses ``base_url`` + ``model`` + ``api_key_env``.
    * ``langgraph`` -- in-backend command parsing + hybrid-RAG question
      answering (LangGraph + LangChain + Langfuse; see
      ``docs/LANGGRAPH_BACKEND.md``). ``base_url`` is the LM Studio ``/v1``
      URL (as for ``openai_compatible``); RAG additionally uses ``qdrant_*``
      + ``embedding_model`` + ``retrieval_top_k`` and honours ``rerank`` /
      ``expansion``. Optional Langfuse tracing via ``langfuse_*`` and
      per-turn runtime audit rows via ``audit_*``.

    Per-field applicability is annotated below. ``timeout_seconds`` applies
    to both -- raise it for ``langgraph`` since the RAG branch can take
    ~10-20s.
    """

    backend: Literal["openai_compatible", "langgraph"] = "openai_compatible"
    # Full /v1 base URL for both backends ("http://localhost:1234/v1").
    base_url: str
    timeout_seconds: float = 30.0

    # --- model ---------------------------------------------------------
    # The set of LLMs we've evaluated for the helmsman's JSON-structured
    # output path. Adding another model means appending here -- keeps
    # config.yaml and the /config UI dropdown in lockstep, and surfaces
    # typos as a clear ValidationError instead of a silent LM Studio 404.
    # openai_compatible sends it directly to the chat-completion endpoint;
    # langgraph applies it to every LLM call (intent classify, command
    # parse, rerank, RAG answer).
    model: Literal[
        "google/gemma-4-e4b",
        "unsloth/gemma-4-e4b-it",
        "qwen/qwen3.5-9b",
        "ministral-3-8b-instruct-2512",
        "google/gemma-4-e2b",
        "nvidia/nemotron-3-nano-4b",
    ]
    api_key_env: str = "LLM_API_KEY"
    max_retries: int = 1

    # --- langgraph only -------------------------------------------------
    # Toggles the RAG-branch LLM reranker. False is the faster path.
    rerank: bool = True
    # Toggles RAG-branch adjacent-chunk expansion (Qdrant scroll for
    # chunk_id +/-1 -- solves the Rule-15 chunk-boundary problem). Independent
    # of ``rerank``; any combination is valid.
    expansion: bool = True
    # Qdrant for the in-backend RAG question branch. ``qdrant_url`` is the
    # Qdrant REST root (e.g. "http://localhost:6333"); leave it unset to run
    # command-only (a question turn then returns a graceful error envelope).
    qdrant_url: str | None = None
    qdrant_collection: str = "maritime_hybrid"
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    # Dense embedding model + Qdrant named vector for the query (must match the
    # collection's dense vector; pinned to bge-m3 / 1024-dim like ingestion).
    embedding_model: str = "text-embedding-bge-m3"
    # Hybrid retrieval breadth before rerank/expansion (each prefetch pulls 2x).
    retrieval_top_k: int = 20
    # Optional Langfuse tracing of every LLM/retrieval step. Keys are read from
    # the env vars named below (blank/unset -> tracing silently disabled).
    langfuse_enabled: bool = False
    langfuse_host: str | None = None
    langfuse_public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    langfuse_secret_key_env: str = "LANGFUSE_SECRET_KEY"
    # Per-turn runtime audit rows (command_runtime / question_runtime /
    # llm_error_runtime) written to the shared SQLite audit log so the Audit
    # page shows live helmsman activity. ``audit_db_path`` should match
    # ``review.db_path`` so runtime + ingestion rows share one table.
    audit_enabled: bool = False
    audit_db_path: str = "./data/ingestion.db"

    def resolved_api_key(self) -> str | None:
        """Return the API key from the env var named by ``api_key_env``."""
        return os.environ.get(self.api_key_env)

    def resolved_qdrant_headers(self) -> dict[str, str]:
        """``api-key`` header for Qdrant (langgraph RAG), or ``{}`` if no key set."""
        key = os.environ.get(self.qdrant_api_key_env)
        return {"api-key": key} if key else {}


class SimulatorRealConfig(_Base):
    host: str = "127.0.0.1"
    port: int = 9100
    connect_timeout_seconds: float = 2.0


class SimulatorMockConfig(_Base):
    initial_heading: float = 0.0
    initial_engine_order: EngineOrder = EngineOrder.STOP
    log_commands: bool = True


class SimulatorConfig(_Base):
    backend: Literal["real", "mock"] = "mock"
    real: SimulatorRealConfig = Field(default_factory=SimulatorRealConfig)
    mock: SimulatorMockConfig = Field(default_factory=SimulatorMockConfig)


class AudioConfig(_Base):
    """Browser-audio (WebRTC) settings.

    Voice input/output is always the **browser**: the control plane serves a
    WebRTC signalling endpoint (``POST /api/webrtc/offer``) and runs a
    STT->LLM->TTS pipeline per browser connection, so the dashboard talks to the
    helmsman and hears its reply over WebRTC. The heavy models are loaded once
    at startup and shared across connections. Audio therefore needs both
    ``api.enabled: true`` and the ``webrtc`` extra (``pip install -e ".[webrtc]"``);
    there is no local-hardware audio path.
    """

    # STT input rate used by the offline benchmark harness (scripts/bench_stt.py).
    sample_rate: int = 16000
    # STUN/TURN servers for WebRTC ICE. The default public STUN server is
    # enough for localhost / same-LAN use; add a TURN server for NAT traversal
    # across networks.
    ice_servers: list[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"]
    )


class LoggingConfig(_Base):
    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["json", "console"] = "json"
    conversation_log_path: Path = Path("./logs/conversations")
    metrics_log_path: Path = Path("./logs/metrics")


class ApiConfig(_Base):
    """Control/observability API for the frontend. Disabled by default."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    # CORS origins for the frontend. Wildcard is fine for local dev; tighten
    # to e.g. ["http://localhost:5173"] (Vite default) for stricter setups.
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])


class DocumentsConfig(_Base):
    """Qdrant document management (list + delete). All fields optional --
    when a field is missing the corresponding endpoint returns HTTP 503 with
    a "configure documents.<field>" message rather than failing silently.

    The qdrant payload field names are configurable because different
    ingestion pipelines store metadata under different keys; defaults match
    the common "document_id / title / source / uploaded_at" convention.

    Uploads live under :class:`ReviewConfig` -- this block is read-only
    qdrant management (list / delete).
    """

    # qdrant REST base URL (e.g. "http://127.0.0.1:6333"), collection name,
    # and the env var holding the qdrant API key (if any).
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    # Payload field names used to group points into documents.
    document_id_field: str = "document_id"
    title_field: str = "title"
    source_field: str = "source"
    uploaded_at_field: str = "uploaded_at"
    # Hard cap on points scrolled when listing documents -- keeps the listing
    # call bounded for large collections. Increase if you genuinely have more.
    scroll_limit: int = 10000
    # Request timeout (seconds) for outbound calls to qdrant.
    request_timeout_seconds: float = 30.0


class ReviewConfig(_Base):
    """In-backend HITL document ingestion + chunk review + audit log.

    Runs the ingestion pipeline in this process: LangChain doc-summary
    (optional Langfuse tracing), local SQLite for the pending-review batches
    and the audit log, direct Qdrant collection management + upserts. Serves
    the five ``/api/review/*`` routes the frontend's Documents / Review /
    Audit pages call. See ``docs/LOCAL_INGESTION.md``. Requires the
    ``langgraph`` pip extra (pypdf + LangChain).

    Write endpoints return HTTP 503 with a clear "configure review.<field>"
    message until ``qdrant_url`` and ``llm_base_url`` are set; the read
    endpoints (pending list, audit log) work immediately.
    """

    # SQLite file holding the pending-review batches and the audit log. The
    # pending table IS the HITL pause state -- batches survive restarts.
    db_path: str = "./data/ingestion.db"
    # LM Studio /v1 root for the doc-summary chat call and the bge-m3
    # embeddings (same server the LLM backend uses).
    llm_base_url: str | None = None
    llm_api_key_env: str = "LLM_API_KEY"
    # Qdrant REST root for collection management + upserts.
    qdrant_url: str | None = None
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    # Dense embedding model; pinned to the collection's 1024-dim vector.
    embedding_model: str = "text-embedding-bge-m3"
    # Pre-fill values for the upload form / defaults when the form omits them.
    default_document_type: str = "PDF"
    default_collection_name: str = "maritime_hybrid"
    default_categories: str = "algemeen"
    default_chunking_strategy: Literal[
        "paragraph_aware", "fixed_size"
    ] = "paragraph_aware"
    # Request timeout (seconds) for outbound calls to Qdrant / LM Studio.
    request_timeout_seconds: float = 60.0
    # Optional Langfuse tracing of the doc-summary call. Field names mirror
    # LlmConfig so the same handler factory serves both.
    langfuse_enabled: bool = False
    langfuse_host: str | None = None
    langfuse_public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    langfuse_secret_key_env: str = "LANGFUSE_SECRET_KEY"

    def resolved_llm_api_key(self) -> str | None:
        """LM Studio API key for the ingestion pipeline, or ``None`` if unset."""
        return os.environ.get(self.llm_api_key_env)

    def resolved_qdrant_headers(self) -> dict[str, str]:
        """``api-key`` header for Qdrant, or ``{}`` if no key set."""
        key = os.environ.get(self.qdrant_api_key_env)
        return {"api-key": key} if key else {}


class AppConfig(_Base):
    """Top-level config object, the single source of truth for the agent."""

    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig = Field(default_factory=VadConfig)
    turn_detection: TurnConfig = Field(default_factory=TurnConfig)
    llm: LlmConfig
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with env-var overrides applied."""
    data = copy.deepcopy(data)
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        data.setdefault("llm", {})["base_url"] = base_url
    sim_backend = os.environ.get("SIMULATOR_BACKEND")
    if sim_backend:
        data.setdefault("simulator", {})["backend"] = sim_backend
    return data


def parse_config(data: dict[str, Any]) -> AppConfig:
    """Apply env overrides to a config mapping and validate it."""
    return AppConfig(**_apply_env_overrides(data))


def load_config(path: str | Path = "./config.yaml") -> AppConfig:
    """Load, env-override, and validate the config file at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return parse_config(data)
