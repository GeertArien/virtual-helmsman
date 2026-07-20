"""Pydantic config schema, YAML loader, and environment-variable overrides.

The schema mirrors ``config.yaml`` one-to-one. ``extra="forbid"`` makes typos
in the config file fail validation rather than being silently ignored.

Shared service blocks -- ``qdrant``, ``lm_studio``, ``langfuse``, ``database``
-- are configured once at the top level and consumed by every subsystem (the
LLM backend, the Documents API, and the HITL ingestion pipeline). The
per-subsystem ``llm`` / ``documents`` / ``review`` blocks keep only what is
unique to them. The subsystems receive flat *runtime* views (``LlmRuntime`` /
``DocumentsRuntime`` / ``IngestionRuntime``) assembled from the shared blocks
via :meth:`AppConfig.llm_runtime` etc., so the backend code reads a single flat
object without knowing the YAML is split into shared blocks.

Environment overrides (applied before validation):

* ``LLM_BASE_URL``      -> ``lm_studio.base_url``
* ``SIMULATOR_BACKEND`` -> ``simulator.backend``

API keys are never config values: the YAML names the *env var* (e.g.
``lm_studio.api_key_env``) and the secret is read from the environment at
runtime.
"""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voice_agent.backends.simulator.base import EngineOrder
from voice_agent.qdrant import api_key_headers


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


# --- shared service blocks --------------------------------------------------


class QdrantConfig(_Base):
    """Qdrant connection, shared by RAG retrieval, ingestion, and Documents.

    ``url`` unset means no Qdrant: the RAG branch returns a graceful error and
    the Documents / ingestion endpoints return HTTP 503 until it's set.
    """

    url: str | None = None
    api_key_env: str = "QDRANT_API_KEY"
    collection: str = "maritime_hybrid"

    def resolved_headers(self) -> dict[str, str]:
        """``api-key`` header from the named env var, or ``{}`` if unset."""
        return api_key_headers(self.api_key_env)


class LmStudioConfig(_Base):
    """The OpenAI-compatible ``/v1`` server (LM Studio), shared by the LLM
    backend (chat) and both RAG + ingestion (bge-m3 embeddings)."""

    # Full ``/v1`` base URL, e.g. "http://localhost:1234/v1". Required: the
    # agent always needs an LLM endpoint.
    base_url: str
    api_key_env: str = "LLM_API_KEY"
    # Dense embedding model id; also the Qdrant dense named-vector name.
    embedding_model: str = "text-embedding-bge-m3"

    def resolved_api_key(self) -> str | None:
        """The API key from the env var named by ``api_key_env`` (or ``None``)."""
        return os.environ.get(self.api_key_env)


class LangfuseConfig(_Base):
    """Optional Langfuse tracing, shared by the LLM backend and ingestion."""

    enabled: bool = False
    host: str | None = None
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"


class DatabaseConfig(_Base):
    """The local SQLite store: HITL pending-review batches + the audit log.

    One path shared by the ingestion pipeline (pending batches + ingestion
    audit rows) and the LLM backend's runtime audit writer, so both land in
    one ``audit_log`` table.
    """

    path: str = "./data/ingestion.db"


class LlmConfig(_Base):
    """LLM backend selection + tuning (connection lives in ``lm_studio``).

    Two backends:

    * ``openai_compatible`` -- direct chat-completion against the ``lm_studio``
      server. Command parsing only; no RAG.
    * ``langgraph`` -- in-backend command parsing + hybrid-RAG question
      answering (LangGraph + LangChain + optional Langfuse; see
      ``docs/LANGGRAPH_BACKEND.md``). RAG uses the shared ``qdrant`` +
      ``lm_studio.embedding_model`` blocks and honours ``rerank`` /
      ``expansion`` / ``retrieval_top_k``.

    ``timeout_seconds`` applies to both -- raise it for ``langgraph`` since the
    RAG branch can take ~10-20s.
    """

    backend: Literal["openai_compatible", "langgraph"] = "openai_compatible"
    # The chat model id sent to the lm_studio server. Free-form string so a new
    # model is a config change, not a code edit. Models evaluated so far for the
    # helmsman's JSON-structured output: google/gemma-4-e4b,
    # unsloth/gemma-4-e4b-it, qwen/qwen3.5-9b, ministral-3-8b-instruct-2512,
    # google/gemma-4-e2b, nvidia/nemotron-3-nano-4b.
    model: str
    timeout_seconds: float = 30.0
    max_retries: int = 1

    # --- langgraph only -------------------------------------------------
    # ``commands_only`` drops the intent classifier and the whole RAG branch:
    # every turn is a single LLM call to the command parser, which refuses
    # questions as out-of-scope. That saves one LLM round-trip (~hundreds of
    # ms) per command and removes the qdrant + embedding-model runtime
    # dependency -- the right trade for a pure conning demo. ``full`` keeps
    # command/question routing with hybrid-RAG answers (issue #21).
    mode: Literal["full", "commands_only"] = "full"
    # Toggles the RAG-branch LLM reranker. False is the faster path.
    rerank: bool = True
    # Toggles RAG-branch adjacent-chunk expansion (Qdrant scroll for
    # chunk_id +/-1 -- solves the Rule-15 chunk-boundary problem). Independent
    # of ``rerank``; any combination is valid.
    expansion: bool = True
    # Hybrid retrieval breadth before rerank/expansion (each prefetch pulls 2x).
    retrieval_top_k: int = 20
    # Per-turn runtime audit rows (command_runtime / question_runtime /
    # llm_error_runtime) written to the shared ``database`` SQLite store so the
    # Audit page shows live helmsman activity alongside ingestion events.
    audit_enabled: bool = False


class SimulatorRealConfig(_Base):
    """Endpoint and link-supervision settings for the real simulator.

    The transport needs two address/port pairs, not one: the link is
    connectionless, so each side independently says where it sends and where it
    listens. The two sides are mirror images -- the simulator sends to the port
    we listen on, and vice versa; change both sides together or the link goes
    quiet with no error anywhere.

    The ports and frame rate are properties of the simulator installation and
    ship with the vendor integration notes -- they are deliberately **not**
    recorded in this repository. ``0`` means "not configured"; the real backend
    refuses to build until real values are set (in a local config file, never
    committed).
    """

    remote_host: str = "127.0.0.1"
    remote_port: int = 0  # we send here; the simulator listens here
    # Loopback by default, deliberately: this socket *believes* whatever it
    # receives -- there is no authentication on the link, so a wide bind lets
    # any host on the network feed the agent fake ship state (and mark the
    # link "connected" with no simulator running). Set 0.0.0.0 (or a specific
    # interface) only when the simulator genuinely runs on another machine.
    local_host: str = "127.0.0.1"
    local_port: int = 0  # we listen here; the simulator sends here

    # How long connect() waits for the first frame before reporting the link as
    # not established. It keeps trying regardless -- this only decides when the
    # state stops being "connecting".
    connect_timeout_seconds: float = 5.0

    # Nominal frame rate of the simulator's broadcast, used to judge link
    # health and to pace the session thread. Comes with the vendor integration.
    expected_fps: float = 0.0

    # Frames are continuous on a live link, so silence is the only loss signal
    # available. Tolerate a few missed frames before declaring the link stale.
    stale_after_missed_frames: float = 20.0

    # Reconnect backoff. The link is cheap to re-establish and retrying is
    # harmless, so this goes on forever -- capped so a long outage does not turn
    # into a long silence after the simulator returns.
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 15.0


class SimulatorMockConfig(_Base):
    initial_heading: float = 0.0
    initial_engine_order: EngineOrder = EngineOrder.STOP
    log_commands: bool = True


class SimulatorConfig(_Base):
    backend: Literal["real", "mock"] = "mock"
    real: SimulatorRealConfig = Field(default_factory=SimulatorRealConfig)
    mock: SimulatorMockConfig = Field(default_factory=SimulatorMockConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_real_keys(cls, data: Any) -> Any:
        """Handle the pre-rename ``simulator.real`` keys (``host``/``port``).

        Those keys configured an integration that was never implemented (the
        old ``real`` backend was all NotImplementedError stubs), yet every
        previously shipped config carried them -- including mock-backend
        configs where the block is inert. With ``extra="forbid"`` they would
        fail validation and stop the agent from starting at all.

        So: a *mock* user's stale block is dropped with a warning -- refusing
        to start over dead keys would punish people the rename cannot affect.
        A *real* user must decide what the new fields should be (the link now
        needs a listen endpoint as well as a send endpoint), so that stays a
        hard error, but one that names the rename instead of pydantic's bare
        "extra inputs are not permitted".
        """
        if not isinstance(data, dict):
            return data
        real = data.get("real")
        if not isinstance(real, dict):
            return data
        legacy = [key for key in ("host", "port") if key in real]
        if not legacy:
            return data
        if data.get("backend") == "real":
            raise ValueError(
                "simulator.real was reworked: 'host'/'port' are now "
                "'remote_host'/'remote_port' (where the simulator listens) "
                "plus 'local_host'/'local_port' (where this agent listens) -- "
                "the link needs both endpoints. See "
                "config.examples/config.real_sim.yaml for a working block."
            )
        for key in legacy:
            real.pop(key)
        warnings.warn(
            "config: dropped obsolete simulator.real keys "
            f"{legacy} (renamed to remote_*/local_*; harmless while "
            "simulator.backend is 'mock', but update the file)",
            stacklevel=2,
        )
        return data


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
    """Qdrant document management (list + delete) -- payload-mapping knobs only.

    The Qdrant connection lives in the shared ``qdrant`` block; this block keeps
    only the payload field names (different ingestion pipelines store metadata
    under different keys) and the listing limits. Uploads live under
    :class:`ReviewConfig` (the HITL ingestion pipeline).
    """

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
    """In-backend HITL document ingestion -- ingestion-specific knobs only.

    The SQLite store (``database``), Qdrant (``qdrant``), the LM Studio server +
    embedding model (``lm_studio``), and Langfuse (``langfuse``) are the shared
    blocks; this block keeps only the upload-form defaults and the ingestion
    request timeout. Write endpoints return HTTP 503 until ``qdrant.url`` is set;
    read endpoints (pending list, audit log) work immediately. Requires the
    ``langgraph`` pip extra (pypdf + LangChain). See ``docs/LOCAL_INGESTION.md``.
    """

    # Pre-fill values for the upload form / defaults when the form omits them.
    default_document_type: str = "PDF"
    default_categories: str = "algemeen"
    default_chunking_strategy: Literal[
        "paragraph_aware", "fixed_size"
    ] = "paragraph_aware"
    # Request timeout (seconds) for outbound calls to Qdrant / LM Studio.
    request_timeout_seconds: float = 60.0


# --- runtime facades --------------------------------------------------------
# Flat views the subsystems consume, assembled from the shared + per-subsystem
# blocks. Keeping the consumer code on a flat object means the YAML can split
# the settings into shared blocks without rippling through every backend.


@dataclass
class LlmRuntime:
    """Flat config the LLM backend (openai_compatible / langgraph) reads."""

    backend: str
    base_url: str
    model: str
    mode: str
    timeout_seconds: float
    max_retries: int
    rerank: bool
    expansion: bool
    retrieval_top_k: int
    qdrant_url: str | None
    qdrant_collection: str
    embedding_model: str
    langfuse_enabled: bool
    langfuse_host: str | None
    langfuse_public_key_env: str
    langfuse_secret_key_env: str
    audit_enabled: bool
    audit_db_path: str
    api_key_env: str
    qdrant_api_key_env: str

    def resolved_api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def resolved_qdrant_headers(self) -> dict[str, str]:
        return api_key_headers(self.qdrant_api_key_env)


@dataclass
class DocumentsRuntime:
    """Flat config the Documents (qdrant list/delete) router reads."""

    qdrant_url: str | None
    qdrant_collection: str
    qdrant_api_key_env: str
    document_id_field: str
    title_field: str
    source_field: str
    uploaded_at_field: str
    scroll_limit: int
    request_timeout_seconds: float


@dataclass
class IngestionRuntime:
    """Flat config the HITL ingestion engine + review router read."""

    db_path: str
    request_timeout_seconds: float
    llm_base_url: str
    llm_api_key_env: str
    qdrant_url: str | None
    qdrant_api_key_env: str
    embedding_model: str
    default_document_type: str
    default_collection_name: str
    default_categories: str
    default_chunking_strategy: str
    langfuse_enabled: bool
    langfuse_host: str | None
    langfuse_public_key_env: str
    langfuse_secret_key_env: str

    def resolved_llm_api_key(self) -> str | None:
        return os.environ.get(self.llm_api_key_env)

    def resolved_qdrant_headers(self) -> dict[str, str]:
        return api_key_headers(self.qdrant_api_key_env)


class AppConfig(_Base):
    """Top-level config object, the single source of truth for the agent."""

    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig = Field(default_factory=VadConfig)
    turn_detection: TurnConfig = Field(default_factory=TurnConfig)
    llm: LlmConfig
    lm_studio: LmStudioConfig
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)

    def llm_runtime(self) -> LlmRuntime:
        """Assemble the flat LLM-backend config from the shared blocks."""
        return LlmRuntime(
            backend=self.llm.backend,
            base_url=self.lm_studio.base_url,
            model=self.llm.model,
            mode=self.llm.mode,
            timeout_seconds=self.llm.timeout_seconds,
            max_retries=self.llm.max_retries,
            rerank=self.llm.rerank,
            expansion=self.llm.expansion,
            retrieval_top_k=self.llm.retrieval_top_k,
            qdrant_url=self.qdrant.url,
            qdrant_collection=self.qdrant.collection,
            embedding_model=self.lm_studio.embedding_model,
            langfuse_enabled=self.langfuse.enabled,
            langfuse_host=self.langfuse.host,
            langfuse_public_key_env=self.langfuse.public_key_env,
            langfuse_secret_key_env=self.langfuse.secret_key_env,
            audit_enabled=self.llm.audit_enabled,
            audit_db_path=self.database.path,
            api_key_env=self.lm_studio.api_key_env,
            qdrant_api_key_env=self.qdrant.api_key_env,
        )

    def documents_runtime(self) -> DocumentsRuntime:
        """Assemble the flat Documents-router config from the shared blocks."""
        return DocumentsRuntime(
            qdrant_url=self.qdrant.url,
            qdrant_collection=self.qdrant.collection,
            qdrant_api_key_env=self.qdrant.api_key_env,
            document_id_field=self.documents.document_id_field,
            title_field=self.documents.title_field,
            source_field=self.documents.source_field,
            uploaded_at_field=self.documents.uploaded_at_field,
            scroll_limit=self.documents.scroll_limit,
            request_timeout_seconds=self.documents.request_timeout_seconds,
        )

    def ingestion_runtime(self) -> IngestionRuntime:
        """Assemble the flat ingestion-engine config from the shared blocks."""
        return IngestionRuntime(
            db_path=self.database.path,
            request_timeout_seconds=self.review.request_timeout_seconds,
            llm_base_url=self.lm_studio.base_url,
            llm_api_key_env=self.lm_studio.api_key_env,
            qdrant_url=self.qdrant.url,
            qdrant_api_key_env=self.qdrant.api_key_env,
            embedding_model=self.lm_studio.embedding_model,
            default_document_type=self.review.default_document_type,
            default_collection_name=self.qdrant.collection,
            default_categories=self.review.default_categories,
            default_chunking_strategy=self.review.default_chunking_strategy,
            langfuse_enabled=self.langfuse.enabled,
            langfuse_host=self.langfuse.host,
            langfuse_public_key_env=self.langfuse.public_key_env,
            langfuse_secret_key_env=self.langfuse.secret_key_env,
        )


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with env-var overrides applied."""
    data = copy.deepcopy(data)
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        data.setdefault("lm_studio", {})["base_url"] = base_url
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
