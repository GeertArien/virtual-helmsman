"""Pure helpers for the LangGraph helmsman backend.

Everything in this module is a plain function over plain data -- no
LangChain, LangGraph, Langfuse, or network. It is a faithful port of the
JavaScript ``Code`` nodes in the n8n ``virtual_helmsman_unified`` workflow
(intent parsing, RRF top-3, LLM-rerank parsing, adjacent-chunk expansion,
RAG-answer parsing, and the canonical reply shaping). Keeping it pure makes
the retrieval/answer logic unit-testable without standing up Qdrant or
LM Studio, and mirrors the split used by the n8n adapter's ``_translate_*``
helpers.

The graph nodes in :mod:`graph` call into these functions; the heavy LLM /
Qdrant calls live in :mod:`graph` and :mod:`retrieval`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from voice_agent.actions.dispatch import BRIDGE_LOST

# Mirrors the n8n "Classify Intent" system prompt. One-word COMMAND/QUESTION.
CLASSIFY_SYSTEM = (
    "Classify the bridge message as one word. Reply COMMAND if it is an order "
    "to the helm or engine (steering, rudder, speed, heading, autopilot, "
    "anchor) or a vessel status request. Reply QUESTION if it asks for "
    "information about maritime rules, COLREGS, VTS, ports or navigation. "
    "Reply with only one word: COMMAND or QUESTION."
)

# Mirrors the n8n "Rerank Chunks" system prompt (RankGPT-style listwise).
RERANK_SYSTEM = (
    "You rank document passages by relevance to a question. Reply with ONLY a "
    'JSON object in the form {"top_3": [i, j, k]} where i, j, k are the 1-based '
    "labels of the three most relevant passages from the list below, in order "
    "of decreasing relevance. No prose, no markdown, just the JSON object."
)

# Mirrors the n8n "Build Prompt" RAG system prompt verbatim so answer quality
# and the rule-isolation behaviour match the workflow exactly.
RAG_SYSTEM_PROMPT = (
    "You are the knowledge assistant of the Virtual Helmsman. You answer "
    "questions about maritime regulations, COLREGS, VTS procedures, port "
    "information, and navigation, exclusively based on the supplied context.\n\n"
    "Rules:\n"
    "1. Answer ONLY based on the given context. If the answer is not there, "
    "say so honestly inside the answer field.\n"
    "2. Answer ALWAYS in English, regardless of the question's language. If the "
    "question is not in English, answer in English and add a brief note that "
    "English is the supported language.\n"
    "3. If the question names a specific rule, article, or number (e.g. "
    "'Rule 15', 'Article III'), use ONLY text that explicitly addresses that "
    "rule or article. Adjacent rules that appear in the same chunk (e.g. the "
    "text of Rule 14 just before a 'Rule 15' header) MUST NOT be used to answer "
    "the question. If the specific rule text is not fully present in the "
    "context, say so honestly rather than citing a neighbouring rule.\n\n"
    "Output format — respond with ONLY a single JSON object with exactly these "
    "two keys, no markdown, no preamble:\n"
    '{\n  "answer": "<your answer text in English>",\n'
    '  "source_chunk_id": "<chunk_id of the chunk that most directly supports '
    'the answer, e.g. chunk_025>"\n}\n'
    "If the answer is not in the context, still emit valid JSON: set answer to "
    "the honest 'not in context' explanation and source_chunk_id to the "
    "chunk_id whose topic is closest to the question (or the first chunk if "
    "nothing is close)."
)

# JSON-schema response_format for the RAG answer call, lifted from the n8n
# "RAG Answer" node. Hard-constrains the model to {answer, source_chunk_id}.
RAG_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The English-language answer derived from the supplied context",
                },
                "source_chunk_id": {
                    "type": "string",
                    "description": "chunk_id of the chunk that most directly supports the answer",
                },
            },
            "required": ["answer", "source_chunk_id"],
            "additionalProperties": False,
        },
    },
}


def latest_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the most recent user-role message's string content, or ``None``.

    Same selection rule as the n8n adapter: the pipeline's user-aggregator
    appends one user message per turn, so the last user-role entry with string
    content is what the helmsman should answer. Robust to context-management
    changes that prepend a system prompt or prior assistant turns.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return None


def strip_code_fence(text: str) -> str:
    """Drop a wrapping ```/```json markdown fence, mirroring the n8n regex."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)
    return s.strip()


def parse_intent(raw: str) -> str:
    """Map a classifier completion to ``"command"`` or ``"question"``.

    Mirrors the n8n "Parse Intent" rule: any casing of QUESTION -> question,
    otherwise command (so an empty/garbled completion defaults to command,
    the safe branch that never touches Qdrant).
    """
    return "question" if "QUESTION" in (raw or "").upper() else "command"


def error_envelope(reason: str, spoken: str = BRIDGE_LOST) -> dict[str, Any]:
    """A HelmsmanResponse-shaped dict encoding an ``error`` action.

    Same contract as the n8n adapter's ``_error_envelope`` -- emitted on any
    failure (LLM/Qdrant unreachable, parse failure) so the downstream
    JsonActionProcessor always sees well-formed JSON and the helmsman speaks
    a graceful fallback instead of crashing the pipeline.
    """
    return {
        "action": {
            "type": "error",
            "error_type": "bridge_error",
            "reason": reason,
            "suggestion": (
                "Check that the LM Studio and Qdrant services are running and "
                "reachable at the configured URLs."
            ),
        },
        "response": spoken,
    }


def command_envelope(raw: str) -> dict[str, Any]:
    """Shape a command-parser completion into an internal HelmsmanResponse.

    Mirrors the n8n "Format Command Reply" node: parse the model JSON and
    return ``{action, response}``. On any parse failure (or a missing
    ``action``) fall back to a ``parse_failure`` error action carrying the
    cleaned model text, so the result is always parseable.
    """
    cleaned = strip_code_fence(raw)
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("action"), dict):
        return {
            "action": parsed["action"],
            "response": parsed.get("response") or cleaned,
        }

    return {
        "action": {
            "type": "error",
            "error_type": "parse_failure",
            "reason": "Model output was not valid JSON",
            "suggestion": "",
        },
        "response": cleaned or BRIDGE_LOST,
    }


def answer_envelope(output: str) -> dict[str, Any]:
    """Internal HelmsmanResponse for the question branch.

    The synthetic ``answer`` action carries no fields; the spoken/displayed
    text is the RAG answer with its citation line already appended. Matches
    the n8n adapter's question-branch translation.
    """
    return {"action": {"type": "answer"}, "response": output}


def map_qdrant_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Qdrant query points into chunk dicts (n8n "Format Results")."""
    chunks: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        payload = p.get("payload") or {}
        chunks.append(
            {
                "rank": i + 1,
                "score": p.get("score"),
                "text": payload.get("text", ""),
                "filename": payload.get("filename"),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "document_type": payload.get("document_type", ""),
                "document_summary": payload.get("document_summary", ""),
            }
        )
    return chunks


def rrf_top3(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RRF bypass path: take the top 3 fused hits unchanged (n8n bypass node)."""
    top = []
    for c in chunks[:3]:
        merged = dict(c)
        merged.update(
            {
                "original_rank": c.get("rank"),
                "rerank_rank": None,
                "rerank_bypassed": True,
            }
        )
        top.append(merged)
    return top


def parse_rerank_indices(raw: str, num_chunks: int) -> list[int]:
    """Parse the rerank LLM's ``{"top_3": [...]}`` into 0-based indices.

    Mirrors the n8n "Apply Rerank" node: 1-based labels -> 0-based positions,
    drop out-of-range, and fall back to ``[0, 1, 2]`` if nothing usable came
    back so the answer branch always has chunks.
    """
    cleaned = strip_code_fence(raw)
    raw_indices: list[int] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("top_3"), list):
            raw_indices = [n for n in parsed["top_3"] if isinstance(n, int)]
    except (ValueError, TypeError):
        raw_indices = []

    indices = [n - 1 for n in raw_indices]
    indices = [n for n in indices if 0 <= n < num_chunks]
    if not indices:
        indices = [i for i in (0, 1, 2) if i < num_chunks]
    return indices[:3]


def apply_rerank(chunks: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    """Select reranked winners by index (n8n "Apply Rerank" projection)."""
    top: list[dict[str, Any]] = []
    for new_idx, idx in enumerate(indices):
        if not (0 <= idx < len(chunks)):
            continue
        c = dict(chunks[idx])
        c.update({"original_rank": chunks[idx].get("rank"), "rerank_rank": new_idx + 1})
        top.append(c)
    return top


def _pad(n: int) -> str:
    return f"{n:03d}"


def neighbour_ids(winners: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Compute adjacent chunk_ids (±1) per filename (n8n "Build Neighbour Requests").

    Returns ``{filename: [chunk_id, ...]}``. Only winners whose chunk_id
    matches ``chunk_<digits>`` and that carry a filename contribute. chunk_000
    has no lower neighbour.
    """
    groups: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for w in winners:
        chunk_id = w.get("chunk_id") or ""
        filename = w.get("filename")
        m = re.match(r"^chunk_(\d+)$", chunk_id)
        if not m or not filename:
            continue
        num = int(m.group(1))
        bucket = seen.setdefault(filename, set())
        ordered = groups.setdefault(filename, [])
        candidates = []
        if num > 0:
            candidates.append(f"chunk_{_pad(num - 1)}")
        candidates.append(f"chunk_{_pad(num + 1)}")
        for cid in candidates:
            if cid not in bucket:
                bucket.add(cid)
                ordered.append(cid)
    return groups


def merge_neighbours(
    winners: list[dict[str, Any]], neighbour_points: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Dedup winners + scrolled neighbour points by ``filename::chunk_id``.

    Mirrors the n8n "Merge Neighbours" node: winners first (preserving order),
    then any neighbour points not already present, tagged ``is_neighbour``.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for w in winners:
        key = f"{w.get('filename')}::{w.get('chunk_id')}"
        if key not in seen:
            seen.add(key)
            merged.append(w)
    for p in neighbour_points:
        payload = p.get("payload") or {}
        chunk_id = payload.get("chunk_id")
        filename = payload.get("filename")
        if not chunk_id or filename == "__no_winners__":
            continue
        key = f"{filename}::{chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "rank": None,
                "score": None,
                "text": payload.get("text", ""),
                "filename": filename or "",
                "page": payload.get("page"),
                "chunk_id": chunk_id,
                "document_type": payload.get("document_type", ""),
                "document_summary": payload.get("document_summary", ""),
                "is_neighbour": True,
            }
        )
    return merged


def build_rerank_user(question: str, chunks: list[dict[str, Any]]) -> str:
    """Render the rerank user message (1-based labelled passages, 800-char cap)."""
    passages = "\n\n".join(
        f"[{i + 1}] {(c.get('text') or '')[:800]}" for i, c in enumerate(chunks)
    )
    return f"Question: {question}\n\nPassages:\n{passages}\n\nReturn JSON: {{\"top_3\": [i, j, k]}}"


def build_rag_user(question: str, chunks: list[dict[str, Any]]) -> str:
    """Render the RAG user message: labelled context blocks + the question."""
    parts = []
    for i, c in enumerate(chunks):
        chunk_id = c.get("chunk_id") or f"chunk_{i}"
        parts.append(
            f"[chunk_id: {chunk_id} | file: {c.get('filename') or '?'} | "
            f"page: {c.get('page') or '?'}]\n{c.get('text') or ''}"
        )
    context = "\n\n---\n\n".join(parts)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def parse_rag_answer(raw: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the RAG completion + resolve citation metadata (n8n "Parse RAG Response")."""
    cleaned = strip_code_fence(raw)
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        parsed = None

    parse_failure = False
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        answer = parsed["answer"].strip()
        src = parsed.get("source_chunk_id")
        source_chunk_id = src.strip() if isinstance(src, str) else None
    else:
        answer = cleaned
        source_chunk_id = None
        parse_failure = True

    source_chunk: dict[str, Any] | None
    if source_chunk_id:
        source_chunk = next(
            (c for c in chunks if c.get("chunk_id") == source_chunk_id),
            chunks[0] if chunks else None,
        )
    else:
        source_chunk = chunks[0] if chunks else None

    citation_reliable = (
        not parse_failure
        and bool(source_chunk_id)
        and bool(source_chunk and source_chunk.get("chunk_id") == source_chunk_id)
    )

    if source_chunk:
        citation = (
            f"{source_chunk.get('filename')}, page {source_chunk.get('page')} "
            f"({source_chunk.get('chunk_id')})"
        )
    else:
        citation = "Source unknown"

    return {
        "answer": answer,
        "citation": citation,
        "source_chunk_id": source_chunk_id
        or (source_chunk.get("chunk_id") if source_chunk else None),
        "source_chunk": source_chunk,
        "citation_reliable": citation_reliable,
        "parse_failure": parse_failure,
        "raw_model_output": raw,
    }


def format_question_output(parsed: dict[str, Any]) -> str:
    """Assemble the user-facing answer string: answer + a Source: line."""
    return f"{parsed.get('answer', '')}\n\nSource: {parsed.get('citation') or 'Source unknown'}"


# --- runtime audit rows (the n8n "Log Runtime Command/Question" analogue) ----
#
# Written per turn to the shared audit store so the Audit page shows live
# helmsman activity, not just ingestion. Field values mirror API.md iteration 9.

RUNTIME_COMMAND = "command_runtime"
RUNTIME_QUESTION = "question_runtime"
RUNTIME_LLM_ERROR = "llm_error_runtime"


def _bool_str(value: Any) -> str:
    return "true" if value else "false"


def command_audit_row(envelope: dict[str, Any]) -> tuple[str, str, str]:
    """``(document_naam, actie, resultaat)`` for a command turn."""
    action = envelope.get("action") or {}
    output = envelope.get("response", "") or ""
    return (
        "n.v.t. (command)",
        RUNTIME_COMMAND,
        f"action_type={action.get('type')} | output={output[:120]}",
    )


def question_audit_row(parsed: dict[str, Any], output: str) -> tuple[str, str, str]:
    """``(document_naam, actie, resultaat)`` for a RAG question turn."""
    chunk = parsed.get("source_chunk") or {}
    filename = chunk.get("filename") or "n.v.t."
    return (
        filename,
        RUNTIME_QUESTION,
        (
            f"chunk={parsed.get('source_chunk_id')} | "
            f"citation_reliable={_bool_str(parsed.get('citation_reliable'))} | "
            f"parse_failure={_bool_str(parsed.get('parse_failure'))} | "
            f"output={(output or '')[:120]}"
        ),
    )


def error_audit_row(reason: str) -> tuple[str, str, str]:
    """``(document_naam, actie, resultaat)`` for a failed turn (LLM/Qdrant down)."""
    return ("onbekend", RUNTIME_LLM_ERROR, f"error={(reason or 'unknown')[:200]}")
