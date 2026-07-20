"""Tests for the pure ingestion ports: cleaning, chunking, metadata, decisions.

These pin the Python ports to the observable behaviour of the n8n Code nodes
(``Clean Text``, the two chunkers, ``Chunk + Complete Metadata``,
``Apply Decisions``, ``Calculate avg_len``) without any I/O.
"""

from __future__ import annotations

import pytest

from voice_agent.kb.ingestion import chunking, metadata
from voice_agent.kb.ingestion.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_fixed_size,
    chunk_paragraph_aware,
    chunk_text,
    clean_pdf_text,
)
from voice_agent.kb.ingestion.metadata import (
    DecisionResult,
    apply_decisions,
    audit_all_rejected,
    audit_llm_error,
    audit_pdf_failed,
    audit_success,
    complete_metadata,
    compute_avg_len,
    new_batch_id,
)


# ---------- clean_pdf_text --------------------------------------------------


def test_clean_removes_repeated_headers() -> None:
    # A short header repeated on every "page" is dropped (freq >= threshold,
    # which drops unconditionally); unique body lines are kept.
    pages = []
    for i in range(8):
        pages.append("ACME Shipping Manual")  # repeated header, < 80 chars
        pages.append(f"Body paragraph {i} with enough unique content to stay.")
    text = "\n".join(pages)
    cleaned = clean_pdf_text(text)
    assert "ACME Shipping Manual" not in cleaned
    assert "Body paragraph 3" in cleaned


def test_clean_keeps_protected_repeats() -> None:
    # In the mid-frequency band (freq > 3 but below the page-scaled
    # threshold), lines that look like content (VHF refs) survive while
    # plain boilerplate is dropped. 4 repeats + a long document put the
    # threshold above 4 so the protected branch is the one that fires.
    lines = []
    for _ in range(4):
        lines.append("Call VHF Channel 12")
        lines.append("plain repeated footer")
    text = "\n".join(lines) + "\n" + "x" * 30000  # high page estimate
    cleaned = clean_pdf_text(text)
    assert "Call VHF Channel 12" in cleaned
    assert "plain repeated footer" not in cleaned


def test_clean_strips_page_number_lines() -> None:
    text = "Real content here.\n12\n- 13 -\nPage 14 of 99\nMore content."
    cleaned = clean_pdf_text(text)
    assert "Real content here." in cleaned
    assert "More content." in cleaned
    assert "\n12\n" not in cleaned
    assert "- 13 -" not in cleaned
    assert "Page 14" not in cleaned


def test_clean_collapses_blank_runs() -> None:
    assert "\n\n\n" not in clean_pdf_text("a\n\n\n\n\nb")


# ---------- chunkers ---------------------------------------------------------


def _sentences(n: int) -> str:
    return " ".join(
        f"Sentence number {i} provides regulatory guidance for vessels." for i in range(n)
    )


def test_fixed_size_windows_and_overlap() -> None:
    text = "x" * 2000
    chunks = chunk_fixed_size(text)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # Consecutive windows share exactly CHUNK_OVERLAP chars.
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]
    # Reconstruction: stripping the overlap prefix re-yields the original.
    rebuilt = chunks[0] + "".join(c[CHUNK_OVERLAP:] for c in chunks[1:])
    assert rebuilt == text


def test_paragraph_aware_respects_max_size() -> None:
    text = "\n\n".join(_sentences(6) for _ in range(10))
    chunks = chunk_paragraph_aware(text)
    assert len(chunks) > 1
    # Overlap prefix (<=75) + merge limit (725) -> hard ceiling 800.
    assert all(len(c) <= CHUNK_SIZE + 1 for c in chunks)


def test_paragraph_aware_overlap_prefix() -> None:
    text = "\n\n".join(_sentences(6) for _ in range(10))
    chunks = chunk_paragraph_aware(text)
    # Each chunk after the first starts with (a word-aligned tail of) the
    # previous chunk's last 75 chars, joined by a newline.
    for prev, cur in zip(chunks, chunks[1:]):
        prefix = cur.split("\n", 1)[0]
        assert prefix in prev[-CHUNK_OVERLAP:]


def test_paragraph_aware_single_paragraph_passthrough() -> None:
    text = "One short paragraph that fits in a single chunk."
    assert chunk_paragraph_aware(text) == [text]


def test_chunk_text_dispatch_and_tags() -> None:
    text = _sentences(3)
    chunks, tag = chunk_text(text, "fixed_size")
    assert tag == "fixed_size"
    chunks, tag = chunk_text(text, "paragraph_aware")
    assert tag == "paragraph_aware_sentence_boundary"
    # Unknown strategies silently coerce to the default (n8n fallback output).
    _, tag = chunk_text(text, "llm_semantic")
    assert tag == "paragraph_aware_sentence_boundary"
    _, tag = chunk_text(text, "")
    assert tag == "paragraph_aware_sentence_boundary"


def test_chunkers_reject_empty_text() -> None:
    with pytest.raises(ValueError):
        chunk_paragraph_aware("")
    with pytest.raises(ValueError):
        chunk_fixed_size("")


# ---------- complete_metadata ------------------------------------------------


def _meta(chunks: list[str], clean: str) -> list[dict]:
    return complete_metadata(
        chunks,
        clean_text=clean,
        filename="COLREGS.pdf",
        doc_summary="Summary.",
        document_type="PDF",
        collection_name="maritime_hybrid",
        categories="colregs, rules",
        strategy_tag="paragraph_aware_sentence_boundary",
    )


def test_metadata_fields() -> None:
    clean = "alpha bravo charlie. " * 100
    chunks = chunk_fixed_size(clean)
    out = _meta(chunks, clean)
    assert [c["chunk_id"] for c in out[:3]] == ["chunk_000", "chunk_001", "chunk_002"]
    assert out[0]["page"] == 1 and out[2]["page"] == 2  # floor(idx/2)+1
    assert out[0]["total_chunks"] == len(chunks)
    assert out[0]["categories"] == ["colregs", "rules"]
    assert out[0]["start_char"] == 0
    assert out[1]["start_char"] > 0
    assert out[0]["words_in_text"] == len(chunks[0].split())
    # point_ids are unique and increasing within the batch.
    ids = [c["point_id"] for c in out]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_metadata_defaults_categories() -> None:
    out = complete_metadata(
        ["text"],
        clean_text="text",
        filename="f.pdf",
        doc_summary="s",
        document_type="PDF",
        collection_name="c",
        categories=None,
        strategy_tag="fixed_size",
    )
    assert out[0]["categories"] == ["algemeen"]


def test_metadata_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _meta([], "x")


def test_batch_id_shape() -> None:
    bid = new_batch_id()
    assert bid.startswith("batch_")
    assert len(bid.split("_")) == 3


# ---------- apply_decisions ----------------------------------------------------


def _chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i:03d}",
            "text": f"original text {i}",
            "chunk_length": 15,
            "words_in_text": 3,
        }
        for i in range(n)
    ]


def test_decisions_approve_reject_edit() -> None:
    edited_text = "e" * 60
    result = apply_decisions(
        _chunks(4),
        [
            {"chunk_id": "chunk_000", "action": "approve"},
            {"chunk_id": "chunk_001", "action": "reject", "reason": "fluff"},
            {"chunk_id": "chunk_002", "action": "edit", "edited_text": edited_text},
            # chunk_003 omitted -> default approve
        ],
    )
    assert result.approved == 1
    assert result.rejected == 1
    assert result.edited == 1
    assert result.default_approved == 1
    kept_ids = [c["chunk_id"] for c in result.kept]
    assert kept_ids == ["chunk_000", "chunk_002", "chunk_003"]
    edited = result.kept[1]
    assert edited["text"] == edited_text
    assert edited["chunk_length"] == 60
    assert edited["words_in_text"] == 1


def test_decisions_short_edit_counts_as_reject() -> None:
    result = apply_decisions(
        _chunks(1), [{"chunk_id": "chunk_000", "action": "edit", "edited_text": "tiny"}]
    )
    assert result.rejected == 1
    assert result.all_rejected


def test_decisions_unknown_action_and_unknown_chunk() -> None:
    result = apply_decisions(
        _chunks(1),
        [
            {"chunk_id": "chunk_000", "action": "promote"},  # unknown -> approve
            {"chunk_id": "chunk_999", "action": "reject"},  # unknown id -> ignored
        ],
    )
    assert result.default_approved == 1
    assert result.rejected == 0
    assert not result.all_rejected


def test_decisions_case_insensitive_actions() -> None:
    result = apply_decisions(_chunks(1), [{"chunk_id": "chunk_000", "action": "REJECT"}])
    assert result.rejected == 1


def test_decisions_empty_list_approves_everything() -> None:
    result = apply_decisions(_chunks(3), [])
    assert len(result.kept) == 3
    assert result.default_approved == 3


# ---------- avg_len + audit texts ----------------------------------------------


def test_avg_len() -> None:
    kept = [{"words_in_text": 10}, {"words_in_text": 20}]
    assert compute_avg_len(kept) == 15.0
    with pytest.raises(ValueError):
        compute_avg_len([])


def test_audit_texts_match_n8n_patterns() -> None:
    result = DecisionResult(kept=[{}] * 9, approved=7, rejected=1, edited=1, default_approved=1)
    s = audit_success("batch_1_a", result, 9)
    assert s == (
        "Succes — HITL batch batch_1_a → approved=8 / edited=1 / rejected=1 "
        "— indexed 9 chunks"
    )
    r = audit_all_rejected("batch_1_a", DecisionResult(rejected=5))
    assert "All rejected — HITL batch batch_1_a dropped (5 chunks rejected)" in r
    assert audit_pdf_failed() == "Fout — PDF extractie mislukt"
    e = audit_llm_error("context length exceeded", 400, 12345)
    assert e == "error=context length exceeded | http=400 | input_chars=12345"
    assert "http=n.v.t." in audit_llm_error("boom", None, 0)


def test_summary_prompt_present() -> None:
    assert "NEVER" in metadata.SUMMARY_SYSTEM
    assert "2-3" in metadata.SUMMARY_SYSTEM
    assert chunking.STRATEGY_TAGS  # tags table exists for both strategies
