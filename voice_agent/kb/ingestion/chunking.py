"""Pure text-processing ports of the n8n ingestion workflow's Code nodes.

Faithful Python ports of three nodes from ``ingestion_with_hitl``:

* ``Clean Text``            -> :func:`clean_pdf_text`
* ``Chunk: Paragraph Aware``-> :func:`chunk_paragraph_aware`
* ``Chunk: Fixed Size``     -> :func:`chunk_fixed_size`

Everything here is a plain function over strings -- no I/O, no LLM -- so the
chunking behaviour (and its quirks, e.g. the sentence-boundary trim that
relies on the 75-char overlap to re-cover trimmed text) is unit-testable and
provably identical to what the n8n pipeline produced. Constants match the
workflow: 800-char windows, 75-char overlap, 725 merge limit, 400 min tail.
"""

from __future__ import annotations

import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 75
MIN_CHUNK_SIZE = 400
MERGE_LIMIT = CHUNK_SIZE - CHUNK_OVERLAP

PARAGRAPH_AWARE = "paragraph_aware"
FIXED_SIZE = "fixed_size"
# The strategy tag written into chunk metadata / qdrant payloads. The
# paragraph-aware tag carries the historical suffix the n8n workflow used, so
# old and new chunks remain filterable on the same value.
STRATEGY_TAGS = {
    PARAGRAPH_AWARE: "paragraph_aware_sentence_boundary",
    FIXED_SIZE: "fixed_size",
}


def clean_pdf_text(text: str) -> str:
    """Strip repeated headers/footers and page numbers from extracted PDF text.

    Port of the n8n ``Clean Text`` node: short lines that repeat across
    (estimated) pages are dropped as boilerplate unless they look like content
    (bullets, "Key: value" lines, VHF/channel references, multi-digit numbers,
    or lines containing ``.``/``@``/``/``); then standalone page-number lines
    are removed and blank runs collapsed.
    """
    lines = text.split("\n")
    line_freq: dict[str, int] = {}
    for line in lines:
        normalized = line.strip()
        if len(normalized) < 80:
            line_freq[normalized] = line_freq.get(normalized, 0) + 1

    # JS Math.round rounds half away from zero; int(x + 0.5) matches for the
    # positive lengths we have here.
    estimated_pages = max(3, int(len(text) / 2500 + 0.5))
    repeat_threshold = max(4, estimated_pages * 0.6)

    kept: list[str] = []
    for line in lines:
        normalized = line.strip()
        if len(normalized) == 0:
            kept.append(line)
            continue
        freq = line_freq.get(normalized, 0)
        if len(normalized) < 80 and freq >= repeat_threshold:
            kept.append("")
            continue
        if len(normalized) < 80 and freq > 3:
            if (
                re.match(r"^[•\-\*]", normalized)
                or re.match(r"^[A-Z][a-z].*:", normalized)
                or re.search(r"VHF|Channel|ZEDIS|Pilot", normalized, re.IGNORECASE)
                or re.search(r"\d{2,}", normalized)
                or re.search(r"[.@/]", normalized)
            ):
                kept.append(line)
            else:
                kept.append("")
            continue
        kept.append(line)

    clean = "\n".join(kept)
    clean = re.sub(r"^[ \t]*\d{1,3}[ \t]*$", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"^[ \t]*-\s*\d{1,3}\s*-[ \t]*$", "", clean, flags=re.MULTILINE)
    clean = re.sub(
        r"^[ \t]*Page\s+\d{1,3}(?:\s+of\s+\d{1,3})?[ \t]*$",
        "",
        clean,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    clean = re.sub(r"p\.\s*\d{1,3}$", "", clean, flags=re.MULTILINE | re.IGNORECASE)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _split_block(block: str, max_size: int) -> list[str]:
    """Recursive paragraph -> line -> sentence -> word split (n8n splitBlock)."""
    for pattern, strip in ((r"\n\n+", True), (r"\n", True), (r"(?<=[.!?;:])\s+", False)):
        parts = re.split(pattern, block)
        if strip:
            parts = [p.strip() for p in parts]
        parts = [p for p in parts if p]
        if len(parts) > 1 and any(len(p) <= max_size for p in parts):
            out: list[str] = []
            for p in parts:
                out.extend(_split_block(p, max_size) if len(p) > max_size else [p])
            return out

    # Hard fallback: cut at the last space before max_size.
    result: list[str] = []
    remaining = block
    while len(remaining) > max_size:
        cut_at = remaining.rfind(" ", 0, max_size + 1)
        if cut_at <= 0:
            cut_at = max_size
        result.append(remaining[:cut_at].strip())
        remaining = remaining[cut_at:].strip()
    if remaining:
        result.append(remaining)
    return result


_SENT_END = re.compile(r"[.!?;:)\]'\"]\s*\n|[.!?]\s+(?=[A-Z(])")


def chunk_paragraph_aware(clean_text: str) -> list[str]:
    """Paragraph-aware chunking -- the v1 default strategy.

    Recursive split on paragraph -> line -> sentence boundaries, merge up to
    ~725 chars, sentence-boundary trim on the last 15% of each non-final
    chunk, tail-merge below 400 chars, then 75-char overlap prefixes.
    """
    if not clean_text:
        raise ValueError("paragraph_aware: no clean_text")

    pieces = _split_block(clean_text, MERGE_LIMIT)

    raw_chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 1 <= MERGE_LIMIT:
            current = f"{current}\n{piece}" if current else piece
        else:
            if current:
                raw_chunks.append(current)
            current = piece
    if current:
        raw_chunks.append(current)

    trimmed: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == len(raw_chunks) - 1:
            trimmed.append(chunk)
            break
        search_from = int(len(chunk) * 0.85)
        tail = chunk[search_from:]
        last_end = -1
        for m in _SENT_END.finditer(tail):
            last_end = search_from + m.start() + len(m.group(0).rstrip())
        if 0 < last_end < len(chunk) - 5:
            trimmed.append(chunk[:last_end].strip())
        else:
            trimmed.append(chunk)

    merged: list[str] = []
    for i, chunk in enumerate(trimmed):
        if i == len(trimmed) - 1 and len(chunk) < MIN_CHUNK_SIZE and merged:
            merged[-1] += "\n" + chunk
        else:
            merged.append(chunk)

    out = [merged[0]]
    for i in range(1, len(merged)):
        prev = merged[i - 1]
        ov = prev[-CHUNK_OVERLAP:]
        nl = ov.find("\n")
        if nl > 0:
            ov = ov[nl + 1 :]
        else:
            sp = ov.find(" ")
            if sp > 0:
                ov = ov[sp + 1 :]
        out.append(ov + "\n" + merged[i])
    return out


def chunk_fixed_size(clean_text: str) -> list[str]:
    """Fixed-size chunking -- the naive 800/75 sliding-window baseline."""
    if not clean_text:
        raise ValueError("fixed_size: no clean_text")
    out: list[str] = []
    pos = 0
    while pos < len(clean_text):
        end = min(pos + CHUNK_SIZE, len(clean_text))
        out.append(clean_text[pos:end])
        if end >= len(clean_text):
            break
        pos = end - CHUNK_OVERLAP
    return out


def chunk_text(clean_text: str, strategy: str) -> tuple[list[str], str]:
    """Dispatch on strategy; unknown values coerce to ``paragraph_aware``.

    Returns ``(chunks, strategy_tag)`` where the tag is what gets written into
    chunk metadata (mirrors the n8n Strategy Switch's fallback output).
    """
    if (strategy or "").lower() == FIXED_SIZE:
        return chunk_fixed_size(clean_text), STRATEGY_TAGS[FIXED_SIZE]
    return chunk_paragraph_aware(clean_text), STRATEGY_TAGS[PARAGRAPH_AWARE]
