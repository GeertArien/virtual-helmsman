"""PDF text extraction for the local ingestion pipeline.

Replaces the n8n ``Extract PDF Text`` node with :mod:`pypdf` (part of the
optional ``langgraph`` extra). Extraction quality differs slightly from
n8n's extractor -- both produce plain text that the cleaning + chunking
steps normalise, so downstream behaviour is equivalent.

The import is deferred so the package (and the test suite) loads without the
extra installed; a missing dependency surfaces as a clear error at upload
time rather than at process start.
"""

from __future__ import annotations

import io


class PdfExtractionError(Exception):
    """Raised when no text could be extracted (image-only, corrupt, encrypted)."""


def extract_pdf_text(content: bytes) -> str:
    """Return the concatenated page text of a PDF, or raise.

    Raises :class:`PdfExtractionError` for anything the operator would
    diagnose as "PDF extractie mislukt" -- unreadable file, password
    protection, or a document with no extractable text (scanned images).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise PdfExtractionError(
            "PDF extraction requires the optional dependency pypdf. "
            'Install it with `pip install -e ".[langgraph]"`.'
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise PdfExtractionError("PDF is password-protected.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 -- pypdf raises a zoo of types
        raise PdfExtractionError(f"PDF could not be parsed: {exc}") from exc

    if not text.strip():
        raise PdfExtractionError(
            "No extractable text found (image-only or empty PDF)."
        )
    return text
