"""SQLite persistence for the local HITL ingestion pipeline.

Two tables in one local database file, replacing the two n8n datatables:

* ``pending_review_chunks`` -- one row per chunk awaiting review (same
  columns as the n8n datatable, minus ``resume_url``: resuming is addressed
  by ``batch_id`` against our own API, not a per-execution callback URL).
  **This table is the HITL pause state**: phase one of the pipeline ends by
  writing it; phase two starts by reading it. It survives restarts, so a
  batch uploaded before a crash is still reviewable after.
* ``audit_log`` -- the ``audit-log-maritime`` analogue (``document_naam`` /
  ``actie`` / ``resultaat`` + auto id/createdAt), serving both the ingestion
  pipeline's audit rows and the UI's ``POST /api/review/audit-event`` writes.

  The Dutch column names are an **explicit decision**, not overlooked n8n
  residue (issue #12 §6): they are the schema of every existing audit
  database and are baked through the API and frontend types. Renaming would
  be a data migration plus an API break for zero functional gain; the
  product's audit audience reads Dutch. Revisit only if the audience changes.

All methods are synchronous and open a short-lived connection per call --
operations are tiny, local, and infrequent (a handful per upload/review),
so connection pooling or WAL tuning would be ceremony. Async callers wrap
them in ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_review_chunks (
    batch_id            TEXT NOT NULL,
    chunk_id            TEXT NOT NULL,
    filename            TEXT NOT NULL,
    collection_name     TEXT NOT NULL,
    chunk_text          TEXT NOT NULL,
    chunk_metadata_json TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (batch_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    document_naam TEXT NOT NULL,
    actie         TEXT NOT NULL,
    resultaat     TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class IngestionStore:
    """Pending-batch + audit-log persistence over one SQLite file."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- pending review --------------------------------------------------

    def write_pending_batch(
        self, batch_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        """Insert one row per chunk; the batch becomes visible to reviewers."""
        created_at = _now_iso()
        rows = [
            (
                batch_id,
                c["chunk_id"],
                c.get("filename", "unknown.pdf"),
                c.get("Collection_Name", ""),
                c.get("text", ""),
                json.dumps(c),
                "pending",
                created_at,
            )
            for c in chunks
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO pending_review_chunks VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )

    def list_pending_batches(self) -> list[dict[str, Any]]:
        """All pending batches grouped for the review UI (newest first).

        Performs the grouping the n8n ``Group by Batch`` node did: one entry
        per batch with its chunks sorted by ``chunk_id``. ``resume_url`` is
        intentionally absent -- the local pipeline resumes by ``batch_id``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_review_chunks WHERE status = 'pending'"
            ).fetchall()

        by_batch: dict[str, dict[str, Any]] = {}
        for r in rows:
            batch = by_batch.setdefault(
                r["batch_id"],
                {
                    "batch_id": r["batch_id"],
                    "filename": r["filename"],
                    "collection_name": r["collection_name"],
                    "created_at": r["created_at"],
                    "pending_chunk_count": 0,
                    "chunks": [],
                },
            )
            try:
                meta = json.loads(r["chunk_metadata_json"])
            except ValueError as exc:
                meta = {"_parse_error": str(exc)}
            batch["chunks"].append(
                {"chunk_id": r["chunk_id"], "text": r["chunk_text"], "metadata": meta}
            )
            batch["pending_chunk_count"] += 1

        batches = sorted(
            by_batch.values(), key=lambda b: b["created_at"], reverse=True
        )
        for b in batches:
            b["chunks"].sort(key=lambda c: c["chunk_id"])
        return batches

    def get_batch_chunks(self, batch_id: str) -> list[dict[str, Any]]:
        """The full metadata dicts for one pending batch, ordered by chunk_id.

        Returns ``[]`` when the batch doesn't exist or was already resumed --
        the caller maps that to a 404, matching the one-shot resume-URL
        semantics of the n8n Wait node.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_metadata_json FROM pending_review_chunks "
                "WHERE batch_id = ? AND status = 'pending' ORDER BY chunk_id",
                (batch_id,),
            ).fetchall()
        return [json.loads(r["chunk_metadata_json"]) for r in rows]

    def delete_batch(self, batch_id: str) -> None:
        """Remove a reviewed batch (the n8n ``Clear Reviewed Rows`` step)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_review_chunks WHERE batch_id = ?", (batch_id,)
            )

    # ---- audit log ---------------------------------------------------------

    def insert_audit(
        self, document_naam: str, actie: str, resultaat: str
    ) -> dict[str, Any]:
        """Insert one audit row; returns it with the generated id/createdAt."""
        created_at = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO audit_log (created_at, document_naam, actie, resultaat) "
                "VALUES (?,?,?,?)",
                (created_at, document_naam, actie, resultaat),
            )
            row_id = cur.lastrowid
        return {
            "id": row_id,
            "createdAt": created_at,
            "document_naam": document_naam,
            "actie": actie,
            "resultaat": resultaat,
        }

    def query_audit(
        self,
        *,
        limit: int | None = None,
        actie: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Filtered audit entries for the Audit page's recent-activity feed.

        Same clamping rules as the n8n ``Filter & Shape Audit Log`` node:
        limit defaults to 50 and caps at 500, an unparseable ``since`` is
        ignored, ordering is newest-first.
        """
        eff_limit = limit if limit is not None and limit > 0 else 50
        eff_limit = min(eff_limit, 500)
        actie_filter = (actie or "").strip()
        since_dt = _parse_ts(since.strip()) if since and since.strip() else None

        with self._connect() as conn:
            total_in_log = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC, id DESC"
            ).fetchall()

        entries: list[dict[str, Any]] = []
        for r in rows:
            if actie_filter and r["actie"] != actie_filter:
                continue
            if since_dt is not None:
                ts = _parse_ts(r["created_at"])
                if ts is None or ts < since_dt:
                    continue
            entries.append(
                {
                    "id": r["id"],
                    "createdAt": r["created_at"],
                    "document_naam": r["document_naam"],
                    "actie": r["actie"],
                    "resultaat": r["resultaat"],
                }
            )
            if len(entries) >= eff_limit:
                break

        return {
            "total_in_log": total_in_log,
            "total_returned": len(entries),
            "applied_filters": {
                "limit": eff_limit,
                "actie": actie_filter or None,
                "since": (since or "").strip() or None,
            },
            "entries": entries,
        }
