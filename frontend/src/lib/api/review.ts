/**
 * HITL chunk review + the audit log: mirrors `voice_agent.kb.review`
 * (in-backend ingestion with human review; audit rows in local SQLite).
 */

import { backendUrl, readError } from './http';

/** A chunk awaiting human review. `text` is the chunk's content; `metadata`
 *  is whatever the ingestion pipeline attached. The UI surfaces a subset
 *  (page, chunk_length, document_summary). */
export interface ReviewChunk {
  chunk_id: string;
  text: string;
  metadata: Record<string, unknown>;
}

/** One batch of chunks waiting for review. The backend tracks pending batches
 *  in local SQLite -- submit decisions via `submitDecisions(batch_id, ...)`. */
export interface PendingBatch {
  batch_id: string;
  filename: string;
  collection_name: string;
  created_at: string;
  pending_chunk_count: number;
  chunks: ReviewChunk[];
}

export interface PendingResponse {
  total_pending_batches: number;
  batches: PendingBatch[];
}

export interface ReviewUploadResponse {
  status: string;
  message: string;
}

/** Per-chunk decision. `approve` is the default for omitted chunks; the
 *  backend applies this to the pending batch identified by `batch_id`. */
export interface ChunkDecision {
  chunk_id: string;
  action: 'approve' | 'reject' | 'edit';
  /** Required when action is "edit". Must be ≥ 50 chars after trim or the
   *  backend rejects the chunk -- enforce on the client. */
  edited_text?: string;
  /** Free-text reason, optional. Not persisted in v1. */
  reason?: string;
}

/** POST /api/review/upload -- forward a file + metadata to the in-backend
 *  ingestion pipeline. Returns 202-equivalent ack; the batch appears in the
 *  pending list after a few seconds of background chunking. */
export interface UploadFields {
  document_type?: string;
  collection_name?: string;
  categories?: string;
  chunking_strategy?: 'paragraph_aware' | 'fixed_size';
}

export async function uploadForReview(
  file: File,
  fields: UploadFields = {}
): Promise<ReviewUploadResponse> {
  const form = new FormData();
  form.append('file', file, file.name);
  if (fields.document_type) form.append('Document_Type', fields.document_type);
  if (fields.collection_name) form.append('Collection_Name', fields.collection_name);
  if (fields.categories !== undefined) form.append('Categories', fields.categories);
  if (fields.chunking_strategy) form.append('Chunking_Strategy', fields.chunking_strategy);
  const res = await fetch(`${backendUrl()}/api/review/upload`, {
    method: 'POST',
    body: form
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as ReviewUploadResponse;
}

/** GET /api/review/pending -- list every batch awaiting review. */
export async function fetchPending(): Promise<PendingResponse> {
  const res = await fetch(`${backendUrl()}/api/review/pending`);
  if (!res.ok) throw await readError(res);
  return (await res.json()) as PendingResponse;
}

/** POST /api/review/{batch_id}/resume -- submit decisions for a batch. The
 *  backend applies them to the pending batch and finalizes (embed + upsert to
 *  Qdrant). */
export async function submitDecisions(
  batchId: string,
  decisions: ChunkDecision[]
): Promise<void> {
  const res = await fetch(
    `${backendUrl()}/api/review/${encodeURIComponent(batchId)}/resume`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch_id: batchId, decisions })
    }
  );
  if (!res.ok) throw await readError(res);
}

// --- Audit log (local SQLite audit feed proxied through /api/review) --------

/** One row from the backend's `audit_log` table. `resultaat` is a free-text
 *  Dutch summary (e.g. "Succes — HITL batch ... → approved=8 / edited=1 ...").
 *  v1 ingestion writes three patterns: success, all-rejected, PDF-extract-fail. */
export interface AuditEntry {
  id: number;
  createdAt: string;
  document_naam: string;
  actie: string;
  resultaat: string;
}

export interface AuditLogResponse {
  total_in_log: number;
  total_returned: number;
  applied_filters: { limit: number; actie: string | null; since: string | null };
  entries: AuditEntry[];
}

export interface AuditLogQuery {
  /** Max rows to return. The backend caps at 500. */
  limit?: number;
  /** Exact-match filter on the `actie` column (e.g. "ingestie_hitl"). */
  actie?: string;
  /** ISO-8601 lower bound on `createdAt`. */
  since?: string;
}

/** GET /api/review/audit-log -- recent ingestion outcomes for the activity feed. */
export async function fetchAuditLog(
  query: AuditLogQuery = {}
): Promise<AuditLogResponse> {
  const params = new URLSearchParams();
  if (query.limit !== undefined) params.set('limit', String(query.limit));
  if (query.actie) params.set('actie', query.actie);
  if (query.since) params.set('since', query.since);
  const qs = params.toString();
  const res = await fetch(
    `${backendUrl()}/api/review/audit-log${qs ? `?${qs}` : ''}`
  );
  if (!res.ok) throw await readError(res);
  return (await res.json()) as AuditLogResponse;
}

/** One UI-side audit row to write via POST /api/review/audit-event. All three
 *  fields are required, non-empty, ≤500 chars (enforced backend-side). */
export interface AuditEventInput {
  document_naam: string;
  actie: string;
  resultaat: string;
}

/** POST /api/review/audit-event -- write a single audit row for a UI event the
 *  backend can't observe itself (e.g. the AI Act Art. 50 transparency
 *  acknowledgement). The backend writes it to the local audit log.
 *
 *  Callers should treat this as **best-effort** and not block on it: the
 *  acknowledgement gate must succeed even when the backend is down. Await/catch
 *  only if you genuinely want to observe the failure. */
export async function logAuditEvent(event: AuditEventInput): Promise<void> {
  const res = await fetch(`${backendUrl()}/api/review/audit-event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(event)
  });
  if (!res.ok) throw await readError(res);
}
