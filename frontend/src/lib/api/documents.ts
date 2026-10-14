/** Qdrant document management: mirrors `voice_agent.kb.documents`. */

import { backendUrl, readError } from './http';

/** One document known to qdrant. `chunk_count` is the number of vector points
 *  that share this document_id (i.e., what gets deleted). */
export interface DocumentInfo {
  document_id: string;
  title: string | null;
  source: string | null;
  chunk_count: number;
  /** Best-effort upload timestamp (ISO 8601). Null when payload didn't carry one. */
  uploaded_at: string | null;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
}

export interface DeleteResponse {
  status: 'deleted';
  document_id: string;
  deleted_chunks: number;
}

/** GET /api/documents -- list distinct documents currently in the collection. */
export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${backendUrl()}/api/documents`);
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as DocumentListResponse;
  return body.documents;
}

/** DELETE /api/documents/{document_id} -- remove all chunks for one document. */
export async function deleteDocument(documentId: string): Promise<DeleteResponse> {
  const res = await fetch(
    `${backendUrl()}/api/documents/${encodeURIComponent(documentId)}`,
    { method: 'DELETE' }
  );
  if (!res.ok) throw await readError(res);
  return (await res.json()) as DeleteResponse;
}
