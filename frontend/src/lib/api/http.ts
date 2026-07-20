/**
 * Shared HTTP plumbing for the typed API client modules: where the backend
 * lives, and how its errors are surfaced.
 */

/** Where the Python control plane is reachable. Override via the URL query
 *  (?api=http://host:port) for quick swaps without rebuilding. */
export function backendUrl(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8765';
  const fromQuery = new URLSearchParams(window.location.search).get('api');
  return fromQuery ?? 'http://127.0.0.1:8765';
}

/** Equivalent for the WebSocket; derived from `backendUrl` so a single override
 *  configures both. */
export function wsUrl(): string {
  const http = backendUrl();
  return http.replace(/^http/i, 'ws') + '/ws/events';
}

/** Friendly Error subclass so UI can show server messages without leaking
 *  raw fetch internals. The backend's 4xx/5xx body (when JSON) is preserved
 *  on `.detail` for components that want to render it. */
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export async function readError(res: Response): Promise<ApiError> {
  let detail: unknown = null;
  let message = `HTTP ${res.status}`;
  try {
    const ct = res.headers.get('content-type') ?? '';
    if (ct.includes('application/json')) {
      detail = await res.json();
      // FastAPI default error shape: { detail: "..." } or { detail: [...] }
      if (detail && typeof detail === 'object' && 'detail' in detail) {
        const d = (detail as { detail: unknown }).detail;
        if (typeof d === 'string') message = d;
      }
    } else {
      const text = await res.text();
      if (text) message = text;
    }
  } catch {
    // ignore -- the bare HTTP code message is fine
  }
  return new ApiError(res.status, message, detail);
}
