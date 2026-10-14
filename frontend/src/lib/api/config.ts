/**
 * Config editor: view + edit `config.yaml`, reload the backend. Mirrors
 * `voice_agent.api.config_router`.
 */

import { backendUrl, readError } from './http';

/** The full AppConfig as a free-form dict -- the structure is whatever
 *  `AppConfig` happens to be on the backend. The frontend reads
 *  /api/config/schema to learn the per-field types and render typed inputs. */
export type ConfigDict = Record<string, unknown>;

/** Subset of JSON Schema fields the config form cares about. The backend
 *  returns `AppConfig.model_json_schema()` which is much larger; we narrow
 *  to the fields we read. Pydantic uses `$defs` for nested models. */
export interface JsonSchemaProperty {
  type?: 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array' | 'null';
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  /** Pydantic emits ``const`` (not ``enum``) for single-value Literals.
   *  The form normalises this to a one-element enum at render time. */
  const?: unknown;
  $ref?: string;
  anyOf?: JsonSchemaProperty[];
  allOf?: JsonSchemaProperty[];
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean | JsonSchemaProperty;
  items?: JsonSchemaProperty;
  minimum?: number;
  maximum?: number;
}

export interface JsonSchema extends JsonSchemaProperty {
  $defs?: Record<string, JsonSchemaProperty>;
}

/** GET /api/config -- raw config.yaml contents, no env-var overrides applied. */
export async function fetchConfig(): Promise<ConfigDict> {
  const res = await fetch(`${backendUrl()}/api/config`);
  if (!res.ok) throw await readError(res);
  return (await res.json()) as ConfigDict;
}

/** GET /api/config/schema -- AppConfig's JSON Schema for form rendering. */
export async function fetchConfigSchema(): Promise<JsonSchema> {
  const res = await fetch(`${backendUrl()}/api/config/schema`);
  if (!res.ok) throw await readError(res);
  return (await res.json()) as JsonSchema;
}

/** PUT /api/config -- validate and write the submitted dict to disk.
 *  Throws ApiError on validation failure; `detail` contains the Pydantic
 *  error list so the caller can highlight bad fields. */
export async function saveConfig(config: ConfigDict): Promise<void> {
  const res = await fetch(`${backendUrl()}/api/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  if (!res.ok) throw await readError(res);
}

/** POST /api/config/reload -- backend `os.execv`'s itself.
 *
 *  Returns immediately; the actual exec happens ~1s later so this response
 *  can flush and the port releases. Callers should then poll /api/health
 *  until the new process is up. */
export async function reloadBackend(): Promise<void> {
  const res = await fetch(`${backendUrl()}/api/config/reload`, { method: 'POST' });
  if (!res.ok) throw await readError(res);
}

/** Poll /api/health until it responds 2xx or `timeoutMs` elapses.
 *  Used after a reload to detect when the new backend is ready. */
export async function waitForBackend(
  timeoutMs: number = 60_000,
  intervalMs: number = 500
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${backendUrl()}/api/health`);
      if (res.ok) return;
    } catch {
      // Connection refused while the old process is dying / new one
      // is binding -- keep polling.
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`Backend did not come back within ${timeoutMs / 1000}s.`);
}
