/**
 * Typed client for the Python control plane, split per backend route family
 * (issue #12 §8) so each module mirrors one backend router:
 *
 * - `http.ts`      -- backend location + error surface (shared plumbing)
 * - `events.ts`    -- event union + reconnecting WebSocket (`api/events.py`)
 * - `session.ts`   -- session identity (`api/app.py`)
 * - `control.ts`   -- chatbox + simulator link (`api/control_router.py`)
 * - `webrtc.ts`    -- browser-audio signalling (`api/webrtc.py`)
 * - `config.ts`    -- config editor + reload (`api/config_router.py`)
 * - `documents.ts` -- qdrant document management (`kb/documents.py`)
 * - `review.ts`    -- HITL review + audit log (`kb/review.py`)
 *
 * Everything re-exports through this index, so consumers keep importing from
 * `$lib/api`.
 */

export * from './config';
export * from './control';
export * from './documents';
export * from './events';
export * from './http';
export * from './review';
export * from './session';
export * from './webrtc';
