/** Session identity: mirrors `GET /api/session` (`voice_agent.api.app`). */

import { backendUrl } from './http';

export interface SessionInfo {
  session_id: string;
  started_at: string;
  stt_backend: string;
  tts_backend: string;
  vad_backend: string;
  turn_backend: string;
  simulator_backend: string;
  llm_model: string;
  subscribers: number;
  events_dropped: number;
  /** Always true: browser audio (WebRTC) is the only voice path, so the
   *  dashboard always offers browser-side mic capture + playback. */
  browser_audio?: boolean;
}

export async function fetchSession(): Promise<SessionInfo> {
  const res = await fetch(`${backendUrl()}/api/session`);
  if (!res.ok) throw new Error(`/api/session: HTTP ${res.status}`);
  return (await res.json()) as SessionInfo;
}
