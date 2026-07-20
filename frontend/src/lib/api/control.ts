/**
 * Control plane: text-command injection + the simulator link. Mirrors
 * `voice_agent.api.control_router`.
 *
 * Voice input is driven by the browser-audio control (see
 * `audio/webrtcAudio.ts`); there is no separate server-mic toggle in the UI.
 */

import type { SimulatorConnectionState } from './events';
import { backendUrl, readError } from './http';

/** POST /api/control/text -- inject a typed command as a user turn. */
export async function sendTextCommand(text: string): Promise<void> {
  const res = await fetch(`${backendUrl()}/api/control/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) throw await readError(res);
}

export interface SimulatorStateResponse {
  state: SimulatorConnectionState;
  ts: string;
}

/** Current link state. Needed on page load: `connection_state` events only
 *  report *changes*, and the next one may be minutes away. */
export async function fetchSimulatorState(): Promise<SimulatorStateResponse> {
  const res = await fetch(`${backendUrl()}/api/control/simulator`);
  if (!res.ok) throw new Error(`/api/control/simulator: HTTP ${res.status}`);
  return (await res.json()) as SimulatorStateResponse;
}

/** Open the link. The returned state may be `connecting` rather than
 *  `connected` -- with no simulator running there is nothing to reach yet, and
 *  the backend keeps trying. That is a normal answer, not a failure. */
export async function connectSimulator(): Promise<SimulatorStateResponse> {
  const res = await fetch(`${backendUrl()}/api/control/simulator/connect`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error(`/api/control/simulator/connect: HTTP ${res.status}`);
  return (await res.json()) as SimulatorStateResponse;
}

/** Close the link and stop reconnecting until asked again. */
export async function disconnectSimulator(): Promise<SimulatorStateResponse> {
  const res = await fetch(`${backendUrl()}/api/control/simulator/disconnect`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error(`/api/control/simulator/disconnect: HTTP ${res.status}`);
  return (await res.json()) as SimulatorStateResponse;
}
