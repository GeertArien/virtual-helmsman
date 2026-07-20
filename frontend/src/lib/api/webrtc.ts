/** Browser-audio (WebRTC) signalling: mirrors `voice_agent.api.webrtc`. */

import { backendUrl, readError } from './http';

/** SDP answer to a WebRTC offer (mirrors SmallWebRTCConnection.get_answer). */
export interface WebRTCAnswer {
  sdp: string;
  type: string;
  pc_id: string;
}

/** POST /api/webrtc/offer — exchange an SDP offer for the agent's answer.
 *  `pc_id` is echoed back on renegotiation. Throws ApiError (e.g. 503 when the
 *  backend lacks the `webrtc` extra). */
export async function postWebRTCOffer(offer: {
  sdp: string;
  type: string;
  pc_id?: string;
  restart_pc?: boolean;
}): Promise<WebRTCAnswer> {
  const res = await fetch(`${backendUrl()}/api/webrtc/offer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(offer)
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as WebRTCAnswer;
}
