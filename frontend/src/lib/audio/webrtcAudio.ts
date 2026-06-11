/**
 * Browser-audio controller over WebRTC (issue #7).
 *
 * Captures the microphone and connects to the backend's per-connection
 * pipeline so the user can speak to the helmsman and hear its reply:
 *
 *   getUserMedia → RTCPeerConnection (mic track out, agent audio track in)
 *   POST /api/webrtc/offer ⇄ SDP answer
 *   agent audio track → <audio> playback
 *
 * The backend runs Pipecat's SmallWebRTCTransport, so WebRTC handles capture,
 * Opus encoding, jitter buffering, and echo cancellation; this client just
 * does signalling + playback. Trickle ICE is collected before the offer is
 * posted (no ICE server endpoint on the backend), which is fine for
 * localhost / same-LAN use.
 */

import { postWebRTCOffer } from '$lib/api';

export type WebRTCStatus = 'idle' | 'connecting' | 'live' | 'error';

export interface WebRTCAudioCallbacks {
  onStatus?: (status: WebRTCStatus, detail?: string) => void;
}

export class WebRTCAudio {
  private pc: RTCPeerConnection | null = null;
  private stream: MediaStream | null = null;
  private audioEl: HTMLAudioElement | null = null;
  private pcId: string | null = null;

  constructor(private readonly cb: WebRTCAudioCallbacks = {}) {}

  /** Acquire the mic, negotiate the peer connection, and start playback. */
  async start(): Promise<void> {
    this.cb.onStatus?.('connecting');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
      });

      const pc = new RTCPeerConnection();
      this.pc = pc;

      // Send the mic and receive the agent's audio over one transceiver:
      // addTrack() creates a sendrecv audio transceiver. Do NOT add a second
      // audio transceiver -- Pipecat's SmallWebRTCConnection switches its
      // input to the newest audio track it sees, and an extra trackless
      // m-line leaves the agent reading a track that never carries RTP.
      for (const track of this.stream.getAudioTracks()) pc.addTrack(track, this.stream);

      // Play the inbound agent track.
      this.audioEl = new Audio();
      this.audioEl.autoplay = true;
      pc.addEventListener('track', (event) => {
        if (this.audioEl) this.audioEl.srcObject = event.streams[0];
      });

      pc.addEventListener('connectionstatechange', () => {
        const s = pc.connectionState;
        if (s === 'connected') this.cb.onStatus?.('live');
        else if (s === 'failed' || s === 'closed') this.cb.onStatus?.('idle');
        else if (s === 'disconnected') this.cb.onStatus?.('error', 'connection lost');
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await this.waitForIceGathering(pc);

      const local = pc.localDescription!;
      const answer = await postWebRTCOffer({ sdp: local.sdp, type: local.type });
      this.pcId = answer.pc_id;
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type as RTCSdpType });
    } catch (err) {
      this.cb.onStatus?.('error', err instanceof Error ? err.message : String(err));
      await this.stop();
    }
  }

  /** Resolve once ICE candidate gathering completes (non-trickle signalling). */
  private waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
      // Safety timeout: some browsers stall on the final state transition.
      setTimeout(resolve, 2000);
    });
  }

  /** Stop streaming and release the mic, peer connection, and playback element. */
  async stop(): Promise<void> {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.audioEl) {
      this.audioEl.srcObject = null;
      this.audioEl = null;
    }
    this.pc?.close();
    this.pc = null;
    this.pcId = null;
    this.cb.onStatus?.('idle');
  }
}
