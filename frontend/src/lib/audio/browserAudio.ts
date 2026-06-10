/**
 * Browser-audio controller (issue #7, phase one).
 *
 * Wires the microphone to the backend's `/ws/audio` raw-PCM bridge and plays
 * received audio back, using two AudioWorklet processors:
 *
 *   getUserMedia → MediaStreamSource → capture-processor → (PCM16) → WS
 *   WS → (PCM16) → playback-processor → destination (speakers)
 *
 * In phase one the backend loops the audio straight back, so this proves the
 * full capture → stream → playback path. The same controller will drive the
 * pipeline-integrated socket in phase two unchanged.
 *
 * The worklet modules are served from `static/` so `addModule()` can fetch
 * them by absolute path.
 */

import { audioWsUrl } from '$lib/api';

export type AudioStatus = 'idle' | 'connecting' | 'live' | 'error';

export interface BrowserAudioCallbacks {
  /** Status transitions, for the UI. */
  onStatus?: (status: AudioStatus, detail?: string) => void;
  /** Mic input peak level in [0, 1], ~per render quantum, for a level meter. */
  onLevel?: (peak: number) => void;
}

export class BrowserAudio {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private capture: AudioWorkletNode | null = null;
  private playback: AudioWorkletNode | null = null;
  private ws: WebSocket | null = null;
  private stopped = false;

  constructor(private readonly cb: BrowserAudioCallbacks = {}) {}

  /** Acquire the mic, open the socket, and start streaming both ways. */
  async start(): Promise<void> {
    this.stopped = false;
    this.cb.onStatus?.('connecting');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
      });
      this.ctx = new AudioContext();
      await this.ctx.audioWorklet.addModule('/audio-capture-worklet.js');
      await this.ctx.audioWorklet.addModule('/audio-playback-worklet.js');

      // Playback path: queue node → speakers.
      this.playback = new AudioWorkletNode(this.ctx, 'playback-processor', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1]
      });
      this.playback.connect(this.ctx.destination);

      // Capture path: mic → capture node (not connected to destination, so the
      // user doesn't hear a raw local copy — only the looped-back stream).
      const source = this.ctx.createMediaStreamSource(this.stream);
      this.capture = new AudioWorkletNode(this.ctx, 'capture-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 0
      });
      this.capture.port.onmessage = (event) => {
        const { pcm, peak } = event.data as { pcm: ArrayBuffer; peak: number };
        if (typeof peak === 'number') this.cb.onLevel?.(peak);
        if (this.ws?.readyState === WebSocket.OPEN && pcm) this.ws.send(pcm);
      };
      source.connect(this.capture);

      await this.openSocket(this.ctx.sampleRate);
    } catch (err) {
      this.cb.onStatus?.('error', err instanceof Error ? err.message : String(err));
      await this.stop();
    }
  }

  private openSocket(sampleRate: number): Promise<void> {
    return new Promise((resolve) => {
      const ws = new WebSocket(audioWsUrl());
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({ type: 'hello', sample_rate: Math.round(sampleRate) }));
        this.cb.onStatus?.('live');
        resolve();
      });
      ws.addEventListener('message', (msg) => {
        if (typeof msg.data === 'string') return; // control replies (e.g. ready)
        // Received PCM16 → hand to the playback worklet (zero-copy transfer).
        const buf = msg.data as ArrayBuffer;
        this.playback?.port.postMessage({ pcm: buf }, [buf]);
      });
      ws.addEventListener('close', () => {
        if (!this.stopped) this.cb.onStatus?.('idle');
      });
      ws.addEventListener('error', () => {
        this.cb.onStatus?.('error', 'audio socket error');
      });
    });
  }

  /** Stop streaming and release the mic, socket, and audio graph. */
  async stop(): Promise<void> {
    this.stopped = true;
    this.ws?.close();
    this.ws = null;
    this.capture?.disconnect();
    this.capture = null;
    this.playback?.disconnect();
    this.playback = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.ctx) {
      await this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    this.cb.onLevel?.(0);
    this.cb.onStatus?.('idle');
  }
}
