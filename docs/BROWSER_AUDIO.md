# Browser audio (`/ws/audio`)

Browser-based microphone capture and audio playback for the dashboard, so the
helmsman can be used from the web UI instead of the machine's local
microphone/speakers. Tracked in issue #7; shipped in phases.

> **Default behaviour is unchanged.** The agent uses local hardware audio
> (Pipecat `LocalAudioTransport`) unless `audio.browser_enabled` is `true`.

## Status — phase one (this PR)

A raw 16-bit PCM **loopback** that proves the full browser ↔ backend audio
path end to end:

```
getUserMedia → AudioWorklet (capture) → PCM16 → /ws/audio ─┐
speakers ← AudioWorklet (playback) ← PCM16 ← /ws/audio ────┘  (server loops back)
```

You enable the mic in the browser, speak, and hear yourself — confirming
capture, streaming, and playback all work. The backend does **not** yet route
this audio into the STT→LLM→TTS pipeline (that's phase two), so the helmsman
doesn't respond to it yet.

### Enable it

```yaml
audio:
  browser_enabled: true   # mounts /ws/audio
  sample_rate: 16000      # pipeline rate; the negotiated rate for phase two
```

With it on, `/api/session` reports `browser_audio: true`, and the Monitor page
shows a **"browser audio"** control (separate from the server-mic toggle in the
chat panel) with a live level meter.

### Wire protocol

A single WebSocket at `/ws/audio`:

- **binary** messages — little-endian PCM16 mono audio frames, both directions.
- **text** messages — a tiny JSON control channel. The client sends
  `{"type": "hello", "sample_rate": N}` on connect; the server replies
  `{"type": "ready", "sample_rate": N, "mode": "loopback"}`.

Capture happens at the browser `AudioContext`'s native rate (announced in
`hello`); loopback is rate-agnostic. The worklets live in
[`frontend/static/`](../frontend/static/) so `audioWorklet.addModule()` can
fetch them by path.

## Phase two (follow-up)

Bind the socket to the pipeline via Pipecat's `FastAPIWebsocketTransport`
(serializer-less raw PCM) so browser audio reaches the helmsman and its TTS
reply streams back. Open questions for that work:

- **Lifecycle:** the pipeline is built once at startup around local hardware.
  Browser audio is per-connection, so this needs either a per-connection
  pipeline or a transport that can be (re)bound to a live WebSocket — and a
  decision on sharing the heavy STT/TTS model instances rather than loading
  them per connection.
- **Sample-rate conversion** between the browser `AudioContext` rate and the
  pipeline's 16 kHz.
- **Barge-in / echo**: the browser plays the bot while the mic is open;
  `getUserMedia` echo cancellation helps, but VAD/turn behaviour over the
  network path needs tuning.
- Unifying the **server-mic toggle** and the browser-audio control once both
  drive the same pipeline.
