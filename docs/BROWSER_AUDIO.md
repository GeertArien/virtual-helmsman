# Browser audio (WebRTC)

Talk to the helmsman from the **web dashboard** — capture the browser's
microphone and play the agent's spoken reply in the browser — instead of using
the machine's local microphone/speakers. Tracked in issue #7.

> **Default behaviour is unchanged.** The agent uses local hardware audio
> (Pipecat `LocalAudioTransport`) unless `audio.browser_enabled` is `true`.
> Requires the `webrtc` extra: `pip install -e ".[webrtc]"`.

## How it works

WebRTC via Pipecat's `SmallWebRTCTransport`, so the browser/transport handle
capture, Opus, jitter buffering, and echo cancellation — the app just does
signalling and runs the pipeline:

```
browser: getUserMedia → RTCPeerConnection ──(SDP offer)──▶ POST /api/webrtc/offer
                                            ◀──(SDP answer)──
mic track  ──▶ SmallWebRTCTransport.input()  → STT → LLM → JSON action → TTS
agent audio ◀── SmallWebRTCTransport.output() ←───────────────────────────┘
```

The browser plays the inbound agent audio track; the action still drives the
simulator and publishes the usual transcript / ship-state / metrics events over
`/ws/events`.

### Shared models, per-connection pipelines

The heavy backends — VAD, turn detector, STT, TTS, the LLM, and the one
simulator — are built **once at startup** into a `SharedBackends`. Each browser
connection assembles its own pipeline (`assemble_task`) bound to that
connection's WebRTC transport, reusing the already-loaded models rather than
reloading them per connect. In browser mode the process has no single local
task; it serves the API and runs one pipeline per live connection.

The existing **server-mic toggle** (`/api/control/mic`) still gates audio: it's
wired into every assembled pipeline as the `MicGate`, so it mutes browser audio
too. The text chatbox is disabled in browser mode (there's no single local task
to inject into) — voice is the input.

### Enable it

```yaml
audio:
  browser_enabled: true
  ice_servers: ["stun:stun.l.google.com:19302"]   # add a TURN server for cross-NAT
api:
  enabled: true     # required — browser audio is served by the control plane
```

With it on, `/api/session` reports `browser_audio: true` and the Monitor page
shows a **"browser audio"** control. Run the agent (`python -m voice_agent.main`)
and the dashboard; click the control, grant mic permission, and speak.

## Signalling endpoint

`POST /api/webrtc/offer`

```jsonc
// request
{ "sdp": "<offer SDP>", "type": "offer", "pc_id": "<optional, on renegotiation>" }
// response (Pipecat SmallWebRTCConnection answer)
{ "sdp": "<answer SDP>", "type": "answer", "pc_id": "<connection id>" }
```

Returns **503** when the `webrtc` extra (aiortc) isn't installed. The frontend
collects ICE candidates before posting the offer (non-trickle), which is fine
for localhost / same-LAN; add a TURN server in `ice_servers` for traversal
across networks.

## Status & caveats

This is the first working version and **needs validation on a real rig** (a
browser + mic + the GPU models + the `webrtc` extra) — the audio path itself
can't be exercised in headless CI. Known follow-ups:

- **Sample-rate conversion** between the WebRTC track rate and the pipeline's
  16 kHz (the transport resamples; confirm against the STT/TTS backends).
- **Concurrency:** the model instances are shared across connections; the
  intended use is one browser at a time (single local user). Concurrent
  connections sharing the same Pipecat service instances is untested.
- **Reconnect / renegotiation** edge cases and TURN configuration for remote
  access.
- Unifying the server-mic toggle UX with the browser-audio control now that
  both drive the pipeline.
