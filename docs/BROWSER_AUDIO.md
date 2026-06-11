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

### Shared models, per-connection services

Pipecat FrameProcessors are **single-use**: once a pipeline is cancelled (a
browser disconnect) every processor in it permanently drops frames. So
`SharedBackends` holds per-pipeline service *factories*, not instances — each
connection assembles a fresh pipeline (`assemble_task`) with fresh
STT/TTS/LLM/VAD/turn services bound to that connection's WebRTC transport.
What *is* shared across connections: the loaded STT model (cached at module
level in `parakeet_onnx`, warmed at startup), the one simulator, and the
event bus. Reconnect cost is ~1 s of service construction, not a model
reload. In browser mode the process serves the API, runs one pipeline per
live connection, plus a standing text-only pipeline for the chatbox.

The **browser-audio control is the mic on/off**: connecting grants the agent
your audio, disconnecting cuts it. Browser pipelines carry no `MicGate`
(`assemble_task(..., gate_mic=False)`) — a second server-side mute on top of
an explicit connect would only look like a deaf agent. The server-mic toggle
UI is gone; `/api/control/mic` remains for the local-hardware mode only.

The **text chatbox still works** in browser mode: typed commands flow through
a standing text-only pipeline (user aggregator → LLM → JSON action →
assistant aggregator, no audio) that runs for the whole session against the
same simulator and event bus. Replies surface in the transcript panel.

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

## Alternative transport: raw PCM over WebSocket (evaluated, not built)

A lighter option than WebRTC is to capture the mic with the **Web Audio API**
(`getUserMedia` → `AudioContext` → an `AudioWorklet`) and stream raw PCM frames
over a WebSocket, playing the agent's reply back through Web Audio. Pipecat
ships a WebSocket transport family (with frame serializers), so this is mostly a
transport swap plus a browser capture/playback worklet rather than a bespoke
protocol. Evaluated against the WebRTC path; **not built** — recorded here so the
trade-off doesn't have to be re-derived.

**Faster on localhost / same-LAN.** Dropping WebRTC removes three costs:

- **Opus encode/decode** on both legs (~20–60 ms of codec algorithmic delay) —
  gone; PCM is sent uncompressed.
- **The adaptive jitter buffer** — WebRTC keeps one on both the inbound
  (`aiortc`) and the browser playback side, and it stays conservatively
  non-zero even on a perfect loopback link. A bare WebSocket has none. This is
  the biggest local saving.
- **Connection setup** — no SDP offer/answer, no ICE/STUN round trip; a
  WebSocket is just a TCP upgrade handshake.

On localhost that reclaims most of the per-turn transport overhead (order of
tens to ~100 ms). The catch is what WebRTC was doing for you:

- **TCP head-of-line blocking.** WebRTC media rides UDP precisely so a lost
  packet is *dropped and concealed*, not stalled. A WebSocket is TCP — one lost
  packet blocks the stream and retransmits, so delay **compounds** under loss.
  Harmless on loopback/LAN (~0 loss); **worse than WebRTC over a real network**,
  where it would also need a hand-rolled jitter buffer and loss concealment.
  This option is therefore strictly a single-local-user play.
- **Echo cancellation reference.** `getUserMedia({ echoCancellation: true })`
  still gives browser AEC/NS/AGC on the captured track, but the AEC reference is
  tied to the WebRTC playback path; routing TTS out through Web Audio may weaken
  it. Minor here — the pipeline already mutes STT while the bot speaks
  (`AlwaysUserMuteStrategy`), so it doesn't rely on AEC to avoid hearing itself.
- **Resampling isn't eliminated, just relocated** — the `AudioContext` runs at
  the hardware rate (usually 48 kHz), so you still convert to the pipeline's
  16 kHz, now in the worklet instead of the transport.

**Verdict.** Legitimate and modestly faster *for the local single-user case*,
and arguably **simpler** (drops the `aiortc` dependency, ICE/STUN config, and the
renegotiation edge cases below) — that simplicity is a better reason to switch
than the latency. But weigh it against [`LATENCY.md`](LATENCY.md): the saving is
tens of milliseconds against a turn whose **LLM eats ~1.6 s**. It ranks well
below folding the two LLM calls into one (≈half the felt latency) and streaming
TTS. Only worth pursuing once those land and the transport edge has actually
been **measured** (a browser capture→playback timestamp loopback test — the
pipeline's `stt + llm_ttft + tts_ttfa` metrics don't instrument the transport
edges, so the cost is currently estimated, not observed).

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
