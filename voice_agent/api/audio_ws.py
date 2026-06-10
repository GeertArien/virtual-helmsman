"""Browser-audio WebSocket endpoint (``/ws/audio``).

Phase one of browser-based audio (issue #7): a raw 16-bit PCM bridge between
the dashboard and the backend. The browser captures the microphone via an
``AudioWorklet``, downsamples to the pipeline rate, and streams little-endian
PCM16 frames over this socket's **binary** channel; the backend streams audio
frames back the same way for browser playback.

In this first slice the backend **loops the audio straight back** -- enough to
prove the full capture -> stream -> playback path end to end (you speak, you
hear yourself) without touching the Pipecat pipeline, which today is bound to
local hardware and built once at startup. Wiring this socket into the pipeline
via Pipecat's ``FastAPIWebsocketTransport`` (so browser audio actually reaches
the helmsman and hears its reply) is the phase-two follow-up.

Text messages on the socket are a tiny JSON control channel: the client sends
``{"type": "hello", "sample_rate": N}`` on connect and the server replies
``{"type": "ready", "sample_rate": N, "mode": "loopback"}`` so the client can
confirm the negotiated rate before streaming.

The route is only mounted when ``audio.browser_enabled`` is true, so the
default local-audio setup is completely unaffected.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_agent.config import AudioConfig
from voice_agent.logging_setup import get_logger


def create_audio_router(audio: AudioConfig) -> APIRouter:
    """Build the ``/ws/audio`` router bound to the audio config."""
    router = APIRouter(tags=["audio"])
    log = get_logger("api.audio")

    @router.websocket("/ws/audio")
    async def audio_ws(ws: WebSocket) -> None:
        await ws.accept()
        # Negotiated rate defaults to the pipeline rate; the client may pin it
        # via the hello message (it knows its own AudioContext rate).
        sample_rate = audio.sample_rate
        bytes_in = 0
        log.info("audio_ws_connected", sample_rate=sample_rate)
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                data = message.get("bytes")
                if data is not None:
                    # Raw PCM16 frame from the mic -> loop straight back for
                    # playback. Phase two replaces this with the pipeline.
                    bytes_in += len(data)
                    await ws.send_bytes(data)
                    continue

                text = message.get("text")
                if text is not None:
                    await _handle_control(ws, text, sample_rate, audio)
        except WebSocketDisconnect:
            pass
        finally:
            log.info("audio_ws_closed", bytes_in=bytes_in)

    return router


async def _handle_control(
    ws: WebSocket, text: str, sample_rate: int, audio: AudioConfig
) -> None:
    """Answer the JSON control channel (currently just the ``hello`` handshake)."""
    try:
        msg = json.loads(text)
    except (ValueError, TypeError):
        return
    if not isinstance(msg, dict) or msg.get("type") != "hello":
        return
    # Honour a client-supplied rate when it's a sane positive integer.
    rate = msg.get("sample_rate")
    negotiated = rate if isinstance(rate, int) and rate > 0 else sample_rate
    await ws.send_text(
        json.dumps(
            {"type": "ready", "sample_rate": negotiated, "mode": "loopback"}
        )
    )
