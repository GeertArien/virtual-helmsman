"""Shared runtime state for the control plane.

A single :class:`ControlState` instance is built when the pipeline is built and
passed to two places:

* the :class:`~voice_agent.api.mic_gate.MicGate` processor, which reads
  ``mic_enabled`` on every audio frame to decide whether to forward or drop it;
* the ``/api/control`` router, which mutates ``mic_enabled`` in response to a
  toggle request and broadcasts an :class:`InputModeChangedEvent` so any open
  browser tab updates immediately.

Kept as a plain dataclass rather than a Pydantic model -- this is internal,
mutable, accessed on the hot audio path, and never serialised directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlState:
    """Mutable knobs the API can flip at runtime.

    ``mic_enabled`` starts False so a fresh session is not listening until the
    cursist explicitly enables the mic. This dovetails with the AI Act Art. 50
    transparency gate in the frontend: the user must acknowledge they are
    talking to an AI system (modal) before any input is possible, and the chat
    box -- which is enabled only while the mic is off -- is the available input
    immediately after acknowledgement. Toggle the mic via ``POST /api/control/mic``.
    """

    mic_enabled: bool = False
