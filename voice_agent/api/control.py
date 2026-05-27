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

    ``mic_enabled`` starts True so the pipeline keeps its historical
    "always listening" behaviour. Toggle it via ``POST /api/control/mic``.
    """

    mic_enabled: bool = True
