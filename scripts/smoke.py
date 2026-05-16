"""Smoke test: full LLM-to-tool-to-simulator path, no audio, no real sim.

Injects a fake ``TranscriptionFrame("steer course two seven zero")``, asserts
the LLM emits ``set_heading`` with ``degrees ~= 270``, and that the mock
simulator's ``command_history`` records it.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("scripts/smoke.py is a scaffold stub")


if __name__ == "__main__":
    main()
