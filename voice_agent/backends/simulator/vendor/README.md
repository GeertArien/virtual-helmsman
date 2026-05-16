# Vendored simulator wrappers

This directory holds the **in-house ship-simulator integration files**, which
are *not* distributed via pip or as a wheel. Drop them in here by hand.

## What goes here

- The in-house Python wrapper class(es) — plain `.py` files that load a managed
  .NET assembly via `pythonnet` (`import clr; clr.AddReference(...)`).
- The managed **.NET DLL** the wrappers reference.

These files **are committed to the repo** (this directory is intentionally not
gitignored), so the `real` simulator backend can build against them.

## How they are used

`voice_agent/backends/simulator/real.py` imports the wrapper class(es) from this
location and adapts them to the `SimulatorClient` protocol
(`voice_agent/backends/simulator/base.py`). The TODO stubs in `real.py` mark
where the actual wrapper method names and constructor signature go — fill those
in during integration.

## Platform requirement

The `real` backend is **Windows-only** because the .NET DLL is loaded via
`pythonnet`. The `mock` backend is platform-agnostic; use it for all STT/TTS/
pipeline development on Linux or Windows.
