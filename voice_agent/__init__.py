"""Virtual Helmsman — a Pipecat voice agent for a ship simulator."""

from __future__ import annotations

from voice_agent._cuda import add_cuda_dll_directories

# Must run before any submodule imports onnxruntime, so the CUDA execution
# provider can load. A safe no-op when the NVIDIA CUDA wheels are absent.
add_cuda_dll_directories()
