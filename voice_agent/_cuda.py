"""Make the pip-installed NVIDIA CUDA runtime libraries discoverable.

``onnxruntime-gpu`` needs the CUDA 12.x runtime and cuDNN 9.x DLLs to load its
CUDA execution provider. This project installs those as the ``nvidia-*-cu12``
pip wheels (the ``cuda`` extra in ``pyproject.toml``) so the environment stays
venv-local and reproducible — no system-wide CUDA toolkit.

The wheels drop their DLLs in ``site-packages/nvidia/<component>/bin``, which is
not on the Windows DLL search path. ONNX Runtime therefore cannot load
``onnxruntime_providers_cuda.dll`` (it fails to resolve transitive dependencies
like ``cublasLt64_12.dll``) until those directories are added here.

``add_cuda_dll_directories`` runs from ``voice_agent/__init__.py`` so it happens
before any submodule imports ``onnxruntime``. It is a safe no-op when the wheels
are absent (a CPU-only or non-Windows environment).
"""

from __future__ import annotations

import os


def add_cuda_dll_directories() -> list[str]:
    """Prepend the bundled NVIDIA CUDA library dirs to the DLL search path.

    Returns the directories that were added (empty if the ``nvidia-*-cu12``
    wheels are not installed).
    """
    try:
        import nvidia
    except ImportError:
        return []

    added: list[str] = []
    for ns_path in getattr(nvidia, "__path__", []):
        try:
            components = sorted(os.listdir(ns_path))
        except OSError:
            continue
        for component in components:
            bindir = os.path.join(ns_path, component, "bin")
            if not os.path.isdir(bindir):
                continue
            # PATH is what the Windows loader searches for a DLL's transitive
            # native dependencies (onnxruntime_providers_cuda.dll ->
            # cublasLt64_12.dll); os.add_dll_directory alone does not cover that.
            path = os.environ.get("PATH", "")
            if bindir not in path.split(os.pathsep):
                os.environ["PATH"] = bindir + os.pathsep + path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bindir)
                except OSError:
                    pass
            added.append(bindir)
    return added
