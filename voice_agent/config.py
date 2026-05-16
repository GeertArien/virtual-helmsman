"""Pydantic config schema and loader.

Loads ``config.yaml``, validates it against the schema, and applies env-var
overrides (``LLM_BASE_URL``, ``LLM_API_KEY``, ``SIMULATOR_BACKEND``).
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path = "./config.yaml"):
    """Load, validate, and env-override the config file."""
    raise NotImplementedError("voice_agent.config.load_config is a scaffold stub")
