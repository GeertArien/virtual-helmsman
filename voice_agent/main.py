"""CLI entrypoint for the virtual helmsman voice agent.

Thin wrapper: parses ``--config``, loads config, builds the pipeline, runs it.
All real logic lives in :mod:`voice_agent.pipeline` and the backends.
"""

from __future__ import annotations


def main() -> None:
    """Parse args, load config, build and run the pipeline."""
    raise NotImplementedError("voice_agent.main.main is a scaffold stub")


if __name__ == "__main__":
    main()
