"""The voice ↔ knowledge-base boundary is structural, not aspirational.

Issue #12 §6: one process serves two unrelated applications (the voice
runtime and the knowledge-base side). ``voice_agent/kb`` documents the
boundary contract; these tests enforce it by walking the import statements of
every module on each side, so a violation fails CI instead of silently
re-entangling the halves.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "voice_agent"

# The voice side of the process. kb modules importing any of these would tie
# the knowledge base to the pipeline runtime.
VOICE_SIDE_PREFIXES = (
    "voice_agent.actions",
    "voice_agent.api",
    "voice_agent.backends",
    "voice_agent.metrics",
    "voice_agent.pipeline",
    "voice_agent.telemetry",
    "voice_agent.main",
)

# The only voice-side modules allowed to import voice_agent.kb, per the
# blessed-crossings contract in voice_agent/kb/__init__.py.
BLESSED_KB_IMPORTERS = {
    "voice_agent/api/app.py",  # mounts create_kb_routers
    "voice_agent/backends/llm/langgraph_helmsman/service.py",  # audit store
}


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _module_files(subpath: str = "") -> list[Path]:
    root = PACKAGE_ROOT / subpath if subpath else PACKAGE_ROOT
    return sorted(p for p in root.rglob("*.py") if "vendor" not in p.parts)


def test_kb_never_imports_the_voice_side() -> None:
    offenders = {
        str(path.relative_to(PACKAGE_ROOT.parent)): sorted(bad)
        for path in _module_files("kb")
        if (
            bad := {
                imp
                for imp in _imports_of(path)
                if imp.startswith(VOICE_SIDE_PREFIXES)
            }
        )
    }
    assert not offenders, f"kb modules import the voice side: {offenders}"


def test_voice_side_crosses_into_kb_only_at_blessed_points() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _module_files():
        rel = str(path.relative_to(PACKAGE_ROOT.parent)).replace("\\", "/")
        if rel.startswith("voice_agent/kb/") or rel in BLESSED_KB_IMPORTERS:
            continue
        bad = {imp for imp in _imports_of(path) if imp.startswith("voice_agent.kb")}
        if bad:
            offenders[rel] = sorted(bad)
    assert not offenders, (
        "unblessed voice->kb imports (extend the contract in "
        f"voice_agent/kb/__init__.py deliberately, or remove them): {offenders}"
    )
