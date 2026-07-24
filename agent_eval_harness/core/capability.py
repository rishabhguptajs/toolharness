"""Canonical capability taxonomy + shell-command classifier.

Detectors operate on `CanonicalCapability`, never on raw per-CLI tool names, so
the same logic works across Claude Code / Cursor / Codex / Gemini. Adapters are
responsible for mapping their raw tool name (+ args) onto a capability; for shell
tools they call `classify_shell_command` to refine SHELL_EXEC into TEST/BUILD/LINT.
"""

from __future__ import annotations

import re
from enum import Enum


class CanonicalCapability(str, Enum):
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    FILE_EDIT = "FILE_EDIT"
    FILE_SEARCH = "FILE_SEARCH"        # glob / find-by-name
    CONTENT_SEARCH = "CONTENT_SEARCH"  # grep / ripgrep
    SHELL_EXEC = "SHELL_EXEC"
    TEST_RUN = "TEST_RUN"              # derived from the shell command
    BUILD_RUN = "BUILD_RUN"           # derived from the shell command
    LINT_RUN = "LINT_RUN"             # derived from the shell command
    WEB_FETCH = "WEB_FETCH"
    WEB_SEARCH = "WEB_SEARCH"
    TASK_MGMT = "TASK_MGMT"           # todo / plan tools
    SUBAGENT = "SUBAGENT"             # spawn a nested agent
    MCP_TOOL = "MCP_TOOL"             # dynamically-registered MCP tool
    UNKNOWN = "UNKNOWN"


# Capabilities that mutate the repository (used by verification / redundancy logic).
MUTATING_CAPABILITIES = frozenset(
    {CanonicalCapability.FILE_WRITE, CanonicalCapability.FILE_EDIT}
)

# Capabilities that constitute a "verification" of a code change.
VERIFICATION_CAPABILITIES = frozenset(
    {CanonicalCapability.TEST_RUN, CanonicalCapability.BUILD_RUN, CanonicalCapability.LINT_RUN}
)


# --- shell command classification -------------------------------------------------
# Ordered: first matching pattern wins. Patterns are intentionally broad; the goal is
# to recognize the *intent* of common toolchains, not to be exhaustive.

_TEST_PATTERNS = [
    r"\bpytest\b",
    r"\bpy\.test\b",
    r"\bpython[0-9.]*\s+-m\s+(pytest|unittest)\b",
    r"\bunittest\b",
    r"\b(npm|pnpm|yarn|bun)\s+(run\s+)?test\b",
    r"\b(jest|vitest|mocha|ava|cypress|playwright)\b",
    r"\bgo\s+test\b",
    r"\bcargo\s+test\b",
    r"\bphpunit\b",
    r"\b(bundle\s+exec\s+)?rspec\b",
    r"\bmvn\s+(test|verify)\b",
    r"\bgradle\w*\s+.*\btest\b",
    r"\bdotnet\s+test\b",
    r"\bctest\b",
]

_LINT_PATTERNS = [
    r"\beslint\b",
    r"\bruff\s+(check|\.)?",
    r"\bflake8\b",
    r"\bpylint\b",
    r"\bmypy\b",
    r"\bpyright\b",
    r"\btsc\b.*--noemit",
    r"\bgolangci-lint\b",
    r"\b(cargo\s+)?clippy\b",
    r"\bprettier\b.*(--check|-c)\b",
    r"\bblack\b.*--check\b",
    r"\bphpstan\b",
    r"\brubocop\b",
]

_BUILD_PATTERNS = [
    r"\b(npm|pnpm|yarn|bun)\s+(run\s+)?build\b",
    r"\btsc\b",
    r"\bwebpack\b",
    r"\b(vite|rollup|esbuild|parcel)\s+build\b",
    r"\bgo\s+build\b",
    r"\bcargo\s+build\b",
    r"\bmake\b",
    r"\bcmake\b",
    r"\bmvn\s+(compile|package|install)\b",
    r"\bgradle\w*\s+(build|assemble)\b",
    r"\bdotnet\s+build\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+(ci|install)\b",
]


def _matches_any(command: str, patterns: list[str]) -> bool:
    return any(re.search(p, command, re.IGNORECASE) for p in patterns)


def classify_shell_command(command: str | None) -> CanonicalCapability:
    """Refine a shell invocation into TEST/BUILD/LINT, else SHELL_EXEC.

    Test is checked before build because e.g. `npm test` also contains build-ish
    substrings; lint before build for the same reason (`tsc --noEmit`).
    """
    if not command:
        return CanonicalCapability.SHELL_EXEC
    if _matches_any(command, _TEST_PATTERNS):
        return CanonicalCapability.TEST_RUN
    if _matches_any(command, _LINT_PATTERNS):
        return CanonicalCapability.LINT_RUN
    if _matches_any(command, _BUILD_PATTERNS):
        return CanonicalCapability.BUILD_RUN
    return CanonicalCapability.SHELL_EXEC


def coerce_capability(value: str | CanonicalCapability | None) -> CanonicalCapability:
    """Best-effort mapping of a string onto the enum (used when parsing traces)."""
    if isinstance(value, CanonicalCapability):
        return value
    if not value:
        return CanonicalCapability.UNKNOWN
    key = str(value).strip().upper()
    try:
        return CanonicalCapability[key]
    except KeyError:
        return CanonicalCapability.UNKNOWN
