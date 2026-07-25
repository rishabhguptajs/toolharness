"""Shared helpers for the real-CLI adapters.

Every adapter ultimately produces the same NormalizedSession, so the fiddly bits
they share — reading NDJSON/JSONL robustly, flattening provider content blocks to
text, synthesizing correlation ids, and normalizing error classes — live here so
each adapter stays a thin format translator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolharness.adapters.base import RunSource


def read_text(source: RunSource) -> str:
    """Return the raw text of a source (in-memory data, path, or stream)."""
    if isinstance(source.data, str):
        return source.data
    if source.path is not None:
        return Path(source.path).read_text()
    if source.stream is not None:
        return "".join(str(chunk) for chunk in source.stream)
    return ""


def load_jsonl(source: RunSource) -> list[dict[str, Any]]:
    """Parse a JSONL/NDJSON source into a list of objects.

    Blank lines and lines that are not JSON objects are skipped rather than
    raising — real CLI logs interleave the occasional non-JSON banner.
    ``source.data`` may already be a parsed list of dicts.
    """
    if isinstance(source.data, list):
        return [d for d in source.data if isinstance(d, dict)]
    records: list[dict[str, Any]] = []
    for line in read_text(source).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def peek_jsonl(source: RunSource, limit: int = 40) -> list[dict[str, Any]]:
    """Parse just the first ``limit`` JSON objects — used by sniff()."""
    if isinstance(source.data, list):
        return [d for d in source.data[:limit] if isinstance(d, dict)]
    out: list[dict[str, Any]] = []
    for line in read_text(source).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
        if len(out) >= limit:
            break
    return out


def flatten_content(content: Any) -> str:
    """Collapse a provider "content" value (string or list of blocks) to text.

    Handles Anthropic-style blocks (``{"type":"text","text":...}``,
    ``input_text``, ``output_text``) and nested ``tool_result`` content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "output"):
            if key in content:
                return flatten_content(content[key])
        return ""
    if isinstance(content, list):
        parts = [flatten_content(item) for item in content]
        return "".join(p for p in parts if p)
    return str(content)


# --- error-class normalization ----------------------------------------------------
# Detectors key on a small set: M3 -> UNKNOWN_TOOL; M2 -> ENOENT/INVALID_ARGS/EISDIR.
# Anything we can't confidently classify stays None (a nonzero exit_code / is_error
# still carries the failure signal for M4).

def normalize_error_class(text: str | None, *, is_error: bool = False) -> str | None:
    if not text:
        return None
    low = text.lower()
    if "no such file or directory" in low or "enoent" in low:
        return "ENOENT"
    if "is a directory" in low or "eisdir" in low:
        return "EISDIR"
    if "unknown tool" in low or "no such tool" in low or "tool not found" in low:
        return "UNKNOWN_TOOL"
    if "invalid argument" in low or "invalid arguments" in low or "invalid input" in low:
        return "INVALID_ARGS"
    if "permission denied" in low or "eacces" in low:
        return "EACCES"
    return None


def status_of(*, is_error: bool, exit_code: int | None) -> str:
    if is_error:
        return "error"
    if exit_code is not None and exit_code != 0:
        return "error"
    return "ok"
