"""Adapters normalize per-CLI run output into a NormalizedSession."""

from toolharness.adapters.base import Adapter, AdapterRegistry, RunSource, default_registry
from toolharness.adapters.claude_code import ClaudeCodeAdapter
from toolharness.adapters.codex import CodexAdapter
from toolharness.adapters.cursor import CursorAdapter
from toolharness.adapters.generic import GenericToolTraceAdapter

# Register built-in adapters on the shared registry.
default_registry.register(GenericToolTraceAdapter())
default_registry.register(ClaudeCodeAdapter())
default_registry.register(CursorAdapter())
default_registry.register(CodexAdapter())

__all__ = [
    "Adapter",
    "AdapterRegistry",
    "RunSource",
    "default_registry",
    "GenericToolTraceAdapter",
    "ClaudeCodeAdapter",
    "CursorAdapter",
    "CodexAdapter",
]
