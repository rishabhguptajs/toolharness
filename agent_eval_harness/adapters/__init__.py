"""Adapters normalize per-CLI run output into a NormalizedSession."""

from agent_eval_harness.adapters.base import Adapter, AdapterRegistry, RunSource, default_registry
from agent_eval_harness.adapters.claude_code import ClaudeCodeAdapter
from agent_eval_harness.adapters.codex import CodexAdapter
from agent_eval_harness.adapters.cursor import CursorAdapter
from agent_eval_harness.adapters.generic import GenericToolTraceAdapter

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
