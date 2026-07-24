"""Adapters normalize per-CLI run output into a NormalizedSession."""

from agent_eval_harness.adapters.base import Adapter, AdapterRegistry, RunSource, default_registry
from agent_eval_harness.adapters.generic import GenericToolTraceAdapter

# Register built-in adapters on the shared registry.
default_registry.register(GenericToolTraceAdapter())

__all__ = [
    "Adapter",
    "AdapterRegistry",
    "RunSource",
    "default_registry",
    "GenericToolTraceAdapter",
]
