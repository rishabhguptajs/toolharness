"""Adapter protocol + a sniff-based registry.

Real CLI adapters (Claude Code, Cursor, Codex, Gemini) land in M5. For M0/M1 the
only concrete adapter is GenericToolTraceAdapter, which the injected-failure test
agents and benchmark converters emit directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_eval_harness.core.capability import CanonicalCapability
from agent_eval_harness.core.model import NormalizedSession


@dataclass
class RunSource:
    """A pointer to raw run output plus any sidecar inputs an adapter needs."""

    # kind: ndjson | jsonl | otel_log | sdk_messages | json_result | generic
    kind: str
    path: Path | None = None
    stream: Iterable[Any] | None = None
    data: Any = None                            # already-parsed in-memory payload
    aux: dict[str, Any] = field(default_factory=dict)  # OTEL sidecar, session dir, exit code, ...

    @classmethod
    def from_path(cls, path: str | Path, kind: str = "generic", **aux: Any) -> RunSource:
        return cls(kind=kind, path=Path(path), aux=aux)


@runtime_checkable
class Adapter(Protocol):
    name: str

    def sniff(self, source: RunSource) -> float:
        """Return 0..1 confidence that this adapter can parse ``source``."""
        ...

    def parse(self, source: RunSource) -> NormalizedSession:
        ...

    def canonicalize_tool(self, raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
        ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(self, adapter: Adapter) -> Adapter:
        self._adapters[adapter.name] = adapter
        return adapter

    def get(self, name: str) -> Adapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown adapter {name!r}; registered: {sorted(self._adapters)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def select(self, source: RunSource) -> Adapter:
        """Pick the highest-confidence adapter, or honor an explicit hint."""
        hint = source.aux.get("adapter")
        if hint:
            return self.get(hint)
        scored = [(a.sniff(source), a) for a in self._adapters.values()]
        scored = [(s, a) for s, a in scored if s > 0]
        if not scored:
            raise ValueError("No adapter could parse the given source.")
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    def parse(self, source: RunSource) -> NormalizedSession:
        return self.select(source).parse(source)


default_registry = AdapterRegistry()
