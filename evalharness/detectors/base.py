"""Detector protocol, shared context, and the result envelope.

A detector reports both its findings *and* the number of "opportunities" it
examined, so the scoring engine can compute a rate rather than a raw count
(1 bad call out of 2 is very different from 1 out of 50).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalharness.core.findings import FailureMode, Finding
from evalharness.core.model import NormalizedSession


@dataclass
class DetectorContext:
    """Shared configuration/services handed to every detector.

    ``judge`` stays ``None`` for the deterministic (M1) detectors; the LLM-judge
    client is injected in M3.
    """

    judge: Any = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorResult:
    mode: FailureMode
    findings: list[Finding] = field(default_factory=list)
    n_opportunities: int = 0
    applicable: bool = True
    confidence: float = 1.0

    @property
    def failing(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict in ("fail", "warn")]


@runtime_checkable
class Detector(Protocol):
    mode: FailureMode
    needs_reference: bool
    works_reference_free: bool
    uses_llm: bool
    version: str

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        ...
