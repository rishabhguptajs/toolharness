"""Failure modes, verdicts, and the Finding record emitted by detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Verdict = Literal["pass", "fail", "warn", "na"]


class FailureMode(str, Enum):
    WRONG_TOOL = "wrong_tool"                    # M1
    WRONG_ARGS = "wrong_args"                    # M2
    HALLUCINATED = "hallucinated"                # M3
    IGNORED_OUTPUT = "ignored_output"            # M4
    REDUNDANT = "redundant"                      # M5
    MISSING_VERIFICATION = "missing_verification"  # M6
    PREMATURE_STOP = "premature_stop"            # M7
    UNSAFE_CALL = "unsafe_call"                  # M8


@dataclass
class EventRef:
    """A pointer back into the source session for the audit trail."""

    seq: int
    call_id: str | None = None
    note: str | None = None


@dataclass
class Finding:
    """A single detected issue.

    Detectors emit only actionable findings (verdict ``fail`` or ``warn``); the
    absence of a finding for a given call/mode means it passed. ``target_call_id``
    of ``None`` denotes a session-level finding.
    """

    mode: FailureMode
    verdict: Verdict
    severity: float                 # 0..1 contribution to the penalty
    confidence: float               # 0..1 (LLM-derived findings are < 1.0)
    rationale: str
    target_call_id: str | None = None
    target_seq: int | None = None
    evidence: list[EventRef] = field(default_factory=list)
    llm_used: bool = False
    detector_version: str = "0.1.0"

    def __post_init__(self) -> None:
        self.severity = max(0.0, min(1.0, float(self.severity)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
