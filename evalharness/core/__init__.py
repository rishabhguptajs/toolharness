"""Core normalized data model shared by every adapter and detector."""

from evalharness.core.capability import CanonicalCapability, classify_shell_command
from evalharness.core.findings import EventRef, FailureMode, Finding, Verdict
from evalharness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from evalharness.core.taskspec import Subgoal, TaskSpec

__all__ = [
    "CanonicalCapability",
    "classify_shell_command",
    "EventRef",
    "FailureMode",
    "Finding",
    "Verdict",
    "EventType",
    "NormalizedEvent",
    "NormalizedSession",
    "ToolCallEvent",
    "ToolResult",
    "ToolSpec",
    "build_session",
    "Subgoal",
    "TaskSpec",
]
