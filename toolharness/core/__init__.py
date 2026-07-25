"""Core normalized data model shared by every adapter and detector."""

from toolharness.core.capability import CanonicalCapability, classify_shell_command
from toolharness.core.findings import EventRef, FailureMode, Finding, Verdict
from toolharness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from toolharness.core.taskspec import Subgoal, TaskSpec

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
