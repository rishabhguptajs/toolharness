"""Core normalized data model shared by every adapter and detector."""

from agent_eval_harness.core.capability import CanonicalCapability, classify_shell_command
from agent_eval_harness.core.findings import EventRef, FailureMode, Finding, Verdict
from agent_eval_harness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from agent_eval_harness.core.taskspec import Subgoal, TaskSpec

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
