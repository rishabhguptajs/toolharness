"""The normalized event stream every adapter produces and every detector consumes.

Design invariants:
  * Ordering is by ``seq`` (stream position), never wall-clock — OTEL/async
    streams interleave timestamps.
  * A ``call_id`` correlates a TOOL_CALL with its TOOL_RESULT. ``build_session``
    attaches each result onto its call and back-fills ``preceding_reasoning``.
  * The ``tool_calls`` list holds the *same* ToolCallEvent objects referenced by
    the TOOL_CALL entries in ``events`` — there is one source of truth, no drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agent_eval_harness.core.capability import CanonicalCapability
from agent_eval_harness.core.taskspec import TaskSpec


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_STOP = "agent_stop"


@dataclass
class ToolResult:
    call_id: str
    status: str = "unknown"          # ok | error | timeout | unknown
    is_error: bool = False
    content: str = ""                # normalized textual output
    error_message: str | None = None
    error_class: str | None = None   # ENOENT | UNKNOWN_TOOL | INVALID_ARGS | ...
    exit_code: int | None = None
    duration_ms: int | None = None
    raw_content: Any = None


@dataclass
class ToolSpec:
    """An advertised tool from the session's registry (may be partial/absent)."""

    name: str
    capability: CanonicalCapability = CanonicalCapability.UNKNOWN
    schema: dict[str, Any] | None = None


@dataclass
class ToolCallEvent:
    call_id: str
    session_id: str
    seq: int
    tool_name: str
    capability: CanonicalCapability
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_tool_name: str = ""
    raw_arguments: Any = None
    turn: int = 0
    timestamp: datetime | None = None
    parent_call_id: str | None = None
    preceding_reasoning: str | None = None   # agent text right before the call
    result: ToolResult | None = None         # attached by build_session
    adapter: str = ""
    raw_event_ref: Any = None                # index/pointer back to source

    @property
    def command(self) -> str | None:
        """The shell command for shell-like tools (best-effort across arg names)."""
        for key in ("command", "cmd", "script", "shell_command"):
            val = self.arguments.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                return " ".join(str(x) for x in val)
        return None

    @property
    def path(self) -> str | None:
        """The primary file path for file-like tools (best-effort across arg names)."""
        for key in ("path", "file_path", "filename", "file", "target_file", "abs_path"):
            val = self.arguments.get(key)
            if isinstance(val, str):
                return val
        return None


@dataclass
class NormalizedEvent:
    event_id: str
    session_id: str
    seq: int
    type: EventType
    turn: int = 0
    timestamp: datetime | None = None
    text: str | None = None                  # USER/AGENT message or stop text
    tool_call: ToolCallEvent | None = None   # when type == TOOL_CALL
    tool_result: ToolResult | None = None    # when type == TOOL_RESULT
    raw: Any = None


@dataclass
class NormalizedSession:
    session_id: str
    adapter: str
    task: TaskSpec
    events: list[NormalizedEvent] = field(default_factory=list)
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    available_tools: list[ToolSpec] = field(default_factory=list)
    final_agent_message: str | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_tool_names(self) -> set[str]:
        return {t.name for t in self.available_tools}

    def tool_spec(self, name: str) -> ToolSpec | None:
        for t in self.available_tools:
            if t.name == name:
                return t
        return None


def build_session(
    *,
    session_id: str,
    adapter: str,
    task: TaskSpec,
    events: list[NormalizedEvent],
    available_tools: list[ToolSpec] | None = None,
    stop_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedSession:
    """Assemble a session from a raw event list.

    Responsibilities (kept here so every adapter gets identical linking behavior):
      1. sort by ``seq``;
      2. attach each ToolResult onto its ToolCallEvent by ``call_id``;
      3. back-fill ``preceding_reasoning`` from the nearest prior AGENT_MESSAGE;
      4. build the ``tool_calls`` convenience view;
      5. derive ``final_agent_message``.
    """
    events = sorted(events, key=lambda e: e.seq)

    calls_by_id: dict[str, ToolCallEvent] = {}
    tool_calls: list[ToolCallEvent] = []
    last_agent_text: str | None = None
    final_agent_message: str | None = None

    for ev in events:
        if ev.type == EventType.AGENT_MESSAGE and ev.text:
            last_agent_text = ev.text
            final_agent_message = ev.text
        elif ev.type == EventType.AGENT_STOP and ev.text:
            final_agent_message = ev.text
        elif ev.type == EventType.TOOL_CALL and ev.tool_call is not None:
            tc = ev.tool_call
            if tc.preceding_reasoning is None:
                tc.preceding_reasoning = last_agent_text
            calls_by_id[tc.call_id] = tc
            tool_calls.append(tc)
            # a tool call consumes the pending reasoning so it isn't reused verbatim
            last_agent_text = None
        elif ev.type == EventType.TOOL_RESULT and ev.tool_result is not None:
            res = ev.tool_result
            call = calls_by_id.get(res.call_id)
            if call is not None:
                call.result = res

    return NormalizedSession(
        session_id=session_id,
        adapter=adapter,
        task=task,
        events=events,
        tool_calls=tool_calls,
        available_tools=available_tools or [],
        final_agent_message=final_agent_message,
        stop_reason=stop_reason,
        metadata=metadata or {},
    )
