"""GenericToolTraceAdapter — the canonical JSON trace format.

This is the schema the injected-failure test agents and the benchmark converters
emit, so validation exercises the *real* detector path rather than a mock. It is
also the reference for what any CLI adapter must ultimately produce.

Trace shape (see fixtures/ for examples)::

    {
      "session_id": "s1",
      "adapter": "generic",
      "stop_reason": "completed",
      "task": { ... TaskSpec.from_dict ... },
      "available_tools": [
        {"name": "read_file", "capability": "FILE_READ",
         "schema": {"required": ["path"], "properties": {"path": {"type": "string"}}}}
      ],
      "events": [
        {"type": "user_message", "text": "..."},
        {"type": "agent_message", "text": "..."},
        {"type": "tool_call", "call_id": "c1", "tool_name": "read_file",
         "capability": "FILE_READ", "arguments": {"path": "a.py"}},
        {"type": "tool_result", "call_id": "c1", "status": "ok",
         "content": "...", "is_error": false},
        {"type": "agent_stop", "text": "done"}
      ]
    }

``seq`` is assigned by array position. ``turn`` auto-increments on each
user/agent message boundary. ``capability`` may be given explicitly or, for shell
tools, refined from the command via ``classify_shell_command``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolharness.adapters.base import RunSource
from toolharness.core.capability import (
    CanonicalCapability,
    classify_shell_command,
    coerce_capability,
)
from toolharness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from toolharness.core.taskspec import TaskSpec

# Default raw-name -> capability map, used when a trace omits an explicit capability.
_DEFAULT_NAME_MAP: dict[str, CanonicalCapability] = {
    "read": CanonicalCapability.FILE_READ,
    "read_file": CanonicalCapability.FILE_READ,
    "write": CanonicalCapability.FILE_WRITE,
    "write_file": CanonicalCapability.FILE_WRITE,
    "create_file": CanonicalCapability.FILE_WRITE,
    "edit": CanonicalCapability.FILE_EDIT,
    "edit_file": CanonicalCapability.FILE_EDIT,
    "apply_patch": CanonicalCapability.FILE_EDIT,
    "str_replace": CanonicalCapability.FILE_EDIT,
    "glob": CanonicalCapability.FILE_SEARCH,
    "find": CanonicalCapability.FILE_SEARCH,
    "grep": CanonicalCapability.CONTENT_SEARCH,
    "ripgrep": CanonicalCapability.CONTENT_SEARCH,
    "search": CanonicalCapability.CONTENT_SEARCH,
    "bash": CanonicalCapability.SHELL_EXEC,
    "shell": CanonicalCapability.SHELL_EXEC,
    "run_command": CanonicalCapability.SHELL_EXEC,
    "run_terminal_cmd": CanonicalCapability.SHELL_EXEC,
    "run_shell_command": CanonicalCapability.SHELL_EXEC,
    "web_fetch": CanonicalCapability.WEB_FETCH,
    "fetch": CanonicalCapability.WEB_FETCH,
    "web_search": CanonicalCapability.WEB_SEARCH,
    "todo": CanonicalCapability.TASK_MGMT,
    "todowrite": CanonicalCapability.TASK_MGMT,
    "task": CanonicalCapability.SUBAGENT,
}


class GenericToolTraceAdapter:
    name = "generic"

    def canonicalize_tool(self, raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
        cap = _DEFAULT_NAME_MAP.get(raw_name.strip().lower(), CanonicalCapability.UNKNOWN)
        if cap == CanonicalCapability.SHELL_EXEC:
            command = _extract_command(args)
            return classify_shell_command(command)
        return cap

    def sniff(self, source: RunSource) -> float:
        data = self._load(source, strict=False)
        if not isinstance(data, dict):
            return 0.0
        if data.get("adapter") == "generic":
            return 1.0
        if isinstance(data.get("events"), list):
            return 0.6  # looks like a generic trace even without the explicit tag
        return 0.0

    def parse(self, source: RunSource) -> NormalizedSession:
        data = self._load(source, strict=True)
        session_id = str(data.get("session_id", "session"))
        task = TaskSpec.from_dict(data.get("task"))

        available_tools = [
            ToolSpec(
                name=t["name"],
                capability=coerce_capability(t.get("capability")),
                schema=t.get("schema"),
            )
            for t in data.get("available_tools", [])
        ]

        events: list[NormalizedEvent] = []
        turn = 0
        for seq, raw in enumerate(data.get("events", [])):
            etype = EventType(raw["type"])
            event = NormalizedEvent(
                event_id=raw.get("event_id", f"{session_id}:{seq}"),
                session_id=session_id,
                seq=seq,
                type=etype,
                turn=raw.get("turn", turn),
                text=raw.get("text"),
                raw=raw,
            )
            if etype in (EventType.USER_MESSAGE, EventType.AGENT_MESSAGE):
                turn += 1
                event.turn = raw.get("turn", turn)
            elif etype == EventType.TOOL_CALL:
                event.tool_call = self._build_call(raw, session_id, seq, turn)
            elif etype == EventType.TOOL_RESULT:
                event.tool_result = _build_result(raw)
            events.append(event)

        return build_session(
            session_id=session_id,
            adapter=self.name,
            task=task,
            events=events,
            available_tools=available_tools,
            stop_reason=data.get("stop_reason"),
            metadata=data.get("metadata", {}),
        )

    # --- helpers ----------------------------------------------------------------

    def _build_call(
        self, raw: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> ToolCallEvent:
        raw_name = raw.get("tool_name", "")
        args = raw.get("arguments", {}) or {}
        explicit_cap = raw.get("capability")
        capability = (
            coerce_capability(explicit_cap)
            if explicit_cap is not None
            else self.canonicalize_tool(raw_name, args)
        )
        # Even when a capability is explicit, refine a generic SHELL_EXEC by command.
        if capability == CanonicalCapability.SHELL_EXEC:
            capability = classify_shell_command(_extract_command(args))
        return ToolCallEvent(
            call_id=str(raw.get("call_id", f"{session_id}:call:{seq}")),
            session_id=session_id,
            seq=seq,
            tool_name=raw_name,
            raw_tool_name=raw.get("raw_tool_name", raw_name),
            capability=capability,
            arguments=args,
            raw_arguments=raw.get("raw_arguments", args),
            turn=raw.get("turn", turn),
            parent_call_id=raw.get("parent_call_id"),
            preceding_reasoning=raw.get("preceding_reasoning"),
            adapter=self.name,
            raw_event_ref=seq,
        )

    @staticmethod
    def _load(source: RunSource, *, strict: bool) -> Any:
        if source.data is not None:
            return source.data
        if source.path is not None:
            try:
                return json.loads(Path(source.path).read_text())
            except (OSError, json.JSONDecodeError):
                if strict:
                    raise
                return None
        if source.stream is not None:
            text = "".join(str(chunk) for chunk in source.stream)
            return json.loads(text) if text else None
        if strict:
            raise ValueError("RunSource has no data, path, or stream to parse.")
        return None


def _extract_command(args: dict[str, Any]) -> str | None:
    for key in ("command", "cmd", "script", "shell_command"):
        val = args.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    return None


def _build_result(raw: dict[str, Any]) -> ToolResult:
    return ToolResult(
        call_id=str(raw.get("call_id", "")),
        status=raw.get("status", "unknown"),
        is_error=bool(raw.get("is_error", raw.get("status") == "error")),
        content=raw.get("content", "") or "",
        error_message=raw.get("error_message"),
        error_class=raw.get("error_class"),
        exit_code=raw.get("exit_code"),
        duration_ms=raw.get("duration_ms"),
        raw_content=raw.get("raw_content", raw.get("content")),
    )
