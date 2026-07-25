"""CodexAdapter — parses Codex CLI runs in two on-disk shapes.

* **exec stream** — the ``thread``/``turn``/``item`` events emitted by
  ``codex exec --json`` on stdout. ``item.started`` opens a call, ``item.completed``
  closes it, correlated by ``item.id``. Item kinds: ``agent_message``,
  ``command_execution`` (shell), ``file_change`` (patch).

* **session rollout** — ``$CODEX_HOME/sessions/**/*.jsonl``, an OpenAI-responses
  style log: ``response_item`` records (``function_call`` / ``function_call_output``
  / ``custom_tool_call`` for ``apply_patch``) correlated by ``call_id``, plus
  ``event_msg`` records (``user_message`` / ``agent_message`` / ``task_complete``).

``parse`` detects the shape; ``sniff`` recognizes both. Neither shape advertises a
tool registry, so M3 falls back to result-signal only (documented degradation).
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalharness.adapters._util import (
    load_jsonl,
    normalize_error_class,
    peek_jsonl,
    status_of,
)
from evalharness.adapters.base import RunSource
from evalharness.core.capability import (
    CanonicalCapability,
    classify_shell_command,
)
from evalharness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    build_session,
)
from evalharness.core.taskspec import TaskSpec

_CODEX_CAP: dict[str, CanonicalCapability] = {
    "exec_command": CanonicalCapability.SHELL_EXEC,
    "shell": CanonicalCapability.SHELL_EXEC,
    "local_shell": CanonicalCapability.SHELL_EXEC,
    "run": CanonicalCapability.SHELL_EXEC,
    "container.exec": CanonicalCapability.SHELL_EXEC,
    "apply_patch": CanonicalCapability.FILE_EDIT,
    "read_file": CanonicalCapability.FILE_READ,
    "write_file": CanonicalCapability.FILE_WRITE,
    "update_plan": CanonicalCapability.TASK_MGMT,
    "web_search": CanonicalCapability.WEB_SEARCH,
    "web_fetch": CanonicalCapability.WEB_FETCH,
}

_EXIT_RE = re.compile(r"(?:exit(?:ed with)? code[:\s]+|process exited with code\s+)(\d+)",
                      re.IGNORECASE)

_EXEC_STREAM_TYPES = {
    "thread.started", "turn.started", "turn.completed", "item.started", "item.completed",
}
_ROLLOUT_TYPES = {"session_meta", "response_item", "event_msg", "turn_context"}


def _cmd_of(args: dict[str, Any]) -> str | None:
    for key in ("cmd", "command", "script"):
        val = args.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    return None


def _capability_for_codex(name: str, args: dict[str, Any]) -> CanonicalCapability:
    cap = _CODEX_CAP.get(name, CanonicalCapability.UNKNOWN)
    if cap == CanonicalCapability.SHELL_EXEC:
        return classify_shell_command(_cmd_of(args))
    return cap


def _exit_code_from(text: str) -> int | None:
    m = _EXIT_RE.search(text or "")
    return int(m.group(1)) if m else None


class CodexAdapter:
    name = "codex"

    def canonicalize_tool(self, raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
        return _capability_for_codex(raw_name, args)

    def sniff(self, source: RunSource) -> float:
        records = peek_jsonl(source)
        if not records:
            return 0.0
        types = {r.get("type") for r in records}
        if types & _EXEC_STREAM_TYPES:
            return 0.95
        if types & _ROLLOUT_TYPES:
            return 0.95
        return 0.0

    def parse(self, source: RunSource) -> NormalizedSession:
        records = load_jsonl(source)
        types = {r.get("type") for r in records}
        if types & _EXEC_STREAM_TYPES:
            return self._parse_exec_stream(records)
        if types & _ROLLOUT_TYPES:
            return self._parse_rollout(records)
        raise ValueError("CodexAdapter: unrecognized source (neither exec-stream nor rollout).")

    # --- exec stream ------------------------------------------------------------

    def _parse_exec_stream(self, records: list[dict[str, Any]]) -> NormalizedSession:
        session_id = "codex-session"
        events: list[NormalizedEvent] = []
        seq = 0
        turn = 0
        stop_reason: str | None = None

        for rec in records:
            rtype = rec.get("type")
            if rtype == "thread.started":
                session_id = str(rec.get("thread_id", session_id))
            elif rtype == "turn.completed":
                stop_reason = "completed"
            elif rtype in ("item.started", "item.completed"):
                item = rec.get("item") or {}
                itype = item.get("type")
                if itype == "agent_message":
                    if rtype == "item.completed" and item.get("text"):
                        turn += 1
                        events.append(NormalizedEvent(
                            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                            type=EventType.AGENT_MESSAGE, turn=turn, text=item["text"], raw=rec,
                        ))
                        seq += 1
                elif itype in ("command_execution", "file_change"):
                    if rtype == "item.started":
                        events.append(self._exec_call(item, session_id, seq, turn))
                        seq += 1
                    else:
                        events.append(self._exec_result(item, session_id, seq, turn))
                        seq += 1

        return build_session(
            session_id=session_id, adapter=self.name, task=TaskSpec.from_dict(None),
            events=events, available_tools=[], stop_reason=stop_reason,
            metadata={"mode": "exec-stream"},
        )

    def _exec_call(
        self, item: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        call_id = str(item.get("id", f"{session_id}:call:{seq}"))
        if item.get("type") == "command_execution":
            command = item.get("command", "")
            name = "exec_command"
            args: dict[str, Any] = {"command": command}
            cap = classify_shell_command(command if isinstance(command, str) else None)
        else:  # file_change
            changes = item.get("changes") or []
            name = "apply_patch"
            first_path = changes[0].get("path") if changes else None
            kinds = {c.get("kind") for c in changes}
            args = {"path": first_path, "changes": changes}
            cap = (CanonicalCapability.FILE_WRITE if kinds == {"add"}
                   else CanonicalCapability.FILE_EDIT)
        call = ToolCallEvent(
            call_id=call_id, session_id=session_id, seq=seq, tool_name=name,
            raw_tool_name=name, capability=cap, arguments=args, raw_arguments=item,
            turn=turn, adapter=self.name, raw_event_ref=seq,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_CALL, turn=turn, tool_call=call, raw=item,
        )

    def _exec_result(
        self, item: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        call_id = str(item.get("id", ""))
        exit_code = item.get("exit_code")
        exit_code = exit_code if isinstance(exit_code, int) else None
        content = str(item.get("aggregated_output", "") or "")
        is_error = (item.get("status") == "failed") or (
            exit_code is not None and exit_code != 0
        )
        result = ToolResult(
            call_id=call_id, status=status_of(is_error=is_error, exit_code=exit_code),
            is_error=is_error, content=content,
            error_message=content if is_error else None,
            error_class=normalize_error_class(content, is_error=is_error),
            exit_code=exit_code, raw_content=item,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_RESULT, turn=turn, tool_result=result, raw=item,
        )

    # --- session rollout --------------------------------------------------------

    def _parse_rollout(self, records: list[dict[str, Any]]) -> NormalizedSession:
        session_id = "codex-session"
        events: list[NormalizedEvent] = []
        seq = 0
        turn = 0
        stop_reason: str | None = None
        metadata: dict[str, Any] = {"mode": "rollout"}

        for rec in records:
            rtype = rec.get("type")
            payload = rec.get("payload") or {}
            ptype = payload.get("type")

            if rtype == "session_meta":
                session_id = str(payload.get("session_id", session_id))
                for key in ("cwd", "cli_version", "model_provider"):
                    if key in payload:
                        metadata[key] = payload[key]
            elif rtype == "event_msg" and ptype == "user_message":
                turn += 1
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.USER_MESSAGE, turn=turn,
                    text=payload.get("message"), raw=rec,
                ))
                seq += 1
            elif rtype == "event_msg" and ptype == "agent_message":
                turn += 1
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.AGENT_MESSAGE, turn=turn,
                    text=payload.get("message"), raw=rec,
                ))
                seq += 1
            elif rtype == "event_msg" and ptype == "task_complete":
                stop_reason = "completed"
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.AGENT_STOP, turn=turn,
                    text=payload.get("last_agent_message"), raw=rec,
                ))
                seq += 1
            elif rtype == "response_item" and ptype in ("function_call", "custom_tool_call"):
                events.append(self._rollout_call(payload, session_id, seq, turn))
                seq += 1
            elif rtype == "response_item" and ptype in (
                "function_call_output", "custom_tool_call_output"
            ):
                events.append(self._rollout_result(payload, session_id, seq, turn))
                seq += 1
            # response_item/message, reasoning, event_msg/token_count -> ignored

        return build_session(
            session_id=session_id, adapter=self.name, task=TaskSpec.from_dict(None),
            events=events, available_tools=[], stop_reason=stop_reason, metadata=metadata,
        )

    def _rollout_call(
        self, payload: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        name = str(payload.get("name", ""))
        raw_args = payload.get("arguments", payload.get("input"))
        args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else {"input": raw_args}
            except json.JSONDecodeError:
                args = {"input": raw_args}  # e.g. apply_patch's patch text
        call = ToolCallEvent(
            call_id=str(payload.get("call_id", f"{session_id}:call:{seq}")),
            session_id=session_id, seq=seq, tool_name=name, raw_tool_name=name,
            capability=_capability_for_codex(name, args),
            arguments=args, raw_arguments=raw_args, turn=turn,
            adapter=self.name, raw_event_ref=seq,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_CALL, turn=turn, tool_call=call, raw=payload,
        )

    def _rollout_result(
        self, payload: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        output = payload.get("output")
        content = output if isinstance(output, str) else json.dumps(output) if output else ""
        exit_code = _exit_code_from(content)
        is_error = exit_code is not None and exit_code != 0
        result = ToolResult(
            call_id=str(payload.get("call_id", "")),
            status=status_of(is_error=is_error, exit_code=exit_code),
            is_error=is_error, content=content,
            error_message=content if is_error else None,
            error_class=normalize_error_class(content, is_error=is_error),
            exit_code=exit_code, raw_content=output,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_RESULT, turn=turn, tool_result=result, raw=payload,
        )
