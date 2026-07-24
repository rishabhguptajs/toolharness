"""ClaudeCodeAdapter — parses Claude Code runs in two sub-modes.

* **SDK / stream-json** (highest fidelity): the NDJSON emitted by
  ``claude -p … --output-format stream-json --verbose``. The ``system/init``
  record advertises the full tool registry; ``assistant`` records carry
  ``tool_use`` blocks with complete arguments; ``user`` records carry
  ``tool_result`` blocks correlated by ``tool_use_id``. This drives M2/M3 fully.

* **OTEL logs** (degraded): the OpenTelemetry log records emitted when
  ``CLAUDE_CODE_ENABLE_TELEMETRY=1`` (``claude_code.tool_decision`` /
  ``claude_code.tool_result``). These carry the tool *name* and result signal but
  **not** the arguments and **no** registry, so M2 degrades and M3 falls back to
  result-signal only — exactly the documented graceful-degradation path. Both the
  OTLP/JSON shape and the ``OTEL_LOGS_EXPORTER=console`` pretty-print are accepted.

``parse`` auto-detects the sub-mode; ``sniff`` recognizes both.
"""

from __future__ import annotations

import re
from typing import Any

from agent_eval_harness.adapters._util import (
    flatten_content,
    load_jsonl,
    normalize_error_class,
    peek_jsonl,
    read_text,
    status_of,
)
from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.core.capability import (
    CanonicalCapability,
    classify_shell_command,
)
from agent_eval_harness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from agent_eval_harness.core.taskspec import TaskSpec

_CLAUDE_CAP: dict[str, CanonicalCapability] = {
    "Read": CanonicalCapability.FILE_READ,
    "Write": CanonicalCapability.FILE_WRITE,
    "Edit": CanonicalCapability.FILE_EDIT,
    "MultiEdit": CanonicalCapability.FILE_EDIT,
    "NotebookEdit": CanonicalCapability.FILE_EDIT,
    "Glob": CanonicalCapability.FILE_SEARCH,
    "Grep": CanonicalCapability.CONTENT_SEARCH,
    "Bash": CanonicalCapability.SHELL_EXEC,
    "BashOutput": CanonicalCapability.SHELL_EXEC,
    "KillShell": CanonicalCapability.SHELL_EXEC,
    "KillBash": CanonicalCapability.SHELL_EXEC,
    "WebFetch": CanonicalCapability.WEB_FETCH,
    "WebSearch": CanonicalCapability.WEB_SEARCH,
    "Task": CanonicalCapability.SUBAGENT,
    "TodoWrite": CanonicalCapability.TASK_MGMT,
    "ExitPlanMode": CanonicalCapability.TASK_MGMT,
}

_EXIT_CODE_RE = re.compile(r"exit code[:\s]+(\d+)", re.IGNORECASE)


def _capability_for_name(raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
    if raw_name.startswith("mcp__"):
        return CanonicalCapability.MCP_TOOL
    cap = _CLAUDE_CAP.get(raw_name, CanonicalCapability.UNKNOWN)
    if cap == CanonicalCapability.SHELL_EXEC:
        command = args.get("command") if isinstance(args, dict) else None
        return classify_shell_command(command if isinstance(command, str) else None)
    return cap


class ClaudeCodeAdapter:
    name = "claude-code"

    def canonicalize_tool(self, raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
        return _capability_for_name(raw_name, args)

    # --- sniff ------------------------------------------------------------------

    def sniff(self, source: RunSource) -> float:
        records = peek_jsonl(source)
        if records:
            # A top-level "tool_call" event is Cursor's shape, never Claude's.
            if any(r.get("type") == "tool_call" for r in records):
                return 0.0
            score = 0.0
            for r in records:
                t = r.get("type")
                if t == "system" and r.get("subtype") == "init" and (
                    "tools" in r or "claude_code_version" in r
                ):
                    score = max(score, 0.95)
                if t == "assistant" and isinstance(r.get("message"), dict):
                    mid = str(r["message"].get("id", ""))
                    if mid.startswith("msg_"):
                        score = max(score, 0.9)
                if t == "result" and ("num_turns" in r or "total_cost_usd" in r):
                    score = max(score, 0.9)
            return score
        text = read_text(source)
        if "com.anthropic.claude_code" in text or 'body: "claude_code.' in text:
            return 0.8
        return 0.0

    # --- parse ------------------------------------------------------------------

    def parse(self, source: RunSource) -> NormalizedSession:
        records = load_jsonl(source)
        if records and any(
            r.get("type") in ("assistant", "result")
            or (r.get("type") == "system" and r.get("subtype") == "init")
            for r in records
        ):
            return self._parse_sdk(records)
        text = read_text(source)
        if "claude_code" in text:
            return self._parse_otel(text)
        raise ValueError("ClaudeCodeAdapter: unrecognized source (neither SDK nor OTEL).")

    # --- SDK / stream-json ------------------------------------------------------

    def _parse_sdk(self, records: list[dict[str, Any]]) -> NormalizedSession:
        session_id = "claude-session"
        available_tools: list[ToolSpec] = []
        stop_reason: str | None = None
        metadata: dict[str, Any] = {"mode": "sdk"}

        events: list[NormalizedEvent] = []
        seq = 0
        turn = 0
        message_turns: dict[str, int] = {}

        for rec in records:
            rtype = rec.get("type")
            if rtype == "system" and rec.get("subtype") == "init":
                session_id = str(rec.get("session_id", session_id))
                available_tools = [
                    ToolSpec(name=n, capability=_capability_for_name(n, {}))
                    for n in rec.get("tools", [])
                    if isinstance(n, str)
                ]
                for key in ("model", "cwd", "permissionMode", "claude_code_version"):
                    if key in rec:
                        metadata[key] = rec[key]
                continue

            if rtype == "assistant":
                msg = rec.get("message") or {}
                mid = str(msg.get("id", ""))
                if mid and mid not in message_turns:
                    turn += 1
                    message_turns[mid] = turn
                cur_turn = message_turns.get(mid, turn)
                for block in msg.get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        events.append(NormalizedEvent(
                            event_id=f"{session_id}:{seq}", session_id=session_id,
                            seq=seq, type=EventType.AGENT_MESSAGE, turn=cur_turn,
                            text=block["text"], raw=rec,
                        ))
                        seq += 1
                    elif btype == "tool_use":
                        events.append(self._sdk_tool_call(block, session_id, seq, cur_turn, rec))
                        seq += 1
                continue

            if rtype == "user":
                msg = rec.get("message") or {}
                for block in msg.get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        events.append(self._sdk_tool_result(block, session_id, seq, turn, rec))
                        seq += 1
                    elif block.get("type") == "text" and block.get("text"):
                        events.append(NormalizedEvent(
                            event_id=f"{session_id}:{seq}", session_id=session_id,
                            seq=seq, type=EventType.USER_MESSAGE, turn=turn,
                            text=block["text"], raw=rec,
                        ))
                        seq += 1
                continue

            if rtype == "result":
                session_id = str(rec.get("session_id", session_id))
                stop_reason = _map_result_stop(rec)
                for key in ("num_turns", "total_cost_usd", "subtype"):
                    if key in rec:
                        metadata[key] = rec[key]
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.AGENT_STOP, turn=turn,
                    text=rec.get("result") if isinstance(rec.get("result"), str) else None,
                    raw=rec,
                ))
                seq += 1
                continue
            # rate_limit_event, system/hook_*, system/thinking_tokens -> ignored

        return build_session(
            session_id=session_id, adapter=self.name, task=TaskSpec.from_dict(None),
            events=events, available_tools=available_tools,
            stop_reason=stop_reason, metadata=metadata,
        )

    def _sdk_tool_call(
        self, block: dict[str, Any], session_id: str, seq: int, turn: int, rec: Any
    ) -> NormalizedEvent:
        raw_name = str(block.get("name", ""))
        args = block.get("input") if isinstance(block.get("input"), dict) else {}
        call = ToolCallEvent(
            call_id=str(block.get("id", f"{session_id}:call:{seq}")),
            session_id=session_id, seq=seq, tool_name=raw_name, raw_tool_name=raw_name,
            capability=_capability_for_name(raw_name, args or {}),
            arguments=args or {}, raw_arguments=block.get("input"),
            turn=turn, adapter=self.name, raw_event_ref=seq,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_CALL, turn=turn, tool_call=call, raw=rec,
        )

    def _sdk_tool_result(
        self, block: dict[str, Any], session_id: str, seq: int, turn: int, rec: Any
    ) -> NormalizedEvent:
        content = flatten_content(block.get("content"))
        is_error = bool(block.get("is_error", False))
        m = _EXIT_CODE_RE.search(content)
        exit_code = int(m.group(1)) if m else None
        result = ToolResult(
            call_id=str(block.get("tool_use_id", "")),
            status=status_of(is_error=is_error, exit_code=exit_code),
            is_error=is_error, content=content,
            error_message=content if is_error else None,
            error_class=normalize_error_class(content, is_error=is_error),
            exit_code=exit_code, raw_content=block.get("content"),
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_RESULT, turn=turn, tool_result=result, raw=rec,
        )

    # --- OTEL logs (degraded) ---------------------------------------------------

    def _parse_otel(self, text: str) -> NormalizedSession:
        session_id = "claude-otel-session"
        events: list[NormalizedEvent] = []
        seq = 0
        for body, attrs in _iter_otel_records(text):
            sid = attrs.get("session.id")
            if sid:
                session_id = sid
            if body == "claude_code.user_prompt":
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.USER_MESSAGE, text=attrs.get("prompt"), raw=attrs,
                ))
                seq += 1
            elif body == "claude_code.tool_decision":
                name = attrs.get("tool_name", "")
                call = ToolCallEvent(
                    call_id=attrs.get("tool_use_id", f"{session_id}:call:{seq}"),
                    session_id=session_id, seq=seq, tool_name=name, raw_tool_name=name,
                    capability=_capability_for_name(name, {}), arguments={},
                    raw_arguments=None, adapter=self.name, raw_event_ref=seq,
                )
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.TOOL_CALL, tool_call=call, raw=attrs,
                ))
                seq += 1
            elif body == "claude_code.tool_result":
                success = str(attrs.get("success", "true")).lower() == "true"
                dur = attrs.get("duration_ms")
                result = ToolResult(
                    call_id=attrs.get("tool_use_id", ""),
                    status="ok" if success else "error", is_error=not success,
                    content="", error_class=None,
                    duration_ms=int(dur) if isinstance(dur, str) and dur.isdigit() else None,
                )
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.TOOL_RESULT, tool_result=result, raw=attrs,
                ))
                seq += 1

        return build_session(
            session_id=session_id, adapter=self.name, task=TaskSpec.from_dict(None),
            events=events, available_tools=[], stop_reason=None,
            metadata={"mode": "otel", "degraded": True},
        )


def _map_result_stop(rec: dict[str, Any]) -> str:
    if rec.get("is_error"):
        return "error"
    terminal = rec.get("terminal_reason")
    if terminal in ("completed", "max_turns", "error"):
        return str(terminal)
    subtype = rec.get("subtype")
    if subtype == "error_max_turns":
        return "max_turns"
    if subtype and subtype != "success":
        return "error"
    return "completed"


# --- OTEL console/OTLP record extraction ------------------------------------------
# Accepts the JS console-exporter pretty-print (unquoted keys, `undefined`, trailing
# commas). Each record starts at a lone "{" line; we pull `body:` and the top-level
# (post-body) `attributes: {…}` block, whose values are all scalars.

_ATTR_RE = re.compile(r'^\s*"?([A-Za-z0-9_.]+)"?:\s*(.+?),?\s*$')


def _iter_otel_records(text: str) -> list[tuple[str, dict[str, str]]]:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln == "{"]
    starts.append(len(lines))
    out: list[tuple[str, dict[str, str]]] = []
    for a, b in zip(starts, starts[1:], strict=False):
        block = lines[a:b]
        body: str | None = None
        body_idx = -1
        for i, ln in enumerate(block):
            m = re.match(r'\s*body:\s*"([^"]+)"', ln)
            if m:
                body = m.group(1)
                body_idx = i
                break
        if body is None:
            continue
        attrs: dict[str, str] = {}
        in_attrs = False
        for ln in block[body_idx:]:
            stripped = ln.strip()
            if not in_attrs:
                if stripped.startswith("attributes:"):
                    in_attrs = True
                continue
            if stripped in ("}", "},"):
                break
            m2 = _ATTR_RE.match(ln)
            if m2:
                key, val = m2.group(1), m2.group(2).strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                attrs[key] = val
        out.append((body, attrs))
    return out
