"""CursorAdapter — parses the NDJSON from ``cursor-agent -p --output-format stream-json``.

Event shapes (observed):
  * ``{"type":"system","subtype":"init", …}``   — session metadata; **no** tool
    registry, so M3 falls back to result-signal only (documented degradation).
  * ``{"type":"user","message":{"content":[{"type":"text","text":…}]}}``
  * ``{"type":"assistant","message":{"content":[{"type":"text","text":…}]}}`` —
    streamed token-by-token; consecutive deltas are coalesced into one message.
  * ``{"type":"tool_call","subtype":"started|completed","call_id":…,
       "tool_call":{"<kind>ToolCall":{"args":{…},"result":{…}}}}`` — ``started``
    becomes a TOOL_CALL, ``completed`` a TOOL_RESULT, correlated by the (unique)
    ``call_id`` string.
  * ``{"type":"result","subtype":"success", "result":…}``
"""

from __future__ import annotations

from typing import Any

from agent_eval_harness.adapters._util import (
    normalize_error_class,
    peek_jsonl,
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
    build_session,
)
from agent_eval_harness.core.taskspec import TaskSpec

# Cursor names each call by the tool-kind key, e.g. "readToolCall".
_CURSOR_CAP: dict[str, CanonicalCapability] = {
    "read": CanonicalCapability.FILE_READ,
    "write": CanonicalCapability.FILE_WRITE,
    "create": CanonicalCapability.FILE_WRITE,
    "edit": CanonicalCapability.FILE_EDIT,
    "multiEdit": CanonicalCapability.FILE_EDIT,
    "delete": CanonicalCapability.FILE_WRITE,
    "glob": CanonicalCapability.FILE_SEARCH,
    "ls": CanonicalCapability.FILE_SEARCH,
    "list": CanonicalCapability.FILE_SEARCH,
    "grep": CanonicalCapability.CONTENT_SEARCH,
    "search": CanonicalCapability.CONTENT_SEARCH,
    "codebaseSearch": CanonicalCapability.CONTENT_SEARCH,
    "shell": CanonicalCapability.SHELL_EXEC,
    "terminal": CanonicalCapability.SHELL_EXEC,
    "webSearch": CanonicalCapability.WEB_SEARCH,
    "webFetch": CanonicalCapability.WEB_FETCH,
    "fetch": CanonicalCapability.WEB_FETCH,
    "todo": CanonicalCapability.TASK_MGMT,
}


def _kind_of(tool_call: dict[str, Any]) -> str:
    """Return the tool-kind stem, e.g. "readToolCall" -> "read"."""
    key = next(iter(tool_call), "")
    return key[: -len("ToolCall")] if key.endswith("ToolCall") else key


def _capability_for_kind(kind: str, args: dict[str, Any]) -> CanonicalCapability:
    cap = _CURSOR_CAP.get(kind, CanonicalCapability.UNKNOWN)
    if cap == CanonicalCapability.SHELL_EXEC:
        command = args.get("command") if isinstance(args, dict) else None
        return classify_shell_command(command if isinstance(command, str) else None)
    return cap


class CursorAdapter:
    name = "cursor"

    def canonicalize_tool(self, raw_name: str, args: dict[str, Any]) -> CanonicalCapability:
        kind = raw_name[: -len("ToolCall")] if raw_name.endswith("ToolCall") else raw_name
        return _capability_for_kind(kind, args)

    def sniff(self, source: RunSource) -> float:
        records = peek_jsonl(source)
        if not records:
            return 0.0
        for r in records:
            if r.get("type") == "tool_call" and isinstance(r.get("tool_call"), dict):
                return 0.95
        # Fallback: Cursor's init has apiKeySource + model but no tool registry,
        # and its result carries request_id without num_turns/total_cost_usd.
        has_bare_init = any(
            r.get("type") == "system" and r.get("subtype") == "init"
            and "tools" not in r and "claude_code_version" not in r
            for r in records
        )
        has_cursor_result = any(
            r.get("type") == "result" and "request_id" in r
            and "num_turns" not in r and "total_cost_usd" not in r
            for r in records
        )
        if has_bare_init and has_cursor_result:
            return 0.55
        return 0.0

    def parse(self, source: RunSource) -> NormalizedSession:
        records = [r for r in peek_jsonl(source, limit=10**9)]
        session_id = "cursor-session"
        stop_reason: str | None = None
        metadata: dict[str, Any] = {}

        events: list[NormalizedEvent] = []
        seq = 0
        turn = 0
        pending_text: list[str] = []

        def flush_text(kind: EventType) -> None:
            nonlocal seq, turn
            if not pending_text:
                return
            text = "".join(pending_text)
            pending_text.clear()
            if not text.strip():
                return
            turn += 1
            events.append(NormalizedEvent(
                event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                type=kind, turn=turn, text=text, raw=None,
            ))
            seq += 1

        for rec in records:
            rtype = rec.get("type")
            if rtype == "system":
                session_id = str(rec.get("session_id", session_id))
                for key in ("model", "cwd", "permissionMode"):
                    if key in rec:
                        metadata[key] = rec[key]
            elif rtype == "user":
                for block in (rec.get("message") or {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        pending_text.append(block.get("text", ""))
                flush_text(EventType.USER_MESSAGE)
            elif rtype == "assistant":
                for block in (rec.get("message") or {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        pending_text.append(block.get("text", ""))
            elif rtype == "tool_call":
                flush_text(EventType.AGENT_MESSAGE)
                sub = rec.get("subtype")
                if sub == "started":
                    events.append(self._tool_call(rec, session_id, seq, turn))
                    seq += 1
                elif sub == "completed":
                    events.append(self._tool_result(rec, session_id, seq, turn))
                    seq += 1
            elif rtype == "result":
                flush_text(EventType.AGENT_MESSAGE)
                session_id = str(rec.get("session_id", session_id))
                stop_reason = "error" if rec.get("is_error") else "completed"
                events.append(NormalizedEvent(
                    event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
                    type=EventType.AGENT_STOP, turn=turn,
                    text=rec.get("result") if isinstance(rec.get("result"), str) else None,
                    raw=rec,
                ))
                seq += 1

        return build_session(
            session_id=session_id, adapter=self.name, task=TaskSpec.from_dict(None),
            events=events, available_tools=[], stop_reason=stop_reason, metadata=metadata,
        )

    def _tool_call(
        self, rec: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        tc = rec.get("tool_call") or {}
        key = next(iter(tc), "")
        inner = tc.get(key) or {}
        args = inner.get("args") if isinstance(inner.get("args"), dict) else {}
        kind = _kind_of(tc)
        call = ToolCallEvent(
            call_id=str(rec.get("call_id", f"{session_id}:call:{seq}")),
            session_id=session_id, seq=seq, tool_name=key or kind, raw_tool_name=key,
            capability=_capability_for_kind(kind, args or {}),
            arguments=args or {}, raw_arguments=inner.get("args"),
            turn=turn, adapter=self.name, raw_event_ref=seq,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_CALL, turn=turn, tool_call=call, raw=rec,
        )

    def _tool_result(
        self, rec: dict[str, Any], session_id: str, seq: int, turn: int
    ) -> NormalizedEvent:
        tc = rec.get("tool_call") or {}
        key = next(iter(tc), "")
        inner = tc.get(key) or {}
        raw_result = inner.get("result")
        result_obj = raw_result if isinstance(raw_result, dict) else {}
        content, is_error, exit_code = _extract_cursor_result(result_obj)
        result = ToolResult(
            call_id=str(rec.get("call_id", "")),
            status=status_of(is_error=is_error, exit_code=exit_code),
            is_error=is_error, content=content,
            error_message=content if is_error else None,
            error_class=normalize_error_class(content, is_error=is_error),
            exit_code=exit_code, raw_content=result_obj,
        )
        return NormalizedEvent(
            event_id=f"{session_id}:{seq}", session_id=session_id, seq=seq,
            type=EventType.TOOL_RESULT, turn=turn, tool_result=result, raw=rec,
        )


def _extract_cursor_result(result_obj: dict[str, Any]) -> tuple[str, bool, int | None]:
    """Return (content, is_error, exit_code) from a Cursor result envelope.

    Success/failure are keyed under ``result.success`` / ``result.failure``.
    """
    if "failure" in result_obj and isinstance(result_obj["failure"], dict):
        f = result_obj["failure"]
        parts = [str(f.get("stdout", "")), str(f.get("stderr", ""))]
        content = "".join(p for p in parts if p) or str(f.get("error", "")) or "failed"
        ec = f.get("exitCode")
        return content, True, ec if isinstance(ec, int) else None
    if "success" in result_obj and isinstance(result_obj["success"], dict):
        s = result_obj["success"]
        for key in ("content", "resultForModel", "stdout", "diffString"):
            val = s.get(key)
            if isinstance(val, str) and val:
                return val, False, s.get("exitCode") if isinstance(s.get("exitCode"), int) else None
        ec = s.get("exitCode")
        return "", False, ec if isinstance(ec, int) else None
    return "", False, None
