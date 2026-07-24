"""Minimal execution harness for the injected-failure test agents.

The point of these agents is validation honesty: rather than hand-authoring
tool *results*, the agents issue tool calls against a tiny in-memory
``MockToolEnv`` that produces **genuine** results — a read of a missing path
really returns ``ENOENT``, an unregistered tool really returns ``UNKNOWN_TOOL``,
a call missing a required field really returns ``INVALID_ARGS``. The detectors
therefore see the same result signals a real CLI would emit, and the whole path
(agent → generic trace → adapter → detectors → scoring) is exercised end to end.

Nothing here uses an LLM or an agent framework: an "agent" is a short scripted
policy (see ``agents.py``) that calls ``TraceRecorder`` methods. An *injection*
perturbs that policy to introduce exactly one failure mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Tool-name families the mock environment understands. Kept small on purpose; the
# agents only use these tools.
_READ_TOOLS = {"read_file", "read"}
_WRITE_TOOLS = {"write_file", "create_file", "write"}
_EDIT_TOOLS = {"edit_file", "edit", "str_replace", "apply_patch"}
_SEARCH_TOOLS = {"grep", "ripgrep", "search"}
_SHELL_TOOLS = {"bash", "run_command", "run_shell_command", "shell", "run"}
_DELETE_TOOLS = {"delete_file", "delete", "rm"}


@dataclass
class MockToolEnv:
    """A pretend workspace: an in-memory file system plus a shell simulator.

    ``tests_failing`` flips the simulated test command from a passing to a failing
    result, so an agent can produce a genuine failing-verification signal for the
    ignored-output injection.
    """

    files: dict[str, str] = field(default_factory=dict)
    tests_failing: bool = False

    def execute(
        self, tool_name: str, args: dict[str, Any], known_tools: set[str]
    ) -> dict[str, Any]:
        # A call to a tool that was never advertised: the runtime rejects it. This
        # is what makes the hallucination injection a *real* unknown-tool result.
        if tool_name not in known_tools:
            return _err(
                "UNKNOWN_TOOL",
                f"No such tool: {tool_name!r}",
                content=f"error: unknown tool {tool_name!r}",
            )

        name = tool_name.strip().lower()
        if name in _READ_TOOLS:
            return self._read(args)
        if name in _WRITE_TOOLS:
            return self._write(args)
        if name in _EDIT_TOOLS:
            return self._edit(args)
        if name in _SEARCH_TOOLS:
            return self._grep(args)
        if name in _SHELL_TOOLS:
            return self._shell(args)
        if name in _DELETE_TOOLS:
            return self._delete(args)
        return _ok(f"{tool_name} ok")

    # --- file tools -------------------------------------------------------------

    def _read(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _path(args)
        if not path:
            return _err("INVALID_ARGS", "path is required")
        if path not in self.files:
            return _err("ENOENT", f"no such file: {path}")
        return _ok(self.files[path])

    def _write(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _path(args)
        if not path:
            return _err("INVALID_ARGS", "path is required")
        self.files[path] = str(args.get("content", ""))
        return _ok(f"wrote {path}")

    def _edit(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _path(args)
        if not path:
            return _err("INVALID_ARGS", "path is required")
        if path not in self.files:
            return _err("ENOENT", f"no such file: {path}")
        old, new = args.get("old"), args.get("new")
        if old is None or new is None:
            return _err("INVALID_ARGS", "edit requires 'old' and 'new'")
        if old not in self.files[path]:
            return _err("EDIT_FAILED", f"pattern not found: {old!r}")
        self.files[path] = self.files[path].replace(str(old), str(new), 1)
        return _ok(f"edited {path}")

    def _delete(self, args: dict[str, Any]) -> dict[str, Any]:
        path = _path(args)
        if path and path in self.files:
            del self.files[path]
        return _ok(f"deleted {path}")

    def _grep(self, args: dict[str, Any]) -> dict[str, Any]:
        pattern = args.get("pattern") or args.get("query")
        if not pattern:
            return _err("INVALID_ARGS", "pattern is required")
        hits = [
            f"{p}: {line}"
            for p, text in self.files.items()
            for line in text.splitlines()
            if str(pattern) in line
        ]
        return _ok("\n".join(hits) if hits else "no matches")

    # --- shell ------------------------------------------------------------------

    def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = _command(args)
        if not command:
            return _err("INVALID_ARGS", "command is required")

        # Test invocation: pass or fail depending on the seeded env state.
        if re.search(r"\b(pytest|py\.test|npm\s+test|go\s+test|jest|vitest)\b", command):
            if self.tests_failing:
                return _fail("1 failed, 2 passed", exit_code=1)
            return _ok("3 passed", exit_code=0)

        # `cat FILE`: reading a file through the shell (an anti-pattern the wrong-tool
        # detector cares about) still returns the file's content, like the real thing.
        m = re.match(r"\s*cat\s+(\S+)", command)
        if m:
            path = m.group(1)
            if path in self.files:
                return _ok(self.files[path])
            return _fail(f"cat: {path}: No such file or directory", exit_code=1)

        # Everything else (rm, git, mkdir, ...) just succeeds — we are validating the
        # detectors' reaction to the *call*, not modelling a real shell.
        return _ok(f"$ {command}\n(ok)", exit_code=0)


# --- result constructors ----------------------------------------------------------


def _ok(content: str, exit_code: int | None = None) -> dict[str, Any]:
    r: dict[str, Any] = {"status": "ok", "is_error": False, "content": content}
    if exit_code is not None:
        r["exit_code"] = exit_code
    return r


def _fail(content: str, exit_code: int = 1) -> dict[str, Any]:
    return {"status": "error", "is_error": True, "content": content, "exit_code": exit_code}


def _err(error_class: str, message: str, content: str | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "is_error": True,
        "content": content if content is not None else f"error: {message}",
        "error_class": error_class,
        "error_message": message,
    }


def _path(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filename", "file", "target_file", "abs_path"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _command(args: dict[str, Any]) -> str | None:
    for key in ("command", "cmd", "script", "shell_command"):
        val = args.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    return None


# --- trace recorder ---------------------------------------------------------------


class TraceRecorder:
    """Accumulates events and renders the canonical generic-trace dict.

    ``call`` executes the tool against the env and records both the ``tool_call``
    and its real ``tool_result``; ``call_id``s are assigned in order.
    """

    def __init__(
        self,
        session_id: str,
        task: dict[str, Any],
        tools: list[dict[str, Any]],
        env: MockToolEnv,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.task = task
        self.tools = tools
        self.env = env
        self.metadata = metadata or {}
        self._known = {t["name"] for t in tools}
        self._events: list[dict[str, Any]] = []
        self._n_calls = 0
        self.stop_reason = "completed"

    def user(self, text: str) -> None:
        self._events.append({"type": "user_message", "text": text})

    def agent(self, text: str) -> None:
        self._events.append({"type": "agent_message", "text": text})

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        self._n_calls += 1
        call_id = f"c{self._n_calls}"
        self._events.append(
            {
                "type": "tool_call",
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
        result = self.env.execute(tool_name, arguments, self._known)
        self._events.append({"type": "tool_result", "call_id": call_id, **result})
        return result

    def stop(self, text: str, reason: str = "completed") -> None:
        self.stop_reason = reason
        self._events.append({"type": "agent_stop", "text": text})

    def build(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "adapter": "generic",
            "stop_reason": self.stop_reason,
            "task": self.task,
            "metadata": self.metadata,
            "available_tools": self.tools,
            "events": self._events,
        }
