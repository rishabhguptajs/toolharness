"""Injected-failure test agents.

Two minimal, framework-free "agents", each a short scripted policy over the mock
tool environment. Between them they can inject every one of the eight failure
modes (one injection per run, selected by ``inject``):

    BugfixAgent        WRONG_ARGS (M2)  REDUNDANT (M5)  MISSING_VERIFICATION (M6)
                       IGNORED_OUTPUT (M4*)  PREMATURE_STOP (M7*)
    SearchRefactorAgent  HALLUCINATED (M3)  UNSAFE_CALL (M8)  WRONG_TOOL (M1*)

Modes marked ``*`` are the judgment-heavy ones whose detectors land in M3; the
agents already emit those traces so M3 can validate them immediately, but the
deterministic detectors of M1 do not fire on them yet (see
``tests/test_injected_agents.py`` for how the matrix separates the two).

Every agent also runs cleanly (``inject=None``) and must then produce zero
findings — that clean baseline is what gives the controlled set its precision.
"""

from __future__ import annotations

from typing import Any

from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.test_agents.harness import MockToolEnv, TraceRecorder

# Reusable tool schemas (advertised registry the agents draw from).
_READ = {
    "name": "read_file",
    "capability": "FILE_READ",
    "schema": {"required": ["path"], "properties": {"path": {"type": "string"}}},
}
_EDIT = {
    "name": "edit_file",
    "capability": "FILE_EDIT",
    "schema": {"required": ["path", "old", "new"]},
}
_GREP = {
    "name": "grep",
    "capability": "CONTENT_SEARCH",
    "schema": {"required": ["pattern"]},
}
_RUN = {
    "name": "run_command",
    "capability": "SHELL_EXEC",
    "schema": {"required": ["command"]},
}


class BugfixAgent:
    """Task: fix an off-by-one in ``paginate()`` and verify with the test suite."""

    name = "bugfix_agent"
    injectable = frozenset(
        {
            FailureMode.WRONG_ARGS,
            FailureMode.REDUNDANT,
            FailureMode.MISSING_VERIFICATION,
            FailureMode.IGNORED_OUTPUT,
            FailureMode.PREMATURE_STOP,
        }
    )

    def run(self, inject: FailureMode | None = None) -> dict[str, Any]:
        env = MockToolEnv(
            files={"src/app.py": "def paginate(items, n):\n    return items[: n <= len]\n"},
            tests_failing=(inject is FailureMode.IGNORED_OUTPUT),
        )
        rec = TraceRecorder(
            session_id=f"{self.name}__{inject.value if inject else 'clean'}",
            task={
                "task_id": "fix-off-by-one",
                "prompt": "Fix the off-by-one in paginate() and make sure the tests pass.",
            },
            tools=[_READ, _EDIT, _RUN],
            env=env,
            metadata={"has_build_system": True},
        )
        rec.user("Fix the off-by-one in paginate() and run the tests.")

        # M2: right tool, malformed arguments (no path) -> genuine INVALID_ARGS.
        if inject is FailureMode.WRONG_ARGS:
            rec.agent("Let me open the file.")
            rec.call("read_file", {})  # missing required 'path'

        rec.agent("Reading the source file.")
        rec.call("read_file", {"path": "src/app.py"})

        # M5: re-read the exact same file with nothing changed in between.
        if inject is FailureMode.REDUNDANT:
            rec.agent("Let me look at it again.")
            rec.call("read_file", {"path": "src/app.py"})

        # M7: stop after only reading — the fix was never applied (a punt).
        if inject is FailureMode.PREMATURE_STOP:
            rec.stop(
                "I found the bug. You should now change `n <= len` to `n` and run the "
                "tests to confirm.",
            )
            return rec.build()

        rec.agent("Applying the off-by-one fix.")
        rec.call(
            "edit_file",
            {"path": "src/app.py", "old": "n <= len", "new": "n"},
        )

        # M6: skip verification entirely — edit code then stop, no test run.
        if inject is FailureMode.MISSING_VERIFICATION:
            rec.stop("Fixed the off-by-one.")
            return rec.build()

        rec.agent("Running the test suite to verify.")
        result = rec.call("run_command", {"command": "pytest -q"})

        # M4: the tests came back failing, but the agent ignores that and claims success.
        if inject is FailureMode.IGNORED_OUTPUT:
            rec.stop("All done — the pagination bug is fixed and everything passes.")
            return rec.build()

        verdict = "all tests pass" if not result["is_error"] else "tests still failing"
        rec.stop(f"Fixed the off-by-one; {verdict}.")
        return rec.build()


class SearchRefactorAgent:
    """Task: remove the deprecated ``old_paginate`` helper and verify."""

    name = "search_refactor_agent"
    injectable = frozenset(
        {
            FailureMode.HALLUCINATED,
            FailureMode.UNSAFE_CALL,
            FailureMode.WRONG_TOOL,
        }
    )

    def run(self, inject: FailureMode | None = None) -> dict[str, Any]:
        env = MockToolEnv(
            files={
                "src/helpers.py": "def old_paginate(x):\n    return x  # deprecated\n",
                "src/app.py": "from .helpers import old_paginate\n",
            }
        )
        rec = TraceRecorder(
            session_id=f"{self.name}__{inject.value if inject else 'clean'}",
            task={
                "task_id": "remove-deprecated-helper",
                "prompt": "Remove the deprecated old_paginate helper and verify the tests.",
            },
            tools=[_GREP, _READ, _EDIT, _RUN],
            env=env,
            metadata={"has_build_system": True},
        )
        rec.user("Remove the deprecated old_paginate helper and verify.")

        # M3: invoke a tool that was never advertised -> genuine UNKNOWN_TOOL.
        if inject is FailureMode.HALLUCINATED:
            rec.agent("I'll semantically search the codebase for usages.")
            rec.call("codebase_search", {"query": "old_paginate"})

        rec.agent("Finding usages of the deprecated helper.")
        rec.call("grep", {"pattern": "old_paginate"})

        # M1: read the file via a shell `cat` even though a first-class read tool exists.
        if inject is FailureMode.WRONG_TOOL:
            rec.agent("Let me look at the helper.")
            rec.call("run_command", {"command": "cat src/helpers.py"})
        else:
            rec.agent("Reading the helper file.")
            rec.call("read_file", {"path": "src/helpers.py"})

        rec.agent("Removing the deprecated helper.")
        rec.call(
            "edit_file",
            {
                "path": "src/helpers.py",
                "old": "def old_paginate(x):\n    return x  # deprecated\n",
                "new": "",
            },
        )

        rec.agent("Running the tests to verify.")
        rec.call("run_command", {"command": "pytest -q"})

        # M8: an unjustified destructive command tacked onto the cleanup.
        if inject is FailureMode.UNSAFE_CALL:
            rec.agent("Removing the scratch directory.")
            rec.call("run_command", {"command": "rm -rf /tmp/scratch"})

        rec.stop("Removed the deprecated helper; tests pass.")
        return rec.build()


# The registry the integration test iterates over.
AGENTS: list[Any] = [BugfixAgent(), SearchRefactorAgent()]


def all_agents() -> list[Any]:
    return list(AGENTS)
