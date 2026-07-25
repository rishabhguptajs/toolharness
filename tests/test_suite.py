"""Bundled task suite: discovery, sandboxed multi-task run (fake CLI), aggregation."""

from __future__ import annotations

import json

from evalharness.suite.runner import (
    BUNDLED_TASKS_DIR,
    discover_tasks,
    format_report,
    run_suite,
)

# A tiny valid Claude-Code SDK trace the fake CLI replays for every task.
_TRACE = "\n".join(json.dumps(r) for r in [
    {"type": "system", "subtype": "init", "session_id": "s",
     "tools": ["Read", "Edit", "Bash"], "cwd": "/w"},
    {"type": "assistant", "message": {"id": "m1", "content": [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "calc.py"}},
    ]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "def add", "is_error": False},
    ]}},
    {"type": "assistant", "message": {"id": "m2", "content": [
        {"type": "tool_use", "id": "t2", "name": "Edit",
         "input": {"file_path": "calc.py", "old_string": "a", "new_string": "b"}},
    ]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t2", "content": "ok", "is_error": False},
    ]}},
    {"type": "assistant", "message": {"id": "m3", "content": [
        {"type": "tool_use", "id": "t3", "name": "Bash",
         "input": {"command": "python -m pytest -q"}},
    ]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t3", "content": "3 passed", "is_error": False},
    ]}},
    {"type": "result", "subtype": "success", "session_id": "s", "num_turns": 3,
     "result": "done"},
])


def _ok_runner(command, cwd, timeout):
    return 0, _TRACE, "", False


def _boom_runner(command, cwd, timeout):
    raise FileNotFoundError("claude: not found")


def test_bundled_tasks_discovered():
    tasks = discover_tasks()
    ids = {t.task_id for t in tasks}
    assert {"add-power", "implement-gcd", "fix-fizzbuzz"} <= ids
    for t in tasks:
        assert t.seed_dir.is_dir()               # each ships a starting repo
        assert (t.seed_dir).glob("*.py")         # with python files
        assert t.spec.prompt.strip()             # and a real prompt


def test_bundled_task_dir_is_in_package():
    assert BUNDLED_TASKS_DIR.name == "tasks"
    assert BUNDLED_TASKS_DIR.is_dir()


def test_run_suite_aggregates_across_tasks():
    report = run_suite("claude-code", command_runner=_ok_runner)
    assert len(report.results) >= 3
    assert all(r.score is not None for r in report.results)
    # Every task ran the Read/Edit/pytest trace -> a real composite per task.
    assert report.composite is not None
    means = report.mode_means()
    # Missing-verification is applicable (a mutation + a test run happened).
    from evalharness.core.findings import FailureMode
    assert means[FailureMode.MISSING_VERIFICATION] is not None


def test_run_suite_captures_task_errors():
    report = run_suite("claude-code", command_runner=_boom_runner)
    assert report.results  # tasks were attempted
    assert all(r.error is not None and r.score is None for r in report.results)
    # With every task errored, there is no composite to report.
    assert report.composite is None


def test_format_report_has_aggregate_line():
    report = run_suite("claude-code", command_runner=_ok_runner)
    text = format_report(report)
    assert "AGGREGATE" in text
    assert "add-power" in text
