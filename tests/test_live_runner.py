"""M7 runner tests: invocation-profile command construction, the safe-by-default
sandbox, and the end-to-end live path driven by a fake CLI (no real binaries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.core.capability import CanonicalCapability
from agent_eval_harness.core.taskspec import TaskSpec
from agent_eval_harness.runner.live import (
    PROFILES,
    InvocationProfile,
    prepare_workdir,
    profile_for,
    run_live,
)

# A minimal Claude-Code SDK stream-json trace the fake CLI will "emit" on stdout.
_FAKE_CLAUDE_TRACE = "\n".join(json.dumps(r) for r in [
    {"type": "system", "subtype": "init", "session_id": "live-1",
     "tools": ["Read", "Edit", "Bash"], "cwd": "/w"},
    {"type": "assistant", "message": {"id": "msg_1", "content": [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "calc.py"}},
    ]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "def add(...)", "is_error": False},
    ]}},
    {"type": "result", "subtype": "success", "session_id": "live-1",
     "num_turns": 1, "result": "done"},
])


def _fake_runner(stdout: str, returncode: int = 0):
    calls: list[list[str]] = []

    def run(command, cwd, timeout):
        calls.append(list(command))
        # The fake CLI ignores cwd/timeout and just replays a canned trace.
        return returncode, stdout, "", False

    run.calls = calls  # type: ignore[attr-defined]
    return run


# --- profiles -----------------------------------------------------------------

def test_profiles_cover_shipped_adapters():
    assert set(PROFILES) == {"claude-code", "cursor", "codex"}


def test_build_command_places_prompt_as_single_arg():
    cmd = PROFILES["claude-code"].build_command("add a power() function")
    assert cmd == ["claude", "-p", "add a power() function",
                   "--output-format", "stream-json", "--verbose"]
    # The prompt is exactly one argv element (never shell-interpolated).
    assert cmd.count("add a power() function") == 1


def test_codex_profile_has_no_post_args():
    assert PROFILES["codex"].build_command("x") == ["codex", "exec", "--json", "x"]


def test_profile_for_unknown_raises():
    with pytest.raises(KeyError):
        profile_for("gemini")


# --- sandbox ------------------------------------------------------------------

def test_prepare_workdir_sandbox_copies_and_isolates(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "calc.py").write_text("x = 1\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "big").write_text("junk")

    workdir, created = prepare_workdir(repo, sandbox=True)
    try:
        assert created is True
        assert workdir != repo
        assert (workdir / "calc.py").read_text() == "x = 1\n"
        assert (workdir / ".git" / "HEAD").exists()  # git history preserved
        assert not (workdir / ".venv").exists()      # heavy dir skipped
        # Mutating the sandbox must not touch the real repo.
        (workdir / "calc.py").write_text("mutated\n")
        assert (repo / "calc.py").read_text() == "x = 1\n"
    finally:
        import shutil
        shutil.rmtree(workdir.parent, ignore_errors=True)


def test_prepare_workdir_in_place_returns_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    workdir, created = prepare_workdir(repo, sandbox=False)
    assert created is False
    assert workdir == repo.resolve()


def test_prepare_workdir_missing_repo_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        prepare_workdir(tmp_path / "nope", sandbox=True)


# --- end-to-end live path (fake CLI) -----------------------------------------

def test_run_live_parses_and_attaches_task(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b): return a + b\n")
    task = TaskSpec.from_dict({
        "task_id": "t", "prompt": "read calc.py",
        "expected_capabilities": ["FILE_READ"],
    })
    runner = _fake_runner(_FAKE_CLAUDE_TRACE)

    result = run_live("claude-code", task, repo=repo, command_runner=runner)

    # The canned trace was captured, parsed, and the task attached.
    assert result.session.task.task_id == "t"
    assert result.session.adapter == "claude-code"
    caps = [c.capability for c in result.session.tool_calls]
    assert CanonicalCapability.FILE_READ in caps
    assert result.returncode == 0
    assert result.sandboxed is True
    assert result.trace_path.exists()
    # The sandbox got cleaned up (default keep_workdir=False).
    assert not result.workdir.exists()
    # The fake CLI saw a claude command with the prompt as one arg.
    assert runner.calls[0][:3] == ["claude", "-p", "read calc.py"]  # type: ignore[attr-defined]


def test_run_live_keeps_workdir_when_requested(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = TaskSpec.from_dict({"task_id": "t", "prompt": "hi"})
    result = run_live(
        "claude-code", task, repo=repo,
        command_runner=_fake_runner(_FAKE_CLAUDE_TRACE), keep_workdir=True,
    )
    try:
        assert result.workdir.exists()
    finally:
        import shutil
        shutil.rmtree(result.workdir.parent, ignore_errors=True)


def test_run_live_requires_repo(tmp_path: Path):
    task = TaskSpec.from_dict({"task_id": "t", "prompt": "hi"})  # no repo.path
    with pytest.raises(ValueError, match="No task repo"):
        run_live("claude-code", task, command_runner=_fake_runner(""))


def test_run_live_rejects_empty_prompt(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = TaskSpec.from_dict({"task_id": "t", "prompt": "   "})
    with pytest.raises(ValueError, match="empty prompt"):
        run_live("claude-code", task, repo=repo, command_runner=_fake_runner(""))


def test_run_live_writes_trace_to_requested_path(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    trace = tmp_path / "captured.jsonl"
    task = TaskSpec.from_dict({"task_id": "t", "prompt": "hi"})
    result = run_live(
        InvocationProfile("claude-code", "claude-code", "claude",
                          ("-p",), ("--output-format", "stream-json", "--verbose")),
        task, repo=repo, trace_path=trace,
        command_runner=_fake_runner(_FAKE_CLAUDE_TRACE),
    )
    assert result.trace_path == trace
    assert trace.read_text() == _FAKE_CLAUDE_TRACE
