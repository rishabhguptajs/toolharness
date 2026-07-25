"""M7 CLI tests: TaskSpec YAML loading, `run --task` reference attachment, and the
composite / per-mode CI gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.core.capability import CanonicalCapability
from evalharness.core.taskspec import TaskSpec
from evalharness.runner.cli import build_parser, evaluate_path, main

FIXTURES = Path(__file__).parent / "fixtures"


# --- TaskSpec.from_yaml --------------------------------------------------------

def test_from_yaml_loads_reference_fields(tmp_path: Path):
    spec = tmp_path / "task.yaml"
    spec.write_text(
        "task_id: demo\n"
        "prompt: do the thing\n"
        "repo:\n  path: /some/repo\n"
        "expected_capabilities: [FILE_READ, TEST_RUN]\n"
        "subgoals:\n  - id: a\n    description: first\n"
        "required_verification: [TEST_RUN]\n"
    )
    task = TaskSpec.from_yaml(spec)
    assert task.task_id == "demo"
    assert task.prompt == "do the thing"
    assert task.repo_path == "/some/repo"
    assert task.expected_capabilities == [
        CanonicalCapability.FILE_READ, CanonicalCapability.TEST_RUN
    ]
    assert task.has_reference is True
    assert task.required_verification == [CanonicalCapability.TEST_RUN]


def test_from_yaml_rejects_non_mapping(tmp_path: Path):
    spec = tmp_path / "bad.yaml"
    spec.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        TaskSpec.from_yaml(spec)


def test_run_task_attaches_reference(tmp_path: Path):
    spec = tmp_path / "task.yaml"
    spec.write_text("task_id: attached\nexpected_capabilities: [FILE_READ]\n")
    _, session = evaluate_path(FIXTURES / "clean_pass.json", task=spec)
    assert session.task.task_id == "attached"
    assert session.task.has_reference is True


# --- CI gates ------------------------------------------------------------------

def test_composite_gate_passes_on_clean(capsys):
    rc = main(["run", str(FIXTURES / "clean_pass.json"), "--fail-under", "50"])
    assert rc == 0


def test_composite_gate_fails_when_below(capsys):
    rc = main(["run", str(FIXTURES / "m8_unsafe_call_fail.json"), "--fail-under", "95"])
    assert rc == 1
    assert "composite" in capsys.readouterr().err


def test_per_mode_gate_fails_on_unsafe(capsys):
    rc = main([
        "run", str(FIXTURES / "m8_unsafe_call_fail.json"),
        "--fail-under-mode", "UNSAFE_CALL=90",
    ])
    assert rc == 1
    assert "unsafe_call" in capsys.readouterr().err


def test_per_mode_gate_accepts_enum_value_form(capsys):
    # lower-case value form resolves the same as the enum name.
    rc = main([
        "run", str(FIXTURES / "m8_unsafe_call_fail.json"),
        "--fail-under-mode", "unsafe_call=90",
    ])
    assert rc == 1


def test_per_mode_gate_skips_not_applicable_modes(capsys):
    # clean_pass has no destructive candidates -> UNSAFE_CALL is n/a -> no gate breach.
    rc = main([
        "run", str(FIXTURES / "clean_pass.json"),
        "--fail-under-mode", "UNSAFE_CALL=100",
    ])
    assert rc == 0


def test_bad_mode_name_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "run", "x.json", "--fail-under-mode", "NOT_A_MODE=50",
        ])


def test_missing_judge_key_reports_cleanly(capsys, monkeypatch):
    """BYO-key: an unset credential must be a readable message, not a traceback."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rc = main(["run", str(FIXTURES / "clean_pass.json"), "--judge", "groq"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "GROQ_API_KEY is not set" in err
    assert "your own API key" in err
