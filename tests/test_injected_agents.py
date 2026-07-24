"""M2 integration tests: the injected-failure agents, end to end.

Each agent runs through the real pipeline (generic adapter -> deterministic
detectors -> scoring). We assert three things:

  * clean runs fire nothing (precision baseline);
  * each deterministic injection is caught with no cross-fire (recall);
  * the agents actually *emit* all eight modes — the three judgment-heavy ones
    (M1/M4/M7) are structurally present now and marked xfail until their M3
    detectors exist, so those cases flip to passing automatically in M3.
"""

from __future__ import annotations

import pytest

from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.test_agents import (
    DETERMINISTIC_MODES,
    JUDGE_PENDING_MODES,
    all_agents,
    fired_modes,
    precision_recall,
    run_controlled_set,
    score_agent_run,
)

AGENTS = {a.name: a for a in all_agents()}


def _agent_for(mode: FailureMode):
    for agent in all_agents():
        if mode in agent.injectable:
            return agent
    raise AssertionError(f"no agent injects {mode}")


# --- clean baselines --------------------------------------------------------------


@pytest.mark.parametrize("agent", all_agents(), ids=lambda a: a.name)
def test_clean_run_fires_nothing(agent):
    score = score_agent_run(agent, None)
    assert score.composite == 100
    assert fired_modes(score) == set()


# --- deterministic injections (caught today) --------------------------------------


@pytest.mark.parametrize("mode", sorted(DETERMINISTIC_MODES, key=lambda m: m.value))
def test_deterministic_injection_is_caught_without_crossfire(mode):
    agent = _agent_for(mode)
    score = score_agent_run(agent, mode)
    fired = fired_modes(score)
    assert mode in fired, f"{mode.value} injection was not caught"
    assert fired == {mode}, f"cross-fire alongside {mode.value}: {fired}"
    assert score.mode_scores[mode].score < 100


# --- judgment-heavy injections (detectors land in M3) -----------------------------


@pytest.mark.parametrize("mode", sorted(JUDGE_PENDING_MODES, key=lambda m: m.value))
@pytest.mark.xfail(reason="detector lands in M3 (LLM judge layer)", strict=False)
def test_judge_pending_injection_is_caught(mode):
    agent = _agent_for(mode)
    score = score_agent_run(agent, mode)
    assert mode in fired_modes(score)


@pytest.mark.parametrize("mode", sorted(JUDGE_PENDING_MODES, key=lambda m: m.value))
def test_judge_pending_injection_does_not_false_fire_deterministically(mode):
    """Until M3 they must stay silent, not trip a deterministic detector."""
    agent = _agent_for(mode)
    score = score_agent_run(agent, mode)
    assert fired_modes(score) & DETERMINISTIC_MODES == set()


# --- the agents emit every one of the eight modes ---------------------------------


def test_agents_cover_all_eight_modes():
    covered = set().union(*(a.injectable for a in all_agents()))
    assert covered == set(FailureMode), set(FailureMode) - covered


def test_wrong_tool_trace_uses_shell_cat_not_read_file():
    trace = _agent_for(FailureMode.WRONG_TOOL).run(FailureMode.WRONG_TOOL)
    tool_names = {"read_file", "grep", "edit_file", "run_command"}
    assert tool_names <= {t["name"] for t in trace["available_tools"]}
    shell_reads = [
        e
        for e in trace["events"]
        if e["type"] == "tool_call"
        and e["tool_name"] == "run_command"
        and str(e["arguments"].get("command", "")).startswith("cat ")
    ]
    assert shell_reads, "wrong-tool injection should read a file via shell cat"


def test_ignored_output_trace_has_failing_test_then_success_claim():
    trace = _agent_for(FailureMode.IGNORED_OUTPUT).run(FailureMode.IGNORED_OUTPUT)
    had_failing_test = any(
        e["type"] == "tool_result" and e.get("is_error") and "fail" in e.get("content", "")
        for e in trace["events"]
    )
    stop = trace["events"][-1]
    assert had_failing_test, "ignored-output injection needs a genuine failing result"
    assert stop["type"] == "agent_stop"
    assert trace["stop_reason"] == "completed"


def test_premature_stop_trace_stops_before_any_edit():
    trace = _agent_for(FailureMode.PREMATURE_STOP).run(FailureMode.PREMATURE_STOP)
    edits = [
        e
        for e in trace["events"]
        if e["type"] == "tool_call" and e["tool_name"] == "edit_file"
    ]
    assert not edits, "premature-stop injection must stop before editing"
    assert trace["events"][-1]["type"] == "agent_stop"


# --- controlled-set metrics -------------------------------------------------------


def test_controlled_set_precision_recall_is_perfect():
    metrics = precision_recall(run_controlled_set())
    assert metrics.recall == 1.0, metrics.cross_fires
    assert metrics.precision == 1.0, metrics.cross_fires
    assert metrics.true_positives == len(DETERMINISTIC_MODES)
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


# --- reproducibility (cumulative verification) ------------------------------------


def test_scoring_is_reproducible():
    agent = _agent_for(FailureMode.UNSAFE_CALL)
    first = score_agent_run(agent, FailureMode.UNSAFE_CALL)
    second = score_agent_run(agent, FailureMode.UNSAFE_CALL)
    assert first.composite == second.composite
    assert {m: s.score for m, s in first.mode_scores.items()} == {
        m: s.score for m, s in second.mode_scores.items()
    }
