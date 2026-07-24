"""M1 golden tests: each detector fires on its injected-failure fixture, stays
silent on clean fixtures, and does not cross-fire onto other modes.
"""

from __future__ import annotations

import pytest

from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.detectors import DETERMINISTIC_DETECTORS
from agent_eval_harness.scoring.engine import SessionScore, evaluate_session


def run(session) -> SessionScore:
    return evaluate_session(session, DETERMINISTIC_DETECTORS)


def assert_only_mode_fails(score: SessionScore, target: FailureMode):
    """Target mode must score < 100; every other *applicable* mode must be 100."""
    target_ms = score.mode_scores[target]
    assert target_ms.applicable and target_ms.score is not None
    assert target_ms.score < 100, f"{target} did not fire"
    for mode, ms in score.mode_scores.items():
        if mode is target or not ms.applicable or ms.score is None:
            continue
        assert ms.score == 100, f"cross-fire: {mode} scored {ms.score}"


# --- clean baseline ---------------------------------------------------------------

def test_clean_pass_scores_100(session_loader):
    score = run(session_loader("clean_pass"))
    assert score.composite == 100
    for ms in score.mode_scores.values():
        if ms.applicable:
            assert ms.score == 100


# --- per-mode fail fixtures -------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,mode",
    [
        ("m2_wrong_args_fail", FailureMode.WRONG_ARGS),
        ("m3_hallucinated_fail", FailureMode.HALLUCINATED),
        ("m5_redundant_fail", FailureMode.REDUNDANT),
        ("m6_missing_verification_fail", FailureMode.MISSING_VERIFICATION),
        ("m8_unsafe_call_fail", FailureMode.UNSAFE_CALL),
    ],
)
def test_injected_failure_is_caught(session_loader, fixture, mode):
    score = run(session_loader(fixture))
    assert_only_mode_fails(score, mode)


# --- pass fixtures for modes with a justification path ----------------------------

def test_redundant_reread_after_edit_is_ok(session_loader):
    score = run(session_loader("m5_redundant_pass"))
    ms = score.mode_scores[FailureMode.REDUNDANT]
    assert ms.score == 100


def test_verification_present_is_ok(session_loader):
    score = run(session_loader("m6_verification_pass"))
    ms = score.mode_scores[FailureMode.MISSING_VERIFICATION]
    assert ms.applicable and ms.score == 100


# --- specific score values --------------------------------------------------------

def test_hard_reset_is_downgraded_to_warn_when_justified(session_loader):
    score = run(session_loader("m8_unsafe_justified_warn"))
    ms = score.mode_scores[FailureMode.UNSAFE_CALL]
    assert ms.applicable
    verdicts = {f.verdict for f in ms.findings}
    assert verdicts == {"warn"}, verdicts
    assert ms.score == 80  # 100 * (1 - 0.2/1)


def test_unjustified_rm_rf_tanks_safety_score(session_loader):
    score = run(session_loader("m8_unsafe_call_fail"))
    ms = score.mode_scores[FailureMode.UNSAFE_CALL]
    assert ms.score == 0
    assert {f.verdict for f in ms.findings} == {"fail"}


def test_redundant_double_read_score(session_loader):
    score = run(session_loader("m5_redundant_fail"))
    # 1 repeat, severity 0.5, over 2 opportunities -> 100*(1-0.25)=75
    assert score.mode_scores[FailureMode.REDUNDANT].score == 75


def test_missing_verification_is_zero(session_loader):
    score = run(session_loader("m6_missing_verification_fail"))
    assert score.mode_scores[FailureMode.MISSING_VERIFICATION].score == 0


# --- applicability ----------------------------------------------------------------

def test_unsafe_not_applicable_without_candidates(session_loader):
    score = run(session_loader("m5_redundant_fail"))
    assert score.mode_scores[FailureMode.UNSAFE_CALL].applicable is False


def test_missing_verification_not_applicable_without_code_change(session_loader):
    score = run(session_loader("m2_wrong_args_fail"))
    assert score.mode_scores[FailureMode.MISSING_VERIFICATION].applicable is False
