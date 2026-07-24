"""M6: benchmark validation over BFCL-derived sessions + agreement metrics."""

from __future__ import annotations

from pathlib import Path

from agent_eval_harness.adapters.benchmarks.bfcl import (
    load_bfcl,
    session_correct,
    session_hallucinated,
    session_wrong_args,
)
from agent_eval_harness.detectors import HallucinatedCallDetector, WrongArgsDetector
from agent_eval_harness.detectors.base import DetectorContext
from agent_eval_harness.detectors.judge import JudgeVerdict, StubJudge
from agent_eval_harness.eval.benchmark import (
    bfcl_report,
    m2_wrong_args_metrics,
    m3_hallucination_metrics,
    relevance_kappa,
)
from agent_eval_harness.eval.metrics import binary_metrics, cohen_kappa

BFCL = Path(__file__).parent / "fixtures" / "benchmarks" / "bfcl"
_CTX = DetectorContext()


# --- pure metrics -----------------------------------------------------------------

def test_binary_metrics_confusion():
    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5 and m.f1 == 0.5


def test_cohen_kappa_perfect_and_chance():
    perfect = cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert perfect.kappa == 1.0
    # rater B constant -> agreement is pure chance -> kappa 0
    chance = cohen_kappa(["a", "b", "a", "b"], ["a", "a", "a", "a"])
    assert abs(chance.kappa) < 1e-9


# --- BFCL loading + session arms --------------------------------------------------

def test_load_bfcl_joins_gold():
    cases = load_bfcl(BFCL / "BFCL_v3_simple.json")
    assert cases
    assert cases[0].functions and cases[0].gold_calls
    assert cases[0].prompt


def test_correct_arm_is_clean():
    m3, m2 = HallucinatedCallDetector(), WrongArgsDetector()
    for case in load_bfcl(BFCL / "BFCL_v3_simple.json"):
        s = session_correct(case)
        assert not m3.evaluate(s, _CTX).failing, f"{case.case_id} false hallucination"
        # simple_307's gold data has a type error; skip that known BFCL quirk
        if case.case_id != "simple_307":
            assert not m2.evaluate(s, _CTX).failing, f"{case.case_id} false wrong-args"


def test_hallucinated_arm_fires_m3():
    m3 = HallucinatedCallDetector()
    cases = load_bfcl(BFCL / "BFCL_v3_multiple.json")
    for case in cases:
        s = session_hallucinated(case)
        assert m3.evaluate(s, _CTX).failing, f"{case.case_id} missed hallucination"


def test_wrong_args_arm_fires_m2():
    m2 = WrongArgsDetector()
    fired = 0
    for case in load_bfcl(BFCL / "BFCL_v3_simple.json"):
        s = session_wrong_args(case)
        if s is None:
            continue
        assert m2.evaluate(s, _CTX).failing, f"{case.case_id} missed wrong-args"
        fired += 1
    assert fired > 0


# --- aggregate metrics ------------------------------------------------------------

def test_m3_metrics_high_recall():
    cases = load_bfcl(BFCL / "BFCL_v3_multiple.json")
    m = m3_hallucination_metrics(cases)
    assert m.recall == 1.0
    assert m.precision >= 0.9


def test_m2_metrics_high_recall():
    cases = load_bfcl(BFCL / "BFCL_v3_simple.json")
    m = m2_wrong_args_metrics(cases)
    assert m.recall == 1.0


# --- judge relevance kappa --------------------------------------------------------

def test_relevance_kappa_with_perfect_judge():
    relevant = load_bfcl(BFCL / "BFCL_v3_simple.json")
    irrelevant = load_bfcl(BFCL / "BFCL_v3_irrelevance.json")
    irr_prompts = {c.prompt for c in irrelevant}

    def policy(req):
        # a "perfect" judge: fail (irrelevant) iff the prompt is an irrelevance case
        is_irrelevant = any(p and p in req.user for p in irr_prompts)
        return JudgeVerdict("fail" if is_irrelevant else "pass", 0.9, "stub")

    judge = StubJudge(policy=policy)
    kappa, skipped = relevance_kappa(relevant, irrelevant, judge)
    assert skipped == 0
    assert kappa.kappa == 1.0
    assert kappa.n == len(relevant) + len(irrelevant)


# --- top-level report -------------------------------------------------------------

def test_bfcl_report_shape():
    report = bfcl_report(BFCL)
    assert report["dataset"] == "bfcl"
    assert set(report["detectors"]) == {"hallucinated", "wrong_args"}
    assert report["detectors"]["hallucinated"]["recall"] == 1.0
    assert report["judge_relevance_kappa"] is None  # no judge configured
