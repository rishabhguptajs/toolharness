"""Run detectors over benchmark-derived sessions and report agreement (M6).

For the deterministic detectors we build a known-correct and known-wrong session
per case and measure precision/recall/F1 (does the detector fire iff the arm is
wrong). For the judgment call — is a tool call even *warranted*? — we compute
Cohen's kappa between the LLM-judge and BFCL's human-curated relevant/irrelevant
labels. The kappa path runs only when a judge is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalharness.adapters.benchmarks.bfcl import (
    BFCLCase,
    load_bfcl,
    session_correct,
    session_hallucinated,
    session_wrong_args,
)
from evalharness.detectors import HallucinatedCallDetector, WrongArgsDetector
from evalharness.detectors.base import Detector, DetectorContext
from evalharness.detectors.judge import JudgeRequest, safe_ask
from evalharness.eval.metrics import (
    BinaryMetrics,
    KappaResult,
    binary_metrics,
    cohen_kappa,
)

_CTX = DetectorContext(judge=None)


def _fires(detector: Detector, session: Any) -> bool:
    return bool(detector.evaluate(session, _CTX).failing)


def m3_hallucination_metrics(cases: list[BFCLCase]) -> BinaryMetrics:
    """Correct call (label False) vs out-of-registry call (label True) -> M3."""
    detector = HallucinatedCallDetector()
    y_true: list[bool] = []
    y_pred: list[bool] = []
    for case in cases:
        if not case.functions:
            continue
        y_true.append(False)
        y_pred.append(_fires(detector, session_correct(case)))
        y_true.append(True)
        y_pred.append(_fires(detector, session_hallucinated(case)))
    return binary_metrics(y_true, y_pred)


def m2_wrong_args_metrics(cases: list[BFCLCase]) -> BinaryMetrics:
    """Gold args (label False) vs dropped-required-arg (label True) -> M2."""
    detector = WrongArgsDetector()
    y_true: list[bool] = []
    y_pred: list[bool] = []
    for case in cases:
        if not case.gold_calls:
            continue
        y_true.append(False)
        y_pred.append(_fires(detector, session_correct(case)))
        broken = session_wrong_args(case)
        if broken is not None:
            y_true.append(True)
            y_pred.append(_fires(detector, broken))
    return binary_metrics(y_true, y_pred)


# --- judge relevance kappa --------------------------------------------------------

_RELEVANCE_SYSTEM = (
    "You audit an AI agent's tool use. Given a user request and the functions "
    "available to the agent, decide whether ANY of the functions is appropriate to "
    "call to address the request. Respond ONLY with JSON: "
    '{"verdict":"pass"|"fail","confidence":0..1,"rationale":"..."}. '
    "verdict=pass means at least one function is relevant; verdict=fail means none "
    "of them fit and the agent should not call any."
)


def _relevance_request(case: BFCLCase) -> JudgeRequest:
    tools = "\n".join(
        f"- {f.get('name')}: {f.get('description', '')}" for f in case.functions
    )
    user = (
        f"User request:\n{case.prompt}\n\nAvailable functions:\n{tools}\n\n"
        "Is any function relevant?"
    )
    return JudgeRequest(kind="bfcl_relevance", system=_RELEVANCE_SYSTEM, user=user)


def relevance_kappa(
    relevant: list[BFCLCase], irrelevant: list[BFCLCase], judge: Any
) -> tuple[KappaResult, int]:
    """Cohen's kappa between the judge and BFCL gold on relevant/irrelevant cases.

    Gold label is "pass" for answerable (relevant) cases and "fail" for irrelevance
    cases. Returns (kappa, n_skipped) where skipped counts judge errors/abstentions.
    """
    gold: list[str] = []
    pred: list[str] = []
    skipped = 0
    for case, gold_label in [(c, "pass") for c in relevant] + [(c, "fail") for c in irrelevant]:
        verdict = safe_ask(judge, _relevance_request(case))
        if verdict is None or verdict.verdict not in ("pass", "fail"):
            skipped += 1
            continue
        gold.append(gold_label)
        pred.append(verdict.verdict)
    return cohen_kappa(gold, pred), skipped


# --- top-level report -------------------------------------------------------------

_CATEGORY_FILES = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}


def load_bfcl_dir(data_dir: str | Path, limit: int | None = None) -> dict[str, list[BFCLCase]]:
    """Load the categories we validate against from a BFCL data directory."""
    data_dir = Path(data_dir)
    out: dict[str, list[BFCLCase]] = {}
    for category, fname in _CATEGORY_FILES.items():
        path = data_dir / fname
        if path.exists():
            cases = load_bfcl(path)
            out[category] = cases[:limit] if limit else cases
    return out


def bfcl_report(
    data_dir: str | Path, judge: Any = None, limit: int | None = None
) -> dict[str, Any]:
    """Build the full BFCL validation report (per-detector P/R/F1 [+ judge kappa])."""
    cats = load_bfcl_dir(data_dir, limit=limit)
    answerable = cats.get("simple", []) + cats.get("multiple", [])
    all_cases = [c for cs in cats.values() for c in cs]

    report: dict[str, Any] = {
        "dataset": "bfcl",
        "n_cases": {k: len(v) for k, v in cats.items()},
        "detectors": {
            "hallucinated": m3_hallucination_metrics(all_cases).as_dict(),
            "wrong_args": m2_wrong_args_metrics(answerable).as_dict(),
        },
        "judge_relevance_kappa": None,
    }
    if judge is not None and cats.get("irrelevance"):
        kappa, skipped = relevance_kappa(answerable, cats["irrelevance"], judge)
        report["judge_relevance_kappa"] = {**kappa.as_dict(), "skipped": skipped}
    return report


def format_report(report: dict[str, Any]) -> str:
    """Render a benchmark report as a compact text table."""
    lines = [
        f"benchmark: {report['dataset']}  cases={report['n_cases']}",
        f"  {'detector':<16}{'P':>7}{'R':>7}{'F1':>7}{'acc':>7}   (tp/fp/fn/tn)",
    ]
    for name, m in report["detectors"].items():
        lines.append(
            f"  {name:<16}{m['precision']:>7.2f}{m['recall']:>7.2f}"
            f"{m['f1']:>7.2f}{m['accuracy']:>7.2f}   "
            f"({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})"
        )
    k = report.get("judge_relevance_kappa")
    if k:
        lines.append(
            f"  judge relevance kappa={k['kappa']:.3f} "
            f"(n={k['n']}, skipped={k['skipped']})"
        )
    else:
        lines.append("  judge relevance kappa: n/a (no judge configured)")
    return "\n".join(lines)
