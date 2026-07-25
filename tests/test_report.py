"""JSON report structure + reproducibility (same input -> identical report)."""

from __future__ import annotations

import json

from evalharness import REPORT_SCHEMA_VERSION
from evalharness.detectors import DETERMINISTIC_DETECTORS
from evalharness.report.json_report import report_dict
from evalharness.scoring.engine import evaluate_session


def _report(session):
    score = evaluate_session(session, DETERMINISTIC_DETECTORS)
    return report_dict(session, score)


def test_report_shape(session_loader):
    rep = _report(session_loader("clean_pass"))
    assert rep["schema_version"] == REPORT_SCHEMA_VERSION
    assert rep["session"]["composite"] == 100
    # all 8 modes are represented in the summary, even the un-evaluated ones
    assert len(rep["session"]["scores"]) == 8
    assert len(rep["tool_calls"]) == 3


def test_unevaluated_modes_flagged(session_loader):
    rep = _report(session_loader("clean_pass"))
    # M1/M4/M7 detectors are not in the deterministic set yet
    assert rep["session"]["scores"]["wrong_tool"]["evaluated"] is False
    assert rep["session"]["scores"]["wrong_args"]["evaluated"] is True


def test_per_call_findings_attached(session_loader):
    rep = _report(session_loader("m8_unsafe_call_fail"))
    calls = rep["tool_calls"]
    unsafe = [c for c in calls if any(f["mode"] == "unsafe_call" for f in c["findings"])]
    assert len(unsafe) == 1
    assert unsafe[0]["findings"][0]["verdict"] == "fail"


def test_report_is_reproducible(session_loader):
    a = json.dumps(_report(session_loader("m5_redundant_fail")), sort_keys=True)
    b = json.dumps(_report(session_loader("m5_redundant_fail")), sort_keys=True)
    assert a == b
