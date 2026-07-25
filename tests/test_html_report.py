"""M4: the self-contained HTML dashboard."""

from __future__ import annotations

import re

from toolharness.adapters.base import RunSource
from toolharness.adapters.generic import GenericToolTraceAdapter
from toolharness.core.findings import FailureMode
from toolharness.detectors import ALL_DETECTORS
from toolharness.report.html_report import (
    build_view,
    render_dashboard,
    write_html_dashboard,
    write_html_report,
)
from toolharness.scoring.engine import evaluate_session
from toolharness.test_agents.agents import BugfixAgent, SearchRefactorAgent

_ADAPTER = GenericToolTraceAdapter()


def _score(agent, inject):
    session = _ADAPTER.parse(RunSource(kind="generic", data=agent.run(inject)))
    return session, evaluate_session(session, ALL_DETECTORS)


def _score_trace(trace):
    session = _ADAPTER.parse(RunSource(kind="generic", data=trace))
    return session, evaluate_session(session, ALL_DETECTORS)


# --- self-containment: the whole point of the deliverable -------------------------


def test_dashboard_is_self_contained():
    html = render_dashboard([_score(BugfixAgent(), FailureMode.IGNORED_OUTPUT)])
    # No external stylesheets, scripts, images, fonts, or fetches.
    assert not re.search(r'(?:src|href)\s*=\s*["\']\s*(?:https?:)?//', html)
    assert "<style>" in html and "<script>" in html
    assert "cdn" not in html.lower()


def test_dashboard_renders_scores_timeline_and_findings():
    session, score = _score(BugfixAgent(), FailureMode.IGNORED_OUTPUT)
    html = render_dashboard([(session, score)])
    assert session.session_id in html
    assert "Timeline" in html
    assert "Ignored output" in html          # mode label
    assert "run_command" in html             # timeline tool
    assert "evidence:" in html               # drill-down evidence trail
    assert "<polygon" in html                # radar rendered


# --- compare view -----------------------------------------------------------------


def test_compare_view_shows_all_sessions():
    a = _score(BugfixAgent(), FailureMode.REDUNDANT)
    b = _score(SearchRefactorAgent(), FailureMode.UNSAFE_CALL)
    html = render_dashboard([a, b])
    assert ">Compare<" in html
    assert a[0].session_id in html and b[0].session_id in html


# --- session-level findings branch (M6/M7 target the session, not a call) ---------


def test_session_level_findings_render():
    session, score = _score(BugfixAgent(), FailureMode.MISSING_VERIFICATION)
    view = build_view(session, score)
    assert view["session_findings"], "expected a session-level finding for M6"
    html = render_dashboard([(session, score)])
    assert "Session-level findings" in html


# --- security: user-controlled strings are escaped --------------------------------


def test_arguments_are_html_escaped():
    trace = {
        "adapter": "generic",
        "task": {"task_id": "t", "prompt": "x"},
        "available_tools": [{"name": "bash", "capability": "SHELL_EXEC",
                             "schema": {"required": ["command"]}}],
        "events": [
            {"type": "tool_call", "call_id": "c1", "tool_name": "bash",
             "arguments": {"command": "echo <script>alert(1)</script>"}},
            {"type": "tool_result", "call_id": "c1", "status": "ok", "content": "hi"},
            {"type": "agent_stop", "text": "done"},
        ],
    }
    html = render_dashboard([_score_trace(trace)])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# --- file writers -----------------------------------------------------------------


def test_write_html_report_and_dashboard(tmp_path):
    single = write_html_report(*(_score(BugfixAgent(), FailureMode.WRONG_ARGS)),
                               tmp_path / "single.html")
    assert single.exists() and single.read_text().startswith("<!doctype html>")

    multi = write_html_dashboard(
        [_score(BugfixAgent(), None), _score(SearchRefactorAgent(), None)],
        tmp_path / "compare.html",
    )
    assert multi.exists() and "Compare" in multi.read_text()
