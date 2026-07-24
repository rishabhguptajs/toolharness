"""M5: parser tests over *captured real* CLI output.

Fixtures under tests/fixtures/real/ are verbatim captures from real runs of each
CLI on the sample repo task ("read README, add a CHANGELOG entry, run pytest").
Each test asserts the adapter auto-selects, parses a scoring-ready session, and
links every result back to its call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.adapters import default_registry
from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.core.capability import CanonicalCapability
from agent_eval_harness.core.model import NormalizedSession

REAL = Path(__file__).parent / "fixtures" / "real"


def _parse(rel: str, adapter: str | None = None) -> NormalizedSession:
    aux = {"adapter": adapter} if adapter else {}
    return default_registry.parse(RunSource(kind="auto", path=REAL / rel, aux=aux))


def _selected(rel: str) -> str:
    return default_registry.select(RunSource(kind="auto", path=REAL / rel)).name


def _sniff_scores(rel: str) -> dict[str, float]:
    src = RunSource(kind="auto", path=REAL / rel)
    return {a.name: a.sniff(src) for a in default_registry._adapters.values()}


# --- auto-selection ---------------------------------------------------------------

@pytest.mark.parametrize("rel,expected", [
    ("claude/stream.jsonl", "claude-code"),
    ("claude/otel.log", "claude-code"),
    ("cursor/stream.jsonl", "cursor"),
    ("codex/stream.jsonl", "codex"),
    ("codex/session.jsonl", "codex"),
])
def test_sniff_selects_right_adapter(rel, expected):
    assert _selected(rel) == expected
    # the generic adapter must not fire on real (JSONL / OTEL) captures
    assert _sniff_scores(rel).get("generic", 0.0) == 0.0


def test_adapter_override_beats_sniff():
    # force the "wrong" adapter; select must honor the explicit hint
    src = RunSource(kind="auto", path=REAL / "cursor/stream.jsonl", aux={"adapter": "codex"})
    assert default_registry.select(src).name == "codex"


# --- result linking (invariant across all adapters) -------------------------------

@pytest.mark.parametrize("rel", [
    "claude/stream.jsonl", "claude/otel.log", "cursor/stream.jsonl",
    "codex/stream.jsonl", "codex/session.jsonl",
])
def test_every_call_links_its_result(rel):
    s = _parse(rel)
    assert s.tool_calls, "expected at least one tool call"
    for call in s.tool_calls:
        assert call.result is not None, f"{call.call_id} has no linked result"
        assert call.result.call_id == call.call_id
        assert call.raw_event_ref is not None  # audit trail back to the source


# --- Claude Code (SDK) ------------------------------------------------------------

def test_claude_sdk_full_fidelity():
    s = _parse("claude/stream.jsonl")
    assert s.adapter == "claude-code"
    assert s.metadata["mode"] == "sdk"
    assert s.stop_reason == "completed"
    # init advertised the full tool registry -> M3 has ground truth
    assert len(s.available_tools) == 30
    assert {"Read", "Write", "Bash", "Edit"} <= s.available_tool_names
    caps = [c.capability for c in s.tool_calls]
    assert caps[0] == CanonicalCapability.FILE_READ
    assert CanonicalCapability.FILE_WRITE in caps
    assert CanonicalCapability.TEST_RUN in caps  # `python3 -m pytest` refined
    # the failing first pytest run is captured with its error signal
    bash_fail = [c for c in s.tool_calls if c.result and c.result.is_error]
    assert bash_fail, "expected the missing-pytest failure to be recorded"


def test_claude_sdk_preceding_reasoning_backfilled():
    s = _parse("claude/stream.jsonl")
    venv_calls = [c for c in s.tool_calls if c.command and ".venv" in (c.command or "")
                  or (c.command and "venv" in c.command)]
    assert venv_calls
    assert any(c.preceding_reasoning and "venv" in c.preceding_reasoning.lower()
               for c in venv_calls)


# --- Claude Code (OTEL, degraded) -------------------------------------------------

def test_claude_otel_degraded():
    s = _parse("claude/otel.log")
    assert s.adapter == "claude-code"
    assert s.metadata.get("degraded") is True
    assert s.available_tools == []            # OTEL carries no registry
    assert s.tool_calls                       # but tool decisions + results survive
    for c in s.tool_calls:
        assert c.arguments == {}              # args are absent in OTEL (documented)


# --- Cursor -----------------------------------------------------------------------

def test_cursor_stream():
    s = _parse("cursor/stream.jsonl")
    assert s.adapter == "cursor"
    assert s.stop_reason == "completed"
    caps = [c.capability for c in s.tool_calls]
    assert CanonicalCapability.FILE_READ in caps
    assert CanonicalCapability.FILE_EDIT in caps
    assert CanonicalCapability.TEST_RUN in caps
    # a shell failure carries its exit code
    failed = [c for c in s.tool_calls if c.result and c.result.is_error]
    assert any(c.result.exit_code == 1 for c in failed)


# --- Codex (both shapes) ----------------------------------------------------------

def test_codex_exec_stream():
    s = _parse("codex/stream.jsonl")
    assert s.adapter == "codex"
    assert s.metadata["mode"] == "exec-stream"
    assert s.stop_reason == "completed"
    caps = [c.capability for c in s.tool_calls]
    assert CanonicalCapability.SHELL_EXEC in caps
    assert any(c.capability in (CanonicalCapability.FILE_WRITE, CanonicalCapability.FILE_EDIT)
               for c in s.tool_calls)


def test_codex_session_rollout():
    s = _parse("codex/session.jsonl")
    assert s.adapter == "codex"
    assert s.metadata["mode"] == "rollout"
    # apply_patch is captured as a file edit with the patch text preserved
    patch = [c for c in s.tool_calls if c.raw_tool_name == "apply_patch"]
    assert patch and patch[0].capability == CanonicalCapability.FILE_EDIT
    assert "input" in patch[0].arguments


def test_codex_shapes_agree_on_tool_count():
    # the two on-disk representations of the same run should see the same calls
    a = _parse("codex/stream.jsonl")
    b = _parse("codex/session.jsonl")
    assert len(a.tool_calls) == len(b.tool_calls)


# --- end-to-end scoring -----------------------------------------------------------

def test_real_capture_scores_end_to_end(tmp_path):
    from agent_eval_harness.detectors import ALL_DETECTORS
    from agent_eval_harness.detectors.base import DetectorContext
    from agent_eval_harness.report.html_report import write_html_report
    from agent_eval_harness.report.json_report import write_json_report
    from agent_eval_harness.scoring.engine import evaluate_session

    s = _parse("cursor/stream.jsonl")
    score = evaluate_session(s, ALL_DETECTORS, DetectorContext(judge=None))
    assert score.composite is not None
    json_path = tmp_path / "r.json"
    html_path = tmp_path / "r.html"
    write_json_report(s, score, json_path)
    write_html_report(s, score, html_path)
    assert json_path.exists() and html_path.stat().st_size > 0
