"""Schema-versioned JSON report (CI/automation consumable).

Two levels of detail:
  * ``session.scores`` — the 8-mode vector + composite (per-session summary);
  * ``tool_calls[]`` — every call with the findings that target it (per-call detail).

Field ordering is stable so report diffs in CI are meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalharness import REPORT_SCHEMA_VERSION
from evalharness.core.findings import FailureMode, Finding
from evalharness.core.model import NormalizedSession, ToolCallEvent
from evalharness.scoring.engine import SessionScore

# Canonical mode ordering (M1..M8) for stable output.
_MODE_ORDER = [
    FailureMode.WRONG_TOOL,
    FailureMode.WRONG_ARGS,
    FailureMode.HALLUCINATED,
    FailureMode.IGNORED_OUTPUT,
    FailureMode.REDUNDANT,
    FailureMode.MISSING_VERIFICATION,
    FailureMode.PREMATURE_STOP,
    FailureMode.UNSAFE_CALL,
]


def _finding_dict(f: Finding) -> dict[str, Any]:
    return {
        "mode": f.mode.value,
        "verdict": f.verdict,
        "severity": round(f.severity, 4),
        "confidence": round(f.confidence, 4),
        "rationale": f.rationale,
        "target_call_id": f.target_call_id,
        "target_seq": f.target_seq,
        "evidence": [
            {"seq": e.seq, "call_id": e.call_id, "note": e.note} for e in f.evidence
        ],
        "llm_used": f.llm_used,
        "detector_version": f.detector_version,
    }


def _call_dict(call: ToolCallEvent, findings: list[Finding]) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "seq": call.seq,
        "turn": call.turn,
        "tool_name": call.tool_name,
        "raw_tool_name": call.raw_tool_name,
        "capability": call.capability.value,
        "arguments": call.arguments,
        "result": (
            {
                "status": call.result.status,
                "is_error": call.result.is_error,
                "error_class": call.result.error_class,
                "exit_code": call.result.exit_code,
            }
            if call.result
            else None
        ),
        "findings": [_finding_dict(f) for f in findings],
    }


def report_dict(session: NormalizedSession, score: SessionScore) -> dict[str, Any]:
    findings_by_call: dict[str | None, list[Finding]] = {}
    session_level: list[Finding] = []
    for f in score.all_findings:
        if f.target_call_id is None:
            session_level.append(f)
        else:
            findings_by_call.setdefault(f.target_call_id, []).append(f)

    scores_block: dict[str, Any] = {}
    for mode in _MODE_ORDER:
        ms = score.mode_scores.get(mode)
        if ms is None:
            scores_block[mode.value] = {"score": None, "applicable": False, "evaluated": False}
        else:
            scores_block[mode.value] = {
                "score": ms.score,
                "confidence": round(ms.confidence, 4),
                "applicable": ms.applicable,
                "evaluated": True,
                "n_opportunities": ms.n_opportunities,
                "n_findings": ms.n_findings,
            }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "session": {
            "session_id": session.session_id,
            "adapter": session.adapter,
            "task_id": session.task.task_id,
            "stop_reason": session.stop_reason,
            "n_tool_calls": len(session.tool_calls),
            "composite": score.composite,
            "scores": scores_block,
        },
        "tool_calls": [
            _call_dict(c, findings_by_call.get(c.call_id, [])) for c in session.tool_calls
        ],
        "session_findings": [_finding_dict(f) for f in session_level],
    }


def write_json_report(
    session: NormalizedSession, score: SessionScore, path: str | Path, *, indent: int = 2
) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report_dict(session, score), indent=indent, sort_keys=False))
    return path
