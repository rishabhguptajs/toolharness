"""Self-contained HTML dashboard (M4).

Renders a Jinja template into a single file with inline CSS/JS and no external
requests — open it straight from disk. It shows, per session:

  * the 8-mode score vector as a bar list and an SVG radar;
  * a tool-call timeline color-coded by the failure modes each call triggered,
    with click-to-expand drill-down (arguments, result, and every finding's
    rationale + evidence trail);
  * mode/verdict filters over the timeline.

``render_dashboard`` takes one or more (session, score) pairs; with more than one
it prepends a compare panel (score vectors + composites side by side) so two
agents on the same task can be read against each other.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from toolharness.core.findings import FailureMode
from toolharness.core.model import NormalizedSession
from toolharness.report.json_report import report_dict
from toolharness.scoring.engine import SessionScore

_TEMPLATES = Path(__file__).parent / "html"

# M1..M8 order, short code, human label, and a distinct accessible color.
_MODE_META: list[tuple[FailureMode, str, str, str]] = [
    (FailureMode.WRONG_TOOL, "M1", "Wrong tool", "#e06c75"),
    (FailureMode.WRONG_ARGS, "M2", "Wrong args", "#e5c07b"),
    (FailureMode.HALLUCINATED, "M3", "Hallucinated", "#c678dd"),
    (FailureMode.IGNORED_OUTPUT, "M4", "Ignored output", "#56b6c2"),
    (FailureMode.REDUNDANT, "M5", "Redundant", "#98c379"),
    (FailureMode.MISSING_VERIFICATION, "M6", "Missing verification", "#61afef"),
    (FailureMode.PREMATURE_STOP, "M7", "Premature stop", "#d19a66"),
    (FailureMode.UNSAFE_CALL, "M8", "Unsafe call", "#ff5c8a"),
]
_MODE_ORDER = [m for m, _, _, _ in _MODE_META]
_CODE = {m: c for m, c, _, _ in _MODE_META}
_LABEL = {m: lbl for m, _, lbl, _ in _MODE_META}
_COLOR = {m: col for m, _, _, col in _MODE_META}

_VERDICT_COLOR = {"fail": "#e06c75", "warn": "#e5c07b", "pass": "#98c379", "na": "#5c6370"}


def _score_rows(score: SessionScore) -> list[dict[str, Any]]:
    rows = []
    for mode in _MODE_ORDER:
        ms = score.mode_scores.get(mode)
        rows.append(
            {
                "code": _CODE[mode],
                "label": _LABEL[mode],
                "color": _COLOR[mode],
                "score": ms.score if ms else None,
                "applicable": bool(ms and ms.applicable),
                "confidence": round(ms.confidence, 2) if ms and ms.applicable else None,
                "n_findings": ms.n_findings if ms else 0,
            }
        )
    return rows


def _radar(score: SessionScore, *, size: int = 220) -> dict[str, Any]:
    """Precompute an 8-axis radar as static SVG geometry (n/a plotted at 100)."""
    cx = cy = size / 2
    radius = size / 2 - 28
    n = len(_MODE_ORDER)
    axes, points = [], []
    for i, mode in enumerate(_MODE_ORDER):
        angle = -math.pi / 2 + i * (2 * math.pi / n)
        ax = cx + radius * math.cos(angle)
        ay = cy + radius * math.sin(angle)
        ms = score.mode_scores.get(mode)
        val = 100 if not ms or ms.score is None else ms.score
        r = radius * val / 100.0
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append(f"{px:.1f},{py:.1f}")
        axes.append(
            {
                "x": round(ax, 1), "y": round(ay, 1),
                "lx": round(cx + (radius + 14) * math.cos(angle), 1),
                "ly": round(cy + (radius + 14) * math.sin(angle), 1),
                "code": _CODE[mode],
            }
        )
    rings = [round(radius * f / 100.0, 1) for f in (25, 50, 75, 100)]
    return {
        "size": size, "cx": cx, "cy": cy,
        "points": " ".join(points), "axes": axes, "rings": rings,
    }


def _timeline(session: NormalizedSession, report: dict[str, Any]) -> list[dict[str, Any]]:
    findings_by_call: dict[str, list[dict[str, Any]]] = {}
    for call in report["tool_calls"]:
        rows = []
        for f in call["findings"]:
            mode = FailureMode(f["mode"])
            rows.append(
                {
                    "mode_code": _CODE[mode],
                    "mode_label": _LABEL[mode],
                    "color": _COLOR[mode],
                    "verdict": f["verdict"],
                    "verdict_color": _VERDICT_COLOR.get(f["verdict"], "#5c6370"),
                    "severity": f["severity"],
                    "confidence": f["confidence"],
                    "rationale": f["rationale"],
                    "llm_used": f["llm_used"],
                    "evidence": f["evidence"],
                }
            )
        findings_by_call[call["call_id"]] = rows

    reasoning_by_call = {c.call_id: c.preceding_reasoning for c in session.tool_calls}
    result_content = {
        c.call_id: (c.result.content if c.result else None) for c in session.tool_calls
    }

    timeline = []
    for call in report["tool_calls"]:
        findings = findings_by_call.get(call["call_id"], [])
        content = result_content.get(call["call_id"]) or ""
        timeline.append(
            {
                "seq": call["seq"],
                "turn": call["turn"],
                "tool_name": call["tool_name"],
                "capability": call["capability"],
                "args_json": json.dumps(call["arguments"], indent=2, default=str),
                "reasoning": reasoning_by_call.get(call["call_id"]),
                "result": call["result"],
                "result_content": content[:2000],
                "findings": findings,
                "worst_verdict": _worst([f["verdict"] for f in findings]),
                "mode_codes": sorted({f["mode_code"] for f in findings}),
            }
        )
    return timeline


def _worst(verdicts: list[str]) -> str:
    for v in ("fail", "warn", "pass"):
        if v in verdicts:
            return v
    return "clean"


def build_view(session: NormalizedSession, score: SessionScore) -> dict[str, Any]:
    report = report_dict(session, score)
    session_findings = []
    for f in report["session_findings"]:
        mode = FailureMode(f["mode"])
        session_findings.append(
            {
                "mode_code": _CODE[mode], "mode_label": _LABEL[mode], "color": _COLOR[mode],
                "verdict": f["verdict"], "verdict_color": _VERDICT_COLOR.get(f["verdict"]),
                "rationale": f["rationale"], "confidence": f["confidence"],
                "llm_used": f["llm_used"], "evidence": f["evidence"],
            }
        )
    return {
        "session_id": session.session_id,
        "adapter": session.adapter,
        "task_id": session.task.task_id,
        "prompt": session.task.prompt,
        "stop_reason": session.stop_reason,
        "composite": score.composite,
        "n_tool_calls": len(session.tool_calls),
        "scores": _score_rows(score),
        "radar": _radar(score),
        "timeline": _timeline(session, report),
        "session_findings": session_findings,
    }


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tojson_pretty"] = lambda v: json.dumps(v, indent=2, default=str)
    return env


def render_dashboard(pairs: list[tuple[NormalizedSession, SessionScore]]) -> str:
    if not pairs:
        raise ValueError("render_dashboard requires at least one (session, score) pair")
    views = [build_view(s, sc) for s, sc in pairs]
    env = _environment()
    template = env.get_template("dashboard.html.j2")
    return template.render(
        views=views,
        compare=len(views) > 1,
        mode_meta=[{"code": c, "label": lbl, "color": col} for _, c, lbl, col in _MODE_META],
        verdict_color=_VERDICT_COLOR,
    )


def write_html_report(
    session: NormalizedSession, score: SessionScore, path: str | Path
) -> Path:
    return write_html_dashboard([(session, score)], path)


def write_html_dashboard(
    pairs: list[tuple[NormalizedSession, SessionScore]], path: str | Path
) -> Path:
    path = Path(path)
    path.write_text(render_dashboard(pairs))
    return path
