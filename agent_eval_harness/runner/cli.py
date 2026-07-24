"""`evalharness` command-line entry point (M1 surface).

    evalharness run <trace.json> [--adapter NAME] [--json OUT] [--fail-under N]

Parses a run into a NormalizedSession, runs the deterministic detectors, writes
the JSON report, and prints the score vector. Exits non-zero when the composite
falls below ``--fail-under`` (CI gate). Real CLI adapters and the HTML report
land in later milestones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_eval_harness.adapters import default_registry
from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.core.model import NormalizedSession
from agent_eval_harness.detectors import DETERMINISTIC_DETECTORS
from agent_eval_harness.report.json_report import report_dict, write_json_report
from agent_eval_harness.scoring.engine import SessionScore, evaluate_session


def evaluate_path(
    path: str | Path, adapter: str | None = None
) -> tuple[SessionScore, NormalizedSession]:
    aux = {"adapter": adapter} if adapter else {}
    source = RunSource(kind="generic", path=Path(path), aux=aux)
    session = default_registry.parse(source)
    score = evaluate_session(session, DETERMINISTIC_DETECTORS)
    return score, session


def _cmd_run(args: argparse.Namespace) -> int:
    score, session = evaluate_path(args.trace, args.adapter)

    if args.json:
        write_json_report(session, score, args.json)

    if args.print_json:
        print(json.dumps(report_dict(session, score), indent=2))
    else:
        print(f"session {score.session_id} [{score.adapter}]  composite={score.composite}")
        for mode, ms in score.mode_scores.items():
            label = "n/a" if ms.score is None else str(ms.score)
            print(f"  {mode.value:<22} {label:>4}  ({ms.n_findings} findings)")

    if args.fail_under is not None and score.composite is not None:
        if score.composite < args.fail_under:
            print(f"FAIL: composite {score.composite} < --fail-under {args.fail_under}",
                  file=sys.stderr)
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a single run trace")
    run.add_argument("trace", help="path to a run trace (generic JSON in M1)")
    run.add_argument("--adapter", default=None,
                     help=f"force adapter; one of {default_registry.names()}")
    run.add_argument("--json", default=None, help="write the JSON report to this path")
    run.add_argument("--print-json", action="store_true", help="print full JSON to stdout")
    run.add_argument("--fail-under", type=int, default=None,
                     help="exit non-zero if composite score is below this threshold")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
