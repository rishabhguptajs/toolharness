"""`evalharness` command-line entry point.

    evalharness run <trace.json> [--adapter N] [--judge P] [--json OUT]
                    [--html OUT] [--fail-under N]
    evalharness compare <trace.json> <trace.json> ... --html OUT [--judge P]

``run`` scores a single trace, writes JSON and/or the self-contained HTML
dashboard, prints the score vector, and gates CI via ``--fail-under``.
``compare`` scores several traces into one dashboard (agent A vs B on a task).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_eval_harness.adapters import default_registry
from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.core.model import NormalizedSession
from agent_eval_harness.detectors import ALL_DETECTORS
from agent_eval_harness.detectors.base import DetectorContext
from agent_eval_harness.detectors.judge import build_judge
from agent_eval_harness.report.html_report import write_html_dashboard, write_html_report
from agent_eval_harness.report.json_report import report_dict, write_json_report
from agent_eval_harness.scoring.engine import SessionScore, evaluate_session


def evaluate_path(
    path: str | Path,
    adapter: str | None = None,
    judge: str | None = None,
    judge_cache: str | Path | None = None,
) -> tuple[SessionScore, NormalizedSession]:
    aux = {"adapter": adapter} if adapter else {}
    source = RunSource(kind="generic", path=Path(path), aux=aux)
    session = default_registry.parse(source)
    ctx = DetectorContext(judge=build_judge(judge, cache_dir=judge_cache))
    score = evaluate_session(session, ALL_DETECTORS, ctx)
    return score, session


def _cmd_run(args: argparse.Namespace) -> int:
    score, session = evaluate_path(args.trace, args.adapter, args.judge, args.judge_cache)

    if args.json:
        write_json_report(session, score, args.json)
    if args.html:
        write_html_report(session, score, args.html)
        print(f"wrote HTML dashboard -> {args.html}")

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


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from agent_eval_harness.eval.benchmark import bfcl_report, format_report

    judge = build_judge(args.judge, cache_dir=args.judge_cache) if args.judge else None
    report = bfcl_report(args.data, judge=judge, limit=args.limit)
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote benchmark report -> {args.json}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    pairs = []
    for trace in args.traces:
        score, session = evaluate_path(trace, args.adapter, args.judge, args.judge_cache)
        pairs.append((session, score))
        print(f"  {score.session_id:<28} composite={score.composite}")
    write_html_dashboard(pairs, args.html)
    print(f"wrote comparison dashboard ({len(pairs)} sessions) -> {args.html}")
    return 0


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--adapter", default=None,
                   help=f"force adapter; one of {default_registry.names()}")
    p.add_argument("--judge", default=None,
                   help="LLM-judge provider for hybrid modes: none (default), "
                        "stub, groq, ollama, openrouter, nvidia")
    p.add_argument("--judge-cache", default=None,
                   help="directory to cache judge verdicts (reproducible re-runs)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a single run trace")
    run.add_argument("trace", help="path to a run trace (generic JSON)")
    _add_judge_args(run)
    run.add_argument("--json", default=None, help="write the JSON report to this path")
    run.add_argument("--html", default=None, help="write the HTML dashboard to this path")
    run.add_argument("--print-json", action="store_true", help="print full JSON to stdout")
    run.add_argument("--fail-under", type=int, default=None,
                     help="exit non-zero if composite score is below this threshold")
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser("compare", help="score several traces into one dashboard")
    compare.add_argument("traces", nargs="+", help="two or more run traces")
    _add_judge_args(compare)
    compare.add_argument("--html", required=True, help="write the comparison dashboard here")
    compare.set_defaults(func=_cmd_compare)

    bench = sub.add_parser("benchmark", help="validate detectors against a public benchmark")
    bench.add_argument("dataset", choices=["bfcl"], help="benchmark to run")
    bench.add_argument("--data", required=True, help="path to the benchmark data directory")
    bench.add_argument("--limit", type=int, default=None, help="cap cases per category")
    _add_judge_args(bench)
    bench.add_argument("--json", default=None, help="write the benchmark report JSON here")
    bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
