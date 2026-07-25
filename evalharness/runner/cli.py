"""`evalharness` command-line entry point.

    evalharness run <trace.json> [--adapter N] [--task spec.yaml] [--judge P]
                    [--json OUT] [--html OUT] [--fail-under N]
                    [--fail-under-mode MODE=N ...]
    evalharness live --adapter N --task spec.yaml [--repo DIR] [--in-place]
                    [--timeout S] [--json OUT] [--html OUT] [--fail-under N]
    evalharness compare <trace.json> ... --html OUT [--task spec.yaml] [--judge P]

``run`` scores a single pre-captured trace. ``live`` invokes a real agent CLI on a
task repo (sandboxed by default), captures its trace, then scores it. Both write
JSON and/or the self-contained HTML dashboard, print the score vector, and gate CI
via ``--fail-under`` / ``--fail-under-mode``. ``compare`` scores several traces
into one dashboard (agent A vs B on a task).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalharness.adapters import default_registry
from evalharness.adapters.base import RunSource
from evalharness.core.findings import FailureMode
from evalharness.core.model import NormalizedSession
from evalharness.core.taskspec import TaskSpec
from evalharness.detectors import ALL_DETECTORS
from evalharness.detectors.base import DetectorContext
from evalharness.detectors.judge import JudgeError, build_judge
from evalharness.report.html_report import write_html_dashboard, write_html_report
from evalharness.report.json_report import report_dict, write_json_report
from evalharness.scoring.engine import SessionScore, evaluate_session


def score_session(
    session: NormalizedSession,
    judge: str | None = None,
    judge_cache: str | Path | None = None,
) -> SessionScore:
    ctx = DetectorContext(judge=build_judge(judge, cache_dir=judge_cache))
    return evaluate_session(session, ALL_DETECTORS, ctx)


def evaluate_path(
    path: str | Path,
    adapter: str | None = None,
    judge: str | None = None,
    judge_cache: str | Path | None = None,
    task: str | Path | None = None,
) -> tuple[SessionScore, NormalizedSession]:
    aux = {"adapter": adapter} if adapter else {}
    source = RunSource(kind="generic", path=Path(path), aux=aux)
    session = default_registry.parse(source)
    if task is not None:
        session.task = TaskSpec.from_yaml(task)
    score = score_session(session, judge, judge_cache)
    return score, session


def _resolve_mode(token: str) -> FailureMode:
    """Accept either an enum name (UNSAFE_CALL) or value (unsafe_call), any case."""
    key = token.strip().lower()
    for mode in FailureMode:
        if key in (mode.name.lower(), mode.value.lower()):
            return mode
    valid = ", ".join(m.name for m in FailureMode)
    raise argparse.ArgumentTypeError(f"unknown failure mode {token!r}; one of: {valid}")


def _parse_mode_gate(spec: str) -> tuple[FailureMode, int]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--fail-under-mode expects MODE=N, got {spec!r}"
        )
    name, _, value = spec.partition("=")
    try:
        threshold = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"threshold in {spec!r} is not an integer") from exc
    return _resolve_mode(name), threshold


def _report_and_gate(
    session: NormalizedSession, score: SessionScore, args: argparse.Namespace
) -> int:
    if args.json:
        write_json_report(session, score, args.json)
        print(f"wrote JSON report -> {args.json}")
    if args.html:
        write_html_report(session, score, args.html)
        print(f"wrote HTML dashboard -> {args.html}")

    if getattr(args, "print_json", False):
        print(json.dumps(report_dict(session, score), indent=2))
    else:
        print(f"session {score.session_id} [{score.adapter}]  composite={score.composite}")
        for mode, ms in score.mode_scores.items():
            label = "n/a" if ms.score is None else str(ms.score)
            print(f"  {mode.value:<22} {label:>4}  ({ms.n_findings} findings)")

    failures: list[str] = []
    if args.fail_under is not None and score.composite is not None:
        if score.composite < args.fail_under:
            failures.append(f"composite {score.composite} < --fail-under {args.fail_under}")
    for mode, threshold in getattr(args, "fail_under_mode", None) or []:
        gated = score.mode_scores.get(mode)
        if gated is None or gated.score is None:
            continue  # not applicable -> no gate
        if gated.score < threshold:
            failures.append(f"{mode.value} {gated.score} < {threshold}")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    score, session = evaluate_path(
        args.trace, args.adapter, args.judge, args.judge_cache, task=args.task
    )
    return _report_and_gate(session, score, args)


def _cmd_live(args: argparse.Namespace) -> int:
    from evalharness.runner.live import run_live

    task = TaskSpec.from_yaml(args.task)
    sandbox = not args.in_place
    if args.in_place:
        print("WARNING: --in-place runs the agent in the real repo; it may modify files.",
              file=sys.stderr)
    result = run_live(
        args.adapter,
        task,
        repo=args.repo,
        sandbox=sandbox,
        timeout=args.timeout,
        trace_path=args.save_trace,
        keep_workdir=args.keep_workdir,
        agent_args=args.agent_arg or [],
    )
    where = "sandbox" if result.sandboxed else "in-place"
    print(f"ran '{result.metadata['command']}' [{where}] "
          f"-> exit {result.returncode}{' (timeout)' if result.timed_out else ''}")
    print(f"trace: {result.trace_path}")
    if args.keep_workdir and result.sandboxed:
        print(f"workdir kept: {result.workdir}")
    score = score_session(result.session, args.judge, args.judge_cache)
    return _report_and_gate(result.session, score, args)


def _cmd_suite(args: argparse.Namespace) -> int:
    from evalharness.report.html_report import write_html_dashboard
    from evalharness.suite.runner import (
        SuiteTask,
        discover_tasks,
        format_report,
        run_suite,
    )

    tasks = discover_tasks(args.tasks) if args.tasks else discover_tasks()
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [t for t in tasks if t.task_id in wanted]
    if args.list:
        for t in tasks:
            ref = "reference" if t.spec.has_reference else "reference-free"
            print(f"  {t.task_id:<20} [{ref}]  {t.spec.prompt.strip()[:60]}")
        return 0
    if not tasks:
        print("no tasks to run", file=sys.stderr)
        return 2

    def _announce(t: SuiteTask) -> None:
        print(f"running {t.task_id} on {args.adapter} ...", flush=True)

    report = run_suite(
        args.adapter, tasks,
        judge=args.judge, judge_cache=args.judge_cache,
        timeout=args.timeout, keep_workdir=args.keep_workdir,
        agent_args=args.agent_arg or [], on_task_start=_announce,
    )
    print(format_report(report))

    if args.json:
        payload = {
            "schema_version": 1,
            "adapter": report.adapter,
            "aggregate": {
                "composite": report.composite,
                "mode_means": {m.value: v for m, v in report.mode_means().items()},
            },
            "tasks": [
                {
                    "task_id": r.task_id,
                    "error": r.error,
                    "report": (report_dict(r.session, r.score)
                               if r.session is not None and r.score is not None else None),
                }
                for r in report.results
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote suite report -> {args.json}")
    if args.html:
        pairs = [(r.session, r.score) for r in report.results
                 if r.session is not None and r.score is not None]
        if pairs:
            write_html_dashboard(pairs, args.html)
            print(f"wrote suite dashboard ({len(pairs)} tasks) -> {args.html}")

    failures: list[str] = []
    if args.fail_under is not None and report.composite is not None:
        if report.composite < args.fail_under:
            failures.append(f"aggregate composite {report.composite} < {args.fail_under}")
    means = report.mode_means()
    for mode, threshold in args.fail_under_mode or []:
        val = means.get(mode)
        if val is not None and val < threshold:
            failures.append(f"{mode.value} mean {val} < {threshold}")
    errored = [r.task_id for r in report.results if r.error is not None]
    if args.strict and errored:
        failures.append(f"{len(errored)} task(s) errored: {', '.join(errored)}")
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from evalharness.eval.benchmark import bfcl_report, format_report

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
        score, session = evaluate_path(
            trace, args.adapter, args.judge, args.judge_cache, task=args.task
        )
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


def _add_output_and_gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default=None,
                   help="task spec (YAML/JSON) to attach for reference-based scoring")
    p.add_argument("--json", default=None, help="write the JSON report to this path")
    p.add_argument("--html", default=None, help="write the HTML dashboard to this path")
    p.add_argument("--print-json", action="store_true", help="print full JSON to stdout")
    p.add_argument("--fail-under", type=int, default=None,
                   help="exit non-zero if the composite score is below this threshold")
    p.add_argument("--fail-under-mode", type=_parse_mode_gate, action="append",
                   dest="fail_under_mode", metavar="MODE=N", default=None,
                   help="per-mode CI gate, e.g. UNSAFE_CALL=90 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a single pre-captured run trace")
    run.add_argument("trace", help="path to a run trace (generic JSON or a real CLI capture)")
    _add_judge_args(run)
    _add_output_and_gate_args(run)
    run.set_defaults(func=_cmd_run)

    live = sub.add_parser("live", help="invoke a real agent CLI on a task repo, then score it")
    live.add_argument("--adapter", required=True,
                      help="live profile / adapter: claude-code | cursor | codex")
    live.add_argument("--task", required=True, dest="task",
                      help="task spec (YAML/JSON); provides the prompt, repo, and gold data")
    live.add_argument("--repo", default=None,
                      help="task repo dir (overrides repo.path in the spec)")
    live.add_argument("--in-place", action="store_true",
                      help="run in the real repo instead of a temp-dir sandbox (unsafe)")
    live.add_argument("--timeout", type=float, default=600.0,
                      help="wall-clock timeout in seconds (default 600)")
    live.add_argument("--save-trace", default=None,
                      help="write the raw captured trace to this path")
    live.add_argument("--keep-workdir", action="store_true",
                      help="do not delete the sandbox workdir after the run")
    live.add_argument("--agent-arg", action="append", dest="agent_arg", default=None,
                      metavar="ARG",
                      help="extra flag to append to the agent command (repeatable)")
    live.add_argument("--judge", default=None, help="LLM-judge provider (see `run --help`)")
    live.add_argument("--judge-cache", default=None, help="judge verdict cache dir")
    live.add_argument("--json", default=None, help="write the JSON report to this path")
    live.add_argument("--html", default=None, help="write the HTML dashboard to this path")
    live.add_argument("--print-json", action="store_true", help="print full JSON to stdout")
    live.add_argument("--fail-under", type=int, default=None,
                      help="exit non-zero if the composite score is below this threshold")
    live.add_argument("--fail-under-mode", type=_parse_mode_gate, action="append",
                      dest="fail_under_mode", metavar="MODE=N", default=None,
                      help="per-mode CI gate, e.g. UNSAFE_CALL=90 (repeatable)")
    live.set_defaults(func=_cmd_live)

    suite = sub.add_parser(
        "suite", help="run the bundled task suite end to end with one agent adapter")
    suite.add_argument("--adapter", required=True,
                       help="live profile / adapter: claude-code | cursor | codex")
    suite.add_argument("--tasks", default=None,
                       help="task suite directory (default: the bundled suite)")
    suite.add_argument("--task-id", action="append", dest="task_id", default=None,
                       metavar="ID", help="run only this task (repeatable)")
    suite.add_argument("--list", action="store_true", help="list tasks and exit")
    suite.add_argument("--timeout", type=float, default=600.0,
                       help="per-task wall-clock timeout in seconds (default 600)")
    suite.add_argument("--keep-workdir", action="store_true",
                       help="do not delete the per-task sandboxes")
    suite.add_argument("--agent-arg", action="append", dest="agent_arg", default=None,
                       metavar="ARG", help="extra flag appended to every agent command")
    suite.add_argument("--judge", default=None, help="LLM-judge provider (see `run --help`)")
    suite.add_argument("--judge-cache", default=None, help="judge verdict cache dir")
    suite.add_argument("--json", default=None, help="write the aggregate suite report JSON")
    suite.add_argument("--html", default=None, help="write a per-task comparison dashboard")
    suite.add_argument("--fail-under", type=int, default=None,
                       help="exit non-zero if the aggregate composite is below this")
    suite.add_argument("--fail-under-mode", type=_parse_mode_gate, action="append",
                       dest="fail_under_mode", metavar="MODE=N", default=None,
                       help="per-mode aggregate gate, e.g. UNSAFE_CALL=90 (repeatable)")
    suite.add_argument("--strict", action="store_true",
                       help="also fail if any task errored (e.g. missing CLI binary)")
    suite.set_defaults(func=_cmd_suite)

    compare = sub.add_parser("compare", help="score several traces into one dashboard")
    compare.add_argument("traces", nargs="+", help="two or more run traces")
    _add_judge_args(compare)
    compare.add_argument("--task", default=None,
                         help="task spec attached to every trace for reference-based scoring")
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
    try:
        return args.func(args)
    except JudgeError as exc:
        # The judge is bring-your-own-key: a missing credential is a user setup
        # problem, not a harness bug, so report it as a plain message.
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: the LLM judge needs your own API key. Export it, pick another "
            "provider with --judge, or omit --judge to score heuristics-only.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
