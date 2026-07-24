"""Discover bundled tasks, run an agent on each in a sandbox, and aggregate scores.

The suite reuses the single-task pieces wholesale: ``run_live`` copies each task's
``seed/`` repo into a throwaway sandbox, invokes the agent CLI, and normalizes the
trace; the standard detectors + scoring engine then score every run. Per-task
failures (a missing CLI binary, a crashed run) are captured, not fatal, so one bad
task never sinks the whole suite.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.core.model import NormalizedSession
from agent_eval_harness.core.taskspec import TaskSpec
from agent_eval_harness.detectors import ALL_DETECTORS
from agent_eval_harness.detectors.base import DetectorContext
from agent_eval_harness.detectors.judge import build_judge
from agent_eval_harness.runner.live import CommandRunner, run_live
from agent_eval_harness.scoring.engine import SessionScore, evaluate_session

BUNDLED_TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class SuiteTask:
    task_id: str
    spec: TaskSpec
    seed_dir: Path


@dataclass
class SuiteRunResult:
    task_id: str
    score: SessionScore | None
    session: NormalizedSession | None
    error: str | None = None


@dataclass
class SuiteReport:
    adapter: str
    results: list[SuiteRunResult] = field(default_factory=list)

    @property
    def ok_results(self) -> list[SuiteRunResult]:
        return [r for r in self.results if r.score is not None]

    @property
    def composite(self) -> int | None:
        """Mean composite across tasks that ran (rounded), or None if none did."""
        vals = [r.score.composite for r in self.ok_results
                if r.score is not None and r.score.composite is not None]
        return round(statistics.mean(vals)) if vals else None

    def mode_means(self) -> dict[FailureMode, int | None]:
        """Per-mode mean over tasks where the mode was applicable."""
        out: dict[FailureMode, int | None] = {}
        for mode in FailureMode:
            vals: list[int] = []
            for r in self.ok_results:
                if r.score is None:
                    continue
                ms = r.score.mode_scores.get(mode)
                if ms is not None and ms.score is not None:
                    vals.append(ms.score)
            out[mode] = round(statistics.mean(vals)) if vals else None
        return out


def discover_tasks(root: Path | str = BUNDLED_TASKS_DIR) -> list[SuiteTask]:
    """Load every ``<root>/<name>/task.yaml`` + its ``seed/`` dir, sorted by id."""
    root = Path(root)
    tasks: list[SuiteTask] = []
    for task_yaml in sorted(root.glob("*/task.yaml")):
        spec = TaskSpec.from_yaml(task_yaml)
        seed_dir = task_yaml.parent / "seed"
        tasks.append(SuiteTask(task_id=spec.task_id, spec=spec, seed_dir=seed_dir))
    return tasks


def _score(
    session: NormalizedSession, judge: str | None, judge_cache: str | Path | None
) -> SessionScore:
    ctx = DetectorContext(judge=build_judge(judge, cache_dir=judge_cache))
    return evaluate_session(session, ALL_DETECTORS, ctx)


def run_suite(
    adapter: str,
    tasks: list[SuiteTask] | None = None,
    *,
    judge: str | None = None,
    judge_cache: str | Path | None = None,
    timeout: float | None = 600.0,
    keep_workdir: bool = False,
    agent_args: list[str] | None = None,
    command_runner: CommandRunner | None = None,
    on_task_start: Callable[[SuiteTask], None] | None = None,
) -> SuiteReport:
    """Run ``adapter`` over every task and return the aggregated report."""
    tasks = tasks if tasks is not None else discover_tasks()
    report = SuiteReport(adapter=adapter)
    for task in tasks:
        if on_task_start is not None:
            on_task_start(task)
        try:
            result = run_live(
                adapter,
                task.spec,
                repo=task.seed_dir,
                sandbox=True,
                timeout=timeout,
                keep_workdir=keep_workdir,
                agent_args=agent_args or [],
                command_runner=command_runner,
            )
            score = _score(result.session, judge, judge_cache)
            report.results.append(SuiteRunResult(task.task_id, score, result.session))
        except Exception as exc:  # noqa: BLE001 - one task must not sink the suite
            report.results.append(
                SuiteRunResult(task.task_id, None, None, error=f"{type(exc).__name__}: {exc}")
            )
    return report


def format_report(report: SuiteReport) -> str:
    """A compact per-task table + the aggregate vector, for the terminal."""
    lines: list[str] = []
    lines.append(f"suite: {len(report.results)} tasks  adapter={report.adapter}")
    lines.append(f"{'task':<20} {'composite':>9}  notes")
    for r in report.results:
        if r.error is not None:
            lines.append(f"{r.task_id:<20} {'ERR':>9}  {r.error}")
        elif r.score is not None:
            comp = "n/a" if r.score.composite is None else str(r.score.composite)
            n = len([f for f in r.score.all_findings if f.verdict in ("fail", "warn")])
            lines.append(f"{r.task_id:<20} {comp:>9}  {n} findings")
    lines.append("-" * 44)
    comp = "n/a" if report.composite is None else str(report.composite)
    lines.append(f"{'AGGREGATE':<20} {comp:>9}  (mean composite over tasks)")
    for mode, val in report.mode_means().items():
        label = "n/a" if val is None else str(val)
        lines.append(f"  {mode.value:<22} {label:>4}")
    return "\n".join(lines)
