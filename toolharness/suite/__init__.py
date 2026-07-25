"""A bundled, self-contained task suite — the Terminal-Bench-style entry point.

Point an adapter at it and it does the rest: each task under ``tasks/<name>/`` ships
its own prompt + gold data (``task.yaml``) and a starting repo (``seed/``) that is
copied into a throwaway sandbox per run. ``toolharness suite --adapter <name>`` runs
the agent on every task, scores each for tool-call reliability, and aggregates.
"""

from toolharness.suite.runner import (
    BUNDLED_TASKS_DIR,
    SuiteReport,
    SuiteRunResult,
    SuiteTask,
    discover_tasks,
    run_suite,
)

__all__ = [
    "BUNDLED_TASKS_DIR",
    "SuiteReport",
    "SuiteRunResult",
    "SuiteTask",
    "discover_tasks",
    "run_suite",
]
