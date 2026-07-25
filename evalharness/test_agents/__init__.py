"""Injected-failure test agents (M2).

Minimal, framework-free agents that deliberately emit one failure mode each
through the canonical generic trace, so the detector path is validated
end-to-end on synthetic ground truth. See ``agents.py`` for the agents and
``matrix.py`` for the controlled-set runner + precision/recall.
"""

from evalharness.test_agents.agents import (
    AGENTS,
    BugfixAgent,
    SearchRefactorAgent,
    all_agents,
)
from evalharness.test_agents.harness import MockToolEnv, TraceRecorder
from evalharness.test_agents.matrix import (
    DETERMINISTIC_MODES,
    HYBRID_MODES,
    Metrics,
    RunRecord,
    fired_modes,
    precision_recall,
    run_controlled_set,
    score_agent_run,
)

__all__ = [
    "AGENTS",
    "BugfixAgent",
    "SearchRefactorAgent",
    "all_agents",
    "MockToolEnv",
    "TraceRecorder",
    "DETERMINISTIC_MODES",
    "HYBRID_MODES",
    "Metrics",
    "RunRecord",
    "fired_modes",
    "precision_recall",
    "run_controlled_set",
    "score_agent_run",
]
