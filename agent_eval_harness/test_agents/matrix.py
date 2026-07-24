"""Run the injected-failure agents through the *real* scoring pipeline and
summarize precision/recall over the controlled set.

Each agent run is parsed by the ``GenericToolTraceAdapter`` and scored by the
deterministic detectors — the same code path production traces take. A mode is
considered "fired" when it is applicable and scored below 100.

The controlled set has ground truth: a clean run should fire nothing, and an
injected run should fire exactly its target mode. We report metrics only over the
modes whose detectors exist today (the deterministic set); the judgment-heavy
modes (M1/M4/M7) are emitted by the agents but their detectors land in M3, so
they are tracked separately as ``judge_pending`` rather than counted as misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.adapters.generic import GenericToolTraceAdapter
from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.detectors import DETERMINISTIC_DETECTORS
from agent_eval_harness.scoring.engine import SessionScore, evaluate_session
from agent_eval_harness.test_agents.agents import all_agents

# Modes with a deterministic detector today (M1 milestone).
DETERMINISTIC_MODES: frozenset[FailureMode] = frozenset(
    {
        FailureMode.WRONG_ARGS,
        FailureMode.HALLUCINATED,
        FailureMode.REDUNDANT,
        FailureMode.MISSING_VERIFICATION,
        FailureMode.UNSAFE_CALL,
    }
)
# Modes the agents inject but whose detectors arrive with the LLM judge in M3.
JUDGE_PENDING_MODES: frozenset[FailureMode] = frozenset(
    {
        FailureMode.WRONG_TOOL,
        FailureMode.IGNORED_OUTPUT,
        FailureMode.PREMATURE_STOP,
    }
)

_ADAPTER = GenericToolTraceAdapter()


def score_agent_run(agent: Any, inject: FailureMode | None) -> SessionScore:
    """Build one agent trace and score it through the production pipeline."""
    trace = agent.run(inject)
    session = _ADAPTER.parse(RunSource(kind="generic", data=trace))
    return evaluate_session(session, DETERMINISTIC_DETECTORS)


def fired_modes(score: SessionScore) -> set[FailureMode]:
    """Modes that produced an actionable finding (applicable and scored < 100)."""
    return {
        mode
        for mode, ms in score.mode_scores.items()
        if ms.applicable and ms.score is not None and ms.score < 100
    }


@dataclass
class RunRecord:
    agent: str
    inject: FailureMode | None
    target: FailureMode | None
    fired: set[FailureMode]
    score: SessionScore

    @property
    def deterministic_target(self) -> bool:
        return self.target in DETERMINISTIC_MODES


def run_controlled_set() -> list[RunRecord]:
    """Run every agent clean plus once per injectable mode."""
    records: list[RunRecord] = []
    for agent in all_agents():
        for inject in (None, *sorted(agent.injectable, key=lambda m: m.value)):
            score = score_agent_run(agent, inject)
            records.append(
                RunRecord(
                    agent=agent.name,
                    inject=inject,
                    target=inject,
                    fired=fired_modes(score),
                    score=score,
                )
            )
    return records


@dataclass
class Metrics:
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    cross_fires: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0


def precision_recall(records: list[RunRecord]) -> Metrics:
    """Precision/recall over the deterministic modes only.

    * A clean run that fires any deterministic mode is a false positive.
    * An injected deterministic run that fires its target is a true positive; if
      it fails to fire, a false negative.
    * A deterministic mode firing on a run where it was not the target (clean or a
      different injection) is a cross-fire / false positive.
    """
    m = Metrics()
    for r in records:
        det_fired = r.fired & DETERMINISTIC_MODES
        if r.inject is None:
            for mode in det_fired:
                m.false_positives += 1
                m.cross_fires.append(f"{r.agent} clean -> {mode.value}")
            continue
        if r.deterministic_target:
            if r.target in det_fired:
                m.true_positives += 1
            else:
                m.false_negatives += 1
        # Any deterministic mode that fired but was not the target is a cross-fire.
        for mode in det_fired:
            if mode is not r.target:
                m.false_positives += 1
                m.cross_fires.append(
                    f"{r.agent} inject={r.inject.value} -> cross-fire {mode.value}"
                )
    return m


def format_report(records: list[RunRecord], metrics: Metrics) -> str:
    lines = ["Injected-failure controlled set:", ""]
    for r in records:
        label = r.inject.value if r.inject else "clean"
        fired = ", ".join(sorted(m.value for m in r.fired)) or "-"
        tag = "" if r.inject is None or r.deterministic_target else "  [judge-pending, M3]"
        lines.append(
            f"  {r.agent:<22} {label:<22} composite={r.score.composite}"
            f"  fired: {fired}{tag}"
        )
    lines += [
        "",
        f"Deterministic modes: precision={metrics.precision:.2f} "
        f"recall={metrics.recall:.2f} "
        f"(tp={metrics.true_positives} fn={metrics.false_negatives} "
        f"fp={metrics.false_positives})",
    ]
    if metrics.cross_fires:
        lines.append("Cross-fires: " + "; ".join(metrics.cross_fires))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    recs = run_controlled_set()
    print(format_report(recs, precision_recall(recs)))
