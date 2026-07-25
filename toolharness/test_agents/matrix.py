"""Run the injected-failure agents through the *real* scoring pipeline and
summarize precision/recall over the controlled set.

Each agent run is parsed by the ``GenericToolTraceAdapter`` and scored by the
detectors — the same code path production traces take. A mode is considered
"fired" when it is applicable and scored below 100.

As of M3 all eight modes have detectors, so the default is ``ALL_DETECTORS`` and
the whole controlled set is graded. The judgment-heavy modes (M1/M4/M7) are
caught by their deterministic anchors here; a judge can be passed to additionally
exercise the escalation path, but the controlled set does not require one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolharness.adapters.base import RunSource
from toolharness.adapters.generic import GenericToolTraceAdapter
from toolharness.core.findings import FailureMode
from toolharness.detectors import ALL_DETECTORS, Detector, DetectorContext
from toolharness.scoring.engine import SessionScore, evaluate_session
from toolharness.test_agents.agents import all_agents

# Modes with a purely deterministic detector.
DETERMINISTIC_MODES: frozenset[FailureMode] = frozenset(
    {
        FailureMode.WRONG_ARGS,
        FailureMode.HALLUCINATED,
        FailureMode.REDUNDANT,
        FailureMode.MISSING_VERIFICATION,
        FailureMode.UNSAFE_CALL,
    }
)
# Modes handled by the hybrid (heuristic-anchored + judge) detectors added in M3.
HYBRID_MODES: frozenset[FailureMode] = frozenset(
    {
        FailureMode.WRONG_TOOL,
        FailureMode.IGNORED_OUTPUT,
        FailureMode.PREMATURE_STOP,
    }
)

_ADAPTER = GenericToolTraceAdapter()


def score_agent_run(
    agent: Any,
    inject: FailureMode | None,
    detectors: list[Detector] | None = None,
    judge: Any = None,
) -> SessionScore:
    """Build one agent trace and score it through the production pipeline."""
    trace = agent.run(inject)
    session = _ADAPTER.parse(RunSource(kind="generic", data=trace))
    ctx = DetectorContext(judge=judge)
    return evaluate_session(session, detectors or ALL_DETECTORS, ctx)


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


def run_controlled_set(
    detectors: list[Detector] | None = None, judge: Any = None
) -> list[RunRecord]:
    """Run every agent clean plus once per injectable mode."""
    records: list[RunRecord] = []
    for agent in all_agents():
        for inject in (None, *sorted(agent.injectable, key=lambda m: m.value)):
            score = score_agent_run(agent, inject, detectors, judge)
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
    """Precision/recall over all eight modes on the controlled set.

    * A clean run that fires anything is a false positive.
    * An injected run that fires its target is a true positive; failing to fire is
      a false negative.
    * Any mode firing on a run where it was not the target is a cross-fire /
      false positive.
    """
    m = Metrics()
    for r in records:
        if r.inject is None:
            for mode in r.fired:
                m.false_positives += 1
                m.cross_fires.append(f"{r.agent} clean -> {mode.value}")
            continue
        if r.target in r.fired:
            m.true_positives += 1
        else:
            m.false_negatives += 1
        for mode in r.fired:
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
        lines.append(
            f"  {r.agent:<22} {label:<22} composite={r.score.composite}  fired: {fired}"
        )
    lines += [
        "",
        f"All modes: precision={metrics.precision:.2f} recall={metrics.recall:.2f} "
        f"(tp={metrics.true_positives} fn={metrics.false_negatives} "
        f"fp={metrics.false_positives})",
    ]
    if metrics.cross_fires:
        lines.append("Cross-fires: " + "; ".join(metrics.cross_fires))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    recs = run_controlled_set()
    print(format_report(recs, precision_recall(recs)))
