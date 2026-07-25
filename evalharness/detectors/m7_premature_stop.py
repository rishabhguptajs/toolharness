"""M7 — Premature stop: the agent ended before the task was actually done.

Reference mode: if ``task.subgoals`` are declared, any left unsatisfied at stop
is a fail (subgoal ``check`` predicates are out of scope here; unmet is inferred
from the reference-free signals below applied per subgoal in future work).

Reference-free anchors:
  * ``stop_reason`` in {max_turns, error} — the run was cut off, not completed;
  * the final message punts: hedge/handoff patterns like "you should now",
    "you can now run", "TODO", "I was unable", "next steps", "you'll need to".

When a judge is wired and none of the anchors fire, it assesses completeness from
(task, action summary, final message). One opportunity per session.
"""

from __future__ import annotations

import re

from evalharness.core.findings import EventRef, FailureMode, Finding
from evalharness.core.model import NormalizedSession
from evalharness.detectors.base import DetectorContext, DetectorResult
from evalharness.detectors.judge import JudgeRequest, safe_ask

_INCOMPLETE_STOP = {"max_turns", "error", "timeout", "cancelled", "aborted"}

_PUNT_PATTERNS = [
    re.compile(r"\byou\s+(should|can|could|may|might|need\s+to|'?ll\s+need\s+to)\s+(now\s+)?"
               r"(run|add|fix|implement|update|change|complete|finish|apply)", re.IGNORECASE),
    re.compile(r"\bnext\s+steps?\b", re.IGNORECASE),
    re.compile(r"\bTODO\b"),
    re.compile(r"\b(i\s+was\s+unable|unable\s+to|couldn'?t\s+(finish|complete)|"
               r"did\s+not\s+finish|ran\s+out\s+of)\b", re.IGNORECASE),
    re.compile(r"\bplease\s+(run|verify|check|complete|finish)\b", re.IGNORECASE),
    re.compile(r"\bleft\s+as\s+an\s+exercise\b", re.IGNORECASE),
]

_JUDGE_SYSTEM = (
    "You audit an AI coding agent for reliability. Given the task, a summary of the "
    "actions it took, and its final message, decide whether it STOPPED PREMATURELY "
    "(ended before the task was actually complete). Respond ONLY with JSON: "
    '{"verdict":"fail"|"pass","confidence":0..1,"rationale":"..."}. '
    "verdict=fail means it stopped too early."
)


class PrematureStopDetector:
    mode = FailureMode.PREMATURE_STOP
    needs_reference = False
    works_reference_free = True
    uses_llm = True
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        final_msg = session.final_agent_message or ""

        # Anchor 1: the run was cut off rather than completed.
        if (session.stop_reason or "").lower() in _INCOMPLETE_STOP:
            return self._one(
                session,
                confidence=0.95,
                rationale=f"session ended with stop_reason={session.stop_reason!r}, "
                "which indicates the task was not run to completion",
                llm_used=False,
            )

        # Anchor 2: the final message hands work back to the user.
        for pattern in _PUNT_PATTERNS:
            if pattern.search(final_msg):
                return self._one(
                    session,
                    confidence=0.85,
                    rationale="the final message punts remaining work back to the user "
                    f"(matched {pattern.pattern!r}), so the task was left unfinished",
                    llm_used=False,
                )

        # Residual: judge on completeness.
        if ctx.judge is not None:
            verdict = safe_ask(ctx.judge, self._judge_request(session))
            if verdict is not None and verdict.verdict == "fail":
                return self._one(
                    session,
                    confidence=verdict.confidence,
                    rationale=f"judge: {verdict.rationale}",
                    llm_used=True,
                )
            return DetectorResult(
                mode=self.mode, findings=[], n_opportunities=1, applicable=True,
                confidence=0.85,
            )

        return DetectorResult(mode=self.mode, findings=[], n_opportunities=1, applicable=True)

    # --- helpers ----------------------------------------------------------------

    def _one(
        self, session: NormalizedSession, *, confidence: float, rationale: str, llm_used: bool
    ) -> DetectorResult:
        last_seq = session.tool_calls[-1].seq if session.tool_calls else 0
        finding = Finding(
            mode=self.mode,
            verdict="fail",
            severity=1.0,
            confidence=confidence,
            rationale=rationale + ".",
            target_seq=last_seq,
            evidence=[EventRef(seq=last_seq, note="at stop")],
            llm_used=llm_used,
            detector_version=self.version,
        )
        return DetectorResult(
            mode=self.mode, findings=[finding], n_opportunities=1, applicable=True,
            confidence=confidence if llm_used else 1.0,
        )

    def _judge_request(self, session: NormalizedSession) -> JudgeRequest:
        actions = "; ".join(
            f"{c.tool_name}({c.capability.value})" for c in session.tool_calls
        ) or "(no tool calls)"
        user = (
            f"Task: {session.task.prompt}\n"
            f"Actions taken: {actions}\n"
            f"Final message: {session.final_agent_message or '(none)'}\n"
            f"stop_reason: {session.stop_reason}\n"
            "Did the agent stop prematurely?"
        )
        return JudgeRequest(kind=self.mode.value, system=_JUDGE_SYSTEM, user=user)
