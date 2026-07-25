"""M4 — Ignored tool output: a result carried information the agent then acted as
if it never saw.

The high-signal, low-false-positive anchors (deterministic):
  * an errored/failing result that is the LAST tool result before a
    ``completed`` stop, with a final message that does not acknowledge the
    failure — the agent stopped as if it had succeeded;
  * an errored result immediately followed by an *identical* retry (same tool +
    args) — the agent didn't adjust based on what the error told it.

When a judge is wired, non-error but salient results (e.g. a test result the
agent should react to) are escalated: "did the next action account for this?"
Opportunities = results that plausibly needed follow-up (errors + salient), so a
run with nothing to react to is simply not applicable.
"""

from __future__ import annotations

import json
import re

from evalharness.core.findings import EventRef, FailureMode, Finding
from evalharness.core.model import NormalizedSession, ToolCallEvent
from evalharness.detectors.base import DetectorContext, DetectorResult
from evalharness.detectors.judge import JudgeRequest, safe_ask

_ACK_FAILURE = re.compile(
    r"\b(fail|failed|failing|error|errored|could\s*not|couldn'?t|unable|did\s*not|"
    r"didn'?t|not\s+pass|still\s+broken|revert|retry)\b",
    re.IGNORECASE,
)

_JUDGE_SYSTEM = (
    "You audit an AI coding agent for reliability. Given a tool result and what the "
    "agent did next, decide whether the agent IGNORED information in that result "
    "that it should have acted on. Respond ONLY with JSON: "
    '{"verdict":"fail"|"pass","confidence":0..1,"rationale":"..."}. '
    "verdict=fail means the output was ignored."
)


def _errored(call: ToolCallEvent) -> bool:
    r = call.result
    if r is None:
        return False
    return bool(r.is_error) or (r.exit_code is not None and r.exit_code != 0)


def _same_key(a: ToolCallEvent, b: ToolCallEvent) -> bool:
    if a.tool_name != b.tool_name:
        return False
    try:
        return json.dumps(a.arguments, sort_keys=True, default=str) == json.dumps(
            b.arguments, sort_keys=True, default=str
        )
    except TypeError:
        return a.arguments == b.arguments


class IgnoredOutputDetector:
    mode = FailureMode.IGNORED_OUTPUT
    needs_reference = False
    works_reference_free = True
    uses_llm = True
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        calls = session.tool_calls
        errored = [c for c in calls if _errored(c)]
        n_opportunities = len(errored)
        if not errored:
            return DetectorResult(mode=self.mode, n_opportunities=0, applicable=False)

        findings: list[Finding] = []
        last_call = calls[-1]
        final_msg = session.final_agent_message or ""
        acknowledges = bool(_ACK_FAILURE.search(final_msg))

        used_judge = False
        for i, call in enumerate(calls):
            if not _errored(call):
                continue

            # Anchor 1: the failing result is the last thing that happened and the
            # agent signed off as complete without acknowledging it.
            is_last = call is last_call
            if is_last and (session.stop_reason in (None, "completed")) and not acknowledges:
                findings.append(
                    self._finding(
                        call,
                        confidence=0.85,
                        rationale=(
                            "the final tool result failed, but the agent stopped as if "
                            "the task succeeded and never acknowledged the failure"
                        ),
                        llm_used=False,
                    )
                )
                continue

            # Anchor 2: identical retry right after the error (didn't adjust).
            nxt = calls[i + 1] if i + 1 < len(calls) else None
            if nxt is not None and _same_key(call, nxt):
                findings.append(
                    self._finding(
                        call,
                        confidence=0.8,
                        rationale=(
                            "the call errored and was immediately retried with identical "
                            "arguments, ignoring what the error reported"
                        ),
                        llm_used=False,
                    )
                )
                continue

            # Residual: let the judge weigh in on an unresolved error mid-run.
            if ctx.judge is not None:
                used_judge = True
                verdict = safe_ask(ctx.judge, self._judge_request(session, call, nxt))
                if verdict is not None and verdict.verdict == "fail":
                    findings.append(
                        self._finding(
                            call,
                            confidence=verdict.confidence,
                            rationale=f"judge: {verdict.rationale}",
                            llm_used=True,
                        )
                    )

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=n_opportunities,
            applicable=True,
            confidence=0.85 if used_judge else 1.0,
        )

    def _judge_request(
        self, session: NormalizedSession, call: ToolCallEvent, nxt: ToolCallEvent | None
    ) -> JudgeRequest:
        result = call.result
        summary = (result.content[:400] if result and result.content else "")
        next_desc = f"{nxt.tool_name} {nxt.arguments}" if nxt else "(agent stopped)"
        user = (
            f"Task: {session.task.prompt}\n"
            f"Tool call: {call.tool_name} -> status="
            f"{result.status if result else 'unknown'} "
            f"error_class={result.error_class if result else None}\n"
            f"Result content: {summary!r}\n"
            f"What the agent did next: {next_desc}\n"
            "Did the agent ignore what this result told it?"
        )
        return JudgeRequest(kind=self.mode.value, system=_JUDGE_SYSTEM, user=user)

    def _finding(
        self, call: ToolCallEvent, *, confidence: float, rationale: str, llm_used: bool
    ) -> Finding:
        return Finding(
            mode=self.mode,
            verdict="fail",
            severity=1.0,
            confidence=confidence,
            rationale=f"{call.tool_name}: {rationale}.",
            target_call_id=call.call_id,
            target_seq=call.seq,
            evidence=[EventRef(seq=call.seq, call_id=call.call_id)],
            llm_used=llm_used,
            detector_version=self.version,
        )
