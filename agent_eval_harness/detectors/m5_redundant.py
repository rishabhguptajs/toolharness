"""M5 — Redundant/repeated calls: same tool + args with no reason (loop signal).

A repeated (capability, canonical-args) key is flagged UNLESS something happened
between the two calls that justifies re-issuing it:
  * a FILE_READ whose path was WRITTEN/EDITED since the previous identical read;
  * (identical-args retries after an error are *not* justified — that's a loop;
    a legitimate retry changes the args and therefore has a different key).

Severity escalates with repeat count, since a call issued three+ times is a
stronger confusion signal than an accidental double.
"""

from __future__ import annotations

import json
import re

from agent_eval_harness.core.capability import MUTATING_CAPABILITIES, CanonicalCapability
from agent_eval_harness.core.findings import EventRef, FailureMode, Finding
from agent_eval_harness.core.model import NormalizedSession, ToolCallEvent
from agent_eval_harness.detectors.base import DetectorContext, DetectorResult


def _canonical_args(call: ToolCallEvent) -> str:
    if call.capability in (CanonicalCapability.SHELL_EXEC, *(
        c for c in CanonicalCapability if c.name.endswith("_RUN")
    )):
        cmd = call.command or ""
        return re.sub(r"\s+", " ", cmd.strip())
    try:
        return json.dumps(call.arguments, sort_keys=True, default=str)
    except TypeError:
        return str(call.arguments)


def _call_key(call: ToolCallEvent) -> str:
    return f"{call.capability.value}|{call.tool_name}|{_canonical_args(call)}"


class RedundantCallDetector:
    mode = FailureMode.REDUNDANT
    needs_reference = False
    works_reference_free = True
    uses_llm = False
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        findings: list[Finding] = []
        seen: dict[str, ToolCallEvent] = {}          # key -> first occurrence
        repeat_count: dict[str, int] = {}
        # For each read key, the path it reads and whether it was mutated since last seen.
        mutated_since: dict[str, bool] = {}          # path -> mutated flag

        for call in session.tool_calls:
            # Update mutation state for write/edit calls.
            if call.capability in MUTATING_CAPABILITIES and call.path:
                mutated_since[call.path] = True

            key = _call_key(call)
            prior = seen.get(key)
            if prior is None:
                seen[key] = call
                continue

            # Justification: re-reading a file that changed since the last read.
            if call.capability == CanonicalCapability.FILE_READ and call.path:
                if mutated_since.get(call.path):
                    mutated_since[call.path] = False  # consume the justification
                    seen[key] = call
                    continue

            repeat_count[key] = repeat_count.get(key, 0) + 1
            n = repeat_count[key]  # 1 = first repeat, 2 = second, ...
            severity = min(1.0, 0.25 * (n + 1))  # 0.5, 0.75, 1.0, ...
            findings.append(
                Finding(
                    mode=self.mode,
                    verdict="fail",
                    severity=severity,
                    confidence=0.9,
                    rationale=(
                        f"{call.tool_name} repeated with identical arguments "
                        f"(repeat #{n}) and no intervening state change to justify it."
                    ),
                    target_call_id=call.call_id,
                    target_seq=call.seq,
                    evidence=[
                        EventRef(seq=prior.seq, call_id=prior.call_id, note="first call"),
                        EventRef(seq=call.seq, call_id=call.call_id, note="repeat"),
                    ],
                    detector_version=self.version,
                )
            )
            seen[key] = call

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=len(session.tool_calls),
            applicable=len(session.tool_calls) > 0,
        )
