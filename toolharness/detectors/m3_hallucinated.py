"""M3 — Hallucinated tool call: the agent invokes a tool that was never available.

Primary signal: ``tool_name`` not in the session's advertised registry.
Corroborating signal (works even when no registry was captured): the result
carries ``error_class == "UNKNOWN_TOOL"``. Registry-based findings are certain
(confidence 1.0); result-only findings are strong but not certain (0.8).
"""

from __future__ import annotations

from toolharness.core.findings import EventRef, FailureMode, Finding
from toolharness.core.model import NormalizedSession
from toolharness.detectors.base import DetectorContext, DetectorResult

_UNKNOWN_TOOL_ERRORS = {"UNKNOWN_TOOL", "NO_SUCH_TOOL", "TOOL_NOT_FOUND", "METHOD_NOT_FOUND"}


class HallucinatedCallDetector:
    mode = FailureMode.HALLUCINATED
    needs_reference = False
    works_reference_free = True
    uses_llm = False
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        findings: list[Finding] = []
        have_registry = bool(session.available_tools)
        known = session.available_tool_names

        for call in session.tool_calls:
            err_class = (call.result.error_class if call.result else None) or ""
            unknown_tool_error = err_class.upper() in _UNKNOWN_TOOL_ERRORS

            if have_registry and call.tool_name not in known:
                findings.append(
                    Finding(
                        mode=self.mode,
                        verdict="fail",
                        severity=1.0,
                        confidence=1.0,
                        rationale=(
                            f"Called {call.tool_name!r}, which is not in the "
                            f"advertised tool registry ({sorted(known)})."
                        ),
                        target_call_id=call.call_id,
                        target_seq=call.seq,
                        evidence=[EventRef(seq=call.seq, call_id=call.call_id)],
                        detector_version=self.version,
                    )
                )
            elif unknown_tool_error:
                findings.append(
                    Finding(
                        mode=self.mode,
                        verdict="fail",
                        severity=1.0,
                        confidence=0.8,
                        rationale=(
                            f"Tool {call.tool_name!r} returned an unknown-tool error "
                            f"({err_class}); likely hallucinated."
                        ),
                        target_call_id=call.call_id,
                        target_seq=call.seq,
                        evidence=[EventRef(seq=call.seq, call_id=call.call_id)],
                        detector_version=self.version,
                    )
                )

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=len(session.tool_calls),
            applicable=len(session.tool_calls) > 0,
            confidence=1.0 if have_registry else 0.8,
        )
