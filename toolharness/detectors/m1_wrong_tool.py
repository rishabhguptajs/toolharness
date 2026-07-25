"""M1 — Wrong tool selected: a first-class capability existed, but the agent
reached for a worse instrument (usually a raw shell command).

Layers (cheap -> expensive):
  1. Anti-pattern heuristics (deterministic, no judge): a shell ``cat`` when a
     FILE_READ tool is advertised; shell ``grep``/``rg`` when CONTENT_SEARCH
     exists; shell ``sed -i``/``echo >`` when FILE_EDIT exists; shell ``find``
     when FILE_SEARCH exists. These are the clear, common cases.
  2. Reference alignment (when ``task.expected_capabilities`` is given): a call
     whose capability is a clear substitution away from the expected sequence.
  3. LLM-judge (optional) for the residue: a shell call that overlaps a
     first-class capability but did not match a hard anti-pattern is sent to the
     judge with (task, reasoning, command, available tools).
"""

from __future__ import annotations

import re

from toolharness.core.capability import CanonicalCapability
from toolharness.core.findings import EventRef, FailureMode, Finding
from toolharness.core.model import NormalizedSession, ToolCallEvent
from toolharness.detectors.base import DetectorContext, DetectorResult
from toolharness.detectors.judge import JudgeRequest, safe_ask

_SHELL_CAPS = {
    CanonicalCapability.SHELL_EXEC,
    CanonicalCapability.TEST_RUN,
    CanonicalCapability.BUILD_RUN,
    CanonicalCapability.LINT_RUN,
}

# (regex over the shell command, capability the agent *should* have used, label).
_FR = CanonicalCapability.FILE_READ
_FE = CanonicalCapability.FILE_EDIT
_CS = CanonicalCapability.CONTENT_SEARCH
_FS = CanonicalCapability.FILE_SEARCH
_ANTIPATTERNS: list[tuple[re.Pattern[str], CanonicalCapability, str]] = [
    (re.compile(r"^\s*cat\s+[^|<>]+$"), _FR, "read a file via shell `cat`"),
    (re.compile(r"\b(grep|egrep|rg|ripgrep)\b"), _CS, "searched content via shell `grep`"),
    (re.compile(r"\bsed\b[^|;&]*-i"), _FE, "edited a file via shell `sed -i`"),
    (re.compile(r"\becho\b[^|;&]*>>?\s*\S"), _FE, "wrote a file via shell redirection"),
    (re.compile(r"^\s*find\s+\S"), _FS, "searched files via shell `find`"),
]

_JUDGE_SYSTEM = (
    "You audit an AI coding agent's tool choices for reliability. Given the task, "
    "the agent's reasoning, one tool call it made, and the tools that were "
    "available, decide whether the agent picked the WRONG tool for the job (a more "
    "appropriate first-class tool existed and should have been used). Respond ONLY "
    'with JSON: {"verdict":"fail"|"pass","confidence":0..1,"rationale":"..."}. '
    "verdict=fail means the wrong tool was chosen."
)


class WrongToolDetector:
    mode = FailureMode.WRONG_TOOL
    needs_reference = False
    works_reference_free = True
    uses_llm = True
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        findings: list[Finding] = []
        capabilities_available = {
            t.capability for t in session.available_tools
        } | {c.capability for c in session.tool_calls}

        used_judge = False
        for call in session.tool_calls:
            if call.capability not in _SHELL_CAPS:
                continue
            command = call.command or ""

            hard = self._antipattern(command, capabilities_available)
            if hard is not None:
                want, label = hard
                findings.append(
                    self._finding(
                        call,
                        verdict="fail",
                        severity=0.7,
                        confidence=0.9,
                        rationale=(
                            f"{label} even though a {want.value} tool was available; "
                            "prefer the first-class tool."
                        ),
                        llm_used=False,
                    )
                )
                continue

            # Residual escalation to the judge (only if one is wired).
            if ctx.judge is not None and self._overlaps_first_class(command, session):
                used_judge = True
                verdict = safe_ask(ctx.judge, self._judge_request(session, call, command))
                if verdict is not None and verdict.verdict == "fail":
                    findings.append(
                        self._finding(
                            call,
                            verdict="fail",
                            severity=0.6,
                            confidence=verdict.confidence,
                            rationale=f"judge: {verdict.rationale}",
                            llm_used=True,
                        )
                    )

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=len(session.tool_calls),
            applicable=len(session.tool_calls) > 0,
            confidence=0.85 if used_judge else 1.0,
        )

    # --- helpers ----------------------------------------------------------------

    @staticmethod
    def _antipattern(
        command: str, available: set[CanonicalCapability]
    ) -> tuple[CanonicalCapability, str] | None:
        for pattern, want, label in _ANTIPATTERNS:
            if pattern.search(command) and want in available:
                return want, label
        return None

    @staticmethod
    def _overlaps_first_class(command: str, session: NormalizedSession) -> bool:
        # A cheap gate so we don't send every shell call to the judge: only ones
        # that look file/search-ish and where a matching first-class tool exists.
        caps = {t.capability for t in session.available_tools}
        looks_fileish = bool(re.search(r"\b(cat|less|head|tail|grep|sed|awk|find|ls)\b", command))
        return looks_fileish and bool(
            caps & {
                CanonicalCapability.FILE_READ,
                CanonicalCapability.FILE_EDIT,
                CanonicalCapability.CONTENT_SEARCH,
                CanonicalCapability.FILE_SEARCH,
            }
        )

    def _judge_request(
        self, session: NormalizedSession, call: ToolCallEvent, command: str
    ) -> JudgeRequest:
        tools = ", ".join(
            sorted(f"{t.name}({t.capability.value})" for t in session.available_tools)
        )
        user = (
            f"Task: {session.task.prompt}\n"
            f"Agent reasoning before the call: {call.preceding_reasoning or '(none)'}\n"
            f"Tool call: {call.tool_name} command={command!r}\n"
            f"Available tools: {tools}\n"
            "Did the agent pick the wrong tool?"
        )
        return JudgeRequest(kind=self.mode.value, system=_JUDGE_SYSTEM, user=user)

    def _finding(
        self,
        call: ToolCallEvent,
        *,
        verdict: str,
        severity: float,
        confidence: float,
        rationale: str,
        llm_used: bool,
    ) -> Finding:
        return Finding(
            mode=self.mode,
            verdict=verdict,  # type: ignore[arg-type]
            severity=severity,
            confidence=confidence,
            rationale=f"{call.tool_name}: {rationale}",
            target_call_id=call.call_id,
            target_seq=call.seq,
            evidence=[EventRef(seq=call.seq, call_id=call.call_id)],
            llm_used=llm_used,
            detector_version=self.version,
        )
