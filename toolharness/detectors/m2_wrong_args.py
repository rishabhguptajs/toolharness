"""M2 — Wrong/malformed arguments: right tool, bad params.

Deterministic signals (M1 scope):
  * schema validation against the tool's advertised JSON schema — missing
    required fields, wrong JSON types;
  * result error class in {ENOENT, INVALID_ARGS, ...} as a strong anchor that the
    arguments were the problem;
  * empty/absent primary path on a file tool.

Semantic "right type, wrong value" (e.g. a valid path to the wrong file) is left
to the LLM-judge in M3.
"""

from __future__ import annotations

from typing import Any

from toolharness.core.capability import CanonicalCapability
from toolharness.core.findings import EventRef, FailureMode, Finding
from toolharness.core.model import NormalizedSession, ToolCallEvent
from toolharness.detectors.base import DetectorContext, DetectorResult

# Normalized error classes that indicate the arguments were at fault.
_ARG_ERROR_CLASSES = {"ENOENT", "INVALID_ARGS", "INVALID_ARGUMENT", "BAD_REQUEST", "EISDIR"}

_FILE_CAPS = {
    CanonicalCapability.FILE_READ,
    CanonicalCapability.FILE_WRITE,
    CanonicalCapability.FILE_EDIT,
}

# JSON-schema type name -> python types.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


class WrongArgsDetector:
    mode = FailureMode.WRONG_ARGS
    needs_reference = False
    works_reference_free = True
    uses_llm = False
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        findings: list[Finding] = []

        for call in session.tool_calls:
            spec = session.tool_spec(call.tool_name)
            schema = spec.schema if spec else None

            problem = self._schema_problem(call.arguments, schema)
            if problem is not None:
                findings.append(self._finding(call, problem, severity=1.0, confidence=1.0))
                continue

            # Result-level anchor: the runtime itself rejected the args.
            err_class = (call.result.error_class if call.result else None) or ""
            if err_class.upper() in _ARG_ERROR_CLASSES:
                findings.append(
                    self._finding(
                        call,
                        f"result error {err_class} indicates bad arguments "
                        f"({call.result.error_message if call.result else ''})".strip(),
                        severity=0.9,
                        confidence=0.9,
                    )
                )
                continue

            # File tool with no usable path argument.
            if call.capability in _FILE_CAPS and not call.path:
                findings.append(
                    self._finding(
                        call,
                        "file tool called without a usable path argument",
                        severity=0.8,
                        confidence=0.85,
                    )
                )

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=len(session.tool_calls),
            applicable=len(session.tool_calls) > 0,
        )

    # --- helpers ----------------------------------------------------------------

    @staticmethod
    def _schema_problem(args: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None
        required = schema.get("required", []) or []
        for field_name in required:
            if field_name not in args or args[field_name] is None:
                return f"missing required argument {field_name!r}"
        props = schema.get("properties", {}) or {}
        for name, spec in props.items():
            if name not in args or args[name] is None:
                continue
            expected = spec.get("type") if isinstance(spec, dict) else None
            allowed = _TYPE_MAP.get(expected) if expected else None
            if allowed is None:
                continue
            value = args[name]
            # bool is a subclass of int — guard against false "integer" matches.
            if expected in ("integer", "number") and isinstance(value, bool):
                return f"argument {name!r} should be {expected}, got boolean"
            if not isinstance(value, allowed):
                return (
                    f"argument {name!r} should be {expected}, got "
                    f"{type(value).__name__}"
                )
        return None

    def _finding(
        self, call: ToolCallEvent, reason: str, *, severity: float, confidence: float
    ) -> Finding:
        return Finding(
            mode=self.mode,
            verdict="fail",
            severity=severity,
            confidence=confidence,
            rationale=f"{call.tool_name}: {reason}.",
            target_call_id=call.call_id,
            target_seq=call.seq,
            evidence=[EventRef(seq=call.seq, call_id=call.call_id)],
            detector_version=self.version,
        )
