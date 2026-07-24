"""M6 — Missing verification step: a code edit was made but nothing ran to check it.

Logic:
  1. Find code-file mutations (FILE_WRITE/FILE_EDIT on a source file).
  2. Decide whether verification is *warranted*: a code file was mutated AND the
     repo has a test/build toolchain (a manifest was touched, a verification tool
     ran at some point, the task declares `required_verification`, or metadata
     flags a build system).
  3. After the last mutation, scan forward for a TEST_RUN / BUILD_RUN / LINT_RUN
     before the session ends. If warranted and none is found -> fail.

Reference mode: if `task.required_verification` names specific capabilities, all
of them must appear after the last mutation.
"""

from __future__ import annotations

from agent_eval_harness.core.capability import (
    VERIFICATION_CAPABILITIES,
    CanonicalCapability,
)
from agent_eval_harness.core.findings import EventRef, FailureMode, Finding
from agent_eval_harness.core.model import NormalizedSession
from agent_eval_harness.detectors.base import DetectorContext, DetectorResult

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".kt", ".swift",
    ".scala", ".m", ".mm", ".sh", ".sql", ".vue", ".svelte",
}
_DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".adoc"}

_MANIFESTS = {
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "makefile",
    "go.mod", "cargo.toml", "pom.xml", "build.gradle", "composer.json",
    "tsconfig.json", "gemfile", "requirements.txt",
}


def _is_code_file(path: str | None) -> bool:
    if not path:
        return False
    lower = path.lower()
    ext = lower[lower.rfind("."):] if "." in lower else ""
    if ext in _DOC_EXTENSIONS:
        return False
    return ext in _CODE_EXTENSIONS


def _references_manifest(path: str | None) -> bool:
    if not path:
        return False
    base = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base in _MANIFESTS


class MissingVerificationDetector:
    mode = FailureMode.MISSING_VERIFICATION
    needs_reference = False
    works_reference_free = True
    uses_llm = False
    version = "0.1.0"

    def evaluate(self, session: NormalizedSession, ctx: DetectorContext) -> DetectorResult:
        calls = session.tool_calls
        mutation_seqs = [
            c.seq
            for c in calls
            if c.capability in (CanonicalCapability.FILE_WRITE, CanonicalCapability.FILE_EDIT)
            and _is_code_file(c.path)
        ]

        if not mutation_seqs:
            return DetectorResult(mode=self.mode, n_opportunities=0, applicable=False)

        warranted = self._is_warranted(session)
        if not warranted:
            return DetectorResult(mode=self.mode, n_opportunities=0, applicable=False)

        last_mutation = max(mutation_seqs)

        # Verification calls occurring after the last code mutation.
        post_verifications = {
            c.capability
            for c in calls
            if c.seq > last_mutation and c.capability in VERIFICATION_CAPABILITIES
        }

        required = set(session.task.required_verification) or None
        missing = self._missing(required, post_verifications)

        findings: list[Finding] = []
        if missing:
            if required:
                reason = (
                    "required verification "
                    f"{sorted(c.value for c in missing)} did not run after the last "
                    "code change"
                )
            else:
                reason = (
                    "code was modified but no test/build/lint ran afterward "
                    "before the session ended"
                )
            findings.append(
                Finding(
                    mode=self.mode,
                    verdict="fail",
                    severity=1.0,
                    confidence=0.85,
                    rationale=reason + ".",
                    target_seq=last_mutation,
                    evidence=[EventRef(seq=last_mutation, note="last code mutation")],
                    detector_version=self.version,
                )
            )

        return DetectorResult(
            mode=self.mode,
            findings=findings,
            n_opportunities=1,  # one "warranted verification" cluster this session
            applicable=True,
        )

    @staticmethod
    def _missing(
        required: set[CanonicalCapability] | None,
        present: set[CanonicalCapability],
    ) -> set[CanonicalCapability]:
        if required:
            return required - present
        # reference-free: any single verification satisfies the requirement
        return set() if present else {CanonicalCapability.TEST_RUN}

    @staticmethod
    def _is_warranted(session: NormalizedSession) -> bool:
        if session.task.required_verification:
            return True
        if session.metadata.get("has_build_system"):
            return True
        for call in session.tool_calls:
            if call.capability in VERIFICATION_CAPABILITIES:
                return True
            if _references_manifest(call.path):
                return True
        return False
