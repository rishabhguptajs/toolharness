"""TaskSpec: optional reference/gold context that switches detectors into
reference-based mode. Every reference field is optional; absent -> reference-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval_harness.core.capability import CanonicalCapability, coerce_capability


@dataclass
class Subgoal:
    id: str
    description: str
    check: str | None = None  # optional shell/predicate identifier for automated checking


@dataclass
class TaskSpec:
    task_id: str
    prompt: str = ""
    repo_path: str | None = None
    repo_git_ref: str | None = None
    expected_capabilities: list[CanonicalCapability] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    required_verification: list[CanonicalCapability] = field(default_factory=list)
    allowed_destructive: list[str] = field(default_factory=list)

    @property
    def has_reference(self) -> bool:
        return bool(
            self.expected_capabilities or self.subgoals or self.required_verification
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskSpec:
        data = data or {}
        repo = data.get("repo") or {}
        subgoals = [
            Subgoal(
                id=str(sg.get("id", i)),
                description=sg.get("description", ""),
                check=sg.get("check"),
            )
            for i, sg in enumerate(data.get("subgoals", []))
        ]
        return cls(
            task_id=str(data.get("task_id", "unknown")),
            prompt=data.get("prompt", ""),
            repo_path=repo.get("path"),
            repo_git_ref=repo.get("git_ref"),
            expected_capabilities=[
                coerce_capability(c) for c in data.get("expected_capabilities", [])
            ],
            subgoals=subgoals,
            required_verification=[
                coerce_capability(c) for c in data.get("required_verification", [])
            ],
            allowed_destructive=list(data.get("allowed_destructive", [])),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        """Load a TaskSpec from a YAML (or JSON) file — the ``--task spec.yaml`` path."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        if raw is not None and not isinstance(raw, dict):
            raise ValueError(f"Task spec {path} must be a mapping, got {type(raw).__name__}.")
        return cls.from_dict(raw)
