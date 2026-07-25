"""BFCL (Berkeley Function-Calling Leaderboard) -> NormalizedSession.

A BFCL case is a *task spec*, not an agent trace: it carries the user question,
the advertised ``function`` schemas, and (for the answerable categories) a
``ground_truth`` in a sibling ``possible_answer/`` file. We turn each case into
scoring-ready sessions by pairing the case's real function registry with a chosen
agent behavior:

  * ``session_correct``      — the agent makes the gold call. Detectors must stay
                               silent (true negative).
  * ``session_hallucinated`` — the agent calls a function absent from the registry.
                               M3 must fire (true positive).
  * ``session_wrong_args``   — the agent drops a required argument. M2 must fire.

The registry + schemas are verbatim from BFCL; only the injected mistake is
synthetic, which is the standard way to measure a deterministic detector's
precision/recall over realistic tool schemas.

Data layout (HuggingFace ``gorilla-llm/Berkeley-Function-Calling-Leaderboard``)::

    <dir>/BFCL_v3_simple.json                     # cases (JSONL)
    <dir>/possible_answer/BFCL_v3_simple.json     # ground-truth (JSONL)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolharness.core.capability import CanonicalCapability
from toolharness.core.model import (
    EventType,
    NormalizedEvent,
    NormalizedSession,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    build_session,
)
from toolharness.core.taskspec import TaskSpec


@dataclass
class BFCLCase:
    case_id: str
    prompt: str
    functions: list[dict[str, Any]]
    # gold_calls: [{func_name: {param: [allowed values...]}}, ...]; empty for irrelevance.
    gold_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def function_names(self) -> list[str]:
        return [f.get("name", "") for f in self.functions]


def _prompt_from_question(question: Any) -> str:
    """BFCL ``question`` is [[{role,content}, ...]]; flatten the user turns."""
    parts: list[str] = []
    if isinstance(question, list):
        for turn in question:
            turns = turn if isinstance(turn, list) else [turn]
            for msg in turns:
                if isinstance(msg, dict) and msg.get("content"):
                    parts.append(str(msg["content"]))
    return "\n".join(parts)


def load_bfcl(cases_path: str | Path, answers_path: str | Path | None = None) -> list[BFCLCase]:
    """Load a BFCL category file, joining ground-truth answers by case id if given."""
    cases_path = Path(cases_path)
    answers: dict[str, list[dict[str, Any]]] = {}
    if answers_path is None:
        # convention: possible_answer/<same-filename> next to the cases file
        guess = cases_path.parent / "possible_answer" / cases_path.name
        if guess.exists():
            answers_path = guess
    if answers_path is not None and Path(answers_path).exists():
        for line in Path(answers_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers[obj["id"]] = obj.get("ground_truth", [])

    cases: list[BFCLCase] = []
    for line in cases_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = obj["id"]
        cases.append(BFCLCase(
            case_id=cid,
            prompt=_prompt_from_question(obj.get("question")),
            functions=obj.get("function", []),
            gold_calls=answers.get(cid, []),
        ))
    return cases


def _registry(case: BFCLCase) -> list[ToolSpec]:
    # BFCL "parameters" is already JSON-schema-shaped ({type,properties,required}),
    # which is exactly what the M2 schema validator consumes.
    return [
        ToolSpec(
            name=f.get("name", ""),
            capability=CanonicalCapability.UNKNOWN,  # arbitrary API fns; M2/M3 use name+schema
            schema=f.get("parameters"),
        )
        for f in case.functions
    ]


def _gold_args(spec: dict[str, Any]) -> dict[str, Any]:
    """First allowed value per parameter, dropping optionals whose only value is ''."""
    args: dict[str, Any] = {}
    for param, allowed in spec.items():
        if not isinstance(allowed, list) or not allowed:
            continue
        chosen = next((v for v in allowed if v != ""), allowed[0])
        if chosen == "" and len(allowed) == 1:
            continue  # optional-absent
        args[param] = chosen
    return args


def _resolve_name(case: BFCLCase, name: str) -> str:
    """Map a gold function name onto the registry, tolerating BFCL's occasional
    namespace drop (gold ``find_closest`` vs registry ``restaurant_search.find_closest``)."""
    names = case.function_names
    if name in names:
        return name
    suffix = [n for n in names if n.endswith("." + name) or n.endswith("_" + name)]
    return suffix[0] if len(suffix) == 1 else name


def _first_gold(case: BFCLCase) -> tuple[str, dict[str, Any]] | None:
    for entry in case.gold_calls:
        for name, spec in entry.items():
            return _resolve_name(case, name), _gold_args(spec if isinstance(spec, dict) else {})
    return None


def _session(case: BFCLCase, calls: list[ToolCallEvent], *, tag: str) -> NormalizedSession:
    events: list[NormalizedEvent] = [
        NormalizedEvent(
            event_id=f"{case.case_id}:0", session_id=case.case_id, seq=0,
            type=EventType.USER_MESSAGE, text=case.prompt,
        )
    ]
    seq = 1
    for call in calls:
        call.seq = seq
        events.append(NormalizedEvent(
            event_id=f"{case.case_id}:{seq}", session_id=case.case_id, seq=seq,
            type=EventType.TOOL_CALL, tool_call=call,
        ))
        seq += 1
        # attach a benign ok-result so result-driven detectors see a completed call
        events.append(NormalizedEvent(
            event_id=f"{case.case_id}:{seq}", session_id=case.case_id, seq=seq,
            type=EventType.TOOL_RESULT,
            tool_result=ToolResult(call_id=call.call_id, status="ok", content=""),
        ))
        seq += 1
    return build_session(
        session_id=case.case_id, adapter="bfcl",
        task=TaskSpec(task_id=case.case_id, prompt=case.prompt),
        events=events, available_tools=_registry(case),
        stop_reason="completed", metadata={"benchmark": "bfcl", "arm": tag},
    )


def _call(case: BFCLCase, name: str, args: dict[str, Any]) -> ToolCallEvent:
    return ToolCallEvent(
        call_id=f"{case.case_id}:call", session_id=case.case_id, seq=1,
        tool_name=name, raw_tool_name=name,
        capability=CanonicalCapability.UNKNOWN, arguments=args, adapter="bfcl",
    )


def session_correct(case: BFCLCase) -> NormalizedSession:
    """The agent makes the gold call (or no call for irrelevance)."""
    gold = _first_gold(case)
    calls = [_call(case, gold[0], gold[1])] if gold else []
    return _session(case, calls, tag="correct")


def session_hallucinated(case: BFCLCase) -> NormalizedSession:
    """The agent calls a function absent from the advertised registry."""
    gold = _first_gold(case)
    name = (gold[0] if gold else (case.function_names[0] if case.function_names else "do_thing"))
    args = gold[1] if gold else {}
    return _session(case, [_call(case, f"{name}__nonexistent", args)], tag="hallucinated")


def session_wrong_args(case: BFCLCase) -> NormalizedSession | None:
    """The agent calls the gold function but drops a required argument.

    Returns None when the gold function has no required arguments to drop (the
    perturbation would be a no-op and can't serve as a positive example).
    """
    gold = _first_gold(case)
    if gold is None:
        return None
    name, args = gold
    spec = next((f for f in case.functions if f.get("name") == name), None)
    required = (spec or {}).get("parameters", {}).get("required", []) if spec else []
    droppable = [r for r in required if r in args]
    if not droppable:
        return None
    broken = {k: v for k, v in args.items() if k != droppable[0]}
    return _session(case, [_call(case, name, broken)], tag="wrong_args")
