"""M3 judge layer: stub, caching/reproducibility, provider factory, the
OpenAI-compatible HTTP client (mocked), and the hybrid detectors' judge paths.
"""

from __future__ import annotations

import json

import pytest

from agent_eval_harness.adapters.base import RunSource
from agent_eval_harness.adapters.generic import GenericToolTraceAdapter
from agent_eval_harness.core.findings import FailureMode
from agent_eval_harness.detectors import (
    IgnoredOutputDetector,
    PrematureStopDetector,
    UnsafeCallDetector,
    WrongToolDetector,
)
from agent_eval_harness.detectors.base import DetectorContext
from agent_eval_harness.detectors.judge import (
    PROVIDERS,
    CachingJudge,
    JudgeError,
    JudgeRequest,
    JudgeVerdict,
    OpenAICompatibleJudge,
    StubJudge,
    build_judge,
    safe_ask,
)

_ADAPTER = GenericToolTraceAdapter()


def sess(trace: dict):
    return _ADAPTER.parse(RunSource(kind="generic", data=trace))


def req(kind: str = "m1", user: str = "u") -> JudgeRequest:
    return JudgeRequest(kind=kind, system="s", user=user)


# --- stub -------------------------------------------------------------------------


def test_stub_default_and_policy():
    stub = StubJudge(default=JudgeVerdict("pass", 0.9, "ok"))
    assert stub.ask(req()).verdict == "pass"

    def policy(r: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict("fail", 0.7, f"saw {r.kind}")

    stub2 = StubJudge(policy=policy)
    v = stub2.ask(req("wrong_tool"))
    assert v.verdict == "fail" and "wrong_tool" in v.rationale
    assert len(stub2.calls) == 1


# --- caching / reproducibility ----------------------------------------------------


class _Counting:
    model = "counter"

    def __init__(self) -> None:
        self.n = 0

    def ask(self, r: JudgeRequest) -> JudgeVerdict:
        self.n += 1
        return JudgeVerdict("fail", 0.9, "rationale", model=self.model)


def test_caching_judge_serves_second_call_from_disk(tmp_path):
    inner = _Counting()
    cj = CachingJudge(inner, tmp_path)
    v1 = cj.ask(req("m4", "same"))
    v2 = cj.ask(req("m4", "same"))
    assert inner.n == 1, "second identical ask must hit the cache, not the backend"
    assert v2.cached and not v1.cached
    assert (v1.verdict, v1.confidence, v1.rationale) == (v2.verdict, v2.confidence, v2.rationale)

    # A fresh wrapper over the same dir still reads the persisted verdict.
    cj2 = CachingJudge(inner, tmp_path)
    v3 = cj2.ask(req("m4", "same"))
    assert inner.n == 1 and v3.cached and v3.verdict == "fail"

    # A different prompt is a cache miss.
    cj.ask(req("m4", "different"))
    assert inner.n == 2


# --- factory ----------------------------------------------------------------------


def test_build_judge_none_and_stub(tmp_path):
    assert build_judge(None) is None
    assert build_judge("none") is None
    j = build_judge("stub")
    assert j is not None and hasattr(j, "ask")
    cached = build_judge("stub", cache_dir=tmp_path)
    assert isinstance(cached, CachingJudge)


def test_build_judge_unknown_provider_raises():
    with pytest.raises(JudgeError):
        build_judge("does-not-exist")


def test_build_judge_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(JudgeError):
        build_judge("groq")


# --- OpenAI-compatible HTTP client (mocked) ---------------------------------------


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_compatible_judge_builds_request_and_parses(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        content = json.dumps({"verdict": "fail", "confidence": 0.83, "rationale": "bad tool"})
        return _FakeResp({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(
        "agent_eval_harness.detectors.judge.urllib.request.urlopen", fake_urlopen
    )
    judge = OpenAICompatibleJudge(PROVIDERS["groq"])
    verdict = judge.ask(req("m1_wrong_tool", "judge this"))

    assert verdict.verdict == "fail" and verdict.confidence == 0.83
    assert captured["url"].endswith("/chat/completions")
    # Authorization header present (case-insensitive key).
    assert any(k.lower() == "authorization" for k in captured["headers"])
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["model"] == "moonshotai/kimi-k2-instruct"
    assert captured["body"]["messages"][-1]["content"] == "judge this"


def test_openai_compatible_judge_raises_on_bad_payload(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")

    def fake_urlopen(request, timeout=None):
        return _FakeResp({"unexpected": True})

    monkeypatch.setattr(
        "agent_eval_harness.detectors.judge.urllib.request.urlopen", fake_urlopen
    )
    with pytest.raises(JudgeError):
        OpenAICompatibleJudge(PROVIDERS["groq"]).ask(req())


# --- safe_ask degradation ---------------------------------------------------------


class _Raiser:
    model = "raiser"

    def ask(self, r: JudgeRequest) -> JudgeVerdict:
        raise JudgeError("backend down")


def test_safe_ask_swallows_backend_failure():
    assert safe_ask(None, req()) is None
    assert safe_ask(_Raiser(), req()) is None


# --- hybrid detectors: judge escalation path --------------------------------------

_TOOLS = [
    {"name": "read_file", "capability": "FILE_READ",
     "schema": {"required": ["path"]}},
    {"name": "edit_file", "capability": "FILE_EDIT",
     "schema": {"required": ["path", "old", "new"]}},
    {"name": "run_command", "capability": "SHELL_EXEC",
     "schema": {"required": ["command"]}},
]


def _kind_policy(fail_kinds: set[str], pass_kinds: set[str] = frozenset()):
    def policy(r: JudgeRequest) -> JudgeVerdict:
        if r.kind in fail_kinds:
            return JudgeVerdict("fail", 0.7, "judged fail")
        if r.kind in pass_kinds:
            return JudgeVerdict("pass", 0.7, "judged justified")
        return JudgeVerdict("na", 0.5, "no opinion")

    return StubJudge(policy=policy)


def test_m1_escalates_non_antipattern_shell_to_judge():
    # `head` overlaps FILE_READ but isn't a hard anti-pattern -> judge decides.
    trace = {
        "adapter": "generic",
        "task": {"task_id": "t", "prompt": "inspect the file"},
        "available_tools": _TOOLS,
        "events": [
            {"type": "user_message", "text": "inspect"},
            {"type": "agent_message", "text": "peeking at the top"},
            {"type": "tool_call", "call_id": "c1", "tool_name": "run_command",
             "arguments": {"command": "head src/app.py"}},
            {"type": "tool_result", "call_id": "c1", "status": "ok", "content": "..."},
            {"type": "agent_stop", "text": "done"},
        ],
    }
    judge = _kind_policy({FailureMode.WRONG_TOOL.value})
    res = WrongToolDetector().evaluate(sess(trace), DetectorContext(judge=judge))
    assert [f.verdict for f in res.findings] == ["fail"]
    assert res.findings[0].llm_used is True
    # Without a judge, the same trace produces no finding (no hard anti-pattern).
    res_none = WrongToolDetector().evaluate(sess(trace), DetectorContext(judge=None))
    assert res_none.findings == []


def test_m4_escalates_midrun_error_to_judge():
    # An errored call that is neither the last nor an identical retry -> judge path.
    trace = {
        "adapter": "generic",
        "task": {"task_id": "t", "prompt": "fix it"},
        "available_tools": _TOOLS,
        "events": [
            {"type": "tool_call", "call_id": "c1", "tool_name": "read_file",
             "arguments": {"path": "missing.py"}},
            {"type": "tool_result", "call_id": "c1", "status": "error", "is_error": True,
             "error_class": "ENOENT", "content": "no such file"},
            {"type": "tool_call", "call_id": "c2", "tool_name": "read_file",
             "arguments": {"path": "other.py"}},
            {"type": "tool_result", "call_id": "c2", "status": "ok", "content": "ok"},
            {"type": "agent_stop", "text": "all good"},
        ],
    }
    judge = _kind_policy({FailureMode.IGNORED_OUTPUT.value})
    res = IgnoredOutputDetector().evaluate(sess(trace), DetectorContext(judge=judge))
    assert any(f.llm_used for f in res.findings)


def test_m7_escalates_ambiguous_stop_to_judge():
    trace = {
        "adapter": "generic",
        "task": {"task_id": "t", "prompt": "do the thing"},
        "available_tools": _TOOLS,
        "stop_reason": "completed",
        "events": [
            {"type": "tool_call", "call_id": "c1", "tool_name": "read_file",
             "arguments": {"path": "a.py"}},
            {"type": "tool_result", "call_id": "c1", "status": "ok", "content": "x"},
            {"type": "agent_stop", "text": "Looks fine to me."},
        ],
    }
    judge = _kind_policy({FailureMode.PREMATURE_STOP.value})
    res = PrematureStopDetector().evaluate(sess(trace), DetectorContext(judge=judge))
    assert [f.verdict for f in res.findings] == ["fail"]
    assert res.findings[0].llm_used is True
    # No judge and no punt anchor -> no finding.
    res_none = PrematureStopDetector().evaluate(sess(trace), DetectorContext(judge=None))
    assert res_none.findings == []


def test_m8_judge_can_only_downgrade_not_suppress():
    trace = {
        "adapter": "generic",
        "task": {"task_id": "t", "prompt": "clean up the workspace"},
        "available_tools": _TOOLS,
        "events": [
            {"type": "agent_message", "text": "removing scratch"},
            {"type": "tool_call", "call_id": "c1", "tool_name": "run_command",
             "arguments": {"command": "rm -rf /tmp/scratch"}},
            {"type": "tool_result", "call_id": "c1", "status": "ok", "content": "(ok)"},
            {"type": "agent_stop", "text": "done"},
        ],
    }
    # Judge says justified -> downgrade fail to warn, but the finding survives.
    judge = _kind_policy(set(), {FailureMode.UNSAFE_CALL.value})
    res = UnsafeCallDetector().evaluate(sess(trace), DetectorContext(judge=judge))
    assert [f.verdict for f in res.findings] == ["warn"]
    assert res.findings[0].llm_used is True

    # No judge -> deterministic fail stands.
    res_none = UnsafeCallDetector().evaluate(sess(trace), DetectorContext(judge=None))
    assert [f.verdict for f in res_none.findings] == ["fail"]
