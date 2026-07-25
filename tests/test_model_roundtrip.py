"""M0 exit criterion: load & round-trip a normalized session; linking works."""

from __future__ import annotations

from toolharness.core.capability import CanonicalCapability
from toolharness.core.model import EventType


def test_clean_pass_parses(session_loader):
    s = session_loader("clean_pass")
    assert s.session_id == "clean_pass"
    assert s.adapter == "generic"
    assert s.stop_reason == "completed"
    assert len(s.tool_calls) == 3


def test_results_link_to_calls(session_loader):
    s = session_loader("clean_pass")
    for call in s.tool_calls:
        assert call.result is not None, f"{call.call_id} has no linked result"
        assert call.result.call_id == call.call_id


def test_preceding_reasoning_backfilled(session_loader):
    s = session_loader("clean_pass")
    first = s.tool_calls[0]
    assert first.preceding_reasoning == "I'll read the file first."


def test_capability_and_shell_classification(session_loader):
    s = session_loader("clean_pass")
    caps = [c.capability for c in s.tool_calls]
    assert caps[0] == CanonicalCapability.FILE_READ
    assert caps[1] == CanonicalCapability.FILE_EDIT
    # bash running pytest must be refined to TEST_RUN, not left as SHELL_EXEC
    assert caps[2] == CanonicalCapability.TEST_RUN


def test_seq_is_monotonic_and_events_present(session_loader):
    s = session_loader("clean_pass")
    seqs = [e.seq for e in s.events]
    assert seqs == sorted(seqs)
    assert s.events[0].type == EventType.USER_MESSAGE
    assert s.events[-1].type == EventType.AGENT_STOP


def test_available_tools_registry(session_loader):
    s = session_loader("clean_pass")
    assert s.available_tool_names == {"read_file", "edit_file", "bash"}
    spec = s.tool_spec("read_file")
    assert spec is not None and spec.schema["required"] == ["path"]
