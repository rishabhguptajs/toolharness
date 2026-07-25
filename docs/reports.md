# Reports

Both `run` and `live` can emit a machine-readable JSON report and/or a self-contained HTML dashboard, in addition to the score table printed to stdout.

## JSON report

`--json PATH` writes a schema-versioned report; `--print-json` prints the same structure to stdout instead of the score table. Field ordering is stable so report diffs in CI are meaningful.

```jsonc
{
  "schema_version": "1.0",
  "session": {
    "session_id": "...",
    "adapter": "claude-code",
    "task_id": "add-power-function",
    "stop_reason": "completed",
    "n_tool_calls": 7,
    "composite": 88,
    "scores": {
      "wrong_tool":             {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 7, "n_findings": 0},
      "wrong_args":             {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 7, "n_findings": 0},
      "hallucinated":           {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 7, "n_findings": 0},
      "ignored_output":         {"score": null, "applicable": false, "evaluated": false},
      "redundant":              {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 7, "n_findings": 0},
      "missing_verification":   {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 1, "n_findings": 0},
      "premature_stop":         {"score": 100, "confidence": 1.0, "applicable": true,  "evaluated": true, "n_opportunities": 1, "n_findings": 0},
      "unsafe_call":            {"score": null, "applicable": false, "evaluated": false}
    }
  },
  "tool_calls": [
    {
      "call_id": "...",
      "seq": 0,
      "turn": 0,
      "tool_name": "Read",
      "raw_tool_name": "Read",
      "capability": "FILE_READ",
      "arguments": {"file_path": "calc.py"},
      "result": {"status": "ok", "is_error": false, "error_class": null, "exit_code": null},
      "findings": []
    }
  ],
  "session_findings": []
}
```

Modes always appear in a fixed order (M1→M8). A mode that wasn't applicable (no opportunities — e.g. no destructive candidates in the whole session) reports `"score": null, "applicable": false, "evaluated": false` rather than a fabricated 100.

### Findings

Each `Finding` (whether attached to a specific tool call or session-level, in `session_findings`) has:

| Field | Meaning |
|-------|---------|
| `mode` | Which of the 8 modes raised it |
| `verdict` | `"pass"` \| `"fail"` \| `"warn"` \| `"na"` — only `fail`/`warn` findings are ever emitted (a pass is the absence of a finding) |
| `severity` | 0–1 contribution to that mode's score penalty |
| `confidence` | 0–1; deterministic findings are typically `1.0`, LLM-derived findings are lower |
| `rationale` | Human-readable explanation |
| `target_call_id` / `target_seq` | Which call this finding is about (`null` for a session-level finding) |
| `evidence` | List of `{seq, call_id, note}` pointers back into the session — the audit trail |
| `llm_used` | Whether the judge was consulted for this specific finding |
| `detector_version` | For tracking detector changes across report versions |

### `suite` and `compare` output shapes

`suite --json` wraps the same per-task report dict inside an aggregate: `{"schema_version": 1, "adapter": ..., "aggregate": {"composite": ..., "mode_means": {...}}, "tasks": [{"task_id": ..., "error": ..., "report": {...}}, ...]}`. A task that errored (e.g. missing CLI binary) has `"report": null` and a non-null `"error"`.

`compare` doesn't have its own JSON report — it scores each trace independently (reuse `run --json` per trace if you want individual JSON alongside the comparison dashboard) and only emits the combined HTML.

## HTML dashboard

`--html PATH` writes a single file — inline CSS/JS, no external requests, safe to open straight from disk or attach to a CI artifact. It includes:

- the 8-mode score bars plus an SVG radar chart
- a tool-call timeline, color-coded by which modes each call triggered
- click-to-expand drill-down per call: reasoning, arguments, result, and every finding's rationale + evidence trail
- mode and verdict filters

`compare --html` and `suite --html` render multiple sessions into the same page, side by side, for agent-A-vs-B or task-by-task comparison.
