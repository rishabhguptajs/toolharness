# agent-eval-harness

An open-source harness that scores **tool-call reliability** for agentic coding
CLIs — not general output quality, but whether the agent makes correct, safe,
well-justified tool calls during a coding task.

Targets (via a pluggable adapter per agent): **Claude Code, Cursor CLI, Codex
CLI, Gemini CLI**. One core scoring engine consumes a normalized event stream
produced by each adapter.

## Failure modes (v1)

| # | Mode | Detection |
|---|------|-----------|
| M1 | Wrong tool selected | LLM-judge + heuristic *(M3 milestone)* |
| M2 | Wrong/malformed arguments | deterministic (schema + result error) |
| M3 | Hallucinated tool call | deterministic (registry membership) |
| M4 | Ignored tool output | heuristic + LLM-judge *(M3 milestone)* |
| M5 | Redundant/repeated calls | deterministic (state tracker) |
| M6 | Missing verification step | rule + LLM-judge on "warranted" |
| M7 | Premature stop | reference rule + LLM-judge *(M3 milestone)* |
| M8 | Unsafe/destructive call w/o justification | deterministic ruleset + judge |

Each mode is scored **0–100**; the primary output is the 8-mode vector plus a
weighted **composite** (safety weighted highest). Two levels of detail:
per-tool-call findings and a per-session summary. Reports emit as JSON (for CI)
and — from the M4 milestone — an HTML dashboard.

## Status

Milestones **M0** (data model + generic adapter) and **M1** (deterministic
detectors, scoring, JSON report, golden fixtures) are implemented. See
`agent_eval_harness/` and `tests/`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
evalharness run tests/fixtures/m8_unsafe_call_fail.json
```

## License

Apache-2.0
