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
| M1 | Wrong tool selected | heuristic anti-patterns + LLM-judge |
| M2 | Wrong/malformed arguments | deterministic (schema + result error) |
| M3 | Hallucinated tool call | deterministic (registry membership) |
| M4 | Ignored tool output | heuristic anchors + LLM-judge |
| M5 | Redundant/repeated calls | deterministic (state tracker) |
| M6 | Missing verification step | rule + LLM-judge on "warranted" |
| M7 | Premature stop | reference rule + heuristic + LLM-judge |
| M8 | Unsafe/destructive call w/o justification | deterministic ruleset + judge (downgrade-only) |

Each mode is scored **0–100**; the primary output is the 8-mode vector plus a
weighted **composite** (safety weighted highest). Two levels of detail:
per-tool-call findings and a per-session summary. Reports emit as JSON (for CI)
and — from the M4 milestone — an HTML dashboard.

## Status

Milestones **M0**–**M3** are implemented: data model + generic adapter (M0);
deterministic detectors, scoring, JSON report, golden fixtures (M1);
injected-failure test agents + end-to-end integration (M2); the **LLM-judge layer
and the three hybrid modes** M1/M4/M7 (M3). All eight modes now have detectors and
golden fixtures, and the controlled injected-failure set is caught at
precision/recall 1.0. See `agent_eval_harness/` and `tests/`.

### The judge

The hybrid modes have deterministic anchors that catch the clear cases with no
model call; ambiguous cases escalate to a **provider-agnostic LLM judge**. The
judge is deliberately independent of the agents under test (never Claude, GPT, or
Gemini) to avoid self-preference bias. One `OpenAICompatibleJudge` covers Groq,
OpenRouter, NVIDIA NIM, and local Ollama — a provider is just
`(base_url, model, api_key_env)`; the default is **Groq + Kimi K2**. Verdicts are
`temperature=0` + seeded and cached on disk, so re-scoring a session is
reproducible and free.

```bash
# heuristic-only (no network, default):
evalharness run tests/fixtures/m1_wrong_tool_fail.json
# with the judge escalation path (needs GROQ_API_KEY):
export GROQ_API_KEY=...
evalharness run <trace.json> --judge groq --judge-cache .judge_cache
```

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
evalharness run tests/fixtures/m8_unsafe_call_fail.json
```

## License

Apache-2.0
