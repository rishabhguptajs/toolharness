# toolharness

[![PyPI](https://img.shields.io/pypi/v/toolharness.svg)](https://pypi.org/project/toolharness/)
[![Python](https://img.shields.io/pypi/pyversions/toolharness.svg)](https://pypi.org/project/toolharness/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

All milestones **M0**–**M7** are implemented: data model + generic adapter (M0);
deterministic detectors, scoring, JSON report, golden fixtures (M1);
injected-failure test agents + end-to-end integration (M2); the **LLM-judge layer
and the three hybrid modes** M1/M4/M7 (M3); a **self-contained HTML dashboard**
(M4); **real CLI adapters** for Claude Code (SDK stream-json + OTEL), Cursor
(stream-json), and Codex (`exec --json` + session rollout), with parser tests over
captured real runs (M5); **BFCL benchmark validation** with per-detector P/R/F1 +
Cohen's κ (M6, see [BENCHMARKS.md](BENCHMARKS.md)); and the **runner + CI gate +
OSS packaging** (M7). All eight modes have detectors and golden fixtures, and the
controlled injected-failure set is caught at precision/recall 1.0. See
`toolharness/` and `tests/`.

> Gemini support is deferred (auth-blocked); its adapter and live profile slot in
> the same way the shipped three do — see [CONTRIBUTING.md](CONTRIBUTING.md).

### HTML dashboard

`toolharness run <trace> --html report.html` writes a single self-contained file
(inline CSS/JS, no external requests) with the 8-mode score bars + an SVG radar, a
tool-call timeline color-coded by the modes each call triggered, click-to-expand
drill-down (reasoning, arguments, result, and every finding's rationale + evidence
trail), and mode/verdict filters. `toolharness compare a.json b.json --html
cmp.html` renders several sessions into one page for agent-A-vs-B comparison.

### The judge

The hybrid modes have deterministic anchors that catch the clear cases with no
model call; ambiguous cases escalate to a **provider-agnostic LLM judge**. The
judge is deliberately independent of the agents under test (never Claude, GPT, or
Gemini) to avoid self-preference bias. One `OpenAICompatibleJudge` covers Groq,
OpenRouter, NVIDIA NIM, and local Ollama — a provider is just
`(base_url, model, api_key_env)`; the default is **Groq + Qwen3.6-27b**. Verdicts are
`temperature=0` + seeded and cached on disk, so re-scoring a session is
reproducible and free.

```bash
# heuristic-only (no network, default):
toolharness run tests/fixtures/m1_wrong_tool_fail.json
# with the judge escalation path (needs GROQ_API_KEY):
export GROQ_API_KEY=...
toolharness run <trace.json> --judge groq --judge-cache .judge_cache
```

## Install

```bash
pip install toolharness
```

That gives you the `toolharness` command. The only dependencies are PyYAML and
Jinja2 — the judge talks HTTP over the standard library, so there is no heavy SDK
to pull in.

### Bring your own judge key

The harness ships **no API key and no hosted service**. Scoring is heuristic-only
by default and makes no network calls at all — everything in the quickstart below
runs offline and free.

The optional LLM-judge escalation reads *your* key from *your* environment
(`GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `NVIDIA_API_KEY`, depending on
`--judge`). The key is only ever sent to the provider you select, and is never
written to the judge cache, the JSON report, or the HTML dashboard. If the
variable is unset the run stops with a plain error rather than silently
downgrading. `--judge ollama` points at a local model and needs no key at all.

### From source

```bash
git clone https://github.com/rishabhguptajs/toolharness
cd toolharness
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Quickstart

```bash
toolharness run tests/fixtures/m8_unsafe_call_fail.json
```

## Benchmark a CLI in one command

The harness ships its own **task suite** — a curated set of coding tasks, each with
a starting repo and gold data baked in. Point an adapter at it and it does the rest:
for every task it copies the seed repo into a throwaway sandbox, runs the agent,
scores its tool-call reliability, and aggregates.

```bash
toolharness suite --adapter claude-code
```

That's the whole thing — no task specs to write, no repo to supply. It runs each
bundled task, prints a per-task table plus an aggregate score vector, and (with
`--html`) drops a dashboard with every task side by side.

```bash
toolharness suite --adapter claude-code \
    --json suite.json --html suite.html \
    --fail-under 70 --fail-under-mode UNSAFE_CALL=90
```

Useful flags: `--list` (show tasks and exit), `--task-id add-power` (run a subset,
repeatable), `--tasks DIR` (point at your own suite instead of the bundled one),
`--strict` (fail if any task errored, e.g. the CLI binary is missing). Adding a
task is just dropping a folder under `toolharness/suite/tasks/<name>/` with a
`task.yaml` and a `seed/` repo. As with `live`, autonomy flags are granted only
inside the sandbox, and each task is bounded by `--timeout`.

## Running an agent on a task (live)

`toolharness live` invokes a real agent CLI on a task repo, captures its trace, and
scores it — the whole pipeline in one command. A **task spec** (YAML) supplies the
prompt, the repo, and any optional gold data (`expected_capabilities`, `subgoals`,
`required_verification`, `allowed_destructive`) that switches detectors into
reference-based mode. See [`examples/`](examples/) for a reference-based and a
reference-free spec.

```bash
toolharness live --adapter claude-code --task examples/add_power_function.yaml \
    --repo ~/code/sample_repo \
    --json report.json --html report.html \
    --fail-under 70 --fail-under-mode UNSAFE_CALL=90
```

**Safe by default:** the task repo is copied into a throwaway temp-dir sandbox
(git history kept, `.venv`/`node_modules`/caches skipped) so the agent never
touches your real tree, and every run is bounded by `--timeout` (default 600s).
`--in-place` opts out with a warning; `--keep-workdir` preserves the sandbox for
inspection. Only `claude-code`, `cursor`, and `codex` have live profiles today
(Gemini is deferred); each is `[binary, *pre_args, prompt, *post_args]` in
`runner/live.py`.

To score a **pre-captured** trace instead of invoking a CLI, use `run` (attach a
spec with `--task` for reference-based scoring):

```bash
toolharness run captured.jsonl --adapter codex --task examples/add_power_function.yaml
```

## CI gate

Both `run` and `live` gate on score: `--fail-under N` on the composite, and
repeatable `--fail-under-mode MODE=N` for a specific dimension (e.g.
`--fail-under-mode UNSAFE_CALL=90`). The command exits non-zero on any breach, with
one `FAIL:` line per gate on stderr — drop it straight into a CI step. Modes that
are *not applicable* to a run (no opportunities) are skipped, never failed. Mode
names accept either the enum name (`UNSAFE_CALL`) or value (`unsafe_call`).

The project's own CI (`.github/workflows/ci.yml`) runs `ruff`, `mypy`, and
`pytest` across Python 3.10–3.12 plus a smoke test of the safety gate.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), including a walkthrough of **how to add an
adapter** for a new agent CLI.

## License

Apache-2.0 — see [LICENSE](LICENSE).
