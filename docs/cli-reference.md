# CLI reference

`toolharness` has five subcommands: `run`, `live`, `suite`, `compare`, `benchmark`. All of them print a human-readable score table by default; `run`/`live`/`suite` also support `--json`, `--html`, and CI gating.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Ran clean, no gate breached |
| `1` | Ran clean, but a `--fail-under` / `--fail-under-mode` / `--strict` gate failed |
| `2` | Setup problem — bad arguments, or the judge's API key env var isn't set |

A `2` is always something you fix before rerunning (a missing `GROQ_API_KEY`, an unknown `--adapter`); a `1` is a real score result that just didn't clear the bar you set.

## `toolharness run`

Score a single pre-captured trace.

```bash
toolharness run <trace> [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` (positional) | — | Path to a run trace — generic JSON, or a real CLI capture (Claude Code stream-json/OTEL, Cursor stream-json, Codex `exec --json`/rollout) |
| `--adapter NAME` | auto-detect | Force a specific adapter instead of sniffing the format. One of the names in [Adapters](adapters.md) |
| `--task PATH` | none | Attach a [task spec](task-specs.md) for reference-based scoring |
| `--judge PROVIDER` | none | Escalate ambiguous cases to an LLM judge — see [Judge](judge.md) |
| `--judge-cache DIR` | none | Cache judge verdicts on disk for reproducible, free re-scoring |
| `--json PATH` | none | Write the [JSON report](reports.md) |
| `--html PATH` | none | Write the self-contained HTML dashboard |
| `--print-json` | off | Print the full JSON report to stdout instead of the score table |
| `--fail-under N` | none | Exit `1` if the composite score is below `N` |
| `--fail-under-mode MODE=N` | none | Exit `1` if a specific mode's score is below `N` (repeatable). `MODE` accepts the enum name (`UNSAFE_CALL`) or value (`unsafe_call`), any case |

```bash
toolharness run tests/fixtures/m8_unsafe_call_fail.json
toolharness run captured.jsonl --adapter codex --task examples/add_power_function.yaml \
    --judge groq --judge-cache .judge_cache --json report.json --html report.html \
    --fail-under 70 --fail-under-mode UNSAFE_CALL=90
```

## `toolharness live`

Invoke a real agent CLI on a task repo, capture its trace, and score it — the whole pipeline in one command.

```bash
toolharness live --adapter NAME --task spec.yaml [options]
```

Only `claude-code`, `cursor`, and `codex` have live invocation profiles (Gemini is deferred — no adapter yet). Each profile is `[binary, *pre_args, prompt, *post_args]`; see `PROFILES` in [`runner/live.py`](../toolharness/runner/live.py).

| Flag | Default | Meaning |
|------|---------|---------|
| `--adapter NAME` | *required* | `claude-code`, `cursor`, or `codex` |
| `--task PATH` | *required* | Task spec YAML — supplies the prompt and (optionally) the repo and gold data |
| `--repo DIR` | spec's `repo.path` | Override the repo the agent runs against |
| `--in-place` | off | Run in the real repo instead of a throwaway sandbox. Prints a warning; the agent's own permission prompts are your only safety net |
| `--timeout N` | `600` | Wall-clock timeout in seconds |
| `--save-trace PATH` | temp file | Write the raw captured trace here too |
| `--keep-workdir` | off | Don't delete the sandbox after the run (for inspection) |
| `--agent-arg ARG` | none | Extra flag appended to the agent's command line (repeatable) |
| `--judge`, `--judge-cache`, `--json`, `--html`, `--print-json`, `--fail-under`, `--fail-under-mode` | — | Same as `run` |

**Safe by default:** the task repo is copied into a temp-dir sandbox (`.venv`/`node_modules`/caches skipped, `.git` kept) before the agent touches anything, and the run is bounded by `--timeout`. Autonomy flags (auto-accept edits, skip permission prompts) are granted to the agent **only inside the sandbox** — an `--in-place` run omits them, so the agent's normal prompts still gate edits to your real repo.

```bash
toolharness live --adapter claude-code --task examples/add_power_function.yaml \
    --repo ~/code/sample_repo --json report.json --html report.html \
    --fail-under 70 --fail-under-mode UNSAFE_CALL=90
```

## `toolharness suite`

Run the bundled task suite end to end against one adapter — no `--task`, no `--repo`, it does the rest.

```bash
toolharness suite --adapter NAME [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--adapter NAME` | *required* | `claude-code`, `cursor`, or `codex` |
| `--tasks DIR` | bundled suite | Point at your own task suite instead |
| `--task-id ID` | all | Run only this task (repeatable) |
| `--list` | off | Print tasks (id, reference/reference-free, prompt preview) and exit |
| `--timeout N` | `600` | Per-task wall-clock timeout |
| `--keep-workdir` | off | Don't delete per-task sandboxes |
| `--agent-arg ARG` | none | Extra flag appended to every agent invocation (repeatable) |
| `--judge`, `--judge-cache` | — | Same as `run` |
| `--json PATH` | none | Write the aggregate suite report (per-task results + mode means) |
| `--html PATH` | none | Write a per-task comparison dashboard |
| `--fail-under N` | none | Exit `1` if the aggregate composite is below `N` |
| `--fail-under-mode MODE=N` | none | Exit `1` if a mode's mean across tasks is below `N` (repeatable) |
| `--strict` | off | Also exit `1` if any task errored (e.g. the CLI binary is missing) — otherwise a task error is reported but not fatal |

A task is just a folder under `toolharness/suite/tasks/<name>/` with a `task.yaml` ([task spec](task-specs.md) format) and a `seed/` starting repo. Three ship today: `add-power`, `implement-gcd`, `fix-fizzbuzz`.

```bash
toolharness suite --adapter claude-code --list
toolharness suite --adapter claude-code --json suite.json --html suite.html \
    --fail-under 70 --fail-under-mode UNSAFE_CALL=90 --strict
```

## `toolharness compare`

Score several traces into one HTML dashboard — agent-A-vs-B, or before/after.

```bash
toolharness compare <trace> <trace> [...] --html OUT [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `traces` (positional) | — | Two or more trace paths |
| `--html PATH` | *required* | Where to write the comparison dashboard |
| `--task PATH` | none | Attach the same task spec to every trace |
| `--adapter`, `--judge`, `--judge-cache` | — | Same as `run`, applied to every trace |

```bash
toolharness compare claude-run.json codex-run.json --html compare.html
```

## `toolharness benchmark`

Validate detectors against a public tool-calling benchmark. Currently supports `bfcl` (Berkeley Function-Calling Leaderboard). See [BENCHMARKS.md](../BENCHMARKS.md) for methodology and published results.

```bash
toolharness benchmark bfcl --data <dir> [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `dataset` (positional) | — | `bfcl` (only option today) |
| `--data DIR` | *required* | Path to the downloaded BFCL data (see BENCHMARKS.md for the layout) |
| `--limit N` | none | Cap cases per category (useful for a quick smoke run) |
| `--json PATH` | none | Write the benchmark report |
| `--adapter`, `--judge`, `--judge-cache` | — | `--judge` enables the judge-relevance κ measurement; without it only deterministic P/R/F1 runs |

```bash
toolharness benchmark bfcl --data ./bfcl_data --limit 50
GROQ_API_KEY=... toolharness benchmark bfcl --data ./bfcl_data --judge groq --json bfcl_report.json
```
