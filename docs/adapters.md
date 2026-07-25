# Adapters

An adapter is the *only* code that touches a CLI's native trace format. Each one normalizes a run into a `NormalizedSession` — an ordered event stream with tool calls and results correlated by `call_id`, tool names collapsed onto a canonical capability taxonomy. Every detector, the scoring engine, and both report formats operate only on that normalized model, so nothing downstream needs to know which agent produced the run.

```
adapters/  ->  NormalizedSession  ->  detectors/ (M1..M8)  ->  scoring/  ->  report/
```

## Shipped adapters

| Adapter | Format | Notes |
|---------|--------|-------|
| `generic` | Plain JSON tool-trace | Used by injected-failure test agents and benchmark converters; the shape `run` falls back to when nothing else matches |
| `claude-code` | SDK `stream-json` + OTEL sidecar | Two sub-modes in one adapter |
| `cursor` | `stream-json` | |
| `codex` | `exec --json` + session rollout | |

Gemini has no adapter yet — auth-blocked, deferred. Its adapter and `live` profile slot in the same way as the shipped three; see [CONTRIBUTING.md](../CONTRIBUTING.md#how-to-add-an-adapter) for the walkthrough.

## Auto-detection vs `--adapter`

Every adapter implements `sniff(source) -> float`, a cheap 0–1 confidence score for "can I parse this." `run`/`live`/`compare` pick the highest-scoring adapter automatically; pass `--adapter NAME` to force one (needed if a trace's shape is ambiguous, or for the `generic` fallback).

## Canonical capability taxonomy

Detectors reason about `CanonicalCapability`, never a CLI's raw tool name — this is what lets the same detector logic work unmodified across Claude Code, Cursor, Codex, and (eventually) Gemini. Adapters are responsible for mapping their raw tool name (+ arguments) onto one of these:

| Capability | Meaning |
|------------|---------|
| `FILE_READ` | Read a file's contents |
| `FILE_WRITE` | Create or overwrite a file |
| `FILE_EDIT` | Modify part of an existing file |
| `FILE_SEARCH` | Find files by name/glob |
| `CONTENT_SEARCH` | Search file contents (grep/ripgrep-style) |
| `SHELL_EXEC` | A shell command that didn't classify as test/build/lint |
| `TEST_RUN` | A shell command recognized as running tests |
| `BUILD_RUN` | A shell command recognized as a build |
| `LINT_RUN` | A shell command recognized as linting/type-checking |
| `WEB_FETCH` | Fetch a URL |
| `WEB_SEARCH` | Web search |
| `TASK_MGMT` | Todo/plan-tracking tools |
| `SUBAGENT` | Spawn a nested agent |
| `MCP_TOOL` | A dynamically-registered MCP tool |
| `UNKNOWN` | Couldn't classify |

`FILE_WRITE`/`FILE_EDIT` are the *mutating* capabilities (used by M5/M6 to reason about repo state); `TEST_RUN`/`BUILD_RUN`/`LINT_RUN` are the *verification* capabilities (used by M6).

## Shell command classification

Raw shell invocations are refined from `SHELL_EXEC` into `TEST_RUN`/`LINT_RUN`/`BUILD_RUN` by pattern-matching the command string against curated regexes for common toolchains — `pytest`, `npm test`, `go test`, `cargo test`, `rspec`, and friends for tests; `eslint`, `ruff`, `mypy`, `clippy`, `rubocop` for lint; `npm run build`, `go build`, `make`, `cmake`, `mvn package` for builds. Test is checked before build (e.g. `npm test` also looks build-ish), and lint before build for the same reason (`tsc --noEmit`). See [`core/capability.py`](../toolharness/core/capability.py) for the exact pattern lists.

## Graceful degradation

Not every CLI's trace carries every piece of data a detector wants:

- **No tool registry captured** → M3 (hallucinated call) falls back to result-signal only (`error_class == "UNKNOWN_TOOL"`) instead of registry membership.
- **No argument data** → M2 (wrong args) degrades to whatever schema/result signal is available.

Adapters are expected to leave these fields empty rather than fabricate data to fill the gap — a detector with less to work with should score with lower confidence, not confidently wrong.
