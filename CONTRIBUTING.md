# Contributing

Thanks for your interest in the harness. It's small, dependency-light, and kept
`ruff` + `mypy` clean with a green `pytest` at every commit.

## Dev setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Before you push

The full green bar is three commands:

```bash
pytest -q
ruff check .
mypy evalharness
```

CI (`.github/workflows/ci.yml`) runs exactly these across Python 3.10–3.12, plus a
smoke test of the CLI safety gate. Keep new code typed and covered.

## Architecture in one paragraph

Adapters are the *only* code that touches a CLI's native output format. Each one
normalizes a run into a `NormalizedSession` (an ordered event stream with tool
calls and results correlated by `call_id`, tool names collapsed to a
`CanonicalCapability` taxonomy). Everything downstream — the eight failure-mode
detectors, the scoring engine, the JSON/HTML reports — operates only on that
normalized model, so it never needs to know which agent produced the run.

```
adapters/  ->  core/NormalizedSession  ->  detectors/ (M1..M8)  ->  scoring/  ->  report/
```

## How to add an adapter

Adding support for a new agent CLI (or a new trace format) means writing one
adapter. Use the shipped ones as templates — `adapters/cursor.py` is the smallest
end-to-end example; `adapters/claude_code.py` shows a two-sub-mode adapter (SDK +
OTEL).

1. **Implement the three-method protocol** (`adapters/base.py::Adapter`):

   ```python
   class MyAgentAdapter:
       name = "my-agent"

       def sniff(self, source: RunSource) -> float:
           # 0..1 confidence this adapter can parse `source`. Cheap checks only —
           # peek at the first few records (see adapters/_util.py::peek_jsonl) and
           # look for a shape only your CLI emits. Return 0.0 if it isn't yours.
           ...

       def parse(self, source: RunSource) -> NormalizedSession:
           # Walk the raw records in stream order and emit NormalizedEvents.
           # Assign `seq` by position (never wall-clock). Correlate each
           # TOOL_RESULT to its TOOL_CALL by a shared call_id — synthesize one
           # if the source lacks it. Finish with core.model.build_session(...).
           ...

       def canonicalize_tool(self, raw_name, args) -> CanonicalCapability:
           # Map the CLI's tool name to the canonical taxonomy. For shell tools,
           # route the command string through core.capability.classify_shell_command
           # so `pytest`/`npm test`/`git push -f` land on TEST_RUN/…/SHELL_EXEC.
           ...
   ```

2. **Register it** in `adapters/__init__.py`:

   ```python
   default_registry.register(MyAgentAdapter())
   ```

   The registry auto-selects the highest-`sniff` adapter; `--adapter my-agent`
   forces yours. Make sure your `sniff` returns `0.0` for *other* agents' shapes so
   you don't hijack their traces (see the Cursor-vs-Claude guard in
   `claude_code.py::sniff`).

3. **Handle graceful degradation.** If the format lacks a tool registry, leave
   `available_tools` empty — M3 falls back to result-signal only. If it omits
   arguments, M2 degrades. Both are expected; don't fabricate data to fill gaps.

4. **Add a captured fixture + a parser test.** Drop a small real capture under
   `tests/fixtures/real/<agent>/` and assert the parse in
   `tests/test_adapters_real.py`: the session's tool calls, their capabilities,
   result correlation, and `stop_reason`. This is the regression backbone for the
   adapter.

5. **(Optional) add a live invocation profile** in `runner/live.py::PROFILES` so
   `evalharness live --adapter my-agent` can drive the real CLI. A profile is just
   `(binary, pre_args, post_args)` assembled as `[binary, *pre, prompt, *post]`.

## Adding a detector or failure mode

The eight v1 modes are locked, but the `Detector` protocol
(`detectors/base.py`) is open. A detector declares whether it needs a reference
(`needs_reference`), can run reference-free (`works_reference_free`), and whether
it calls the judge (`uses_llm`); it returns `DetectorResult` with per-target
`Finding`s. Deterministic anchors first, judge only for the ambiguous residue —
and never wire the judge to an agent-under-test model (self-preference bias; see
`detectors/judge.py`). Add a `pass` and a `fail` golden fixture under
`tests/fixtures/` and assert in `tests/test_detectors.py` that your detector fires
on the fail case and stays silent on the pass case.

## Commit conventions

- One focused change per commit, with a clear message.
- Do **not** add co-author / self-attribution trailers to commits or PRs.
