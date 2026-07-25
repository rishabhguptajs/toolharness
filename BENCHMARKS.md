# Benchmark validation (M6)

We validate the detectors against the **Berkeley Function-Calling Leaderboard
(BFCL v3)** — a public, human-curated tool-calling dataset — and report
per-detector precision/recall/F1 plus a Cohen's κ for the LLM-judge.

## What is measured

A BFCL case is a *task spec*: a user question, the advertised `function` schemas,
and (for answerable categories) a `ground_truth` answer. The benchmark adapter
(`evalharness/adapters/benchmarks/bfcl.py`) pairs each case's **real
function registry** with a chosen agent behavior to build scoring-ready sessions
with a known label:

| Arm | Construction | Label | Detector under test |
|-----|--------------|-------|---------------------|
| correct | agent makes the gold call | negative (should stay silent) | M2, M3 |
| hallucinated | agent calls a function **absent** from the registry | positive | **M3** hallucinated |
| wrong-args | agent calls the gold function but **drops a required arg** | positive | **M2** wrong-args |

The registry and JSON schemas are verbatim from BFCL; only the injected mistake is
synthetic. This measures a deterministic detector's precision/recall over
realistic, diverse tool schemas.

**Judge κ (relevance):** BFCL's `irrelevance` category (no function fits → the
agent should call nothing) versus the answerable categories (a function does fit)
gives a human-curated relevant/irrelevant gold label. We ask the LLM-judge "is any
function relevant to this request?" and compute Cohen's κ between the judge and the
gold label. This is the organic-agreement measure the perturbation arms can't give.

## Results (full BFCL v3 sets)

Categories: `simple` (400), `multiple` (200), `irrelevance` (240).

| Detector | Precision | Recall | F1 | Confusion (tp/fp/fn/tn) |
|----------|-----------|--------|----|--------------------------|
| M3 hallucinated | 1.000 | 1.000 | 1.000 | 840 / 0 / 0 / 840 |
| M2 wrong-args | 0.998 | 1.000 | 0.999 | 600 / 1 / 0 / 599 |

Two "errors" surfaced, both **BFCL data quirks, not detector bugs** — a useful
sanity check on the detectors:

- `simple_363` — the gold answer calls `find_closest` while the registry names the
  function `restaurant_search.find_closest` (gold drops the namespace). The adapter
  now resolves gold names against the registry by dotted/underscore suffix, so this
  no longer counts against M3.
- `simple_307` — the gold answer sets `venue: true` (a boolean) for a parameter the
  schema types as `string`. M2 **correctly** flags the type mismatch; the gold data
  is wrong. This is the single remaining M2 false positive.

**Judge relevance κ:** requires a configured judge (see below); not run in CI. On a
labeled subset the command prints `judge relevance kappa=<κ> (n=…, skipped=…)`,
where `skipped` counts judge errors/abstentions.

## Reproduce

The full BFCL files are not vendored (they are large); a 12-case-per-category slice
lives under `tests/fixtures/benchmarks/bfcl/` for the test suite. To run against the
full sets, download them from HuggingFace
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) into a directory laid out as:

```
<dir>/BFCL_v3_simple.json
<dir>/BFCL_v3_multiple.json
<dir>/BFCL_v3_irrelevance.json
<dir>/possible_answer/BFCL_v3_simple.json
<dir>/possible_answer/BFCL_v3_multiple.json
```

Then:

```bash
# deterministic P/R/F1 (no judge, reproducible)
evalharness benchmark bfcl --data <dir>

# add the judge relevance kappa (needs a non-self-preference judge, e.g. Groq)
GROQ_API_KEY=... evalharness benchmark bfcl --data <dir> --judge groq --json bfcl_report.json
```

## Limitations

- The M2/M3 arms measure detector correctness on **controlled perturbations** of
  real schemas, not on organic model mistakes. The judge-κ path is the organic
  measure; extending the perturbation arms with captured real model errors is future
  work (ties into the real-CLI adapters from M5).
- Only the deterministic modes (M2, M3) have BFCL-derived ground truth today.
  ToolBench/API-Bank converters for the multi-step modes are not yet implemented.
