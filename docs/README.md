# Documentation

- **[CLI reference](cli-reference.md)** — every subcommand and flag: `run`, `live`, `suite`, `compare`, `benchmark`.
- **[Failure modes](failure-modes.md)** — what each of M1–M8 detects, how it decides, and how scores are computed.
- **[Task specs](task-specs.md)** — the YAML format that supplies a prompt, a repo, and (optionally) gold data for reference-based scoring.
- **[Judge](judge.md)** — the optional LLM-judge escalation: providers, keys, caching, and why it's never Claude/GPT/Gemini.
- **[Adapters](adapters.md)** — how per-CLI output is normalized, the canonical capability taxonomy, and adapter auto-detection.
- **[Reports](reports.md)** — the JSON report schema and what the HTML dashboard shows.

For contributing (dev setup, architecture, how to add an adapter or detector), see [CONTRIBUTING.md](../CONTRIBUTING.md). For the BFCL validation methodology and results, see [BENCHMARKS.md](../BENCHMARKS.md).
