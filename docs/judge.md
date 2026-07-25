# The judge

Three of the eight modes (M1, M4, M6, M7 — wrong tool, ignored output, missing verification, premature stop) are hybrid: deterministic anchors catch the clear cases with no model call, and only the ambiguous residue escalates to an LLM judge. Everything else stays fully deterministic. See [Failure modes](failure-modes.md) for exactly which cases anchor vs. escalate.

**Default is heuristic-only.** Without `--judge`, no network call is ever made — the harness scores purely on deterministic anchors and reference-mode rules.

## Why it's never Claude, GPT, or Gemini

The judge is deliberately independent of the agents under test. Wiring it to one of the same model families being scored would risk self-preference bias — the judge would be evaluating output produced by a close relative of itself. The default and every built-in provider point at a neutral open model instead (currently Qwen, served over Groq).

## Providers

One `OpenAICompatibleJudge` covers every backend, since Groq, OpenRouter, NVIDIA NIM, and local Ollama all speak the same `/v1/chat/completions` shape. A provider is just `(base_url, model, api_key_env)`.

| `--judge` | Model | Env var | Notes |
|-----------|-------|---------|-------|
| *(omitted)* | — | — | Heuristic-only, no network calls |
| `groq` | `qwen/qwen3.6-27b` | `GROQ_API_KEY` | Default hosted provider |
| `ollama` | `qwen2.5` | *(none)* | Local — `http://localhost:11434`, no key needed |
| `openrouter` | `qwen/qwen-2.5-72b-instruct` | `OPENROUTER_API_KEY` | |
| `nvidia` | `moonshotai/kimi-k2-instruct` | `NVIDIA_API_KEY` | NVIDIA NIM |
| `stub` | — | — | Deterministic test double; not for real scoring |

## Bring your own key

The harness ships **no API key and no hosted service**. The key is read from *your* environment at the moment the judge is built, sent only to the provider you selected, and never written to the judge cache, the JSON report, or the HTML dashboard.

```bash
export GROQ_API_KEY=...
toolharness run <trace> --judge groq --judge-cache .judge_cache
```

If the environment variable for the provider you named isn't set, the run stops immediately with a plain error and exit code `2` — not a downgrade to heuristic-only, and not a traceback:

```
error: groq: environment variable GROQ_API_KEY is not set
hint: the LLM judge needs your own API key. Export it, pick another
provider with --judge, or omit --judge to score heuristics-only.
```

## Reproducibility and caching

Every request is `temperature=0` with a fixed `seed`. Hosted open models aren't bit-reproducible on their own, but `--judge-cache DIR` writes each verdict to disk keyed on `(model, seed, kind, system prompt, user content)` — so a cache hit is exact and free, and re-scoring a session with the same trace, task, and provider always reproduces the same report. Delete the cache directory (or a specific `.json` file in it) to force a fresh verdict.

## Failure handling

A judge backend failure (network error, timeout, malformed response) during scoring doesn't crash the run — it's swallowed and treated as "no judgment," so detection degrades gracefully to whatever the deterministic anchors already found. This is different from a *missing API key*, which fails fast at startup (see above) rather than silently degrading, since that's a setup problem you'd want to know about immediately.
