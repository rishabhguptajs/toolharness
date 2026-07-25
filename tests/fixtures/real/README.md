# Captured-real fixtures

These are captures from real runs of each agent CLI on one sample-repo task
("read the README, add a CHANGELOG entry, run pytest"). They exist to test the
**parsers** in `toolharness/adapters/` against output shapes that no
hand-written fixture would get right — nested content blocks, result linking,
degraded OTEL records, and the two on-disk shapes Codex emits.

## They are scrubbed, not verbatim

Captured agent output embeds a lot of the machine it ran on. Before this repo
was made public these files were scrubbed of:

- **Local paths** — the capturing user's home directory and repo location,
  including paths appearing as JSON *keys* (Codex records file changes keyed by
  absolute path), and the sandbox temp-dir names that leak an account hash.
- **That machine's configuration** — installed skills, plugins, MCP servers,
  slash commands, custom agents, and the local timezone.
- **Third-party prompt text** — the CLIs inject their own system prompts and
  instruction blocks (`<skills_instructions>`, `<permissions instructions>`,
  and similar) into the transcript. Those are the vendors' content, not ours to
  redistribute, so they are replaced with a placeholder.
- **Identifiers** — session, turn, request, and call IDs are replaced with
  deterministic pseudonyms. The mapping is consistent within and across files,
  so `call_id` → result linking still holds; that linkage is exactly what
  `test_every_call_links_its_result` checks.
- **Tool registries** — the Claude capture advertised that user's full registry,
  including unreleased internal tools. It now carries the public tool set, which
  still covers every tool the run actually calls (what M3 needs to detect
  hallucinated calls).

Event structure, ordering, timing, tool arguments, and results are untouched.

## Adding a capture

Re-scrub before committing. Keep the same task so the fixtures stay comparable,
and check the result with:

```bash
git grep -nIE "$USER|$HOME" -- tests/fixtures/real/
```
