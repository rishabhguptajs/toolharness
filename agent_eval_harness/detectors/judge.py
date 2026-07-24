"""Provider-agnostic LLM-judge with caching.

The judge is deliberately independent of the agents under test (Claude Code,
Cursor, Codex/OpenAI, Gemini/Google) to avoid self-preference bias — the default
is a neutral open model (Kimi K2) served over an OpenAI-compatible endpoint.

Because Groq, OpenRouter, NVIDIA NIM, Cerebras and local Ollama all speak the
same ``/v1/chat/completions`` API, one ``OpenAICompatibleJudge`` covers every
backend; a provider is just ``(base_url, model, api_key_env)``. Tests use the
``StubJudge`` so they stay hermetic and free.

Reproducibility: ``temperature=0`` + a fixed ``seed`` + an on-disk cache keyed on
``(model, system, user, seed)``. Hosted open models are not bit-reproducible, but
once a verdict is cached, re-scoring a session yields identical output — which is
what the report reproducibility guarantee relies on.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

Verdict = Literal["pass", "fail", "warn", "na"]

_USER_AGENT = "agent-eval-harness/0.1 (+https://github.com/rishabhguptajs/agent-eval-harness)"


class JudgeError(RuntimeError):
    """Raised when a real backend cannot produce a verdict."""


@dataclass
class JudgeRequest:
    """A single judging question. ``kind`` scopes the cache and aids debugging."""

    kind: str                      # e.g. "m1_wrong_tool"
    system: str                    # rubric / role
    user: str                      # the concrete case to judge
    max_tokens: int = 512

    def cache_key(self, model: str, seed: int) -> str:
        h = hashlib.sha256()
        for part in (model, str(seed), self.kind, self.system, self.user):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()


@dataclass
class JudgeVerdict:
    verdict: Verdict
    confidence: float
    rationale: str
    model: str = "stub"
    cached: bool = False
    raw: Any = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@runtime_checkable
class Judge(Protocol):
    model: str

    def ask(self, req: JudgeRequest) -> JudgeVerdict: ...


# --- stub (tests / offline) -------------------------------------------------------


class StubJudge:
    """Deterministic judge for tests. Either returns a fixed verdict or defers to a
    ``policy`` callable so a test can encode the decision it expects.
    """

    def __init__(
        self,
        default: JudgeVerdict | None = None,
        policy: Callable[[JudgeRequest], JudgeVerdict] | None = None,
        model: str = "stub",
    ) -> None:
        self.model = model
        self._default = default or JudgeVerdict("na", 0.5, "stub: no opinion", model=model)
        self._policy = policy
        self.calls: list[JudgeRequest] = []

    def ask(self, req: JudgeRequest) -> JudgeVerdict:
        self.calls.append(req)
        verdict = self._policy(req) if self._policy else self._default
        verdict.model = self.model
        return verdict


# --- OpenAI-compatible backend ----------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None   # None => no auth header (e.g. local Ollama)
    seed: int = 7


# Ready-to-use presets. All OpenAI-compatible; switch with one config value.
PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        "groq", "https://api.groq.com/openai/v1",
        "qwen/qwen3.6-27b", "GROQ_API_KEY",
    ),
    "ollama": ProviderConfig(
        "ollama", "http://localhost:11434/v1", "qwen2.5", None,
    ),
    "openrouter": ProviderConfig(
        "openrouter", "https://openrouter.ai/api/v1",
        "qwen/qwen-2.5-72b-instruct", "OPENROUTER_API_KEY",
    ),
    "nvidia": ProviderConfig(
        "nvidia", "https://integrate.api.nvidia.com/v1",
        "moonshotai/kimi-k2-instruct", "NVIDIA_API_KEY",
    ),
}


class OpenAICompatibleJudge:
    """Talks to any OpenAI-compatible chat-completions endpoint via stdlib urllib
    (no third-party HTTP dependency)."""

    def __init__(self, config: ProviderConfig, timeout: float = 60.0) -> None:
        self.config = config
        self.model = config.model
        self.timeout = timeout
        self._api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not self._api_key:
            raise JudgeError(
                f"{config.name}: environment variable {config.api_key_env} is not set"
            )

    def ask(self, req: JudgeRequest) -> JudgeVerdict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "seed": self.config.seed,
            "max_tokens": req.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        # A User-Agent is required: some providers front their API with Cloudflare,
        # which returns 403 to the default "Python-urllib" agent.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise JudgeError(f"{self.config.name} request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise JudgeError(f"{self.config.name}: unparseable response: {exc}") from exc

        return _verdict_from_json(parsed, model=self.model)


def _verdict_from_json(parsed: dict[str, Any], *, model: str) -> JudgeVerdict:
    verdict = str(parsed.get("verdict", "na")).strip().lower()
    if verdict not in ("pass", "fail", "warn", "na"):
        verdict = "na"
    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    rationale = str(parsed.get("rationale", "")).strip() or "(no rationale)"
    return JudgeVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        model=model,
        raw=parsed,
    )


# --- caching wrapper --------------------------------------------------------------


class CachingJudge:
    """Wraps any judge with an on-disk JSON cache so verdicts are reproducible and
    a re-run costs nothing."""

    def __init__(self, inner: Judge, cache_dir: str | Path, seed: int = 7) -> None:
        self.inner = inner
        self.model = inner.model
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ask(self, req: JudgeRequest) -> JudgeVerdict:
        key = req.cache_key(self.model, self.seed)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return JudgeVerdict(
                verdict=data["verdict"],
                confidence=data["confidence"],
                rationale=data["rationale"],
                model=data.get("model", self.model),
                cached=True,
                raw=data.get("raw"),
            )
        verdict = self.inner.ask(req)
        path.write_text(
            json.dumps(
                {
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "model": verdict.model,
                    "raw": verdict.raw,
                }
            )
        )
        return verdict


def safe_ask(judge: Judge | None, req: JudgeRequest) -> JudgeVerdict | None:
    """Ask the judge, swallowing backend failures so detection degrades to
    heuristic-only rather than crashing a scoring run. ``None`` => no judgment."""
    if judge is None:
        return None
    try:
        return judge.ask(req)
    except JudgeError:
        return None


# --- factory ----------------------------------------------------------------------


def build_judge(
    provider: str | None,
    *,
    cache_dir: str | Path | None = None,
) -> Judge | None:
    """Build a judge from a provider name (``groq``/``ollama``/...). ``None`` (or
    ``"none"``) returns ``None`` — detectors then run heuristic-only."""
    if not provider or provider.lower() == "none":
        return None
    if provider.lower() == "stub":
        judge: Judge = StubJudge()
    else:
        try:
            config = PROVIDERS[provider.lower()]
        except KeyError as exc:
            raise JudgeError(
                f"Unknown judge provider {provider!r}; known: {sorted(PROVIDERS)}"
            ) from exc
        judge = OpenAICompatibleJudge(config)
    if cache_dir is not None:
        seed = getattr(getattr(judge, "config", None), "seed", 7)
        judge = CachingJudge(judge, cache_dir, seed=seed)
    return judge
