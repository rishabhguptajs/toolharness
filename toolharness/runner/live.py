"""Live CLI invocation: run a real agent CLI on a task repo, capture its output,
and hand the raw trace to the matching adapter.

This is the opt-in path behind ``toolharness live``. It is deliberately *safe by
default* and deliberately *un-clever*:

* **Sandbox by default.** The task repo is copied into a throwaway temp directory
  (git history preserved, heavy build dirs skipped) and the agent runs *there*, so
  a misbehaving agent cannot mutate the user's real working tree. ``--in-place``
  opts out with an explicit warning.
* **Always time-bounded.** Every invocation runs under a wall-clock ``--timeout``.
* **No container / network magic.** We shell out to a CLI the user already has
  installed and trust their own agent's permission model for everything else — the
  harness scores the run, it does not try to jail it.

The command construction (``InvocationProfile.build_command``) and the sandbox
setup (``prepare_workdir``) are pure and unit-tested with a fake echo-CLI; the real
CLIs need their own binaries + API keys and so are exercised manually, not in CI.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from toolharness.adapters import default_registry
from toolharness.adapters.base import RunSource
from toolharness.core.model import NormalizedSession
from toolharness.core.taskspec import TaskSpec

# Directories never worth copying into a sandbox (heavy, regenerable).
_SANDBOX_IGNORE = shutil.ignore_patterns(
    ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.pyc",
)


@dataclass(frozen=True)
class InvocationProfile:
    """How to invoke one agent CLI so its output is parseable by ``adapter``.

    The command is assembled as
    ``[binary, *pre_args, prompt, *post_args, *(sandbox_args if sandboxed), *extra]``
    — the prompt is always passed as a single argv element, never shell-interpolated.

    ``sandbox_args`` are the flags that let the agent act *autonomously* (auto-accept
    edits, run commands without approval). They are applied **only inside the
    throwaway temp-dir sandbox**, where the copy is the safety boundary; an
    ``--in-place`` run omits them so the agent's own permission prompts still gate
    edits to the user's real repo.
    """

    name: str
    adapter: str
    binary: str
    pre_args: tuple[str, ...] = ()
    post_args: tuple[str, ...] = ()
    sandbox_args: tuple[str, ...] = ()

    def build_command(
        self, prompt: str, *, sandboxed: bool = False, extra_args: Sequence[str] = ()
    ) -> list[str]:
        autonomy = list(self.sandbox_args) if sandboxed else []
        return [self.binary, *self.pre_args, prompt, *self.post_args, *autonomy, *extra_args]


# The three profiles whose adapters shipped in M5. Gemini is intentionally absent
# (auth-blocked in M5); add it here once the GeminiAdapter lands. The ``sandbox_args``
# grant headless autonomy and are only used when running in the sandbox.
PROFILES: dict[str, InvocationProfile] = {
    "claude-code": InvocationProfile(
        name="claude-code",
        adapter="claude-code",
        binary="claude",
        pre_args=("-p",),
        post_args=("--output-format", "stream-json", "--verbose"),
        sandbox_args=("--permission-mode", "bypassPermissions"),
    ),
    "cursor": InvocationProfile(
        name="cursor",
        adapter="cursor",
        binary="cursor-agent",
        pre_args=("-p",),
        post_args=("--output-format", "stream-json"),
        sandbox_args=("--force",),
    ),
    "codex": InvocationProfile(
        name="codex",
        adapter="codex",
        binary="codex",
        pre_args=("exec", "--json"),
        sandbox_args=("--dangerously-bypass-approvals-and-sandbox",),
    ),
}


def profile_for(name: str) -> InvocationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"No live invocation profile for {name!r}; available: {sorted(PROFILES)}"
        ) from exc


@dataclass
class LiveResult:
    session: NormalizedSession
    trace_path: Path
    workdir: Path
    returncode: int
    timed_out: bool
    stdout: str = ""
    stderr: str = ""
    sandboxed: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


# A command runner is injectable so tests can substitute a fake CLI. It returns
# (returncode, stdout, stderr, timed_out).
CommandRunner = Callable[[Sequence[str], Path, float | None], "tuple[int, str, str, bool]"]


def _subprocess_runner(
    command: Sequence[str], cwd: Path, timeout: float | None
) -> tuple[int, str, str, bool]:
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
        return 124, out, err, True
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"CLI binary {command[0]!r} not found on PATH; install it or pass --adapter "
            f"with a pre-captured trace instead."
        ) from exc
    return proc.returncode, proc.stdout, proc.stderr, False


def prepare_workdir(repo: Path, *, sandbox: bool = True) -> tuple[Path, bool]:
    """Return the directory the agent should run in.

    With ``sandbox`` (the default), copy ``repo`` into a fresh temp dir so the real
    tree is never touched; heavy/regenerable dirs are skipped, ``.git`` is kept.
    Returns ``(workdir, created)`` — ``created`` is True only for a fresh sandbox
    (so the caller knows whether cleanup is theirs to do).
    """
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise NotADirectoryError(f"Task repo {repo} is not a directory.")
    if not sandbox:
        return repo, False
    tmp = Path(tempfile.mkdtemp(prefix="toolharness-live-"))
    dest = tmp / repo.name
    shutil.copytree(repo, dest, ignore=_SANDBOX_IGNORE, symlinks=True)
    return dest, True


def run_live(
    profile: InvocationProfile | str,
    task: TaskSpec,
    *,
    repo: str | Path | None = None,
    sandbox: bool = True,
    timeout: float | None = 600.0,
    trace_path: str | Path | None = None,
    keep_workdir: bool = False,
    agent_args: Sequence[str] = (),
    command_runner: CommandRunner | None = None,
) -> LiveResult:
    """Invoke an agent CLI on a task repo and parse the captured trace.

    ``repo`` defaults to the TaskSpec's ``repo_path``. The prompt comes from the
    TaskSpec. The raw stdout is written to ``trace_path`` (or a temp file) and then
    normalized by the profile's adapter; the returned session has ``task`` attached
    so downstream detectors run in reference-based mode when the spec has gold data.
    """
    prof = profile if isinstance(profile, InvocationProfile) else profile_for(profile)
    run_cmd = command_runner or _subprocess_runner

    repo_path = Path(repo) if repo is not None else (
        Path(task.repo_path) if task.repo_path else None
    )
    if repo_path is None:
        raise ValueError("No task repo: pass --repo or set repo.path in the task spec.")
    if not task.prompt.strip():
        raise ValueError("Task spec has an empty prompt; nothing to run.")

    workdir, created = prepare_workdir(repo_path, sandbox=sandbox)
    cleanup_root = workdir.parent if (created and not keep_workdir) else None
    try:
        command = prof.build_command(
            task.prompt, sandboxed=created, extra_args=agent_args
        )
        returncode, stdout, stderr, timed_out = run_cmd(command, workdir, timeout)

        if trace_path is not None:
            trace = Path(trace_path)
        else:
            fd = tempfile.NamedTemporaryFile(
                prefix=f"toolharness-{prof.name}-", suffix=".trace", delete=False
            )
            trace = Path(fd.name)
            fd.close()
        trace.write_text(stdout)

        source = RunSource(kind="live", path=trace, aux={"adapter": prof.adapter})
        session = default_registry.parse(source)
        session.task = task

        return LiveResult(
            session=session,
            trace_path=trace,
            workdir=workdir,
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            sandboxed=created,
            metadata={"command": " ".join(command), "profile": prof.name},
        )
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)
