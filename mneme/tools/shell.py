"""The `shell` tool — execute a shell command, return stdout/stderr/exit code.

v0.9 / M2: the SCHEMA literal is gone — the registry derives the schema
from the @tool-decorated `shell(command)` wrapper below. The lower-level
`execute()` / `format_for_llm()` stay byte-identical to v0.8 (signatures,
return types, error labels, byte caps — `tests/test_shell_tool.py` is
unchanged). Confirmation and audit are still layered on top by
`mneme/agent.py`, NOT here — this module remains pure execution.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from .registry import ToolResult, tool

# When the LLM sees stdout it sees at most this many bytes; beyond that we
# slice and set `truncated=True`. Lets the model read small files inline
# without ballooning the prompt on `find /`. The CLI shows the user a much
# shorter preview (handled in mneme/cli.py).
STDOUT_LIMIT = 8 * 1024
STDERR_LIMIT = 4 * 1024

# Default timeout — keep it short. Long-running commands (servers, watchers)
# should never be invoked through this tool; if they are, the timeout fires
# and we report it cleanly rather than hang the agent loop.
DEFAULT_TIMEOUT = 30.0

@dataclass(frozen=True)
class ShellResult:
    """Result of one `shell` invocation. Carries the *raw* stdout/stderr
    bytes-as-text along with the byte counts before truncation — agent layer
    decides what to show the LLM vs the user."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    stdout_bytes: int            # length of full stdout before any truncation
    stderr_bytes: int
    truncated: bool              # True iff stdout OR stderr was clipped
    duration_ms: int


def execute(command: str, *, timeout: float = DEFAULT_TIMEOUT) -> ShellResult:
    """Run `command` under /bin/sh and return a ShellResult.

    Never raises. Timeouts and OSErrors are reported via exit_code = -1 and
    a specific message in stderr so the LLM (and audit log) can distinguish
    them from a command that ran and returned non-zero. stdin is closed
    (DEVNULL) so commands like `cat` without arguments don't block forever.
    """
    start = time.time()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # `exc.stdout` / `exc.stderr` may be None or bytes; coerce safely.
        partial_out = _coerce(exc.stdout)
        partial_err = _coerce(exc.stderr)
        return _build_result(
            command=command,
            exit_code=-1,
            stdout=partial_out,
            stderr=f"timeout after {timeout:.0f}s\n{partial_err}",
            start=start,
        )
    except OSError as exc:
        # subprocess raises OSError if /bin/sh is missing or fork fails.
        # Surface error TYPE explicitly (lessons/developer.md v0.6 lesson:
        # don't lump unrelated failures into one generic message).
        return _build_result(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"OSError launching shell: {exc}",
            start=start,
        )

    return _build_result(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        start=start,
    )


def format_for_llm(result: ShellResult) -> str:
    """Render a ShellResult as the `content` of a tool_result block.

    Includes exit code, then truncated stdout, then truncated stderr (in that
    order — the LLM should see the exit code first to know if the command
    failed). Truncation is noted explicitly so the model knows the output
    was clipped and doesn't hallucinate the rest.
    """
    out, out_truncated = _clip(result.stdout, STDOUT_LIMIT)
    err, err_truncated = _clip(result.stderr, STDERR_LIMIT)
    parts = [f"exit_code: {result.exit_code}"]
    if out:
        suffix = f"\n... [stdout truncated, {result.stdout_bytes} bytes total]" \
            if out_truncated else ""
        parts.append(f"stdout:\n{out}{suffix}")
    else:
        parts.append("stdout: (empty)")
    if err:
        suffix = f"\n... [stderr truncated, {result.stderr_bytes} bytes total]" \
            if err_truncated else ""
        parts.append(f"stderr:\n{err}{suffix}")
    return "\n\n".join(parts)


# -- @tool wrapper: registers `shell` in the registry ---------------------


@tool
def shell(command: str) -> ToolResult:
    """Execute a shell command on the user's local machine. Returns stdout, stderr, and the exit code. Use for reading files, listing directories, or any non-destructive inspection. Every call requires user confirmation, so prefer one well-formed command over many.

    Args:
        command: The shell command to execute (passed to /bin/sh -c).
    """  # noqa: E501
    # `execute` is documented as "never raises" but the v0.8 agent.py wrapped
    # it in try/except to preserve a "ShellError: <Type>: <msg>" prefix
    # against test-time monkeypatching. Keep that exact contract here so
    # the registry's generic catch-all never sees a shell error first.
    try:
        result = execute(command)
    except Exception as exc:
        msg = f"ShellError: {type(exc).__name__}: {exc}"
        return ToolResult(
            content=msg,
            is_error=True,
            audit={"error": type(exc).__name__, "exit_code": None},
        )
    content = format_for_llm(result)
    return ToolResult(
        content=content,
        is_error=result.exit_code != 0,
        audit={
            "exit_code": result.exit_code,
            "stdout_bytes": result.stdout_bytes,
            "stderr_bytes": result.stderr_bytes,
            "truncated": result.truncated,
            "duration_ms": result.duration_ms,
        },
    )


def _build_result(
    *, command: str, exit_code: int, stdout: str, stderr: str, start: float,
) -> ShellResult:
    out_bytes = len(stdout.encode("utf-8", errors="replace"))
    err_bytes = len(stderr.encode("utf-8", errors="replace"))
    truncated = out_bytes > STDOUT_LIMIT or err_bytes > STDERR_LIMIT
    return ShellResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_bytes=out_bytes,
        stderr_bytes=err_bytes,
        truncated=truncated,
        duration_ms=int((time.time() - start) * 1000),
    )


def _coerce(value) -> str:
    """Bytes / None / str -> str. Used for TimeoutExpired's partial output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """Return (clipped_text, was_truncated). Clip on bytes, not chars, so
    very large unicode text doesn't sneak past the byte limit."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="replace"), True
