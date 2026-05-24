"""The `python_exec` tool — run Python code in a fresh subprocess.

v0.9 / M2 design pins (Spike 2):
  - subprocess + sys.executable -c <code>, capture stdout/stderr, hard timeout
  - NOT exec() + restricted globals (signal-based timeouts can't cross
    threads + historical CVE pattern around frame manipulation)
  - NOT docker (violates PRINCIPLE 3 — native-first)
  - Description tells the LLM the truth: real subprocess, host access,
    confirm_cb is the only barrier.
"""

from __future__ import annotations

import subprocess
import sys
import time

from .registry import ToolResult, tool

PYTHON_EXEC_TIMEOUT_DEFAULT = 10.0
PYTHON_EXEC_TIMEOUT_MAX = 60.0
PYTHON_EXEC_TIMEOUT_MIN = 0.1
PY_STDOUT_LIMIT = 8 * 1024
PY_STDERR_LIMIT = 4 * 1024


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """Bytes-based clip — mirrors shell.py for parity."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="replace"), True


def _format_for_llm(exit_code: int, stdout: str, stderr: str,
                    stdout_bytes: int, stderr_bytes: int) -> str:
    """Same exit-code-first / stdout-before-stderr ordering as shell."""
    out, out_truncated = _clip(stdout, PY_STDOUT_LIMIT)
    err, err_truncated = _clip(stderr, PY_STDERR_LIMIT)
    parts = [f"exit_code: {exit_code}"]
    if out:
        suffix = (f"\n... [stdout truncated, {stdout_bytes} bytes total]"
                  if out_truncated else "")
        parts.append(f"stdout:\n{out}{suffix}")
    else:
        parts.append("stdout: (empty)")
    if err:
        suffix = (f"\n... [stderr truncated, {stderr_bytes} bytes total]"
                  if err_truncated else "")
        parts.append(f"stderr:\n{err}{suffix}")
    return "\n\n".join(parts)


@tool
def python_exec(code: str, timeout: float = 10.0) -> ToolResult:
    """Run Python source code. The code runs in a real subprocess on the user's machine (`python -c`), so it has full host access — confirmation is the only safety barrier. Returns stdout, stderr, and the exit code.

    Args:
        code: Python source to execute.
        timeout: Wall-clock seconds before the subprocess is killed (clamped to [0.1, 60]).
    """  # noqa: E501
    # Clamp timeout into [MIN, MAX]. The schema's "default" was 10.0; the
    # MAX is a hard ceiling regardless of what the model asks for.
    if not isinstance(timeout, (int, float)) or timeout != timeout:  # NaN guard
        timeout = PYTHON_EXEC_TIMEOUT_DEFAULT
    timeout = max(PYTHON_EXEC_TIMEOUT_MIN, min(float(timeout), PYTHON_EXEC_TIMEOUT_MAX))

    start = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        # Match shell's "timeout after Ns" with int(timeout).
        msg = f"PythonExecError: timeout after {int(timeout)}s"
        return ToolResult(
            content=msg,
            is_error=True,
            audit={
                "exit_code": -1,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "truncated": False,
                "duration_ms": duration_ms,
                "error": "Timeout",
            },
        )
    except OSError as exc:
        duration_ms = int((time.time() - start) * 1000)
        msg = f"PythonExecError: OSError launching python: {exc}"
        return ToolResult(
            content=msg,
            is_error=True,
            audit={
                "exit_code": -1,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "truncated": False,
                "duration_ms": duration_ms,
                "error": "OSError",
            },
        )
    except Exception as exc:
        duration_ms = int((time.time() - start) * 1000)
        msg = f"PythonExecError: {type(exc).__name__}: {exc}"
        return ToolResult(
            content=msg,
            is_error=True,
            audit={
                "exit_code": -1,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "truncated": False,
                "duration_ms": duration_ms,
                "error": type(exc).__name__,
            },
        )

    duration_ms = int((time.time() - start) * 1000)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    stdout_bytes = len(stdout.encode("utf-8", errors="replace"))
    stderr_bytes = len(stderr.encode("utf-8", errors="replace"))
    truncated = stdout_bytes > PY_STDOUT_LIMIT or stderr_bytes > PY_STDERR_LIMIT
    content = _format_for_llm(completed.returncode, stdout, stderr,
                              stdout_bytes, stderr_bytes)
    return ToolResult(
        content=content,
        is_error=completed.returncode != 0,
        audit={
            "exit_code": completed.returncode,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "truncated": truncated,
            "duration_ms": duration_ms,
        },
    )
