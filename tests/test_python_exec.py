"""v0.9 / M2 — python_exec: subprocess isolation + clamp + DEVNULL.

Boundary numbers from the phase-1 plan:
  C20-C27: happy zero-exit, non-zero exit + is_error=True, stdout/stderr
           capture, timeout label literal, exit-code-first formatting,
           stdout 8 KB / stderr 4 KB byte caps with marker, OSError path
           surfaces type, real subprocess no-shell.
  R-PE-1..3: ratchets — DEVNULL stdin so `python -c "input()"` doesn't
             hang, sys.executable used (not bare 'python'), code arg
             passed via -c (not stdin/file).
  R-PE-4 (LEAD must-fix #5): timeout clamp split into THREE sub-tests:
         (a) NaN -> default 10.0,
         (b) below MIN -> 0.1,
         (c) above MAX -> 60.0.
  R-PE-5 (nice-to-have): exit_code on timeout is -1 (not the OS-level code).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from mneme.tools import python_exec as pe_mod
from mneme.tools.registry import execute


# Capture timeout actually used by subprocess.run, so the clamp tests can
# verify the boundary directly (not via wall-clock).


@pytest.fixture
def spy_subprocess_run(monkeypatch):
    """Monkeypatch subprocess.run inside python_exec module to return a
    fake CompletedProcess and capture (args, timeout, stdin, kwargs)."""
    captured: dict = {}

    def fake_run(args, *, capture_output=True, text=True, timeout=None,
                 stdin=None, check=False, **kw):
        captured["args"] = args
        captured["timeout"] = timeout
        captured["stdin"] = stdin
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        captured["other"] = kw
        return _CP(stdout=captured.get("_stdout", ""),
                   stderr=captured.get("_stderr", ""),
                   returncode=captured.get("_rc", 0))

    monkeypatch.setattr(pe_mod.subprocess, "run", fake_run)
    return captured


class _CP:
    """Stand-in for subprocess.CompletedProcess with the fields python_exec
    reads. Using a custom class so we don't have to import the real one."""
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# -- C20: happy zero-exit returns formatted content with exit_code header -


def test_C20_zero_exit_formats_content_with_exit_code_first(spy_subprocess_run):
    spy_subprocess_run["_stdout"] = "hello"
    spy_subprocess_run["_stderr"] = ""
    spy_subprocess_run["_rc"] = 0
    tr = execute("python_exec", {"code": "print('hello')"})
    assert tr.is_error is False
    assert tr.content.startswith("exit_code: 0")
    assert "stdout:\nhello" in tr.content
    # No stderr section when stderr is empty (matches shell's empty handling).
    assert tr.audit["exit_code"] == 0


# -- C21: non-zero exit is_error=True --------------------------------------


def test_C21_nonzero_exit_is_error_true(spy_subprocess_run):
    spy_subprocess_run["_rc"] = 2
    spy_subprocess_run["_stderr"] = "Traceback (...)"
    tr = execute("python_exec", {"code": "raise SystemExit(2)"})
    assert tr.is_error is True
    assert "exit_code: 2" in tr.content
    assert "stderr:\nTraceback" in tr.content
    assert tr.audit["exit_code"] == 2


# -- C22: stdout AND stderr captured separately ----------------------------


def test_C22_stdout_and_stderr_captured_separately(spy_subprocess_run):
    spy_subprocess_run["_stdout"] = "out-line"
    spy_subprocess_run["_stderr"] = "err-line"
    spy_subprocess_run["_rc"] = 0
    tr = execute("python_exec", {"code": "..."})
    assert "stdout:\nout-line" in tr.content
    assert "stderr:\nerr-line" in tr.content


# -- C23: timeout label literal --------------------------------------------


def test_C23_timeout_returns_specific_label_literal(monkeypatch):
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10.0,
                                        output=b"", stderr=b"")
    monkeypatch.setattr(pe_mod.subprocess, "run", boom)
    tr = execute("python_exec", {"code": "import time; time.sleep(99)"})
    assert tr.is_error is True
    assert tr.audit["error"] == "Timeout"
    # Default timeout is 10s -> 'timeout after 10s' (int formatted).
    assert "PythonExecError: timeout after 10s" in tr.content


# -- C24: stdout truncation marker (8 KB cap, real bytes recorded) --------


def test_C24_stdout_over_8kb_clipped_with_marker_literal(spy_subprocess_run):
    n = 10_000
    spy_subprocess_run["_stdout"] = "X" * n
    spy_subprocess_run["_stderr"] = ""
    spy_subprocess_run["_rc"] = 0
    tr = execute("python_exec", {"code": "..."})
    assert "[stdout truncated, 10000 bytes total]" in tr.content
    assert tr.audit["stdout_bytes"] == n
    assert tr.audit["truncated"] is True


# -- C25: stderr truncation marker (4 KB cap) ------------------------------


def test_C25_stderr_over_4kb_clipped_with_marker_literal(spy_subprocess_run):
    n = 5_000
    spy_subprocess_run["_stdout"] = ""
    spy_subprocess_run["_stderr"] = "Y" * n
    spy_subprocess_run["_rc"] = 1
    tr = execute("python_exec", {"code": "..."})
    assert "[stderr truncated, 5000 bytes total]" in tr.content
    assert tr.audit["stderr_bytes"] == n
    assert tr.audit["truncated"] is True


# -- C26: OSError launching python surfaces error type --------------------


def test_C26_oserror_launching_python_surfaces_error_type(monkeypatch):
    def boom(*a, **kw):
        raise OSError("simulated fork failure")
    monkeypatch.setattr(pe_mod.subprocess, "run", boom)
    tr = execute("python_exec", {"code": "..."})
    assert tr.is_error is True
    assert tr.audit["error"] == "OSError"
    assert "PythonExecError: OSError launching python:" in tr.content


# -- C27: real subprocess, no shell — argv passed to python -c -----------


def test_C27_subprocess_run_uses_python_minus_c_with_argv_not_shell(spy_subprocess_run):
    execute("python_exec", {"code": "print('x')"})
    args = spy_subprocess_run["args"]
    # argv form: [sys.executable, '-c', code]. Never a shell=True path.
    assert isinstance(args, list)
    assert args[0] == sys.executable
    assert args[1] == "-c"
    assert args[2] == "print('x')"
    # shell kwarg was not passed in (defense in depth — the keyword is
    # absent from `other`).
    assert "shell" not in spy_subprocess_run["other"]


# -- R-PE-1: stdin is DEVNULL so input() does not block -------------------


def test_R_PE_1_stdin_is_devnull_so_input_does_not_hang(spy_subprocess_run):
    """Without DEVNULL, `python -c "input()"` would block on stdin until
    timeout (10s). DEVNULL closes stdin immediately -> EOFError raised by
    input() and the process exits quickly. Verify subprocess.run was
    called with stdin=subprocess.DEVNULL."""
    execute("python_exec", {"code": "input()"})
    assert spy_subprocess_run["stdin"] == subprocess.DEVNULL


# -- R-PE-2: sys.executable used (not bare 'python') ----------------------


def test_R_PE_2_sys_executable_used_for_argv(spy_subprocess_run):
    """sys.executable is the canonical path to THIS python interpreter.
    Using bare 'python' would shell-search the path and could miss venv."""
    execute("python_exec", {"code": "pass"})
    assert spy_subprocess_run["args"][0] == sys.executable


# -- R-PE-3: code arg passed via -c, not stdin / not file -----------------


def test_R_PE_3_code_passed_via_dash_c_flag(spy_subprocess_run):
    """We pass code as the third argv element after '-c'. NOT via stdin
    (we close it) and NOT via a temp file (would leave forensics on disk)."""
    execute("python_exec", {"code": "import os; print(os.getcwd())"})
    assert spy_subprocess_run["args"][1] == "-c"
    assert spy_subprocess_run["args"][2] == "import os; print(os.getcwd())"


# -- R-PE-4a (Lead must-fix #5 split): NaN -> default 10.0 ----------------


def test_R_PE_4a_python_exec_timeout_nan_falls_back_to_default(spy_subprocess_run):
    """NaN passes through the schema (float type), but inside the tool the
    `timeout != timeout` guard catches it and reinstates DEFAULT=10.0."""
    execute("python_exec", {"code": "...", "timeout": float("nan")})
    assert spy_subprocess_run["timeout"] == 10.0


# -- R-PE-4b: below MIN clamps to 0.1 -------------------------------------


def test_R_PE_4b_python_exec_timeout_below_min_clamps_to_0_1(spy_subprocess_run):
    """0.0 is below the MIN floor of 0.1 -> clamped UP to 0.1. Mutation
    target: a bug that reverses min/max would set it to 60.0."""
    execute("python_exec", {"code": "...", "timeout": 0.0})
    assert spy_subprocess_run["timeout"] == 0.1


# -- R-PE-4c: above MAX clamps to 60.0 ------------------------------------


def test_R_PE_4c_python_exec_timeout_above_max_clamps_to_60(spy_subprocess_run):
    """999.0 is above the MAX ceiling of 60.0 -> clamped DOWN to 60.0.
    Mutation target: dropping the upper clamp would let a malicious model
    request a 999s timeout and hang the agent."""
    execute("python_exec", {"code": "...", "timeout": 999.0})
    assert spy_subprocess_run["timeout"] == 60.0


# -- R-PE-5 (nice-to-have): timeout exit_code is -1 -----------------------


def test_R_PE_5_timeout_exit_code_is_minus_one_audit_field(monkeypatch):
    """The audit log must clearly distinguish 'we killed it' (-1) from
    'the script exited with code N'. -1 is the chosen sentinel."""
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10.0,
                                        output=b"", stderr=b"")
    monkeypatch.setattr(pe_mod.subprocess, "run", boom)
    tr = execute("python_exec", {"code": "..."})
    assert tr.audit["exit_code"] == -1


# -- Extra ratchet: schema constants match the docstring ------------------


def test_python_exec_module_constants_pin_design_numbers():
    """Sanity ratchet: the constants the dev exposed match the design.
    A mutation that changes any of these is a regression."""
    assert pe_mod.PYTHON_EXEC_TIMEOUT_DEFAULT == 10.0
    assert pe_mod.PYTHON_EXEC_TIMEOUT_MIN == 0.1
    assert pe_mod.PYTHON_EXEC_TIMEOUT_MAX == 60.0
    assert pe_mod.PY_STDOUT_LIMIT == 8 * 1024
    assert pe_mod.PY_STDERR_LIMIT == 4 * 1024
