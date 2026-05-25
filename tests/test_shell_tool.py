"""B7 / B20 / B21: the `shell` tool in isolation.

These tests run actual subprocesses (small, fast, sandbox-safe) plus a few
that monkeypatch `subprocess.run` to force timeout / OSError paths. The
layering invariant in B21 (shell does NOT touch events.jsonl, does NOT
accept a confirm_cb) is the strongest mutation guard — without it a future
refactor could silently push audit into the tool layer.
"""

from __future__ import annotations

import inspect
import subprocess


from mneme.tools import shell as shell_tool


# -- B7 unit: timeout path returns exit=-1 with specific stderr label --------


def test_timeout_returns_minus_one_with_timeout_label(monkeypatch):
    def fake_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="sleep 100", timeout=30.0,
                                        output=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = shell_tool.execute("sleep 100", timeout=30.0)
    assert r.exit_code == -1
    assert "timeout after 30s" in r.stderr   # specific label, not generic
    assert r.stdout == ""


def test_oserror_path_surfaces_error_type(monkeypatch):
    """B7 sibling: lessons/developer.md v0.6 — surface the exception TYPE,
    not 'unknown error'."""
    def boom(*_a, **_kw):
        raise OSError("simulated fork failure")

    monkeypatch.setattr(subprocess, "run", boom)
    r = shell_tool.execute("anything")
    assert r.exit_code == -1
    assert "OSError" in r.stderr   # the type appears verbatim
    assert "fork failure" in r.stderr


# -- B7 unit: real subprocess, non-zero exit ---------------------------------


def test_nonzero_exit_preserved_with_no_truncation():
    r = shell_tool.execute("false")    # exits 1, no output
    assert r.exit_code == 1
    assert r.stdout == ""
    assert r.truncated is False


def test_zero_exit_with_stdout_captured():
    r = shell_tool.execute("echo hello")
    assert r.exit_code == 0
    assert r.stdout.strip() == "hello"
    assert r.stderr == ""


# -- B20: stdout > 8 KB is truncated for the LLM but real bytes recorded ----


def test_stdout_over_8kb_clips_to_8kb_but_records_real_size():
    n = 10_000   # > STDOUT_LIMIT (8 KB)
    r = shell_tool.execute(f"python3 -c \"import sys; sys.stdout.write('X'*{n})\"")
    assert r.exit_code == 0
    # Raw stdout in ShellResult preserved at full length — the LLM-view
    # truncation happens in format_for_llm so the raw stays inspectable.
    assert len(r.stdout) == n
    assert r.stdout_bytes == n
    # truncated flag fires because raw size > STDOUT_LIMIT
    assert r.truncated is True
    # The LLM view IS clipped to 8 KB + a "[stdout truncated, N bytes total]"
    # marker. v0.6 lesson: pin the marker substring so a regression that
    # drops the marker is caught.
    llm_view = shell_tool.format_for_llm(r)
    assert "[stdout truncated, 10000 bytes total]" in llm_view
    # Hard bound: the LLM view must not be larger than the raw output by
    # more than a small framing margin (exit-code header + suffix).
    assert len(llm_view) < n   # clipped: we never feed the full 10 KB


def test_format_for_llm_includes_exit_code_first():
    """Order matters: the LLM should see exit_code before stdout so it knows
    immediately whether to interpret the output as success/failure."""
    r = shell_tool.execute("echo ok")
    out = shell_tool.format_for_llm(r)
    assert out.startswith("exit_code: 0")


# -- B21: layering invariant — shell.execute writes nothing, has no confirm -


def test_execute_signature_does_not_accept_confirm_or_audit_args():
    """If a future refactor pushes confirm/audit into the shell layer, this
    catches it — the architect's PRINCIPLE 1 layering boundary."""
    sig = inspect.signature(shell_tool.execute)
    assert "confirm_cb" not in sig.parameters
    assert "audit" not in sig.parameters
    assert "events_path" not in sig.parameters
    assert "trace_id" not in sig.parameters


def test_execute_does_not_write_to_events_jsonl(tmp_path, monkeypatch):
    """Read-only contract: calling shell.execute must not append to any file
    even if events_path env vars are around. We pre-create an events.jsonl
    and snapshot bytes-before / bytes-after; equality means no write."""
    ep = tmp_path / "events.jsonl"
    ep.write_text("seed-line\n", encoding="utf-8")
    before = ep.read_bytes()
    monkeypatch.chdir(tmp_path)   # in case anything tries cwd-relative writes
    r = shell_tool.execute("echo audit-check")
    after = ep.read_bytes()
    assert before == after, "shell.execute appended to events.jsonl"
    assert r.exit_code == 0


# -- ratchet: format_for_llm distinguishes empty vs non-empty stderr --------


def test_format_for_llm_shows_empty_stdout_marker_when_no_output():
    r = shell_tool.execute("true")
    out = shell_tool.format_for_llm(r)
    assert "stdout: (empty)" in out


# -- B7 unit: stdin DEVNULL — cat-with-no-args returns immediately -----------


def test_stdin_is_devnull_so_cat_does_not_block():
    """If stdin weren't DEVNULL, `cat` with no args would wait on stdin
    forever and hit the timeout. With DEVNULL it returns immediately at
    EOF. Use a 5s timeout so failure is fast, not 30s."""
    r = shell_tool.execute("cat", timeout=5.0)
    assert r.exit_code == 0
    assert r.stdout == ""
