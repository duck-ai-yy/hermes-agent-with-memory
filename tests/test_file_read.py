"""v0.9 / M2 — file_read tool: sandbox + truncation + binary refusal.

Boundary numbers from the phase-1 plan:
  C1-C8: happy-path read, max_bytes truncation, hard 256 KB ceiling
         regardless of arg, UTF-8 with replace, binary refusal via NUL
         probe, sandbox rules (HOME / /tmp / /var/tmp / CWD subtree),
         not-found / not-regular-file / permission denied.
  R-FR-1: relative path resolution (../) is checked AFTER resolve().
  R-FR-2: SKIPPED — symlink TOCTOU is a v0.9 documented gap (Lead must-fix
          #4: skip, not xfail).
  R-FR-3: directory passed in -> NotRegularFile error class.
  R-FR-4 (nice-to-have): UTF-8 multi-byte boundary at cap doesn't crash
          (errors='replace' is what saves us).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mneme.tools import file_read as fr_mod
from mneme.tools.registry import execute


# -- helpers --------------------------------------------------------------


def _write(p: Path, data: bytes | str) -> Path:
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)
    return p


# -- C1: read happy path returns content via execute() ---------------------


def test_C1_read_existing_file_returns_content_and_not_is_error(tmp_path, monkeypatch):
    """Read a small file under tmp_path (which we make CWD so sandbox lets us).
    Verify content matches and is_error=False, audit fields are populated."""
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path / "hello.txt", "hello world\n")
    tr = execute("file_read", {"path": str(p)})
    assert tr.is_error is False
    assert tr.content == "hello world\n"
    assert tr.audit["path"] == str(p.resolve())
    assert tr.audit["bytes_read"] == len("hello world\n")
    assert tr.audit["truncated"] is False


# -- C2: max_bytes truncation marker + bytes_read clip --------------------


def test_C2_max_bytes_arg_truncates_with_marker_literal(tmp_path, monkeypatch):
    """A 10_000-byte file read with max_bytes=100 should give back 100
    bytes of content + a literal "... [file truncated, 10000 bytes total]"
    marker (design-pinned wording)."""
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path / "big.txt", "A" * 10_000)
    tr = execute("file_read", {"path": str(p), "max_bytes": 100})
    assert tr.is_error is False
    assert tr.content.startswith("A" * 100)
    assert "... [file truncated, 10000 bytes total]" in tr.content
    assert tr.audit["truncated"] is True
    assert tr.audit["bytes_read"] == 100


# -- C3: hard MAX_FILE_BYTES ceiling silently caps max_bytes --------------


def test_C3_max_bytes_above_hard_ceiling_is_silently_clamped(tmp_path, monkeypatch):
    """A model asking for max_bytes=10**9 against a LARGE file (1 MB) must
    cap at 256 KB. Mutation guard for M7: `cap = max_bytes` without the
    min() clamp would let the model balloon memory. Test forces the
    distinction by making the file genuinely bigger than the hard cap."""
    monkeypatch.chdir(tmp_path)
    # 1 MB file; hard cap is 256 KB.
    payload = "Z" * (1024 * 1024)
    p = _write(tmp_path / "huge.txt", payload)
    tr = execute("file_read", {"path": str(p), "max_bytes": 10**9})
    assert tr.is_error is False
    # The hard 256 KB cap fires: content body trimmed to 256 KB + marker.
    assert tr.audit["truncated"] is True
    assert tr.audit["bytes_read"] == 256 * 1024
    # The truncation marker reports the REAL file size, not the cap.
    assert f"[file truncated, {1024 * 1024} bytes total]" in tr.content
    assert fr_mod.MAX_FILE_BYTES == 256 * 1024


# -- C4: file exactly at MAX_FILE_BYTES is not truncated ------------------


def test_C4_file_at_hard_ceiling_exactly_no_truncation_marker(tmp_path, monkeypatch):
    """A 262144-byte file with max_bytes large is read in full (cap+1 = 262145
    bytes from disk, but the file is exactly 262144 -> not truncated)."""
    monkeypatch.chdir(tmp_path)
    payload = "Y" * (256 * 1024)
    p = _write(tmp_path / "ceiling.bin", payload)
    tr = execute("file_read", {"path": str(p), "max_bytes": 256 * 1024})
    assert tr.is_error is False
    assert "... [file truncated" not in tr.content
    assert tr.audit["truncated"] is False
    assert len(tr.content) == 256 * 1024


# -- C5: binary file refused via NUL byte probe ---------------------------


def test_C5_binary_file_with_nul_in_first_512_refused(tmp_path, monkeypatch):
    """A file whose first 512 bytes contain \\x00 is refused with a
    BinaryRefused error — protects the model from being shown garbage and
    keeps the LLM from confusing binary content for text."""
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path / "bin.dat", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    tr = execute("file_read", {"path": str(p)})
    assert tr.is_error is True
    assert tr.audit["error"] == "BinaryRefused"
    assert "binary file refused" in tr.content


# -- C6: not found ---------------------------------------------------------


def test_C6_file_not_found_returns_specific_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = execute("file_read", {"path": str(tmp_path / "missing.txt")})
    assert tr.is_error is True
    assert tr.audit["error"] == "FileNotFound"
    assert "file not found" in tr.content


# -- C7: path outside sandbox is refused with PathNotAllowed --------------


def test_C7_path_outside_sandbox_refused(tmp_path, monkeypatch):
    """/etc/passwd is not under HOME / /tmp / /var/tmp / cwd — refused.
    cwd here is tmp_path (set via monkeypatch.chdir) so /etc is outside."""
    monkeypatch.chdir(tmp_path)
    tr = execute("file_read", {"path": "/etc/passwd"})
    assert tr.is_error is True
    assert tr.audit["error"] == "PathNotAllowed"
    assert "path not allowed" in tr.content


# -- C8: directory passed in (R-FR-3 merged) ------------------------------


def test_C8_directory_passed_in_returns_not_regular_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "adir"
    d.mkdir()
    tr = execute("file_read", {"path": str(d)})
    assert tr.is_error is True
    assert tr.audit["error"] == "NotRegularFile"
    assert "not a regular file" in tr.content


# -- R-FR-1: relative path traversal blocked AFTER resolve() --------------


def test_R_FR_1_relative_dotdot_path_blocked_when_escapes_sandbox(tmp_path, monkeypatch):
    """`../../etc/passwd` resolves to /etc/passwd from cwd=tmp_path; the
    sandbox check runs on the RESOLVED path. The traversal must be blocked
    not because of textual `..` but because the resolved real path is not
    under any allowed root."""
    monkeypatch.chdir(tmp_path)
    tr = execute("file_read", {"path": "../../../../../../etc/passwd"})
    assert tr.is_error is True
    # /etc/passwd is outside the sandbox; should be either PathNotAllowed
    # or FileNotFound depending on whether the file actually exists on the
    # test host. PathNotAllowed is checked BEFORE existence in the impl,
    # so PathNotAllowed wins.
    assert tr.audit["error"] == "PathNotAllowed"


# -- R-FR-2: symlink TOCTOU is a documented v0.9 gap ----------------------


@pytest.mark.skip(reason="v0.9 documented gap: symlink TOCTOU (Spike 3). "
                         "A symlink to a sandboxed path that gets swapped to "
                         "/etc/passwd between resolve() and open() can leak. "
                         "Tracked for a future hardening pass; xfail would "
                         "flap CI, so skip is the correct mark.")
def test_R_FR_2_symlink_toctou_documented_v09_gap():
    pass


# -- R-FR-4 (nice-to-have): UTF-8 cap boundary doesn't crash --------------


def test_R_FR_4_utf8_multibyte_at_cap_boundary_uses_replace_errors(
    tmp_path, monkeypatch,
):
    """A file whose UTF-8 multi-byte sequence straddles the cap byte must
    decode without crashing (errors='replace' is what makes this safe)."""
    monkeypatch.chdir(tmp_path)
    # "ñ" is 0xC3 0xB1 in UTF-8. Build a file where the cut falls between
    # the two bytes by using max_bytes that lands mid-sequence.
    body = ("a" * 9) + "ñ" + ("b" * 90)
    p = _write(tmp_path / "u8.txt", body)
    # cap at 10 bytes — the 0xC3 of ñ is included, 0xB1 is dropped.
    tr = execute("file_read", {"path": str(p), "max_bytes": 10})
    assert tr.is_error is False
    # Truncation marker fires (file is bigger than 10 bytes).
    assert "... [file truncated" in tr.content
    # The first 9 'a's are intact; the truncated ñ is replaced (U+FFFD or similar).
    assert tr.content.startswith("a" * 9)


# -- Extra ratchet: sandbox HOME path is reachable ------------------------


def test_home_dir_under_sandbox_is_allowed(tmp_path, monkeypatch):
    """If a file lives under HOME, it's readable (assuming HOME is itself
    a real path)."""
    home = Path.home()
    # Use a file we KNOW exists under HOME — the test runner's HOME is
    # somewhere persistent. Create a temp file there.
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", dir=home, delete=False, encoding="utf-8",
    ) as fh:
        fh.write("home content")
        p = Path(fh.name)
    try:
        tr = execute("file_read", {"path": str(p)})
        assert tr.is_error is False
        assert tr.content == "home content"
    finally:
        p.unlink(missing_ok=True)
