"""The `file_read` tool — read a text file from a sandboxed set of roots.

v0.9 / M2 design pins (verbatim error prefixes — do not edit):
  - allowed roots: HOME, /tmp, /var/tmp, CWD subtree
  - 256 KB hard ceiling; truncation marker literal
  - UTF-8 with errors="replace"; first 512 bytes probed for NUL -> binary refused

Layering: this module is pure execution. confirm + audit live in the
agent layer (same boundary shell.py holds).
"""

from __future__ import annotations

from pathlib import Path

from .registry import ToolResult, tool

# Hard ceiling regardless of the model's `max_bytes` argument. Defense in
# depth against a hallucinated max_bytes=10**9 — the LLM never gets to
# spike memory by asking for a huge cap.
MAX_FILE_BYTES = 256 * 1024


def _is_under(path: Path, root: Path) -> bool:
    """True if `path` is `root` or a descendant. Both args must be resolved."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_allowed(resolved: Path) -> bool:
    """v0.9 sandbox: HOME / /tmp / /var/tmp / CWD subtree."""
    roots = [
        Path.home().resolve(),
        Path("/tmp").resolve(),
        Path("/var/tmp").resolve(),
        Path.cwd().resolve(),
    ]
    return any(_is_under(resolved, r) for r in roots)


@tool
def file_read(path: str, max_bytes: int = 65536) -> ToolResult:
    """Read a UTF-8 text file from the user's machine. Returns the file content (truncated if large). Reads are restricted to the user's home directory, /tmp, /var/tmp, or the current working directory. Binary files are refused.

    Args:
        path: Absolute or relative path to the file to read.
        max_bytes: Maximum bytes to return (capped at 256 KB).
    """  # noqa: E501
    try:
        p = Path(path).resolve()
    except Exception as exc:
        msg = f"FileReadError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": path, "error": type(exc).__name__})

    abs_str = str(p)

    if not _path_allowed(p):
        msg = f"FileReadError: path not allowed: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "PathNotAllowed"})

    if not p.exists():
        msg = f"FileReadError: file not found: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "FileNotFound"})

    if not p.is_file():
        msg = f"FileReadError: not a regular file: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "NotRegularFile"})

    # Binary probe: first 512 bytes contain \x00 -> refuse.
    try:
        with open(p, "rb") as f:
            probe = f.read(512)
    except PermissionError:
        msg = f"FileReadError: permission denied: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "PermissionDenied"})
    except OSError as exc:
        msg = f"FileReadError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": type(exc).__name__})

    if b"\x00" in probe:
        msg = f"FileReadError: binary file refused: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "BinaryRefused"})

    cap = min(max_bytes, MAX_FILE_BYTES)
    # Cap is on bytes, not characters: read cap+1 bytes to detect truncation,
    # then decode with errors="replace" to stay safe on partial multi-byte
    # sequences at the boundary.
    try:
        total_size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(cap + 1)
    except PermissionError:
        msg = f"FileReadError: permission denied: {abs_str}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": "PermissionDenied"})
    except OSError as exc:
        msg = f"FileReadError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": type(exc).__name__})
    except Exception as exc:
        msg = f"FileReadError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"path": abs_str, "error": type(exc).__name__})

    truncated = len(raw) > cap
    payload = raw[:cap].decode("utf-8", errors="replace")
    if truncated:
        payload += f"\n... [file truncated, {total_size} bytes total]"

    return ToolResult(
        content=payload,
        is_error=False,
        audit={
            "path": abs_str,
            "bytes_read": min(len(raw), cap),
            "truncated": truncated,
        },
    )
