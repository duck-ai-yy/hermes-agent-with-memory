"""Low-level SQLite access: connection, schema init, transaction context.

The vec0 extension is loaded here so `vec_slices` and the normal tables live
in one file and one transaction (see docs/MEMORY.md).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

_SCHEMA = Path(__file__).with_name("schema.sql")
_EMBED_DIM = 768


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the sqlite-vec extension loaded."""
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    cx.enable_load_extension(True)
    sqlite_vec.load(cx)
    cx.enable_load_extension(False)
    cx.execute("PRAGMA foreign_keys = ON")
    return cx


def init_db(cx: sqlite3.Connection) -> None:
    """Apply schema.sql and create the vec0 virtual table (idempotent)."""
    cx.executescript(_SCHEMA.read_text())
    cx.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_slices "
        f"USING vec0(slice_id TEXT PRIMARY KEY, embedding FLOAT[{_EMBED_DIM}])"
    )
    cx.commit()


@contextmanager
def tx(cx: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Transaction context manager: commit on success, rollback on error."""
    try:
        yield cx
        cx.commit()
    except Exception:
        cx.rollback()
        raise


def db_file(cx: sqlite3.Connection) -> Path | None:
    """Return the on-disk path of the main database, or None for in-memory."""
    for row in cx.execute("PRAGMA database_list"):
        if row[1] == "main":
            return Path(row[2]) if row[2] else None
    return None


def events_path(cx: sqlite3.Connection) -> Path | None:
    """Locate events.jsonl next to the database file (None if in-memory)."""
    path = db_file(cx)
    return path.with_name("events.jsonl") if path else None


def snapshot(cx: sqlite3.Connection, dest: Path) -> None:
    """Native SQLite .backup to `dest` — see PRINCIPLES.md principle 4."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_cx = sqlite3.connect(dest)
    try:
        cx.backup(dest_cx)
    finally:
        dest_cx.close()
