"""Text embedding with a local cache (see docs/MEMORY.md).

Same text is embedded at most once — PRINCIPLES.md principle 2.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time

from ..llm import client as _llm


def text_hash(text: str) -> str:
    """Stable 16-hex-char id for a piece of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def embed(text: str, cx: sqlite3.Connection) -> bytes:
    """Return the embedding for `text`, hitting `embeddings_cache` first."""
    h = text_hash(text)
    row = cx.execute(
        "SELECT vector FROM embeddings_cache WHERE text_hash = ?", (h,)
    ).fetchone()
    if row is not None:
        return row[0]

    vector = _llm.get_client().embed(text)
    cx.execute(
        "INSERT OR IGNORE INTO embeddings_cache(text_hash, vector, created_at) "
        "VALUES (?, ?, ?)",
        (h, vector, int(time.time())),
    )
    cx.commit()
    return vector
