"""Delete path: remove a slice and everything it owns.

edges are cleared automatically via `edges.slice_id ON DELETE CASCADE`.
Orphan nodes are intentionally left in place (see docs/MEMORY.md trade-offs).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..trace import events
from . import store


@dataclass(frozen=True)
class ForgetResult:
    slice_text: str
    edges_removed: int
    vectors_removed: int


def forget(slice_id: str, consent: bool, cx: sqlite3.Connection) -> ForgetResult:
    """Delete a slice with cascade. Requires explicit consent."""
    if not consent:
        raise PermissionError("forget requires explicit consent")

    row = cx.execute("SELECT text FROM slices WHERE id = ?", (slice_id,)).fetchone()
    if row is None:
        raise KeyError(slice_id)
    slice_text = row[0]

    edge_count = cx.execute(
        "SELECT COUNT(*) FROM edges WHERE slice_id = ?", (slice_id,)
    ).fetchone()[0]

    with store.tx(cx):
        cx.execute("DELETE FROM vec_slices WHERE slice_id = ?", (slice_id,))
        # edges are removed by ON DELETE CASCADE when the slice goes.
        cx.execute("DELETE FROM slices WHERE id = ?", (slice_id,))

    result = ForgetResult(slice_text, edge_count, 1)
    ep = store.events_path(cx)
    if ep is not None:
        events.append(
            ep,
            kind="forget",
            slice_id=slice_id,
            cascade={"edges": edge_count, "vectors": 1},
        )
    return result
