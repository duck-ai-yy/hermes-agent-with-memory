"""Read path: vector top-K + graph expansion, stably ranked.

Stable ordering (score DESC, id ASC) is load-bearing: it keeps the prompt
suffix identical across identical queries, preserving cache hits
(PRINCIPLES.md principle 2).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import graph
from .embed import embed

_VEC_WEIGHT = 0.7
_GRAPH_WEIGHT = 0.3
_CHAR_BUDGET = 6000


@dataclass(frozen=True)
class Slice:
    id: str
    role: str
    text: str
    created_at: int


class _GraphDB:
    """Adapter exposing `graph.bfs`'s minimal `exec().all()` shape over sqlite3."""

    def __init__(self, cx: sqlite3.Connection) -> None:
        self._cx = cx

    def exec(self, sql: str, *params: object) -> "_Rows":
        return _Rows(self._cx.execute(sql, params).fetchall())


class _Rows:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list[tuple]:
        return self._rows


def recall(query: str, cx: sqlite3.Connection, k: int = 10, hops: int = 2) -> list[Slice]:
    """Retrieve slices relevant to `query` via vector search + graph expansion."""
    qvec = embed(query, cx)

    # 1. Vector top-K.
    vec_rows = cx.execute(
        "SELECT slice_id, distance FROM vec_slices "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (qvec, k),
    ).fetchall()
    vec_scores = {row[0]: 1.0 / (1.0 + row[1]) for row in vec_rows}

    # 2. Seed nodes: concepts touched by the seed slices.
    seed_nodes: set[str] = set()
    if vec_scores:
        marks = ",".join("?" * len(vec_scores))
        for src, dst in cx.execute(
            f"SELECT DISTINCT src, dst FROM edges WHERE slice_id IN ({marks})",
            list(vec_scores),
        ).fetchall():
            seed_nodes.add(src)
            seed_nodes.add(dst)

    # 3. Graph expansion + 4. slices reachable from the expanded node set.
    graph_scores: dict[str, float] = {}
    reachable = graph.bfs(list(seed_nodes), hops, _GraphDB(cx))
    if reachable:
        marks = ",".join("?" * len(reachable))
        nodes = list(reachable)
        for (sid,) in cx.execute(
            f"SELECT DISTINCT slice_id FROM edges "
            f"WHERE src IN ({marks}) OR dst IN ({marks})",
            nodes + nodes,
        ).fetchall():
            graph_scores[sid] = 1.0

    # 5. Merge + rescore.
    scored = [
        (
            _VEC_WEIGHT * vec_scores.get(sid, 0.0)
            + _GRAPH_WEIGHT * graph_scores.get(sid, 0.0),
            sid,
        )
        for sid in vec_scores.keys() | graph_scores.keys()
    ]
    # 6. Stable ordering: score DESC, id ASC.
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    # 7. Fetch slices, truncating to the character budget.
    result: list[Slice] = []
    budget = _CHAR_BUDGET
    for _, sid in scored:
        row = cx.execute(
            "SELECT id, role, text, created_at FROM slices WHERE id = ?", (sid,)
        ).fetchone()
        if row is None:
            continue
        if result and budget - len(row[2]) < 0:
            break
        budget -= len(row[2])
        result.append(Slice(row[0], row[1], row[2], row[3]))
    return result
