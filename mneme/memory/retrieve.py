"""Read path: vector top-K + graph expansion, stably ranked.

Stable ordering (score DESC, id ASC) is load-bearing: it keeps the prompt
suffix identical across identical queries, preserving cache hits
(PRINCIPLES.md principle 2).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace

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
    """Adapter exposing graph.bfs's minimal exec().all() shape over sqlite3."""

    def __init__(self, cx: sqlite3.Connection) -> None:
        self._cx = cx

    def exec(self, sql: str, *params: object):
        rows = self._cx.execute(sql, params).fetchall()
        return SimpleNamespace(all=lambda: rows)


def recall(
    query: str,
    cx: sqlite3.Connection,
    k: int = 10,
    hops: int = 2,
    exclude: set[str] | None = None,
) -> list[Slice]:
    """Retrieve slices relevant to `query` via vector search + graph expansion.

    `exclude` lets the caller drop specific slice ids — used by the agent to
    keep the just-ingested user message out of its own retrieved context
    (otherwise the LLM sees "you said X" right after the user said X, and
    treats the turn as a repeat).
    """
    exclude = exclude or set()
    qvec = embed(query, cx)

    vec_rows = cx.execute(
        "SELECT slice_id, distance FROM vec_slices "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (qvec, k),
    ).fetchall()
    vec_scores = {row[0]: 1.0 / (1.0 + row[1]) for row in vec_rows}

    seed_nodes: set[str] = set()
    if vec_scores:
        marks = ",".join("?" * len(vec_scores))
        for src, dst in cx.execute(
            f"SELECT DISTINCT src, dst FROM edges WHERE slice_id IN ({marks})",
            list(vec_scores),
        ):
            seed_nodes.update((src, dst))

    graph_scores: dict[str, float] = {}
    reachable = list(graph.bfs(list(seed_nodes), hops, _GraphDB(cx)))
    if reachable:
        marks = ",".join("?" * len(reachable))
        for (sid,) in cx.execute(
            f"SELECT DISTINCT slice_id FROM edges WHERE src IN ({marks}) OR dst IN ({marks})",
            reachable + reachable,
        ):
            graph_scores[sid] = 1.0

    scored = sorted(
        (
            (-(_VEC_WEIGHT * vec_scores.get(sid, 0.0)
               + _GRAPH_WEIGHT * graph_scores.get(sid, 0.0)), sid)
            for sid in vec_scores.keys() | graph_scores.keys()
        )
    )

    result: list[Slice] = []
    budget = _CHAR_BUDGET
    for _, sid in scored:
        if sid in exclude:
            continue
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
