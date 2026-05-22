"""Graph traversal over the `edges` table — pure stdlib, no graph library.

This module is intentionally complete: it is a finished, dependency-free
algorithm (see PRINCIPLES.md principle 3). Everything else in memory/ is a
skeleton, but this is the real implementation.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol


class _DB(Protocol):
    def exec(self, sql: str, *params: object) -> "_Rows": ...


class _Rows(Protocol):
    def all(self) -> list[tuple]: ...


def bfs(seed_ids: list[str], hops: int, db: _DB) -> set[str]:
    """Return all node ids reachable from `seed_ids` within `hops` edges.

    Edges are treated as undirected for reachability (a relation connects two
    concepts regardless of direction). Each hop issues one indexed query.
    """
    if hops <= 0 or not seed_ids:
        return set(seed_ids)

    visited: set[str] = set(seed_ids)
    frontier: deque[tuple[str, int]] = deque((nid, 0) for nid in seed_ids)

    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        rows = db.exec(
            "SELECT src, dst FROM edges WHERE src = ? OR dst = ?", node, node
        ).all()
        for src, dst in rows:
            for neighbor in (src, dst):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, depth + 1))

    return visited
