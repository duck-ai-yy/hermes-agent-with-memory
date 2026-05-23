"""Write path: turn a message into a slice + vector + concept graph.

Two-stage transaction (see docs/MEMORY.md):
  A) slices + vec_slices  — always succeeds.
  B) nodes + edges        — best-effort; failure is logged, never rolled back
     into A, so a flaky LLM call cannot break the conversation.
"""

from __future__ import annotations

import logging
import sqlite3
import time

from ..ids import ulid
from ..trace import events
from . import concept, store
from .embed import embed

log = logging.getLogger("mneme.ingest")


def _save(text: str, role: str, turn_id: str, cx: sqlite3.Connection) -> str:
    sid = ulid()
    now = int(time.time())
    vector = embed(text, cx)

    # Stage A — slice + searchable vector. Must succeed.
    with store.tx(cx):
        cx.execute(
            "INSERT INTO slices(id, role, text, turn_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, role, text, turn_id, now),
        )
        cx.execute(
            "INSERT INTO vec_slices(slice_id, embedding) VALUES (?, ?)", (sid, vector)
        )

    # Stage B — concept graph. Best-effort: a failure here leaves stage A intact.
    # Skipped for assistant messages: the cost of a second LLM call per turn is
    # not justified by the marginal recall gain (PRINCIPLES.md principle 2).
    # Assistant slices stay retrievable via stage A's vector index.
    nodes_added = edges_added = 0
    if role == "user":
        try:
            graph = concept.extract(text)
            with store.tx(cx):
                for node in graph.nodes:
                    cx.execute(
                        "INSERT OR IGNORE INTO nodes(id, name, kind, first_seen) "
                        "VALUES (?, ?, ?, ?)",
                        (ulid(), node.name, node.kind, now),
                    )
                name_to_id: dict[str, str] = {}
                for node in graph.nodes:
                    row = cx.execute(
                        "SELECT id FROM nodes WHERE name = ?", (node.name,)
                    ).fetchone()
                    if row is not None:
                        name_to_id[node.name] = row[0]
                nodes_added = len(name_to_id)
                for edge in graph.edges:
                    src, dst = name_to_id.get(edge.src), name_to_id.get(edge.dst)
                    if src and dst:
                        cx.execute(
                            "INSERT INTO edges(id, src, dst, type, slice_id, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (ulid(), src, dst, edge.type, sid, now),
                        )
                        edges_added += 1
        except concept.ConceptExtractionFailed as exc:
            log.warning("concept extraction failed for slice %s: %s", sid, exc)

    ep = store.events_path(cx)
    if ep is not None:
        events.append(
            ep,
            kind="ingest",
            slice_id=sid,
            role=role,
            turn_id=turn_id,
            nodes=nodes_added,
            edges=edges_added,
        )
    return sid


def save_user_message(text: str, turn_id: str, cx: sqlite3.Connection) -> str:
    """Persist a user message and its concept graph. Return the new slice id."""
    return _save(text, "user", turn_id, cx)


def save_assistant_message(text: str, turn_id: str, cx: sqlite3.Connection) -> str:
    """Persist an assistant reply so it can be retrieved in later turns."""
    return _save(text, "assistant", turn_id, cx)
