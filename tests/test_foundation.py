"""ids, store, graph, events — the layers with no LLM dependency."""

from __future__ import annotations

import time

import pytest

from mneme.ids import ulid
from mneme.memory import graph, store
from mneme.memory.retrieve import _GraphDB
from mneme.trace import events


def test_ulid_shape_and_uniqueness():
    a = ulid()
    assert len(a) == 26
    assert len({ulid() for _ in range(1000)}) == 1000


def test_ulid_is_chronologically_sortable():
    early = ulid()
    time.sleep(0.005)
    late = ulid()
    assert early < late


def test_init_db_is_idempotent(cx):
    store.init_db(cx)  # second apply must not raise
    tables = {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"slices", "nodes", "edges", "embeddings_cache"} <= tables


def test_snapshot_writes_a_file(cx, tmp_path):
    dest = tmp_path / "snap" / "backup.sqlite"
    store.snapshot(cx, dest)
    assert dest.exists() and dest.stat().st_size > 0


def test_events_append_and_explain_merge(tmp_path):
    log = tmp_path / "events.jsonl"
    events.append(log, kind="trace", id="T1", query="hi", model="m")
    events.append(log, kind="trace", id="T1", response_hash="rh", citation_quality="explicit")
    events.append(log, kind="ingest", slice_id="S1")

    record = events.explain(log, "T1")
    assert record["query"] == "hi"
    assert record["response_hash"] == "rh"
    assert record["citation_quality"] == "explicit"

    with pytest.raises(KeyError):
        events.explain(log, "missing")


def test_graph_bfs_respects_hop_limit(cx):
    now = 0
    for nid in ("a", "b", "c", "d"):
        cx.execute("INSERT INTO nodes(id,name,kind,first_seen) VALUES (?,?,?,?)",
                   (nid, nid, "concept", now))
    cx.execute("INSERT INTO slices(id,role,text,created_at) VALUES ('s','user','t',0)")
    chain = [("a", "b"), ("b", "c"), ("c", "d")]
    for i, (src, dst) in enumerate(chain):
        cx.execute("INSERT INTO edges(id,src,dst,type,slice_id,created_at) VALUES (?,?,?,?,?,?)",
                   (f"e{i}", src, dst, "MENTIONS", "s", now))
    cx.commit()

    db = _GraphDB(cx)
    assert graph.bfs(["a"], hops=1, db=db) == {"a", "b"}
    assert graph.bfs(["a"], hops=2, db=db) == {"a", "b", "c"}
    assert graph.bfs(["a"], hops=9, db=db) == {"a", "b", "c", "d"}
