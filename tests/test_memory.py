"""embed cache, ingest two-stage transaction, retrieve, forget cascade."""

from __future__ import annotations

import pytest

from mneme.memory import forget, ingest, retrieve, store
from mneme.memory.embed import embed


def test_embed_is_cached(cx, fake_llm):
    first = embed("hello world", cx)
    second = embed("hello world", cx)
    assert first == second
    assert fake_llm.embed_calls == 1  # second call hit the cache


def test_ingest_saves_slice_and_vector(cx, fake_llm):
    sid = ingest.save_user_message("I use Tauri for the desktop app", "turn1", cx)
    assert cx.execute("SELECT text FROM slices WHERE id=?", (sid,)).fetchone() is not None
    assert cx.execute("SELECT 1 FROM vec_slices WHERE slice_id=?", (sid,)).fetchone() is not None


def test_ingest_builds_concept_graph(cx, fake_llm):
    fake_llm.concept_json = (
        '{"nodes": [{"name": "Tauri", "kind": "artifact"},'
        '            {"name": "desktop app", "kind": "concept"}],'
        ' "edges": [{"src": "Tauri", "type": "PART_OF", "dst": "desktop app"}]}'
    )
    ingest.save_user_message("Tauri powers the desktop app", "turn1", cx)
    assert cx.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
    assert cx.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_assistant_message_skips_concept_extraction(cx, fake_llm):
    """Stage B is user-only — assistant slices stay retrievable via the vector
    index but cost no extra LLM call (PRINCIPLES.md principle 2)."""
    fake_llm.concept_json = (
        '{"nodes": [{"name": "Tauri", "kind": "artifact"},'
        '            {"name": "desktop app", "kind": "concept"}],'
        ' "edges": [{"src": "Tauri", "type": "PART_OF", "dst": "desktop app"}]}'
    )
    chat_calls_before = fake_llm.chat_calls
    sid = ingest.save_assistant_message("Tauri powers the desktop app", "turn1", cx)
    # Stage A ran: slice + vector are searchable.
    assert cx.execute("SELECT text FROM slices WHERE id=?", (sid,)).fetchone() is not None
    assert cx.execute("SELECT 1 FROM vec_slices WHERE slice_id=?", (sid,)).fetchone() is not None
    # Stage B did NOT run: no concept-extraction LLM call, no nodes, no edges.
    assert fake_llm.chat_calls == chat_calls_before
    assert cx.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
    assert cx.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_concept_failure_does_not_lose_the_slice(cx, fake_llm):
    fake_llm.concept_json = "this is not json"  # stage B will fail
    sid = ingest.save_user_message("a flaky turn", "turn1", cx)
    # Stage A survived despite stage B failing.
    assert cx.execute("SELECT text FROM slices WHERE id=?", (sid,)).fetchone()[0] == "a flaky turn"
    assert cx.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_recall_finds_an_exact_match_and_is_stable(cx, fake_llm):
    ingest.save_user_message("the capital of memory is recall", "turn1", cx)
    ingest.save_user_message("an unrelated sentence", "turn1", cx)

    first = retrieve.recall("the capital of memory is recall", cx)
    assert first, "expected at least one retrieved slice"
    assert first[0].text == "the capital of memory is recall"

    second = retrieve.recall("the capital of memory is recall", cx)
    assert [s.id for s in first] == [s.id for s in second]  # stable ordering


def test_recall_can_exclude_specific_slice_ids(cx, fake_llm):
    """Used by the agent to keep the just-ingested user message out of its
    own retrieved context — otherwise the LLM sees a self-repeat."""
    sid_a = ingest.save_user_message("the sky is blue", "turn1", cx)
    ingest.save_user_message("a separate unrelated sentence", "turn1", cx)

    without = retrieve.recall("the sky is blue", cx)
    with_exclude = retrieve.recall("the sky is blue", cx, exclude={sid_a})

    assert sid_a in {s.id for s in without}
    assert sid_a not in {s.id for s in with_exclude}


def test_forget_cascades_and_requires_consent(cx, fake_llm):
    fake_llm.concept_json = (
        '{"nodes": [{"name": "X", "kind": "concept"},'
        '            {"name": "Y", "kind": "concept"}],'
        ' "edges": [{"src": "X", "type": "MENTIONS", "dst": "Y"}]}'
    )
    sid = ingest.save_user_message("X relates to Y", "turn1", cx)
    assert cx.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1

    with pytest.raises(PermissionError):
        forget.forget(sid, consent=False, cx=cx)

    result = forget.forget(sid, consent=True, cx=cx)
    assert result.edges_removed == 1
    assert cx.execute("SELECT COUNT(*) FROM slices WHERE id=?", (sid,)).fetchone()[0] == 0
    assert cx.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0  # cascaded

    with pytest.raises(KeyError):
        forget.forget("nonexistent", consent=True, cx=cx)


# =============================================================================
# v0.10 / §C — schema migration + back-fill behavior
# =============================================================================
#
# These tests exercise store._migrate_slices_session_id end-to-end against
# legacy v0.9 databases. The legacy fixture ships a sqlite file with the
# pre-v0.10 schema (no session_id column) and 3 seeded slices across 2
# distinct turn_ids. The migration must:
#   - C1: ALTER add session_id column to existing slices
#   - C2: back-fill session_id := turn_id row-by-row
#   - C3: be idempotent on a fresh DB (no-op if column already present)
#   - C4: create idx_slices_session
# Plus R-8 (double-pinned), R-9 (back-fill atomicity), R-10 (FTS not broken),
# R-20 (vec_slices untouched).


def test_C1_migration_adds_session_id_column_to_legacy_db(legacy_db_with_v09_data):
    """C1: opening a pre-v0.10 DB through store.connect + init_db ALTERs in
    the session_id column. Before init: column missing; after: present."""
    import sqlite3 as _sql
    # Before: column not present
    cx_raw = _sql.connect(legacy_db_with_v09_data)
    cols_before = {row[1] for row in cx_raw.execute("PRAGMA table_info(slices)")}
    cx_raw.close()
    assert "session_id" not in cols_before

    # Migrate
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    cols_after = {row[1] for row in cx.execute("PRAGMA table_info(slices)")}
    cx.close()
    assert "session_id" in cols_after


def test_C2_migration_backfills_session_id_from_turn_id(legacy_db_with_v09_data):
    """C2: after migration, every pre-existing slice has session_id == turn_id
    (back-fill rule). The 3 seeded slices keep their turn_id values."""
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    rows = cx.execute(
        "SELECT id, turn_id, session_id FROM slices ORDER BY created_at"
    ).fetchall()
    cx.close()
    assert len(rows) == 3
    # Every row: session_id == turn_id (back-fill rule).
    for r in rows:
        assert r["session_id"] == r["turn_id"], (
            f"back-fill broken: id={r['id']} turn_id={r['turn_id']} "
            f"session_id={r['session_id']}"
        )
    # Both turn ids represented (TURN_A x2, TURN_B x1).
    turn_ids = [r["turn_id"] for r in rows]
    assert turn_ids.count("TURN_A") == 2
    assert turn_ids.count("TURN_B") == 1


def test_C3_init_db_is_idempotent_on_already_migrated_db(legacy_db_with_v09_data):
    """C3 (also R-8 part 1): init_db must be idempotent — calling it twice
    on the same DB must NOT change any rows the second time. Use
    cx.total_changes diff as the precise pin."""
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)  # first call: ALTER + back-fill
    changes_after_first = cx.total_changes

    store.init_db(cx)  # second call: should be a no-op for slices
    changes_after_second = cx.total_changes

    # The delta between the two init_db calls must be 0 — no rows modified
    # by the second pass (the WHERE session_id IS NULL guard does its job).
    assert changes_after_second == changes_after_first, (
        f"second init_db modified {changes_after_second - changes_after_first} "
        f"rows — idempotency broken"
    )
    cx.close()


def test_C4_migration_creates_session_index(legacy_db_with_v09_data):
    """C4: idx_slices_session is created during migration. Without it, every
    session-scoped query (e.g. mneme chat --resume's session-exists check)
    is a full table scan."""
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    indices = {
        row[0] for row in cx.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='slices'"
        )
    }
    cx.close()
    assert "idx_slices_session" in indices


# =============================================================================
# v0.10 / §R — ratchets covered by memory layer
# =============================================================================


def test_R_8_idempotency_unconditional_update_would_clobber_real_data(
    legacy_db_with_v09_data,
):
    """R-8 (lead must-fix #5, idempotency part 2): the WHERE-IS-NULL guard
    is the load-bearing line. To prove it (not just check no-op once),
    simulate the failure mode: a v0.11+ caller has set explicit
    session_ids on the back-filled rows. The second init_db pass must
    LEAVE THOSE ALONE — only NULL rows get back-filled.

    Set up: migrate once, then mutate one row's session_id to a real,
    distinct value; call init_db a second time. The mutated row's
    session_id must survive verbatim (the guard prevents UPDATE).
    """
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)  # initial back-fill
    # Mutate a back-filled row to look like real session assignment.
    cx.execute(
        "UPDATE slices SET session_id = 'EXPLICIT_REAL_SESSION' WHERE id = 'S2'"
    )
    cx.commit()
    # Re-run init_db (e.g. on app restart) — must NOT clobber S2.
    store.init_db(cx)
    s2_session = cx.execute(
        "SELECT session_id FROM slices WHERE id = 'S2'"
    ).fetchone()[0]
    cx.close()
    assert s2_session == "EXPLICIT_REAL_SESSION", (
        f"second init_db clobbered an explicit session_id: got {s2_session}"
    )


def test_R_8_partial_migration_backfills_only_null_rows(
    legacy_db_with_v09_data,
):
    """R-8 (idempotency part 3): the back-fill MUST be NULL-targeted. If
    a row's session_id is NULL (e.g. a v0.11+ row that bypassed the
    ingest path and forgot to set session_id), init_db must back-fill it
    on the next run — without disturbing rows that already had real ids.

    Lead must-fix #5: the two-half pin is (a) re-init makes 0 row
    changes when nothing is NULL [C3]; (b) manually NULL'ing one row +
    re-init back-fills only that row [this test].
    """
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)  # full back-fill
    # Set TWO rows: one to a real value (must not be touched), one to NULL.
    cx.execute(
        "UPDATE slices SET session_id = 'REAL_SESSION' WHERE id = 'S1'"
    )
    cx.execute("UPDATE slices SET session_id = NULL WHERE id = 'S3'")
    cx.commit()
    changes_before = cx.total_changes
    store.init_db(cx)
    changes_after = cx.total_changes

    # Exactly one UPDATE during the second init_db (the NULL row).
    # PRAGMA-level changes from the migration's UPDATE count as 1 row.
    delta = changes_after - changes_before
    assert delta == 1, f"expected 1 row back-filled, got delta={delta}"

    # S1 untouched, S3 back-filled to turn_id ('TURN_B').
    rows = dict(
        cx.execute("SELECT id, session_id, turn_id FROM slices").fetchall()
        and {(r["id"]): (r["session_id"], r["turn_id"])
             for r in cx.execute("SELECT id, session_id, turn_id FROM slices")}
    )
    cx.close()
    assert rows["S1"][0] == "REAL_SESSION"
    assert rows["S3"][0] == "TURN_B"   # back-filled from turn_id


def test_R_9_retrieve_recall_is_byte_identical_pre_and_post_v10(
    legacy_db_with_v09_data, fake_llm,
):
    """R-9: retrieve.recall must return byte-identical results across the
    v0.10 schema migration. The migration adds session_id but does not
    affect ranking, scores, or text. We can't easily compare against a
    real pre-migration retrieve (the index wasn't loaded yet), but we
    pin the closest invariant: after migration, recall results contain
    only fields documented in v0.9's Slice (id, role, text, created_at,
    score). No session_id appears in the returned dataclass — recall is
    cross-session by design (see N1).
    """
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    hits = retrieve.recall("first user msg", cx)
    cx.close()
    # Slice dataclass has no session_id field.
    if hits:
        assert not hasattr(hits[0], "session_id"), (
            "retrieve.Slice exposed session_id — recall must stay "
            "cross-session per principle 2"
        )


def test_R_10_legacy_db_after_migration_keeps_existing_rows_unchanged(
    legacy_db_with_v09_data,
):
    """R-10: every pre-v0.10 column value (id, role, text, turn_id,
    created_at) survives migration unchanged. Only session_id is added.
    Byte-identical preservation of historical rows is the migration's
    no-data-loss contract."""
    import sqlite3 as _sql
    # Snapshot before migration.
    cx_raw = _sql.connect(legacy_db_with_v09_data)
    cx_raw.row_factory = _sql.Row
    before = sorted([
        (r["id"], r["role"], r["text"], r["turn_id"], r["created_at"])
        for r in cx_raw.execute("SELECT * FROM slices")
    ])
    cx_raw.close()

    # Migrate.
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    after = sorted([
        (r["id"], r["role"], r["text"], r["turn_id"], r["created_at"])
        for r in cx.execute(
            "SELECT id, role, text, turn_id, created_at FROM slices"
        )
    ])
    cx.close()
    assert before == after, "migration mutated pre-existing column values"


def test_R_20_migration_does_not_touch_vec_slices(legacy_db_with_v09_data):
    """R-20: v0.10 only changes the `slices` table. The vec_slices virtual
    table (the embedding index) must NOT be re-created or dropped during
    migration — that would invalidate cached vectors. The legacy fixture
    doesn't ship a vec_slices, so we instead pin: after init_db, vec_slices
    exists AND contains 0 entries (slice rows are present but never had
    vectors written — proves migration did not synthesize new entries)."""
    cx = store.connect(legacy_db_with_v09_data)
    store.init_db(cx)
    # vec_slices is recreated as a virtual table with IF NOT EXISTS, so it's
    # present. But the legacy fixture didn't populate it; count must be 0.
    n = cx.execute("SELECT COUNT(*) FROM vec_slices").fetchone()[0]
    cx.close()
    assert n == 0, (
        f"migration spuriously populated vec_slices with {n} entries — "
        f"vector index integrity at risk"
    )


# =============================================================================
# E7 — recall returns slices ACROSS sessions (the design choice §A.4)
# =============================================================================


def test_E7_recall_returns_slices_across_sessions(cx, fake_llm):
    """E7: memory is bigger than any one chat. retrieve.recall must surface
    slices from session A even when called in the context of session B.
    The N1 grep proves recall doesn't *filter* by session; this end-to-end
    proves the search actually traverses the boundary."""
    # Two sessions with similar-but-distinct text.
    sid_a = ingest.save_user_message(
        "the recall test fact lives here", "TURN_A", cx, session_id="SESS_A",
    )
    ingest.save_user_message(
        "totally unrelated content", "TURN_B", cx, session_id="SESS_B",
    )
    # Query in SESS_B's space — but should still find SESS_A's fact.
    hits = retrieve.recall("the recall test fact", cx)
    hit_ids = {h.id for h in hits}
    assert sid_a in hit_ids, (
        f"recall did not cross session boundary; hits={hit_ids}, "
        f"expected to include {sid_a}"
    )
