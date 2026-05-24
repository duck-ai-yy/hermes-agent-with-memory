"""v0.10 negative-form contracts: must-not / silent / no-leak (E + N + R-).

These are the invariants that fail if "implementation does the right thing
by accident but a future refactor silently regresses." The plan calls them
out per architect §E + lead must-fix #7 (N1-N5).

The pin pattern: a positive-form assertion would be "the thing happened";
here we pin "the thing did NOT happen" by snapshotting before/after,
spying on the relevant call site, or grep'ing for forbidden literals.
"""

from __future__ import annotations

import inspect
import json
import sqlite3

from mneme import agent
from mneme.memory import ingest, retrieve, store


# -- helpers ----------------------------------------------------------------


def _events_records(path):
    if not path.exists():
        return []
    return [
        json.loads(line) for line in
        path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# =============================================================================
# §E — negative contracts (silently / must-not / never)
# =============================================================================


def test_E1_session_id_whitespace_only_at_http_layer_silently_mints_new(
    http_client,
):
    """E1: design §7 says whitespace-only session_id at the HTTP layer is
    treated as 'mint a new one' silently. The server must NOT reject the
    request with a 400 — that would be a behavior-breaking change for
    clients passing accidental empty strings. The test asserts both: 200
    response AND a fresh ULID was returned (different from any literal
    whitespace input)."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": "   "},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 26-char ULID was minted, not echoed back as whitespace.
    assert isinstance(body["session_id"], str)
    assert body["session_id"].strip() != ""
    assert len(body["session_id"]) == 26


def test_E2_session_id_empty_string_at_http_layer_silently_mints_new(http_client):
    """E2 (variant of E1): empty string is also treated as 'mint new' —
    the design comment 'None or whitespace-only -> mint a new ULID' is
    inclusive of the bare empty case."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": ""},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] != ""
    assert len(body["session_id"]) == 26


def test_E3_server_does_not_raise_sessionnotfound_for_unknown_id(http_client):
    """E3: §7 says the server is STATELESS w.r.t. sessions — any non-empty
    string is accepted silently. Unknown ids do NOT raise SessionNotFound
    (that's a CLI-only contract because the CLI has on-disk state to check
    against). HTTP /chat must accept arbitrary string ids."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": "not-a-real-session-but-i-set-it-anyway"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Server echoes back the caller's id verbatim — does not mint a new one
    # for an "unknown" id (because the server holds no concept of "known").
    assert body["session_id"] == "not-a-real-session-but-i-set-it-anyway"


def test_E4_server_echoes_caller_session_id_verbatim(http_client):
    """E4: when the caller sets session_id, the response body has that exact
    string — no normalization, no mutation. Pin the round-trip is identity."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": "client-chose-this"},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "client-chose-this"


def test_E6_cli_resume_unknown_session_id_does_not_create_session_record(
    repl_runner,
):
    """E6 (CLI variant of E3, must-not form): `--resume <unknown>` must exit
    1 AND must NOT silently create a session for the unknown id. After the
    failed invocation, the DB still has zero rows."""
    from mneme.ids import ulid as _ulid
    bogus = _ulid()
    result = repl_runner("never reached\n/quit\n", extra_args=["--resume", bogus])
    assert result.exit_code == 1
    from mneme import paths
    cx = store.connect(paths.DB_PATH)
    n = cx.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    cx.close()
    assert n == 0


# =============================================================================
# §R — silent / must-not ratchets (R-2..5)
# =============================================================================


def test_R_2_session_id_never_logged_as_null_in_events(cx, fake_llm, tmp_path):
    """R-2: every event that takes session_id must record a real string,
    never JSON null. v0.10 callers may pass session_id=None to the public
    API (legacy contract), but the boundary inside the agent maps None ->
    turn_id BEFORE any event is appended. So events.jsonl must contain
    zero records where session_id is None/null."""
    # Call with explicit None — exercises the None-fallback branch.
    agent.respond("legacy caller", "TURN_LEGACY", cx, session_id=None)
    ev = _events_records(tmp_path / "events.jsonl")
    # Records that ought to carry session_id: trace (open + close), ingest,
    # tool_audit, tool_result. Any of those with session_id == None fails.
    for r in ev:
        if "session_id" in r:
            assert r["session_id"] is not None, (
                f"session_id is null in event: {r}"
            )
        # Some events (e.g. open trace) must carry session_id even when not
        # asked — verify the contract: trace events carry it.
    # Stronger: every trace event has a non-null session_id.
    traces = [r for r in ev if r.get("kind") == "trace"]
    assert traces, "no trace events written"
    for t in traces:
        assert t.get("session_id") == "TURN_LEGACY", (
            f"expected session_id == 'TURN_LEGACY' (None fallback to turn_id), got {t}"
        )


def test_R_3_legacy_callers_no_session_id_still_persist_a_real_string(
    cx, fake_llm,
):
    """R-3 (silent): callers that omit session_id entirely (legacy v0.7-v0.9
    pattern) must NOT leave the slices.session_id column as NULL. The
    fallback `session_id := turn_id` is applied at the ingest boundary."""
    sid = ingest.save_user_message("legacy api", "TURN_X", cx)  # no session_id kw
    row = cx.execute(
        "SELECT session_id FROM slices WHERE id=?", (sid,)
    ).fetchone()
    assert row[0] is not None, "session_id leaked as NULL for legacy caller"
    assert row[0] == "TURN_X"


def test_R_4_no_real_tool_emits_session_id_in_its_audit_dict():
    """R-4 (must-not, audit-shape): tool_audit + tool_result events DO carry
    session_id at the agent layer (by design — see R-15 and lead's
    audit-by-session goal). But the registry merges any audit-dict field
    from the tool into the event flat — so a tool that ever returned
    audit={"session_id": ...} would HIJACK the audit log's session
    attribution.

    The strongest contract we can pin without changing mneme/ is "no
    shipped tool emits session_id in its audit dict". We grep every
    tools/*.py for the pattern. (The cleaner long-term fix is to merge
    audit FIRST, then set session_id LAST in agent._run_tool_call — see
    tester finding logged for v0.11.)
    """
    import pathlib
    tools_dir = pathlib.Path(__file__).resolve().parent.parent / "mneme" / "tools"
    offenders: list[tuple[str, int, str]] = []
    for py in tools_dir.glob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            # Allow incidental occurrences in comments? Be strict: any
            # literal "session_id" in tools/ is a smell because tools live
            # below the session boundary.
            if '"session_id"' in line or "'session_id'" in line:
                offenders.append((py.name, i, line.strip()))
    assert offenders == [], (
        f"tool source mentions 'session_id' — risks hijacking the audit "
        f"log's session attribution: {offenders}"
    )


def test_R_5_close_trace_session_id_matches_open_trace_session_id(
    cx, fake_llm, tmp_path,
):
    """R-5 (must-not diverge): for one trace_id, the open and close trace
    rows MUST report the same session_id. A divergence here would silently
    mis-attribute the turn after merging — the cardinal sin of audit logs."""
    reply = agent.respond("hi", "TURN1", cx, session_id="SESS_SAME")
    ev = _events_records(tmp_path / "events.jsonl")
    rows = [r for r in ev if r.get("kind") == "trace"
            and r.get("id") == reply.trace_id]
    assert len(rows) == 2, f"expected 2 trace rows, got {len(rows)}"
    assert {r["session_id"] for r in rows} == {"SESS_SAME"}


# =============================================================================
# §N — "explicitly not done" pins (lead must-fix #7)
# =============================================================================


def test_N1_retrieve_recall_source_does_not_reference_session_id():
    """N1: `retrieve.recall` source code must NOT mention `session_id`.
    Retrieval is across the whole memory (intentionally cross-session, see
    architect's §A.4), so adding a session_id filter would silently truncate
    recall to within-session — a behavior regression that's near-impossible
    to spot without this grep.
    """
    src = inspect.getsource(retrieve.recall)
    assert "session_id" not in src, (
        f"retrieve.recall references session_id — recall must be cross-session "
        f"per principle 2 (memory is bigger than any one chat). Source:\n{src}"
    )


def test_N3_tool_registry_warn_event_does_not_carry_session_id(
    cx, fake_llm, tmp_path,
):
    """N3 (re-interpreted): the only non-per-turn event kinds emitted by
    mneme (tool_registry_warn, audit from the LLM client) must NOT carry
    a session_id field — those events live outside the per-turn context.

    The plan's original N3 was about a `snapshot` event, but snapshot()
    currently emits no event of any kind. We pin the closest analogue:
    registry events are not per-turn, so they don't get session attribution.
    """
    # Synthesize a registry warn event by triggering one. The fixture
    # framework doesn't naturally emit these, so grep the source instead:
    # tool_registry.append is called with kind="tool_registry_warn" and we
    # need that call site to NOT include session_id.
    import mneme.tools.registry as reg_mod
    src = inspect.getsource(reg_mod)
    # Locate the tool_registry_warn append site and verify no session_id
    # kwarg is passed there.
    # Simple: grep the call form for "tool_registry_warn" then ensure no
    # session_id literal nearby.
    idx = src.find('"tool_registry_warn"')
    assert idx >= 0, "tool_registry_warn emission site missing"
    # Look at the surrounding 300 chars (the call payload).
    window = src[idx:idx + 300]
    assert "session_id" not in window, (
        f"tool_registry_warn carries session_id in window:\n{window}"
    )


def test_N4_server_chat_does_not_select_validate_session_id_against_slices(
    http_client, monkeypatch,
):
    """N4 (lead must-fix #7): the /chat handler must NOT issue a SELECT to
    validate session_id against the slices table — the server is stateless.
    Spy on cx.execute and grep for any SELECT against slices with a
    session_id WHERE clause; the count must be zero.

    The CLI's `--resume` path DOES select against slices (that's correct);
    the server path must not (that's the architect's §7 contract).

    sqlite3.Connection.execute is read-only at the instance level, so we
    wrap the connection in a delegator that intercepts execute().
    """
    executed_sql: list[str] = []
    real_connect = store.connect

    class _ExecSpy:
        def __init__(self, real_cx):
            self._cx = real_cx

        def execute(self, sql, *a, **kw):
            executed_sql.append(sql)
            return self._cx.execute(sql, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._cx, name)

    def spy_connect(path):
        return _ExecSpy(real_connect(path))

    monkeypatch.setattr(store, "connect", spy_connect)

    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": "client-supplied"},
    )
    assert resp.status_code == 200, resp.text
    # No SELECT against slices that filters on session_id.
    suspect = [
        sql for sql in executed_sql
        if "SELECT" in sql.upper() and "slices" in sql.lower()
        and "session_id" in sql.lower()
    ]
    assert suspect == [], (
        f"server validated session_id against slices table: {suspect}"
    )


def test_N5_migration_introduces_no_new_tables(tmp_path):
    """N5 (lead must-fix #7): the v0.10 migration must not create any new
    SQL tables. It only ALTERs `slices` (adds session_id column) and adds
    an index. Pinning this prevents an unintended schema sprawl regression.

    Compare sqlite_master table list before vs after init_db on a fresh DB.
    """
    db_path = tmp_path / "fresh.sqlite"
    cx = sqlite3.connect(db_path)
    # The v0.9-shape: same as a pre-init blank file (no tables).
    before = {
        row[0] for row in cx.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    cx.close()

    # Run the v0.10 init via the public store.connect + init_db.
    cx = store.connect(db_path)
    store.init_db(cx)
    after = {
        row[0] for row in cx.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    cx.close()

    # The post-init table set must equal what existed before PLUS the
    # documented schema tables (slices, nodes, edges, embeddings_cache,
    # vec_slices). v0.10 must NOT add a new table beyond those.
    documented_v09 = {
        "slices", "nodes", "edges", "embeddings_cache", "vec_slices",
    }
    # vec_slices is sqlite-vec virtual; allow its associated shadow tables.
    # sqlite-internal helper tables (sqlite_*) are also not migration sprawl.
    extra = {
        name for name in after - before - documented_v09
        if not name.startswith("vec_slices_") and not name.startswith("sqlite_")
    }
    assert extra == set(), (
        f"v0.10 migration introduced new tables: {extra}"
    )


# =============================================================================
# Extra: ingest event for assistant slice carries the agent-supplied session_id
# (not turn_id — they may differ when the agent loop sets session explicitly).
# =============================================================================


def test_R_ingest_assistant_event_session_id_matches_agent_supplied(
    cx, fake_llm, tmp_path,
):
    """Belt-and-braces: when agent.respond is called with explicit session_id
    different from turn_id, both ingest events (user + assistant) carry the
    explicit session_id, not turn_id."""
    agent.respond("x", "TURN_X", cx, session_id="SESS_X")
    ev = _events_records(tmp_path / "events.jsonl")
    ing = [r for r in ev if r.get("kind") == "ingest"]
    assert len(ing) == 2
    assert all(r["session_id"] == "SESS_X" for r in ing)
    assert all(r["turn_id"] == "TURN_X" for r in ing)
