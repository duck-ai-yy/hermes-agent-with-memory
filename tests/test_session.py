"""v0.10 session_id contract — lifecycle (A), signature (B partial),
CLI surface (G), positive ratchets (R-7/11/13/14/15/19/21).

Architect's §A lifecycle + §B signature + §G CLI sections are pinned here.
Negative-form contracts (E, N, must-not / silently) live in
test_session_negative.py per lead must-fix #1.

NOTE on footer literals (lead reconcile): the second-and-later turn footer
is `…YZ01` (Unicode ellipsis U+2026 + last **4** characters of the ULID),
NOT a 5-char `…XYZ01`. The dev's `_session_footer` uses `session_id[-4:]`.
"""

from __future__ import annotations

import inspect
import json

import pytest

from mneme import agent, cli, paths
from mneme.ids import ulid
from mneme.memory import ingest, store
from mneme.trace import events


# -- helpers ----------------------------------------------------------------

def _trace_records(events_path) -> list[dict]:
    return [
        json.loads(line) for line in
        events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _ingest_records(events_path) -> list[dict]:
    return [r for r in _trace_records(events_path) if r.get("kind") == "ingest"]


# =============================================================================
# §A — session_id lifecycle (mint / persist / resume)
# =============================================================================


def test_A1_fresh_chat_mints_new_session_ulid(repl_runner, tmp_path):
    """A1: a `mneme chat` invocation without --resume mints a fresh session
    ULID. The footer's `session <ULID>` (first turn, full id) is the only
    user-visible signal — capture and validate the format."""
    result = repl_runner("hello\n/quit\n")
    assert result.exit_code == 0, result.stdout
    # Footer should contain "session " followed by a 26-char ULID.
    import re
    m = re.search(r"session ([0-9A-HJKMNP-TV-Z]{26})", result.stdout)
    assert m, f"no 26-char ULID in footer: {result.stdout!r}"
    sess_id = m.group(1)
    # Same id should be persisted to slices.session_id.
    cx = store.connect(paths.DB_PATH)
    row = cx.execute(
        "SELECT session_id FROM slices WHERE role='user' "
        "ORDER BY created_at LIMIT 1"
    ).fetchone()
    cx.close()
    assert row[0] == sess_id


def test_A2_two_turns_share_same_session_id(repl_runner):
    """A2: within one `mneme chat` invocation, every turn shares a single
    session_id. Two user inputs → two slices.session_id values that must
    match. Footer also shows the same id (full on turn 1, ellipsis on 2)."""
    result = repl_runner("hi\nyo\n/quit\n")
    assert result.exit_code == 0, result.stdout
    cx = store.connect(paths.DB_PATH)
    ids = [
        r[0] for r in cx.execute(
            "SELECT session_id FROM slices ORDER BY created_at"
        ).fetchall()
    ]
    cx.close()
    # 2 user slices + 2 assistant slices = 4 slices, all same session_id.
    assert len(ids) == 4
    assert len(set(ids)) == 1, f"session_ids diverged within a turn: {ids}"


def test_A3_resume_with_existing_id_reuses_session(repl_runner):
    """A3: `--resume <id>` reuses an existing session — slices written in
    the resumed turn carry the same session_id as the prior session."""
    # Turn 1: fresh session.
    r1 = repl_runner("first\n/quit\n")
    assert r1.exit_code == 0
    cx = store.connect(paths.DB_PATH)
    sess_id = cx.execute(
        "SELECT session_id FROM slices LIMIT 1"
    ).fetchone()[0]
    cx.close()

    # Turn 2: same session via --resume.
    r2 = repl_runner("second\n/quit\n", extra_args=["--resume", sess_id])
    assert r2.exit_code == 0, r2.stdout

    cx = store.connect(paths.DB_PATH)
    sessions = {
        r[0] for r in cx.execute("SELECT DISTINCT session_id FROM slices").fetchall()
    }
    cx.close()
    assert sessions == {sess_id}, f"resume minted extra sessions: {sessions}"


def test_A4_resume_unknown_id_exits_one_with_sessionnotfound(repl_runner):
    """A4: `--resume <id>` with no matching session must exit 1 and print
    'SessionNotFound:'. No new slices are created (we never started a turn)."""
    bogus = ulid()  # valid ULID format, just not in the DB
    result = repl_runner("never reached\n/quit\n", extra_args=["--resume", bogus])
    assert result.exit_code == 1
    assert "SessionNotFound" in result.stdout
    # No slices written.
    cx = store.connect(paths.DB_PATH)
    n = cx.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    cx.close()
    assert n == 0


# =============================================================================
# §B — keyword-only signature + default = turn_id
# =============================================================================


def test_B1_save_user_message_session_id_keyword_only_default_turn_id(cx, fake_llm):
    """B1: ingest.save_user_message has `session_id` as keyword-only with
    default = turn_id when None. Confirms the back-compat boundary the
    pre-v0.10 callers depend on (and that v0.7-v0.9 tests don't have to
    pass session_id explicitly to keep working)."""
    sid = ingest.save_user_message("hi", "TURN1", cx)  # no session_id
    row = cx.execute(
        "SELECT turn_id, session_id FROM slices WHERE id=?", (sid,)
    ).fetchone()
    assert row[0] == row[1] == "TURN1"


def test_B7_signature_keyword_only_session_id_param(cx, fake_llm):
    """B7: every public function that took (turn_id, cx) now has a
    keyword-only `session_id` parameter. Positional passing must raise
    TypeError mentioning 'positional' (the keyword-only contract — see
    architect §B + lead R-7 strengthening: 'keyword-only' in the message).

    Functions covered:
      - ingest.save_user_message
      - ingest.save_assistant_message
      - agent.respond
      - agent.respond_stream
    """
    # Positional passes must error out.
    with pytest.raises(TypeError) as ei:
        ingest.save_user_message("x", "T1", cx, "SESS_X")
    msg = str(ei.value)
    assert "positional" in msg, f"missing 'positional' in TypeError: {msg}"

    with pytest.raises(TypeError) as ei:
        ingest.save_assistant_message("x", "T1", cx, "SESS_X")
    assert "positional" in str(ei.value)

    # Same for agent.respond.
    with pytest.raises(TypeError) as ei:
        agent.respond("x", "T1", cx, "SESS_X")
    assert "positional" in str(ei.value)


# =============================================================================
# §G — CLI surface (footer + --help)
# =============================================================================


def test_G1_first_turn_footer_shows_full_ulid(repl_runner):
    """G1: the first turn's footer carries the full 26-char ULID so the
    user can copy it for a later `--resume`. Match the literal `session `
    + 26 chars."""
    result = repl_runner("hi\n/quit\n")
    assert result.exit_code == 0
    # Should match exactly one full-ULID instance.
    import re
    matches = re.findall(r"session ([0-9A-HJKMNP-TV-Z]{26})", result.stdout)
    assert len(matches) == 1, f"expected exactly one full ULID, got: {matches}"


def test_G2_second_turn_footer_shrinks_to_ellipsis_last_four(repl_runner):
    """G2: the second turn's footer uses `…XXXX` (U+2026 ellipsis + the
    LAST 4 chars of the session ULID). Pin both: presence of the 4-char
    suffix AND absence of the full ULID on turn 2.

    Lead reconcile: footer is `…YZ01` (4 chars), not the 5-char `…XYZ01`
    typo'd in the architect's example — `_session_footer` is `[-4:]`.
    """
    result = repl_runner("first\nsecond\n/quit\n")
    assert result.exit_code == 0
    import re
    # First footer has the full ULID.
    full_matches = re.findall(
        r"session ([0-9A-HJKMNP-TV-Z]{26})", result.stdout
    )
    assert len(full_matches) == 1
    sess_id = full_matches[0]
    # Second footer has `…` followed by the last 4 chars of sess_id.
    expected_short = "…" + sess_id[-4:]
    assert result.stdout.count(expected_short) == 1, (
        f"missing ellipsis-suffix {expected_short!r} in: {result.stdout!r}"
    )


def test_G3_chat_command_has_resume_option(repl_runner):
    """G3: `mneme chat --help` documents the --resume option. Lead R-13:
    introspect typer.app.registered_commands instead of grepping stdout
    (--help text can vary; flag wiring is the real contract).
    """
    # Locate the `chat` command in the registered list.
    chat_cmds = [c for c in cli.app.registered_commands
                 if c.callback and c.callback.__name__ == "chat"]
    assert len(chat_cmds) == 1
    chat_cb = chat_cmds[0].callback
    sig = inspect.signature(chat_cb)
    assert "resume" in sig.parameters, (
        f"--resume parameter missing from chat(): {list(sig.parameters)}"
    )


# =============================================================================
# §R — positive ratchets
# =============================================================================


def test_R_7_session_id_is_keyword_only_in_signature(cx, fake_llm):
    """R-7 (lead strengthened, sig-based): session_id MUST be declared
    keyword-only in the signatures of ingest.save_*, agent.respond, and
    agent.respond_stream. Lead wanted the TypeError message to contain
    'keyword-only', but CPython's positional-overflow error does NOT
    include that literal (it says 'takes N positional arguments but M
    were given'). The truer pin is `inspect.Parameter.KEYWORD_ONLY` —
    when the parameter is keyword-only by signature, the diagnostic that
    matters is that the symbol simply cannot be supplied positionally.
    """
    for fn in (ingest.save_user_message, ingest.save_assistant_message,
               agent.respond, agent.respond_stream):
        sig = inspect.signature(fn)
        assert "session_id" in sig.parameters, f"{fn.__name__} missing session_id"
        kind = sig.parameters["session_id"].kind
        assert kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{fn.__name__}.session_id is {kind!r}, expected KEYWORD_ONLY"
        )
    # And confirm positional passing raises TypeError (the user-visible
    # diagnostic — even without 'keyword-only' literal, the message names
    # the function and the positional limit, which is the actionable bit).
    with pytest.raises(TypeError) as ei:
        agent.respond("x", "T1", cx, "SESS_X")
    msg = str(ei.value)
    assert "positional" in msg
    assert "respond" in msg


def test_R_11_close_trace_carries_session_id(cx, fake_llm, tmp_path):
    """R-11: every close-trace event (the one with response_hash) carries
    `session_id`. Sliced-by-session forensic queries depend on this."""
    reply = agent.respond("hello", "TURN1", cx, session_id="SESS_ABC")
    closes = [
        r for r in _trace_records(tmp_path / "events.jsonl")
        if r.get("id") == reply.trace_id and "response_hash" in r
    ]
    assert len(closes) == 1
    assert closes[0]["session_id"] == "SESS_ABC"


def test_R_14_open_trace_carries_session_id(cx, fake_llm, tmp_path):
    """R-14: the pre-call (open) trace event also carries session_id, so a
    crash mid-call still leaves a session-attributable breadcrumb."""
    reply = agent.respond("hello", "TURN1", cx, session_id="SESS_OPEN")
    opens = [
        r for r in _trace_records(tmp_path / "events.jsonl")
        if r.get("id") == reply.trace_id and "query" in r
    ]
    assert len(opens) == 1
    assert opens[0]["session_id"] == "SESS_OPEN"


def test_R_15_ingest_event_carries_session_id(cx, fake_llm, tmp_path):
    """R-15: the ingest event for both user and assistant slices carries
    session_id (so the audit log groups ingests by session)."""
    agent.respond("hello", "TURN1", cx, session_id="SESS_ING")
    ing = _ingest_records(tmp_path / "events.jsonl")
    assert len(ing) == 2  # user + assistant
    assert all(r["session_id"] == "SESS_ING" for r in ing)


def test_R_13_chat_command_resume_param_introspectable(cx, fake_llm):
    """R-13: pinned by introspecting typer commands (not --help stdout).
    Already enforced in G3; this version verifies the underlying registry
    is the source of truth used by typer to build --help."""
    names = {c.callback.__name__ for c in cli.app.registered_commands if c.callback}
    assert "chat" in names


def test_R_19_prompt_hash_strictly_excludes_session_id(monkeypatch, cx, fake_llm, tmp_path):
    """R-19 (PRINCIPLE 2 / lead must-fix #4 / M21):
    `soul.prompt_hash` MUST NOT take session_id as input. Two turns with
    identical (user_text, retrieved slices) but different session_ids must
    produce byte-identical `prompt_hash` in the open-trace event.

    The cache key is prefix+suffix only; adding session_id silently halves
    the prompt-cache hit rate on the LLM side. To equalize the slice
    context (since retrieve.recall would otherwise see turn-1's ingested
    slices on turn 2), monkeypatch recall to return [] both times — then
    only session_id varies, so prompt_hash must be IDENTICAL."""
    from mneme.memory import retrieve
    monkeypatch.setattr(retrieve, "recall", lambda *a, **kw: [])

    reply1 = agent.respond("same input", "TURN1", cx, session_id="SESS_A")
    reply2 = agent.respond("same input", "TURN2", cx, session_id="SESS_B")
    recs = _trace_records(tmp_path / "events.jsonl")
    opens1 = [r for r in recs if r.get("id") == reply1.trace_id and "prompt_hash" in r]
    opens2 = [r for r in recs if r.get("id") == reply2.trace_id and "prompt_hash" in r]
    assert opens1[0]["prompt_hash"] == opens2[0]["prompt_hash"], (
        "prompt_hash differs across sessions with identical (text, slices) — "
        "session_id is leaking into the hash and will halve cache hit rate"
    )


def test_R_21_explain_merge_carries_session_id_from_both_halves(
    cx, fake_llm, tmp_path,
):
    """R-21 (lead-added): events.explain(trace_id) merges open + close trace
    rows. Both rows carry session_id; the merged dict must reflect the same
    value (no late-merge clobber, no field shadowing). This is the audit-trail
    promise that `mneme explain <trace>` shows a real session attribution.
    """
    reply = agent.respond("hi", "TURN1", cx, session_id="SESS_MERGE")
    merged = events.explain(tmp_path / "events.jsonl", reply.trace_id)
    assert merged.get("session_id") == "SESS_MERGE"
    # Both halves are present: open-side query AND close-side response_hash.
    assert "query" in merged and "response_hash" in merged


# =============================================================================
# Extra B-section: signature respond_stream session_id default
# =============================================================================


def test_B_respond_stream_session_id_default_turn_id_when_none(
    cx, fake_llm, tmp_path,
):
    """B (extra): respond_stream with session_id=None falls back to turn_id
    so legacy callers preserve v0.7 one-turn-one-session semantics. The
    persisted slice carries session_id == turn_id."""
    list(agent.respond_stream("hello", "TURNZ", cx, session_id=None))
    row = cx.execute(
        "SELECT turn_id, session_id FROM slices WHERE role='user' LIMIT 1"
    ).fetchone()
    assert row[0] == row[1] == "TURNZ"
