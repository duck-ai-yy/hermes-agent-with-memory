"""v0.13 HTTP /chat SSE streaming — full test plan.

The contract under test is encoded literally in `mneme/server.py` (helpers
`_wants_sse`, `_make_sse_frame`, `_error_frame_data`, `_SSE_HEADERS`,
`_sse_event_stream`, and the SSE branch of the `chat` handler). Reading
ratchet -- every literal byte that appears in the on-the-wire contract is
pinned here so a silent refactor cannot drift the response shape.

Six unit groups (Unit A-F), one ratchet matrix per group. Numbering uses the
lead-reviewed phase-1 plan (A1-A7, B1-B3, C1-C5, D1-D9, E1-E7, F1-F7) plus
R-* ratchet items, the lead's 5 must-fix entries, and 2 dev micro-decisions
(SQLite thread affinity + start/error field order).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import pytest

from mneme.llm.client import BudgetExceeded
from mneme.server import (
    _error_frame_data,
    _make_sse_frame,
    _wants_sse,
)


# -- helpers ----------------------------------------------------------------


_ULID_RE = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in
        path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ============================================================================
# Unit A — _make_sse_frame (pure helper)
# ============================================================================


def test_B1_make_sse_frame_returns_bytes_with_canonical_layout():
    """B1: minimal happy path. The frame is exactly
    `event: <type>\\ndata: <json>\\n\\n` UTF-8 bytes -- no CRLF, no padding."""
    out = _make_sse_frame("delta", {"text": "hello"})
    assert isinstance(out, bytes)
    # Raw byte equality. Defends against accidental CR injection, BOM, etc.
    assert out == b'event: delta\ndata: {"text":"hello"}\n\n'


def test_B2_make_sse_frame_uses_compact_json_no_spaces():
    """B2: lead must-fix #1 echo -- the json payload is single-line, compact
    `(",", ":")` separators (no `, ` or `: ` padding). Critical because some
    SSE parsers split on data: <space> exactly once."""
    out = _make_sse_frame("done", {"a": 1, "b": 2})
    # No `", "` between fields, no `": "` between key and value.
    assert b'"a":1,"b":2' in out
    assert b'", "' not in out
    assert b'": "' not in out


def test_B3_make_sse_frame_preserves_non_ascii_raw_utf8():
    """B3: dev contract -- `ensure_ascii=False` -> CJK round-trips raw,
    NOT as `\\uXXXX`. Pin the byte sequence for U+4F60 U+597D (你好)."""
    out = _make_sse_frame("delta", {"text": "你好"})
    assert "你好".encode("utf-8") in out
    # Must NOT have the unicode escape form.
    assert b"\\u4f60" not in out
    assert b"\\u597d" not in out


def test_R_B_7_make_sse_frame_uses_LF_not_CRLF():
    """R-B-7 (lead must-fix #5 byte hygiene): explicit assert that no CRLF
    appears anywhere in the frame. SSE spec allows both but the design pins
    LF only. A silent switch to CRLF would still parse on most clients but
    breaks every byte-level test below."""
    out = _make_sse_frame("delta", {"text": "x"})
    assert b"\r" not in out
    assert b"\r\n" not in out


def test_R_B_8_make_sse_frame_terminates_with_exactly_two_LFs():
    """R-B-8: end-of-frame separator is EXACTLY `\\n\\n` (two LFs), not three,
    not one. SSE spec: a frame is terminated by a blank line, i.e., two
    consecutive LFs. Concatenating frames must produce parseable stream."""
    out = _make_sse_frame("delta", {"text": "x"})
    # Last two bytes are LF.
    assert out[-2:] == b"\n\n"
    # And no triple-LF anywhere (would emit an extra empty frame).
    assert b"\n\n\n" not in out


def test_R_B_9_make_sse_frame_concatenation_is_valid_sse_stream():
    """R-B-9: two frames concatenated split cleanly on `\\n\\n` and yield
    the original two records. Defends against any helper that forgets a
    terminator or adds an extra one."""
    f1 = _make_sse_frame("start", {"a": 1})
    f2 = _make_sse_frame("done", {"b": 2})
    combined = (f1 + f2).decode("utf-8")
    # Strip trailing separator if any then split.
    parts = combined.rstrip("\n").split("\n\n")
    assert len(parts) == 2
    assert parts[0].startswith("event: start\n")
    assert parts[1].startswith("event: done\n")


# ============================================================================
# Unit B — _wants_sse (pure helper)
# ============================================================================


def test_A1_wants_sse_exact_token():
    """A1: bare `text/event-stream` -> True."""
    assert _wants_sse("text/event-stream") is True


def test_A2_wants_sse_none_returns_false():
    """A2: missing Accept header silently degrades to JSON path."""
    assert _wants_sse(None) is False


def test_A3_wants_sse_empty_string_returns_false():
    """A3: empty string -> False (no token at all)."""
    assert _wants_sse("") is False


def test_A4_wants_sse_application_json_returns_false():
    """A4: legacy JSON callers must keep their JSON response."""
    assert _wants_sse("application/json") is False


def test_A5_wants_sse_star_slash_star_returns_false():
    """A5: `*/*` (curl default) is NOT a match -- silent degrade."""
    assert _wants_sse("*/*") is False


def test_A6_wants_sse_weighted_token_returns_false():
    """A6: lead must-fix area -- `text/event-stream;q=0.9` is the SAME literal
    string by token but the design pins exact-match-after-strip-lower. q-value
    suffixes must NOT be treated as SSE. (`;q=0.9` is part of the first token
    when split on `,` so it does not match the bare literal.)"""
    assert _wants_sse("text/event-stream;q=0.9") is False


def test_A7_wants_sse_case_insensitive_match():
    """A7: `Text/Event-Stream` -> True (the design lowercases the token)."""
    assert _wants_sse("Text/Event-Stream") is True


def test_R_A_8_wants_sse_with_whitespace_padding():
    """R-A-8: leading/trailing whitespace on the token is stripped."""
    assert _wants_sse("  text/event-stream  ") is True


def test_R_A_9_wants_sse_first_token_wins():
    """R-A-9: when multiple types are advertised, only the FIRST one matters.
    `application/json, text/event-stream` -> False even though SSE is in the
    list. This is per the design literal `accept_raw.split(",")[0]`."""
    assert _wants_sse("application/json, text/event-stream") is False


def test_R_A_10_wants_sse_garbage_string_returns_false():
    """R-A-10: random garbage silently degrades -- the server NEVER 4xx's
    a bad Accept header (the JSON path is the safe fallback)."""
    assert _wants_sse("not a media type at all !!!") is False


def test_R_A_11_wants_sse_with_charset_param_returns_false():
    """R-A-11 (nice-to-have #1): `text/event-stream;charset=utf-8` is NOT a
    match -- the design pins exact equality to the bare literal token. A
    browser that emits this would silently degrade to JSON. This is a
    behavioral pin: future relaxation should be a deliberate design
    decision, not a silent accident."""
    assert _wants_sse("text/event-stream;charset=utf-8") is False


# ============================================================================
# Unit C — _sse_event_stream + chat handler SSE branch (end-to-end driver)
# ============================================================================
#
# These tests speak to the real handler through `sse_client.post_sse(...)`.
# `fake_llm` is auto-wired by the fixture so respond_stream() emits 3 chunks
# of `fake_llm.reply` per turn (`_stream_chunks`).


def test_C1_happy_path_yields_start_then_deltas_then_done(sse_client):
    """C1: minimal happy path. The SSE stream is exactly [start, delta*, done]
    in that order. No error frame. close-trace event on disk."""
    frames, _raw = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    events = [f["event"] for f in frames]
    assert events[0] == "start"
    assert events[-1] == "done"
    middle = events[1:-1]
    assert middle and all(e == "delta" for e in middle), events


def test_C2_start_frame_carries_session_id_and_turn_id(sse_client):
    """C2: start frame data is exactly `{session_id, turn_id}`. Both are
    well-formed ULIDs (because no session_id passed)."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    start = sse_client.parse_data(frames[0])
    assert set(start.keys()) == {"session_id", "turn_id"}
    assert _ULID_RE.fullmatch(start["session_id"])
    assert _ULID_RE.fullmatch(start["turn_id"])


def test_C3_start_frame_field_order_session_id_then_turn_id(sse_client):
    """C3 + dev micro-decision #2: pin start data field order on RAW BYTES.
    Same trick as F4 -- find the literal `"session_id"` / `"turn_id"` byte
    positions and assert session_id appears first."""
    frames, _raw = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    raw_start = (b"event: start\ndata: " +
                 frames[0]["data"].encode("utf-8"))
    p_sess = raw_start.find(b'"session_id"')
    p_turn = raw_start.find(b'"turn_id"')
    assert p_sess > 0 and p_turn > 0
    assert p_sess < p_turn, (
        f"start frame data field order broken: session_id@{p_sess} "
        f"turn_id@{p_turn} in {raw_start!r}"
    )


def test_C4_delta_frame_carries_text_only(sse_client):
    """C4: each delta frame's data is `{text: <chunk>}` -- nothing else.
    Catches mutations that add session_id or turn_id to every delta."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    deltas = [sse_client.parse_data(f) for f in frames if f["event"] == "delta"]
    assert deltas
    for d in deltas:
        assert set(d.keys()) == {"text"}, d


def test_C5_explicit_accept_header_opens_sse_path_writes_slices(
    sse_client, tmp_path,
):
    """C5 (lead must-fix #2): force `Accept: text/event-stream` (the
    fixture's post_sse always does), then SELECT slices.session_id and
    verify the SSE path produced the same sid that the `start` frame
    advertised."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    start_sid = sse_client.parse_data(frames[0])["session_id"]

    from mneme import paths
    from mneme.memory import store
    cx = store.connect(paths.DB_PATH)
    db_sessions = {
        r[0] for r in cx.execute("SELECT DISTINCT session_id FROM slices")
    }
    cx.close()
    assert db_sessions == {start_sid}, (
        f"start frame sid {start_sid!r} not the one written to slices: "
        f"{db_sessions}"
    )


def test_D1_done_frame_contains_trace_id_and_session_id(sse_client):
    """D1: done frame must include trace_id and session_id, both non-empty."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    assert done.get("trace_id")
    assert done.get("session_id")


def test_D2_done_field_order_pinned_literal(sse_client):
    """D2: done frame field insertion order matches design §3 literal:
    trace_id, citation_quality, session_id, [cost_usd], [prompt_tokens],
    [completion_tokens], [total_tokens], iters. Pin on raw bytes."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done_data = frames[-1]["data"]
    # All optional fields may or may not appear; the required ones must
    # appear in the pinned order.
    must_order = [
        "trace_id", "citation_quality", "session_id",
    ]
    # Optional fields (token / cost). Filter to those actually present in
    # the response so absence does not break the order check.
    parsed = json.loads(done_data)
    tail = [k for k in
            ("cost_usd", "prompt_tokens", "completion_tokens",
             "total_tokens", "iters") if k in parsed]
    expected = must_order + tail
    positions = [done_data.find(f'"{k}"') for k in expected]
    assert all(p >= 0 for p in positions), (
        f"done frame missing key: positions={dict(zip(expected, positions))}"
    )
    assert positions == sorted(positions), (
        f"done frame field order broken: {dict(zip(expected, positions))} "
        f"in {done_data}"
    )


def test_D3_done_trace_id_matches_close_trace_on_disk(sse_client):
    """D3: done.trace_id MUST equal the id of the close-trace event written
    by respond_stream._close_turn (lead must-fix #1 a/b)."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    trace_rows = [r for r in ev
                  if r.get("kind") == "trace" and r.get("id") == done["trace_id"]]
    # open + close
    assert len(trace_rows) == 2, trace_rows


def test_D4_done_session_id_matches_start_session_id(sse_client):
    """D4: start and done frames carry the same session_id."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    start = sse_client.parse_data(frames[0])
    done = sse_client.parse_data(frames[-1])
    assert start["session_id"] == done["session_id"]


def test_D5_done_carries_iters_field(sse_client):
    """D5: design §3 -- `iters` is always recorded (>= 1) because _close_turn
    writes it unconditionally."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    assert "iters" in done
    assert done["iters"] >= 1


def test_D6_done_has_cost_when_close_trace_recorded_it(sse_client):
    """D6: when close-trace wrote `cost_usd`, done echoes it. fake_llm uses
    provider='ollama' so cost_usd=0.0 is recorded."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    # FakeLLM provider is "ollama" so a cost row IS written. Pin both fields.
    assert "cost_usd" in done
    assert "total_tokens" in done


def test_D7_done_omits_cost_when_close_trace_skipped_it(
    sse_client, monkeypatch, fake_llm,
):
    """D7 (nice-to-have #3 -- monkeypatch _next_usage rather than new
    fixture): when last_usage is None (provider didn't report tokens),
    close-trace skips cost/token fields and done frame omits them too."""
    # Override usage so respond_stream's last_usage stays None.
    def _no_usage(self, text):  # noqa: ARG001
        self.last_usage = None
        return None
    monkeypatch.setattr(
        type(fake_llm), "_next_usage", _no_usage, raising=True,
    )
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    # iters is always there, but token / cost fields must be absent.
    assert "iters" in done
    assert "cost_usd" not in done
    assert "prompt_tokens" not in done
    assert "completion_tokens" not in done
    assert "total_tokens" not in done


def test_D8_done_strict_byte_field_order_via_raw_find(sse_client):
    """D8: dev micro-decision #2 generalised -- done frame's RAW BYTES have
    the keys in monotone-increasing order. Same defense as F4."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done_raw = frames[-1]["data"].encode("utf-8")
    keys = ["trace_id", "citation_quality", "session_id"]
    positions = [done_raw.find(f'"{k}"'.encode("utf-8")) for k in keys]
    assert positions == sorted(positions)


def test_D9_done_has_non_empty_trace_id_and_error_frame_omits_it(
    sse_client, fake_llm,
):
    """D9 (lead must-fix #3 -- negative + positive together):
      - done frame: `trace_id` key present AND non-empty AND matches close-trace
      - error frame: `trace_id` key MUST NOT appear in the data
    Combined into one test so the error/happy paths can never lie about
    one another (B6/B13 v0.8 lesson)."""
    # Happy path first.
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    assert "trace_id" in done
    assert isinstance(done["trace_id"], str) and done["trace_id"]
    # Match close-trace on disk.
    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    close_rows = [r for r in ev
                  if r.get("kind") == "trace" and r.get("id") == done["trace_id"]
                  and "citation_quality" in r]
    assert len(close_rows) == 1

    # Now error path -- force BudgetExceeded via fake_llm.chat patch.
    real_chat = fake_llm.chat

    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    try:
        frames2, _ = sse_client.post_sse(
            {"messages": [{"role": "user", "content": "boom"}]}
        )
    finally:
        fake_llm.chat = real_chat  # type: ignore[method-assign]
    # Locate the error frame.
    err_frames = [f for f in frames2 if f["event"] == "error"]
    assert err_frames, [f["event"] for f in frames2]
    err = sse_client.parse_data(err_frames[0])
    assert "trace_id" not in err, err


def test_async_bridge_done_emitted_after_close_trace_disk_write(sse_client):
    """LEAD MUST-FIX #1: contract -- `done` SSE event MUST be yielded AFTER
    the close-trace event is on disk. Inspecting events.jsonl from inside
    the test (after the response is fully consumed) is enough to confirm
    that ordering, because the server only yields `done` after merging the
    on-disk close-trace into the frame data. (If `done` yielded before the
    file write, the merge would see an empty dict and the test would catch
    the missing iters field.)"""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])
    # iters comes from the on-disk close-trace (server.py:258).
    assert "iters" in done and done["iters"] >= 1
    # And the close-trace row IS on disk now.
    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    close_rows = [r for r in ev
                  if r.get("kind") == "trace"
                  and r.get("id") == done["trace_id"]
                  and "citation_quality" in r]
    assert len(close_rows) == 1


def test_R_1_two_sse_calls_same_session_id_share_session_in_db(sse_client):
    """R-1 (R-NEW echo of v0.10 F3): two SSE calls with the same
    client-supplied sid produce slices that all carry that sid in the DB."""
    sid = "sse-client-sid-42"
    for content in ("first", "second"):
        frames, _ = sse_client.post_sse(
            {"messages": [{"role": "user", "content": content}]}, sid=sid,
        )
        start = sse_client.parse_data(frames[0])
        assert start["session_id"] == sid

    from mneme import paths
    from mneme.memory import store
    cx = store.connect(paths.DB_PATH)
    sessions = {r[0] for r in cx.execute(
        "SELECT DISTINCT session_id FROM slices")}
    cx.close()
    assert sessions == {sid}


def test_R_2_sse_minted_session_id_is_well_formed_ulid(sse_client):
    """R-2: when no sid supplied, server mints a real ULID into start frame."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    sid = sse_client.parse_data(frames[0])["session_id"]
    assert _ULID_RE.fullmatch(sid)


def test_R_3_sse_path_writes_session_id_into_trace_events(sse_client):
    """R-3 (echo of v0.10 F5 on SSE path): the trace events on disk carry
    session_id == the one in the start/done frames."""
    sid = "sse-traceme-77"
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "audit me"}]}, sid=sid,
    )
    done = sse_client.parse_data(frames[-1])

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    trace_rows = [r for r in ev
                  if r.get("kind") == "trace"
                  and r.get("id") == done["trace_id"]]
    assert len(trace_rows) == 2  # open + close
    for r in trace_rows:
        assert r["session_id"] == sid, r


def test_R_4_done_iters_matches_close_trace_on_disk(sse_client):
    """R-4: `iters` value in the done frame MUST equal the close-trace
    `iters` on disk. The SSE handler reads via events.explain so this is
    transitive, but pin it so a mutation that hard-codes iters=1 fires."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done = sse_client.parse_data(frames[-1])

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    close = [r for r in ev
             if r.get("kind") == "trace"
             and r.get("id") == done["trace_id"]
             and "citation_quality" in r][0]
    assert done["iters"] == close["iters"]


def test_R_11_start_frame_field_order_negative_swap_would_fail(sse_client):
    """R-11 (dev micro-decision #2 negative): explicit negative -- prove
    that if a mutation swapped the start frame fields to `{turn_id,
    session_id}`, this test would fail. We check that session_id comes
    BEFORE turn_id strictly (not >=)."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    data = frames[0]["data"]
    p_sess = data.find('"session_id"')
    p_turn = data.find('"turn_id"')
    assert p_sess >= 0 and p_turn >= 0
    # Strict ordering -- p_sess MUST be < p_turn.
    assert p_sess < p_turn


def test_R_13_done_frame_only_emitted_once(sse_client):
    """R-13: there is exactly one `done` frame per request. Catches a
    mutation that yields done twice (e.g. in both the finally and the
    happy path)."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    done_count = sum(1 for f in frames if f["event"] == "done")
    assert done_count == 1


def test_E1_session_id_none_in_request_mints_new_ulid_on_sse(sse_client):
    """E1: SSE path matches the JSON path on session_id minting rules:
    None / missing -> new ULID."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    start = sse_client.parse_data(frames[0])
    assert _ULID_RE.fullmatch(start["session_id"])


def test_E2_session_id_empty_string_mints_new_ulid_on_sse(sse_client):
    """E2: empty string -> mint new ULID."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}, sid="",
    )
    start = sse_client.parse_data(frames[0])
    assert _ULID_RE.fullmatch(start["session_id"])


def test_E3_session_id_whitespace_only_mints_new_ulid_on_sse(sse_client):
    """E3: `"   "` -> mint new ULID (whitespace-only branch in chat handler)."""
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}, sid="   ",
    )
    start = sse_client.parse_data(frames[0])
    assert _ULID_RE.fullmatch(start["session_id"])


def test_E4_session_id_unknown_string_accepted_silently_on_sse(sse_client):
    """E4: arbitrary unknown sid string round-trips verbatim. The server is
    stateless about session existence (E6 v0.10 lesson)."""
    sid = "totally-unknown-id-99999"
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}, sid=sid,
    )
    start = sse_client.parse_data(frames[0])
    assert start["session_id"] == sid


def test_E6_no_user_message_returns_400_json_even_when_sse_requested(
    sse_client,
):
    """E6: design §5 -- pre-stream validation (no user msg) returns 400 +
    JSON even when SSE was requested. The SSE channel is NOT opened."""
    # Use raw .stream because post_sse parses on a 200 contract.
    with sse_client.stream(
        "POST", "/chat",
        json={"messages": []},
        headers={"Accept": "text/event-stream"},
    ) as resp:
        body = b"".join(resp.iter_bytes())
        status = resp.status_code
        ct = resp.headers.get("content-type", "")
    assert status == 400, body
    # Content-Type is JSON not SSE.
    assert "application/json" in ct.lower(), ct
    # And the body does NOT start with `event: ` (no SSE leak).
    assert not body.startswith(b"event: ")


def test_E7_no_user_message_returns_400_when_only_assistant_msgs(sse_client):
    """E7: only assistant messages -> 400. The handler filters to user
    messages first then errors if empty."""
    with sse_client.stream(
        "POST", "/chat",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
        headers={"Accept": "text/event-stream"},
    ) as resp:
        status = resp.status_code
        body = b"".join(resp.iter_bytes())
    assert status == 400, body


# ============================================================================
# Unit D — chat handler header / negotiation
# ============================================================================


def test_G3_sse_response_status_is_200(sse_client):
    """G3: SSE response status is exactly 200 (StreamingResponse default)."""
    sse_client.post_sse({"messages": [{"role": "user", "content": "hi"}]})
    assert sse_client.post_sse.last_status == 200  # type: ignore[attr-defined]


def test_G4_must_fix_5_three_sse_headers_complete(sse_client):
    """G4 (lead MUST-FIX #5): SSE response carries ALL THREE headers --
    Content-Type, Cache-Control, X-Accel-Buffering. Literal values pinned
    per design §2. Missing one defeats either proxy buffering (no nginx
    survival) or caching (proxy may serve stale data)."""
    sse_client.post_sse({"messages": [{"role": "user", "content": "hi"}]})
    h = sse_client.post_sse.last_headers  # type: ignore[attr-defined]
    # Case-insensitive header access via dict from TestClient response.
    h_lower = {k.lower(): v for k, v in h.items()}
    assert h_lower.get("content-type") == "text/event-stream; charset=utf-8"
    assert h_lower.get("cache-control") == "no-cache"
    assert h_lower.get("x-accel-buffering") == "no"


def test_R_NEW_v010_json_response_still_works_byte_identical_on_post_merge(
    http_client,
):
    """R-NEW (lead MUST-FIX #4): the v0.10 JSON path keeps the F4 field
    order pin alive after the v0.13 merge. Plain JSON POST -> response
    body field order: text, trace_id, citation_quality, session_id.
    This is the closest single test to the byte-identical KPI."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    raw = resp.text
    expected = ["text", "trace_id", "citation_quality", "session_id"]
    positions = [raw.find(f'"{k}"') for k in expected]
    assert all(p > 0 for p in positions), positions
    assert positions == sorted(positions), (
        f"v0.10 JSON field order broken: {dict(zip(expected, positions))}"
    )
    # Bonus: response Content-Type is application/json, NOT SSE.
    assert "application/json" in resp.headers.get("content-type", "").lower()


def test_R_5_silent_degrade_returns_json_not_status_alone(http_client):
    """R-5 (lead should-fix #7): Accept silent-degrade pins Content-Type
    application/json, not just status 200. Without this, a header swap
    to SSE would still pass an over-loose status-only check."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "").lower()
    # And NOT SSE.
    assert "text/event-stream" not in resp.headers.get("content-type", "").lower()


def test_R_15_no_tools_leak_on_both_json_and_sse_paths(
    sse_client, fake_llm,
):
    """R-15 (lead should-fix #5): no-tools invariant holds on BOTH paths.
    `tools_seen` after a call must be `[None]` (filtered to agent calls)
    on the SSE path, just like the JSON path (v0.10 B22). One test
    covers both."""
    # Plant a scripted tool call that should NEVER be consumed.
    from mneme.llm.client import ToolCall
    fake_llm.tool_call_script = [
        {"text": "ghost",
         "tool_calls": [ToolCall(id="t1", name="shell",
                                 arguments={"command": "ls"})],
         "stop_reason": "tool_use"},
    ]

    # SSE path.
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert frames[-1]["event"] == "done"
    # Filter to agent-loop chat calls (not concept extraction).
    agent_tools = [
        t for t, msgs in zip(fake_llm.tools_seen, fake_llm.messages_seen)
        if msgs and "STRICT JSON" not in (msgs[0].get("content") or "")
    ]
    # Should still be exactly [None] -- no tools declared via SSE.
    assert agent_tools == [None], agent_tools
    # Scripted tool call was NOT consumed (script length intact).
    assert len(fake_llm.tool_call_script) == 1


# ============================================================================
# Unit E — disconnect handling (F5 split: a + b)
# ============================================================================


def test_F5a_disconnect_before_first_chunk_writes_close_trace_no_aborted_event(
    sse_client, disconnect_mocker,
):
    """F5a (lead should-fix #1): disconnect detected BEFORE any delta byte
    is on the wire (bytes_streamed == 0). Contract:
      - close-trace IS written by respond_stream (mandatory)
      - NO `http_stream_aborted` event is written (telemetry gate guards
        on bytes_streamed > 0)
      - Done frame is NOT yielded (disconnected path returns early)
    """
    disconnect_mocker.disconnect_immediately()
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    # No done frame on the disconnected path.
    assert not any(f["event"] == "done" for f in frames)

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    # close-trace still written.
    close_rows = [r for r in ev
                  if r.get("kind") == "trace" and "citation_quality" in r]
    assert close_rows, ev
    # http_stream_aborted NOT written (bytes_streamed == 0 branch).
    aborted = [r for r in ev if r.get("kind") == "http_stream_aborted"]
    assert aborted == [], aborted


def test_F5b_disconnect_mid_stream_writes_close_trace_and_http_aborted(
    sse_client, disconnect_mocker,
):
    """F5b (lead should-fix #1): disconnect AFTER >=1 delta byte. Contract:
      - close-trace IS written
      - `http_stream_aborted` IS written (bytes_streamed > 0)
      - aborted event carries `trace_id`, `session_id`, `bytes_streamed`
    """
    # Allow some initial frames to land before flipping the disconnect.
    disconnect_mocker.disconnect_after(2)  # 2 False checks, then True
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    # At least the start frame on the wire.
    assert frames and frames[0]["event"] == "start"
    # No done frame (disconnected path).
    assert not any(f["event"] == "done" for f in frames)

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    close_rows = [r for r in ev
                  if r.get("kind") == "trace" and "citation_quality" in r]
    assert close_rows, ev
    aborted = [r for r in ev if r.get("kind") == "http_stream_aborted"]
    assert len(aborted) == 1, ev
    rec = aborted[0]
    assert rec.get("trace_id")
    assert rec.get("session_id")
    assert isinstance(rec.get("bytes_streamed"), int)
    assert rec["bytes_streamed"] > 0


def test_E5_disconnect_does_not_emit_error_frame(
    sse_client, disconnect_mocker,
):
    """E5: disconnect path NEVER yields an `error` SSE frame. The error
    frame is reserved for mid-stream EXCEPTIONS (BudgetExceeded etc.); a
    clean client hang-up is silent."""
    disconnect_mocker.disconnect_immediately()
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert not any(f["event"] == "error" for f in frames)


def test_N_2_disconnect_with_exotic_session_id_byte_roundtrip(
    sse_client, disconnect_mocker,
):
    """N-2 (nice-to-have #2): SSE path with an exotic session_id (emoji +
    multi-KB) round-trips verbatim into the start frame. Disconnect
    detection happens at request level; the sid encoding path must not
    panic when bytes contain non-BMP chars."""
    exotic = "rocket-" + ("\U0001F680" * 3) + "-" + ("x" * 2048)
    disconnect_mocker.never_disconnect()
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "hi"}]}, sid=exotic,
    )
    assert frames[0]["event"] == "start"
    start = sse_client.parse_data(frames[0])
    assert start["session_id"] == exotic


# ============================================================================
# Unit F — error path (mid-stream exceptions)
# ============================================================================


def test_F1_budget_exceeded_yields_error_frame_no_done(sse_client, fake_llm):
    """F1: BudgetExceeded raised mid-stream -> single `error` frame, NO
    `done` frame. Design §5 literal."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "boom"}]}
    )
    err = [f for f in frames if f["event"] == "error"]
    assert len(err) == 1, [f["event"] for f in frames]
    assert not any(f["event"] == "done" for f in frames)


def test_F2_error_frame_data_fields_pinned(sse_client, fake_llm):
    """F2 (lead must-fix #1c -- close-trace skip rules for BudgetExceeded):
    error frame data has EXACTLY `{error_type, message, session_id}` keys."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "boom"}]}, sid="my-sid",
    )
    err = [f for f in frames if f["event"] == "error"][0]
    data = sse_client.parse_data(err)
    assert set(data.keys()) == {"error_type", "message", "session_id"}
    assert data["error_type"] == "BudgetExceeded"
    assert "daily token budget exhausted" in data["message"]
    assert data["session_id"] == "my-sid"


def test_F3_error_frame_field_order_pinned(sse_client, fake_llm):
    """F3 (dev micro-decision #2 negative variant): error frame fields in
    pinned order `error_type, message, session_id`."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "boom"}]}
    )
    err = [f for f in frames if f["event"] == "error"][0]
    raw = err["data"]
    positions = [raw.find(f'"{k}"') for k in
                 ("error_type", "message", "session_id")]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions), positions


def test_F4a_error_frame_short_message_not_truncated(sse_client, fake_llm):
    """F4a (lead should-fix #8 branch 1): str(exc) <= 500 -> NO truncation
    suffix appended. Direct unit on _error_frame_data is the cleanest pin."""
    exc = ValueError("a short msg")
    data = _error_frame_data(exc, "sid-x")
    assert data["message"] == "a short msg"
    assert "(truncated)" not in data["message"]
    # Total length sanity.
    assert len(data["message"]) <= 500


def test_F4b_error_frame_long_message_truncated_with_ellipsis_suffix():
    """F4b (lead should-fix #8 branch 2): str(exc) > 500 -> sliced to first
    500 chars plus U+2026 + `(truncated)`. Total length <= 513."""
    exc = ValueError("x" * 600)
    data = _error_frame_data(exc, "sid-x")
    msg = data["message"]
    # Total length is 500 + len("…(truncated)") = 500 + 12 = 512.
    assert len(msg) <= 513, len(msg)
    # First 500 chars are the original.
    assert msg.startswith("x" * 500)
    # Suffix is U+2026 + "(truncated)".
    assert msg.endswith("…(truncated)")
    # And the U+2026 is a single codepoint NOT three ASCII dots.
    assert "..." not in msg


def test_F7_close_trace_NOT_written_on_budget_exceeded_path(
    sse_client, fake_llm,
):
    """F7: lead must-fix #1c -- when BudgetExceeded fires BEFORE the first
    real chunk (i.e., during the first generator pull), `respond_stream`
    never reaches `_close_turn`. So NO close-trace row appears on disk.
    (Pre-call trace IS written; the open-trace happens before chat()).
    """
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "boom"}]}
    )
    assert any(f["event"] == "error" for f in frames)

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    # Close-trace rows carry citation_quality. There MUST be none for this
    # turn -- BudgetExceeded fires before _close_turn.
    close_rows = [r for r in ev
                  if r.get("kind") == "trace" and "citation_quality" in r]
    assert close_rows == [], close_rows


def test_E8_error_path_no_http_stream_aborted_event(sse_client, fake_llm):
    """E8 (lead must-fix #1 echo + plan E8): the error path NEVER writes
    `http_stream_aborted` either (close-trace itself was not written, so
    nothing to "abort" semantically). Pin on disk."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    sse_client.post_sse(
        {"messages": [{"role": "user", "content": "boom"}]}
    )
    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    aborted = [r for r in ev if r.get("kind") == "http_stream_aborted"]
    assert aborted == [], aborted


def test_E9_arbitrary_exception_still_yields_error_frame(sse_client, fake_llm):
    """E9: any exception (not just BudgetExceeded) maps to an error frame
    with `error_type == type(exc).__name__`. Pin the generic branch."""
    class WeirdError(RuntimeError):
        pass

    def boom(*args, **kwargs):  # noqa: ARG001
        raise WeirdError("something broke")
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    err = [f for f in frames if f["event"] == "error"]
    assert len(err) == 1
    data = sse_client.parse_data(err[0])
    assert data["error_type"] == "WeirdError"
    assert "something broke" in data["message"]


def test_N_1_sse_call_does_not_write_tool_audit_or_tool_result(
    sse_client, fake_llm,
):
    """N-1 (lead should-fix #6): SSE path NEVER writes `tool_audit` /
    `tool_result` events. Byte-snapshot events.jsonl before/after the SSE
    call and grep for the kinds."""
    from mneme.llm.client import ToolCall
    fake_llm.tool_call_script = [
        {"text": "ghost",
         "tool_calls": [ToolCall(id="t1", name="shell",
                                 arguments={"command": "ls"})],
         "stop_reason": "tool_use"},
    ]
    from mneme import paths
    before = paths.EVENTS_PATH.read_text(encoding="utf-8") \
        if paths.EVENTS_PATH.exists() else ""
    sse_client.post_sse({"messages": [{"role": "user", "content": "hi"}]})
    after = paths.EVENTS_PATH.read_text(encoding="utf-8")
    new = after[len(before):]
    # Neither kind appears in the post-call additions.
    assert "tool_audit" not in new
    assert "tool_result" not in new


def test_R_9_error_frame_message_str_exc_exact(sse_client, fake_llm):
    """R-9: error frame `message` is `str(exc)` verbatim (modulo 500-char
    truncation). For BudgetExceeded(1000, 100) the message matches the
    literal in mneme/llm/client.py:70."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    err = [f for f in frames if f["event"] == "error"][0]
    data = sse_client.parse_data(err)
    # Matches the literal in BudgetExceeded.__init__.
    assert data["message"] == "daily token budget exhausted: 1,000/100"


def test_R_10_error_frame_session_id_echoes_caller_supplied_sid(
    sse_client, fake_llm,
):
    """R-10: error frame `session_id` is the same as what the request
    supplied (or the minted ULID if none). Bug guard: if the server
    accidentally used turn_id in error.session_id, this fires."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}, sid="my-sid-99",
    )
    err = [f for f in frames if f["event"] == "error"][0]
    data = sse_client.parse_data(err)
    assert data["session_id"] == "my-sid-99"


def test_R_12_error_frame_data_has_exactly_three_keys(sse_client, fake_llm):
    """R-12 (negative double pin of D9): error frame data set is EXACTLY
    {error_type, message, session_id}. No extra keys (trace_id, etc.)
    Catches a mutation that adds trace_id to error data."""
    def boom(*args, **kwargs):  # noqa: ARG001
        raise BudgetExceeded(1_000, 100)
    fake_llm.chat = boom  # type: ignore[method-assign]
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    err = [f for f in frames if f["event"] == "error"][0]
    data = sse_client.parse_data(err)
    assert set(data.keys()) == {"error_type", "message", "session_id"}


# ============================================================================
# Source-grep ratchets -- belt-and-braces invariants the runtime layer alone
# cannot catch (v0.10 lesson: grep + runtime double pin)
# ============================================================================


def test_R_grep_sse_path_passes_allowed_tools_empty_explicit():
    """Belt-and-braces grep ratchet: the SSE _sse_event_stream MUST pass
    `allowed_tools=[]` to respond_stream, even though `confirm_cb=None`
    alone gates tools off. Self-flag finding: M-16 mutation (drop
    `allowed_tools=[]`) does not affect observable behavior because the
    confirm_cb=None guard fires first -- but per v0.9 lesson, **belt-
    and-braces invariants need source-level pins too**.
    """
    src = (Path(__file__).parent.parent / "mneme/server.py").read_text(
        encoding="utf-8"
    )
    # The SSE branch literal:
    assert "confirm_cb=None, allowed_tools=[]," in src, (
        "SSE branch lost the explicit allowed_tools=[] belt-and-braces guard"
    )
    # And the JSON branch keeps the same guarantee.
    # (v0.10 R-15 covers JSON; this asserts the literal stays.)
    assert "allowed_tools=[]" in src


def test_R_grep_pool_max_workers_one_for_sqlite_affinity():
    """Belt-and-braces grep ratchet for SQLite thread affinity. The dev
    finding: `ThreadPoolExecutor(max_workers=1)` keeps every sync hop on
    the same OS thread, satisfying SQLite's default
    `check_same_thread=True`. Mutation M-21 (`max_workers=4`) doesn't
    fire observably in the small-load test environment, but the
    invariant is real and load-dependent. Pin it at the source level so
    a refactor that bumps max_workers gets caught by CI."""
    src = (Path(__file__).parent.parent / "mneme/server.py").read_text(
        encoding="utf-8"
    )
    assert "ThreadPoolExecutor(max_workers=1)" in src, (
        "SQLite thread affinity guard ThreadPoolExecutor(max_workers=1) lost"
    )


# ============================================================================
# G-section -- KPI: v0.10 baseline tests untouched (zero modification)
# ============================================================================
#
# Lead should-fix #2: explicitly name the KPI files. The actual git-diff
# check sits in test_R_16 below (skip-if-no-origin per should-fix #3).


def test_R_16_v010_baseline_tests_byte_identical_to_main():
    """R-16 (lead should-fix #3 -- skipif sandbox lacks origin/main):
    `git diff main -- tests/test_server_chat.py tests/test_server_chat_session.py`
    MUST be empty. Demonstrates the v0.13 work added zero edits to v0.10
    baseline tests (byte-identical regression KPI)."""
    # Sandbox detection: skip if origin/main is unreachable.
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        # Try plain `main` (local).
        res2 = subprocess.run(
            ["git", "rev-parse", "--verify", "main"],
            capture_output=True, text=True,
        )
        if res2.returncode != 0:
            pytest.skip("no origin/main or main ref available in sandbox")
        base = "main"
    else:
        base = "origin/main"

    files = [
        "tests/test_server_chat.py",
        "tests/test_server_chat_session.py",
    ]
    diff = subprocess.run(
        ["git", "diff", base, "--"] + files,
        capture_output=True, text=True,
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout == "", (
        f"KPI violation: v0.10 baseline tests modified vs {base}:\n"
        f"{diff.stdout[:2000]}"
    )


# ============================================================================
# Smoke -- async bridge / sanity for the running loop fixture
# ============================================================================


def test_event_loop_runs_with_sse_client(sse_client):
    """Smoke check: the TestClient runs the async handler successfully even
    when respond_stream is dispatched through asyncio.to_thread+pool. If
    this ever fails, every other test fails with the same error."""
    # Should not raise.
    frames, _ = sse_client.post_sse(
        {"messages": [{"role": "user", "content": "smoke"}]}
    )
    assert frames[0]["event"] == "start"
    assert frames[-1]["event"] == "done"


# Bring asyncio into scope so ruff doesn't complain about unused import.
assert asyncio
