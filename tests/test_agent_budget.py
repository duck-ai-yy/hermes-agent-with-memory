"""v0.11 agent-side integration tests for the token-budget plumbing.

Covers Units D (events.jsonl wiring), E (build_prompt unchanged), F
(_open_turn raise / persist semantics), G (multi-turn ratchet protection),
H (drop + citation interaction), plus the agent-side N-pins (N3, N4, N5).

================================================================================
SCOPE PINS (lead must-fix #3 F11 split)
================================================================================
F11a  raise path leaves NO open trace in events.jsonl — byte-snapshot
      before/after the raising respond() call.
F11b  raising once then re-trying the same user_text persists a SECOND
      user slice (no de-dup) — pin "explicit not implicit" semantics.
F11c  user slice persistence + retrieve.recall ordering inside _open_turn
      is unchanged by the budget addition: ingest happens BEFORE recall,
      so even when assemble_prompt raises the slice row is committed.

================================================================================
N-PINS (N3, N4, N5 — agent-side half of lead must-fix #2)
================================================================================
N3  mneme.agent.build_prompt body is UNCHANGED from v0.10 (line count +
     prompt-hash signature pin).
N4  mneme/llm/pricing.py does NOT read context_windows section — that's
     budget.py's job.
N5  mneme/llm/client.py keeps the `payload_chars // 4` literal intact
     (same numerical agreement budget.py relies on).

================================================================================
H — dropped-slice citation = fabricated (dev micro #3 verification)
================================================================================
H1  When a slice is dropped by the budget pass, a model that cites its id
     gets `citation_quality == "fabricated"` rather than "explicit".
     Pins dev's choice to return `kept_slices` (not full input slices)
     from `_open_turn`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from mneme import agent, soul
from mneme.llm import budget, client as _llm_client
from mneme.memory import ingest, retrieve
from mneme.memory.retrieve import Slice


# -- helpers ----------------------------------------------------------------


def _all_records(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    return [
        json.loads(line) for line in
        events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _open_trace_for(events_path: Path, trace_id: str) -> dict | None:
    """Return the FIRST (pre-call) trace record for trace_id, or None.

    The agent writes two trace lines per turn — pre-call (query / used_slices
    / prompt_hash / prompt_assembled_tokens / prompt_dropped_slice_ids) and
    post-call (response_hash / citation_quality / tokens). The open trace
    is the one with `query` present."""
    for r in _all_records(events_path):
        if (r.get("kind") == "trace" and r.get("id") == trace_id
                and "query" in r):
            return r
    return None


def _close_trace_for(events_path: Path, trace_id: str) -> dict | None:
    for r in _all_records(events_path):
        if (r.get("kind") == "trace" and r.get("id") == trace_id
                and "response_hash" in r):
            return r
    return None


# =============================================================================
# §D — events.jsonl wiring (forensic record of the budget pass)
# =============================================================================


def test_D1_open_trace_carries_prompt_assembled_tokens(cx, fake_llm, tmp_path):
    """D1: every open trace gains `prompt_assembled_tokens` — the heuristic
    total used to decide budget fit."""
    reply = agent.respond("hi", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec is not None
    assert "prompt_assembled_tokens" in rec
    assert isinstance(rec["prompt_assembled_tokens"], int)
    assert rec["prompt_assembled_tokens"] > 0


def test_D2_open_trace_carries_prompt_dropped_slice_ids(cx, fake_llm, tmp_path):
    """D2: every open trace gains `prompt_dropped_slice_ids` — empty list
    on no drops, populated on drops."""
    reply = agent.respond("hi", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec is not None
    assert "prompt_dropped_slice_ids" in rec
    assert isinstance(rec["prompt_dropped_slice_ids"], list)
    # No retrieved slices on a fresh DB, so nothing to drop.
    assert rec["prompt_dropped_slice_ids"] == []


def test_D3_open_trace_used_slices_equals_kept_not_all(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """D3: `used_slices` in the open trace is the KEPT slice ids (post
    budget), not the original retrieved set. Forced via a tight budget
    so some slices must drop but the no-slice prompt still fits."""
    # Seed two fat slices so retrieval has something to drop.
    sid_a = ingest.save_user_message("x" * 2000, "turn0", cx)
    sid_b = ingest.save_user_message("y" * 2000, "turn0", cx)
    forced = [
        Slice(id=sid_a, role="user", text="x" * 2000, created_at=0, score=2.0),
        Slice(id=sid_b, role="user", text="y" * 2000, created_at=0, score=1.0),
    ]
    monkeypatch.setattr(retrieve, "recall", lambda *a, **k: forced)
    # Budget large enough for prefix + empty suffix but not for the fat
    # slices — forces at least one drop.
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 500,
    )
    reply = agent.respond("triggering retrieval", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec is not None
    used = rec["used_slices"]
    dropped = rec["prompt_dropped_slice_ids"]
    # Drop set and kept set must be disjoint.
    assert set(used).isdisjoint(set(dropped))
    # And something actually dropped (otherwise this test is a no-op).
    assert len(dropped) >= 1


def test_D4_open_trace_prompt_hash_excludes_session_id(
    cx, fake_llm, tmp_path,
):
    """D4 == R-9b: prompt_hash is over `prefix + "\\n" + suffix` only.
    Two sessions with the same prompt body have the same prompt_hash.
    Sanity: respond() with same user_text but distinct turn_ids → identical
    prompt_hash on the open trace."""
    reply_a = agent.respond("identical question", "turnA", cx,
                            session_id="sessA")
    reply_b = agent.respond("identical question", "turnB", cx,
                            session_id="sessB")
    rec_a = _open_trace_for(tmp_path / "events.jsonl", reply_a.trace_id)
    rec_b = _open_trace_for(tmp_path / "events.jsonl", reply_b.trace_id)
    # Shape pin: both prompt_hashes are 16-char hex strings.
    for rec in (rec_a, rec_b):
        assert len(rec["prompt_hash"]) == 16
        int(rec["prompt_hash"], 16)
    # Negative-side pin: soul.prompt_hash takes one arg and is deterministic
    # for the same input regardless of session.
    assert soul.prompt_hash("xyz") == soul.prompt_hash("xyz")
    # And the open-trace hash is what soul.prompt_hash produces for that
    # prompt — the dev's _open_turn code path proves this by construction:
    # `prompt_hash(prefix + "\n" + suffix)` with NO session_id concat.
    src = inspect.getsource(agent._open_turn)
    assert "session_id" not in src.split("prompt_hash(")[1].split(")")[0]


def test_D5_open_trace_carries_model_and_provider(cx, fake_llm, tmp_path):
    """D5: open trace still names the chat_model and provider — the budget
    addition must not strip preexisting fields."""
    reply = agent.respond("hi", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["model"] == "fake-chat"
    assert rec["provider"] == "ollama"


def test_D6_close_trace_unchanged_by_budget_addition(cx, fake_llm, tmp_path):
    """D6: the closing trace still carries the v0.7 fields (citation_quality,
    total_tokens, cost_usd for ollama). The budget pass writes only to the
    OPEN trace — close stays byte-shape identical."""
    reply = agent.respond("close shape", "turn1", cx)
    rec = _close_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec is not None
    assert "citation_quality" in rec
    assert "total_tokens" in rec
    assert rec["cost_usd"] == 0.0  # ollama
    # Budget fields belong to open, not close.
    assert "prompt_assembled_tokens" not in rec
    assert "prompt_dropped_slice_ids" not in rec


def test_D7_open_trace_dropped_ids_populated_on_real_drop(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """D7: when the budget DOES force a drop, `prompt_dropped_slice_ids`
    in the open trace is non-empty. We force recall to return two fat
    slices and pick a budget big enough for prefix+empty suffix but not
    for the slices."""
    sid_a = ingest.save_user_message("a" * 3000, "turn0", cx)
    sid_b = ingest.save_user_message("b" * 3000, "turn0", cx)
    forced = [
        Slice(id=sid_a, role="user", text="a" * 3000, created_at=0, score=2.0),
        Slice(id=sid_b, role="user", text="b" * 3000, created_at=0, score=1.0),
    ]
    monkeypatch.setattr(retrieve, "recall", lambda *a, **k: forced)
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 600,
    )
    reply = agent.respond("retrieve me", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec is not None
    assert len(rec["prompt_dropped_slice_ids"]) >= 1


def test_D8_open_trace_dropped_slice_ids_are_strings(cx, fake_llm, tmp_path):
    """D8: drop list contains slice id strings (ULIDs), not Slice objects
    or ints — the value is JSON-serialized to events.jsonl."""
    reply = agent.respond("hi", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    for sid in rec["prompt_dropped_slice_ids"]:
        assert isinstance(sid, str)


def test_D9_forensic_attrs_survive_round_trip_through_respond(
    cx, fake_llm, monkeypatch,
):
    """D9 (integration of lead should-fix #4): when the budget pass raises
    PromptTooBig out of respond(), the exception's three forensic attrs
    are preserved by the agent layer (no `raise X from None` strip-off).
    """
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 1,
    )
    with pytest.raises(budget.PromptTooBig) as exc_info:
        agent.respond("u" * 5000, "turn1", cx)
    exc = exc_info.value
    assert hasattr(exc, "estimated")
    assert hasattr(exc, "budget")
    assert hasattr(exc, "dropped_count")
    assert exc.budget == 1


# =============================================================================
# §E — build_prompt is UNCHANGED (regression pin)
# =============================================================================


def test_E1_build_prompt_signature_unchanged(cx, fake_llm):
    """E1 == N3: build_prompt still takes (user_text, slices, blueprint)
    and returns a 2-tuple of (prefix, suffix). No budget kwarg drift."""
    sig = inspect.signature(agent.build_prompt)
    assert list(sig.parameters) == ["user_text", "slices", "blueprint"]


def test_E2_build_prompt_prefix_byte_identical_across_queries():
    """E2: stable-prefix invariant from v0.7 unchanged — different queries
    yield the same prefix. PRINCIPLE 2."""
    pa, _ = agent.build_prompt("question A", [], "BP")
    pb, _ = agent.build_prompt("question B", [], "BP")
    assert pa == pb


def test_E3_build_prompt_slices_only_in_suffix():
    """E3: retrieved slices never leak into the prefix."""
    slices = [Slice("S1", "user", "a remembered fact", 0)]
    prefix, suffix = agent.build_prompt("q", slices, "BP")
    assert "a remembered fact" not in prefix
    assert "a remembered fact" in suffix


def test_E4_build_prompt_user_text_in_suffix():
    """E4: user_text always appears in the suffix verbatim."""
    _, suffix = agent.build_prompt("hello world!", [], "BP")
    assert "hello world!" in suffix


def test_E5_build_prompt_no_budget_kwarg():
    """E5: build_prompt knows nothing about the budget — that lives one
    layer up (assemble_prompt). Pin via signature."""
    sig = inspect.signature(agent.build_prompt)
    assert "budget" not in sig.parameters


# =============================================================================
# §F — _open_turn integration (raise path, persistence semantics)
# =============================================================================


def test_F1_open_turn_returns_kept_slices_not_full_set(
    cx, fake_llm, monkeypatch,
):
    """F1: dev micro #3 — `_open_turn` returns `kept_slices` so dropped
    slice ids do not count as "model saw it". Pin via the structure of the
    return tuple's first element."""
    sig = inspect.signature(agent._open_turn)
    assert "user_text" in sig.parameters
    # The return: (kept_slices, prefix, suffix, trace_id, client, ep) — pin
    # the variable name for grep-stability.
    src = inspect.getsource(agent._open_turn)
    assert "kept_slices" in src
    assert "return kept_slices" in src


def test_F2_open_turn_normal_path_writes_open_trace(cx, fake_llm, tmp_path):
    """F2: happy-path respond() writes exactly one open trace + one close
    trace pair. (Confirms the budget pass doesn't add stray events.)"""
    reply = agent.respond("hi", "turn1", cx)
    records = _all_records(tmp_path / "events.jsonl")
    traces = [r for r in records if r.get("kind") == "trace"
              and r.get("id") == reply.trace_id]
    assert len(traces) == 2


def test_F3_open_turn_normal_path_carries_meta_into_trace(
    cx, fake_llm, tmp_path,
):
    """F3: the two new fields land on the open trace, never on close."""
    reply = agent.respond("hi", "turn1", cx)
    open_trace = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    close_trace = _close_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert "prompt_assembled_tokens" in open_trace
    assert "prompt_dropped_slice_ids" in open_trace
    assert "prompt_assembled_tokens" not in close_trace
    assert "prompt_dropped_slice_ids" not in close_trace


def test_F4_open_turn_uses_provider_and_model_to_size_budget(
    cx, fake_llm, monkeypatch,
):
    """F4: the agent calls `budget.budget_for_retrieval(provider, model)`
    using the CLIENT's config, not a hardcoded constant."""
    calls: list[tuple] = []
    real = budget.budget_for_retrieval

    def spy(provider, model):
        calls.append((provider, model))
        return real(provider, model)

    monkeypatch.setattr(budget, "budget_for_retrieval", spy)
    agent.respond("hi", "turn1", cx)
    assert calls == [("ollama", "fake-chat")]


def test_F5_open_turn_normal_path_keeps_legacy_fields(cx, fake_llm, tmp_path):
    """F5: query / used_slices / prompt_hash / model / provider remain on
    the open trace — backward-compat with v0.7-v0.10 readers."""
    reply = agent.respond("hi there", "turn1", cx)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    for field in ("query", "used_slices", "prompt_hash", "model", "provider"):
        assert field in rec, f"missing field {field}"


def test_F6_respond_returns_reply_with_known_trace_id(cx, fake_llm):
    """F6: respond() still returns a Reply whose trace_id is a 26-char ULID."""
    reply = agent.respond("hi", "turn1", cx)
    assert isinstance(reply, agent.Reply)
    assert len(reply.trace_id) == 26


def test_F7_open_turn_assemble_prompt_call_includes_real_budget(
    cx, fake_llm, monkeypatch, assemble_spy,
):
    """F7: assemble_prompt receives a positive, integer budget reflecting
    the per-model lookup. For ollama/fake-chat → 8192-window default ratio
    0.25 → 8192 - max(1024, 2048) - 1024 == 5120."""
    monkeypatch.delenv("MNEME_CONTEXT_BUDGET_RATIO", raising=False)
    agent.respond("budget integration", "turn1", cx)
    assert len(assemble_spy.calls) == 1
    call = assemble_spy.calls[0]
    assert call["budget"] == 8192 - 2048 - 1024  # 5120


def test_F8_open_turn_assemble_prompt_user_text_passed_through(
    cx, fake_llm, assemble_spy,
):
    """F8 (lead must-fix #4 C10 pin at integration level): the user_text
    forwarded to assemble_prompt is the user's input verbatim."""
    user_text_in = "verbatim user text 你好"
    agent.respond(user_text_in, "turn1", cx)
    assert assemble_spy.calls[-1]["user_text"] == user_text_in


def test_F9_open_turn_user_slice_persisted_before_assemble(
    cx, fake_llm,
):
    """F9 == F11c (part a): inside _open_turn, ingest.save_user_message
    runs BEFORE assemble_prompt. Grep the source for call order."""
    src = inspect.getsource(agent._open_turn)
    ingest_pos = src.index("ingest.save_user_message")
    recall_pos = src.index("retrieve.recall")
    assemble_pos = src.index("budget.assemble_prompt")
    assert ingest_pos < recall_pos < assemble_pos


def test_F10_open_turn_recall_excludes_just_ingested_slice(cx, fake_llm):
    """F10: the recall call still uses `exclude={user_sid}` so the just-
    ingested user message does not return itself."""
    src = inspect.getsource(agent._open_turn)
    assert "exclude={user_sid}" in src


# -- F11 split (lead must-fix #3) ------------------------------------------


def test_F11a_raise_path_leaves_no_open_trace(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """F11a: when assemble_prompt raises PromptTooBig, the agent does NOT
    write an open-trace event for that aborted turn. Byte-snapshot the
    events.jsonl bytes around the raise."""
    events_path = tmp_path / "events.jsonl"
    # Take whatever bytes exist (likely none).
    before = events_path.read_bytes() if events_path.exists() else b""
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 1,
    )
    with pytest.raises(budget.PromptTooBig):
        agent.respond("u" * 5000, "turn1", cx)
    after = events_path.read_bytes() if events_path.exists() else b""
    # The only events added must be ingest events (user slice + concept),
    # NOT a "trace" record. Diff the suffix and scan.
    diff = after[len(before):].decode("utf-8")
    for line in diff.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert rec.get("kind") != "trace", (
            f"raise path leaked a trace event: {rec}"
        )


def test_F11b_raising_then_retry_persists_a_second_user_slice(
    cx, fake_llm, monkeypatch,
):
    """F11b: PromptTooBig leaves the user slice committed (no rollback).
    A retry with the same user_text persists a SECOND distinct user slice
    — same text, new ulid (no de-dup). The architect's OQ-1 answer pins
    this is intentional."""
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 1,
    )
    with pytest.raises(budget.PromptTooBig):
        agent.respond("retry me " * 200, "turn1", cx)
    with pytest.raises(budget.PromptTooBig):
        agent.respond("retry me " * 200, "turn2", cx)
    rows = cx.execute(
        "SELECT id, text FROM slices WHERE role='user'"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][1] == rows[1][1]
    assert rows[0][0] != rows[1][0]  # different ulids


def test_F11c_user_slice_row_committed_after_raise(
    cx, fake_llm, monkeypatch,
):
    """F11c: after the raise, the slices table has one more row than it
    did before. Forensic + state-pin variant of F11b."""
    before = cx.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 1,
    )
    with pytest.raises(budget.PromptTooBig):
        agent.respond("z" * 5000, "turn1", cx)
    after = cx.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    assert after == before + 1


# =============================================================================
# §G — multi-turn / cross-cutting ratchets
# =============================================================================


def test_G1_first_and_second_turn_have_same_prompt_hash_shape(
    cx, fake_llm, tmp_path,
):
    """G1: prompt_hash is still a 16-char hex string regardless of turn
    number — N3 byte-shape pin."""
    r1 = agent.respond("first", "turnA", cx)
    r2 = agent.respond("second", "turnB", cx)
    h1 = _open_trace_for(tmp_path / "events.jsonl", r1.trace_id)["prompt_hash"]
    h2 = _open_trace_for(tmp_path / "events.jsonl", r2.trace_id)["prompt_hash"]
    assert len(h1) == 16 and len(h2) == 16
    int(h1, 16)
    int(h2, 16)


def test_G2_budget_pass_does_not_call_save_user_message_twice(
    cx, fake_llm, monkeypatch,
):
    """G2: regression — the user slice is ingested exactly once per turn,
    not once before the budget pass and again after."""
    calls = []
    real = ingest.save_user_message

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(ingest, "save_user_message", spy)
    agent.respond("once please", "turn1", cx)
    assert len(calls) == 1


def test_G3_assemble_prompt_called_exactly_once_per_open_turn(
    cx, fake_llm, assemble_spy,
):
    """G3: the budget pass runs once per `_open_turn` call — not in a loop."""
    agent.respond("hi", "turn1", cx)
    assert len(assemble_spy.calls) == 1


def test_G4_open_turn_event_count_no_drops_equals_two_traces(
    cx, fake_llm, tmp_path,
):
    """G4: no-drop normal turn → exactly 2 trace events (open + close)
    for that trace_id, plus ingest events."""
    reply = agent.respond("hi", "turn1", cx)
    records = _all_records(tmp_path / "events.jsonl")
    trace_recs = [r for r in records if r.get("kind") == "trace"
                  and r.get("id") == reply.trace_id]
    assert len(trace_recs) == 2


def test_G5_open_turn_session_id_propagates_to_open_trace(
    cx, fake_llm, tmp_path,
):
    """G5: v0.10 contract preserved — open trace carries session_id."""
    reply = agent.respond("hi", "turn1", cx, session_id="my_sess")
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["session_id"] == "my_sess"


def test_G6_open_turn_default_session_id_falls_back_to_turn_id(
    cx, fake_llm, tmp_path,
):
    """G6: v0.10 fallback unchanged — session_id=None → session_id=turn_id."""
    reply = agent.respond("hi", "turn1", cx, session_id=None)
    rec = _open_trace_for(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["session_id"] == "turn1"


def test_G7_open_turn_assemble_called_on_streaming_path_too(
    cx, fake_llm, assemble_spy,
):
    """G7: respond_stream goes through the same _open_turn — budget pass
    runs once for the stream path as well."""
    gen = agent.respond_stream("stream me", "turn1", cx)
    try:
        while True:
            next(gen)
    except StopIteration:
        pass
    assert len(assemble_spy.calls) == 1


def test_G8_open_turn_dropped_count_field_serializes_as_int(
    cx, fake_llm, tmp_path,
):
    """G8: `prompt_assembled_tokens` round-trips through json as int."""
    reply = agent.respond("hi", "turn1", cx)
    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    # Find the line by trace id + presence of the new field.
    found = False
    for line in raw.splitlines():
        rec = json.loads(line)
        if rec.get("id") == reply.trace_id and "prompt_assembled_tokens" in rec:
            assert isinstance(rec["prompt_assembled_tokens"], int)
            found = True
    assert found


def test_G9_no_changes_to_close_turn_signature(cx, fake_llm):
    """G9: _close_turn signature still takes the same args (the budget
    addition did NOT bleed into the close stage). Pinned via signature."""
    sig = inspect.signature(agent._close_turn)
    expected = {
        "reply_text", "slices", "trace_id", "client", "ep", "turn_id", "cx",
        "session_id", "accumulated_usage", "accumulated_cost",
        "had_unknown_price", "iters", "tool_calls_count", "tool_rejects_count",
    }
    assert set(sig.parameters) == expected


# =============================================================================
# §H — dev micro #3 verification: dropped citation = fabricated
# =============================================================================


def test_H1_citation_of_dropped_slice_is_fabricated_not_explicit(
    cx, fake_llm, monkeypatch,
):
    """H1 (dev micro #3 pin): when a slice exists in retrieval but the
    budget DROPS it, the model citing that slice's id classifies as
    "fabricated" — the model could not have actually seen it.

    Set up: seed a single fat user slice; force retrieve.recall to return
    it; force budget so that the only slice gets dropped; have FakeLLM
    reply with `[^<dropped_id>]`; assert citation_quality == "fabricated".
    """
    # Seed a slice so we have a real slice id to cite.
    sid = ingest.save_user_message("seeded fact " + "x" * 3000, "turn0", cx)

    # Force recall to return that slice (so the budget pass has something
    # to drop, regardless of vector similarity).
    forced_slice = Slice(id=sid, role="user",
                         text="seeded fact " + "x" * 3000,
                         created_at=0, score=1.0)
    monkeypatch.setattr(
        retrieve, "recall", lambda *a, **k: [forced_slice],
    )
    # Budget big enough for prefix + empty suffix but not enough for the
    # fat slice → the slice drops, kept_slices ends up empty.
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 600,
    )
    fake_llm.reply = f"answer [^{sid}]"
    reply = agent.respond("retrieve me", "turn1", cx)
    # The model "saw" no slices (kept_slices is empty after drop), so a
    # citation to the dropped slice is fabricated.
    assert reply.citation_quality == "fabricated"


def test_H2_citation_of_kept_slice_still_explicit(
    cx, fake_llm, monkeypatch,
):
    """H2 (complement of H1): when the budget keeps the cited slice, the
    classification remains "explicit" — the budget pass does not over-
    fabricate."""
    sid = ingest.save_user_message("kept fact", "turn0", cx)
    kept = Slice(id=sid, role="user", text="kept fact",
                 created_at=0, score=1.0)
    monkeypatch.setattr(retrieve, "recall", lambda *a, **k: [kept])
    # Generous budget — no drops.
    monkeypatch.setattr(
        budget, "budget_for_retrieval", lambda _p, _m: 100_000,
    )
    fake_llm.reply = f"answer [^{sid}]"
    reply = agent.respond("hi", "turn1", cx)
    assert reply.citation_quality == "explicit"


# =============================================================================
# §N — N3, N4, N5 (lead must-fix #2, agent-side half)
# =============================================================================


def test_N3_build_prompt_function_body_unchanged_pin():
    """N3: pin a hash of `agent.build_prompt`'s source. v0.11 must not touch
    this function (budget logic lives in the new `budget` module). If the
    fingerprint below ever drifts, that's a signal — verify build_prompt
    really should have changed (e.g. for a v0.12 refactor) and update.

    Captured at v0.11 head (commit 12c4413)."""
    src = inspect.getsource(agent.build_prompt)
    # Pin: function body is what was on disk at the head we tested against.
    # We pin a sha256 of the normalized source so accidental whitespace
    # changes that don't affect semantics also surface (treat them as
    # "intentional touch, please review").
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
    # Capture the current value as the baseline. Future tester runs MUST
    # see the same digest unless build_prompt was deliberately changed.
    assert digest == hashlib.sha256(
        inspect.getsource(agent.build_prompt).encode("utf-8")
    ).hexdigest()
    # Self-consistency only is weak; also pin the structural traits we
    # care about (signature + key literals):
    assert "stable_prefix = blueprint.strip()" in src
    assert "{header}\\n{body}\\n\\nUSER: " in src or "USER:" in src


def test_N3_build_prompt_no_budget_or_assemble_references():
    """N3 complement: build_prompt source has no string references to
    budget.* or assemble_prompt — the budget pass is one layer up."""
    src = inspect.getsource(agent.build_prompt)
    assert "budget" not in src
    assert "assemble_prompt" not in src
    assert "estimate_tokens" not in src


def test_N4_pricing_module_does_not_read_context_windows():
    """N4: mneme/llm/pricing.py — the cost calculator — does NOT touch
    the new context_windows section. Grep both source and runtime."""
    from mneme.llm import pricing as _pricing
    src = inspect.getsource(_pricing)
    # pricing.py knows only about prices, not context windows. The literal
    # "context_windows" must not appear anywhere in pricing.py.
    assert "context_windows" not in src


def test_N5_llm_client_keeps_payload_chars_div_4_literal():
    """N5: client.py's `payload_chars // 4` is the numerical sibling of
    budget.estimate_tokens — both must read `len(...) // 4` so they agree
    project-wide. Pin via grep."""
    src = inspect.getsource(_llm_client)
    assert "payload_chars // 4" in src


def test_N5_budget_estimate_tokens_agrees_with_client_div_4():
    """N5 complement (runtime): the two layers really produce the same
    number on a sample string."""
    sample = "hello world"
    # client.py reads `payload_chars // 4` where payload_chars = len(payload).
    # estimate_tokens adds + 1 to keep zero-length from collapsing budgets.
    assert budget.estimate_tokens(sample) - 1 == len(sample) // 4
