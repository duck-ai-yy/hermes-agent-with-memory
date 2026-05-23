"""embed cache, ingest two-stage transaction, retrieve, forget cascade."""

from __future__ import annotations

import pytest

from mneme.memory import forget, ingest, retrieve
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
