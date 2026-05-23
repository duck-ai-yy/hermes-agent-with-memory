"""End-to-end chat turn with the fake LLM, plus prompt-assembly invariants."""

from __future__ import annotations

import json

from mneme import agent
from mneme.memory.retrieve import Slice
from mneme.trace import events


def test_build_prompt_keeps_retrieval_out_of_the_prefix():
    slices = [Slice("S1", "user", "a remembered fact", 0)]
    prefix_a, _ = agent.build_prompt("question one", slices, "BLUEPRINT")
    prefix_b, suffix_b = agent.build_prompt("question two", [], "BLUEPRINT")
    # Prefix is byte-identical regardless of query or retrieved slices.
    assert prefix_a == prefix_b
    # Retrieved slices appear only in the suffix.
    assert "a remembered fact" not in prefix_a
    assert "question two" in suffix_b


def test_respond_runs_a_full_turn(cx, fake_llm):
    reply = agent.respond("what did I decide about Tauri?", "turn1", cx)

    assert reply.text == fake_llm.reply
    assert len(reply.trace_id) == 26
    # Default fake reply has no citation markers.
    assert reply.citation_quality == "coarse"

    # Both the user message and the assistant reply were persisted.
    roles = {r[0] for r in cx.execute("SELECT role FROM slices")}
    assert {"user", "assistant"} <= roles


def test_respond_writes_a_resolvable_trace(cx, fake_llm, tmp_path):
    reply = agent.respond("trace me", "turn1", cx)
    record = events.explain(tmp_path / "events.jsonl", reply.trace_id)
    assert record["query"] == "trace me"
    assert record["model"] == "fake-chat"
    assert record["citation_quality"] == "coarse"
    assert "prompt_hash" in record


def test_respond_marks_explicit_when_citation_resolves(cx, fake_llm):
    """An [^id] that matches a retrieved slice id classifies as "explicit"."""
    from mneme.memory import ingest

    # Seed a slice so retrieval has something to return, then grab its id.
    ingest.save_user_message("Tauri powers the desktop app", "turn0", cx)
    sid = cx.execute(
        "SELECT id FROM slices WHERE role='user' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()[0]

    fake_llm.reply = f"the answer [^{sid}]"
    reply = agent.respond("Tauri powers the desktop app", "turn1", cx)
    assert reply.citation_quality == "explicit"


def test_respond_marks_fabricated_when_citation_does_not_resolve(cx, fake_llm):
    """An [^id] that does not match any retrieved slice id is fabricated."""
    fake_llm.reply = "the answer [^DOES_NOT_EXIST]"
    reply = agent.respond("anything", "turn1", cx)
    assert reply.citation_quality == "fabricated"


def test_respond_stream_yields_chunks_then_returns_reply(cx, fake_llm):
    """The generator yields incremental chunks; StopIteration.value is the Reply."""
    fake_llm.reply = "streamed response"
    gen = agent.respond_stream("hi", "turn1", cx)
    chunks: list[str] = []
    reply = None
    while True:
        try:
            chunks.append(next(gen))
        except StopIteration as stop:
            reply = stop.value
            break

    assert len(chunks) >= 2  # FakeLLM splits into multiple chunks
    assert "".join(chunks) == "streamed response"
    assert reply.text == "streamed response"
    assert len(reply.trace_id) == 26
    # Assistant message persisted just like non-streaming respond.
    roles = {r[0] for r in cx.execute("SELECT role FROM slices")}
    assert {"user", "assistant"} <= roles


def test_respond_stream_writes_real_token_counts_to_trace(cx, fake_llm, tmp_path):
    """Token fields land on the closing trace event so `stats` can sum them."""
    gen = agent.respond_stream("count me", "turn1", cx)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        reply = stop.value

    record = events.explain(tmp_path / "events.jsonl", reply.trace_id)
    assert record["prompt_tokens"] == fake_llm.last_usage.prompt_tokens
    assert record["completion_tokens"] == fake_llm.last_usage.completion_tokens
    assert record["total_tokens"] == fake_llm.last_usage.total_tokens


def test_respond_non_stream_also_writes_token_counts_to_trace(cx, fake_llm, tmp_path):
    """Symmetry: non-streaming respond captures usage the same way."""
    reply = agent.respond("usage too", "turn1", cx)
    record = events.explain(tmp_path / "events.jsonl", reply.trace_id)
    assert "prompt_tokens" in record
    assert record["total_tokens"] == fake_llm.last_usage.total_tokens


def test_closing_trace_event_records_provider_for_budget_filtering(cx, fake_llm, tmp_path):
    """The closing trace event carries `provider` alongside `total_tokens` so
    `sum_cloud_tokens_since` can filter ollama vs cloud without joining lines."""
    reply = agent.respond("any", "turn1", cx)

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    closing = [
        r for r in records
        if r.get("id") == reply.trace_id and "total_tokens" in r
    ]
    assert len(closing) == 1
    assert closing[0]["provider"] == fake_llm.config.provider
