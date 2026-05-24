"""B14 / B15 / B16: per-turn cost + trace accumulation across loop rounds.

v0.7 contract: one close-trace per turn carries (provider, total_tokens,
cost_usd?). v0.8 must hold that invariant even when the loop makes N>1 LLM
calls. Cost is summed across rounds; if ANY round had unknown pricing the
entire turn's cost_usd is omitted (carries v0.7 boundary 1/2 forward).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mneme import agent
from mneme.llm.client import ToolCall, Usage


def _events(events_path) -> list[dict]:
    return [
        json.loads(line) for line in
        events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _close_traces(events_path, trace_id) -> list[dict]:
    return [
        r for r in _events(events_path)
        if r.get("id") == trace_id and "response_hash" in r
    ]


def _accept(name, args):  # noqa: ARG001
    return True


def _shell_call(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="shell", arguments={"command": "echo x"})


# -- B14: cost + token totals are SUMS across all loop rounds ---------------


def test_three_round_loop_sums_tokens_and_cost_into_one_close_trace(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """Pricing math: openai gpt-4o-mini is $0.15/1M input, $0.60/1M output.
    Three rounds with usages (10,20), (30,40), (50,60). Expected total
    tokens = 210; expected cost = sum of per-round cost via pricing.cost_usd.
    Anything other than a single close-trace with that exact sum fails."""
    fake_llm.config = SimpleNamespace(
        chat_model="gpt-4o-mini", embed_model="fake-embed", provider="openai",
    )
    # tool_call_script: 3 agent rounds. First two emit a tool call, third
    # is the final text response.
    fake_llm.tool_call_script = [
        {"text": "ok 1", "tool_calls": [_shell_call("t1")], "stop_reason": "tool_use"},
        {"text": "ok 2", "tool_calls": [_shell_call("t2")], "stop_reason": "tool_use"},
        {"text": "final answer", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    # usage_script applies to EVERY chat() pop. Ingest calls concept
    # extraction once before and once after the loop (None == auto), and
    # the agent loop pops one entry per round in between.
    fake_llm.usage_script = [
        None,                            # concept extract for user msg
        Usage(10, 20),                   # agent round 1
        Usage(30, 40),                   # agent round 2
        Usage(50, 60),                   # agent round 3 (final)
        None,                            # concept extract for assistant msg
    ]

    reply = agent.respond("hi", "turn1", cx, confirm_cb=_accept)
    closes = _close_traces(tmp_path / "events.jsonl", reply.trace_id)
    # B16 happens inline here: still exactly one close-trace.
    assert len(closes) == 1
    rec = closes[0]
    assert rec["prompt_tokens"] == 10 + 30 + 50
    assert rec["completion_tokens"] == 20 + 40 + 60
    assert rec["total_tokens"] == 210
    # Pricing: (10*0.15 + 20*0.60 + 30*0.15 + 40*0.60 + 50*0.15 + 60*0.60) / 1e6
    expected = (
        (10 + 30 + 50) * 0.15 + (20 + 40 + 60) * 0.60
    ) / 1_000_000
    assert abs(rec["cost_usd"] - expected) < 1e-12
    # iters / tool counters mirror the loop shape
    assert rec["iters"] == 3
    assert rec["tool_calls"] == 2
    assert rec["tool_rejects"] == 0


# -- B15: one unknown-priced round → entire turn's cost_usd OMITTED ---------


def test_one_unknown_priced_round_omits_cost_usd_for_whole_turn(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """v0.7 boundary 1/2: a turn whose cost includes any 'unknown' price must
    OMIT cost_usd entirely from the close-trace — historical sums in `stats`
    must never include a partial value. Token totals stay (they're known)."""
    fake_llm.config = SimpleNamespace(
        chat_model="gpt-4o-mini", embed_model="fake-embed", provider="openai",
    )
    fake_llm.tool_call_script = [
        {"text": "a", "tool_calls": [_shell_call("t1")], "stop_reason": "tool_use"},
        {"text": "b", "tool_calls": [_shell_call("t2")], "stop_reason": "tool_use"},
        {"text": "final", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    fake_llm.usage_script = [
        None,
        Usage(10, 20), Usage(30, 40), Usage(50, 60),
        None,
    ]
    # Inject "unknown price" on the SECOND round by patching pricing.cost_usd
    # to return None for that one call. agent._price_call swallows None into
    # (0.0, True) which flips had_unknown_price for the turn.
    real_cost = agent.pricing.cost_usd
    call_n = {"i": 0}

    def patched(provider, model, usage):
        call_n["i"] += 1
        if call_n["i"] == 2:
            return None
        return real_cost(provider, model, usage)

    monkeypatch.setattr(agent.pricing, "cost_usd", patched)

    reply = agent.respond("hi", "turn1", cx, confirm_cb=_accept)
    rec = _close_traces(tmp_path / "events.jsonl", reply.trace_id)[0]
    # Tokens still summed across all rounds.
    assert rec["total_tokens"] == 210
    # cost_usd OMITTED — must not be present in the close-trace dict.
    assert "cost_usd" not in rec, (
        f"cost_usd leaked into close-trace with unknown-price round: {rec}"
    )


# -- B16: exactly one close-trace per turn, regardless of loop length -------


def test_long_loop_writes_exactly_one_close_trace_record(
    cx, fake_llm, tmp_path,
):
    """Even a 4-round loop must produce exactly one close-trace row. Catches
    the mutation 'emit a close-trace per round' (would inflate stats sums)."""
    fake_llm.tool_call_script = [
        {"text": "r1", "tool_calls": [_shell_call("t1")], "stop_reason": "tool_use"},
        {"text": "r2", "tool_calls": [_shell_call("t2")], "stop_reason": "tool_use"},
        {"text": "r3", "tool_calls": [_shell_call("t3")], "stop_reason": "tool_use"},
        {"text": "r4 final", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("hi", "turn1", cx, confirm_cb=_accept)
    closes = _close_traces(tmp_path / "events.jsonl", reply.trace_id)
    assert len(closes) == 1
    assert closes[0]["iters"] == 4
    assert closes[0]["tool_calls"] == 3


# -- Ratchet: zero-round (no tools) still writes exactly one close-trace ----


def test_no_tool_path_still_writes_one_close_trace(cx, fake_llm, tmp_path):
    """v0.7 regression guard inside v0.8: confirm_cb=None path still gives
    1 close-trace, 1 iter, 0 tool_calls."""
    reply = agent.respond("hi", "turn1", cx)
    closes = _close_traces(tmp_path / "events.jsonl", reply.trace_id)
    assert len(closes) == 1
    assert closes[0]["iters"] == 1
    assert closes[0]["tool_calls"] == 0
