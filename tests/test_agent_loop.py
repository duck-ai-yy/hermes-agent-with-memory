"""v0.8 agent loop behaviour: tool-call round-trip, iter cap, rejection,
audit/result events, error paths. The hard part of M1 is that the *second*
LLM call must see a tool_result block in the messages list — happy-path
tests that only count chat_calls would lie. So most of these spy on
`fake_llm.messages_seen` directly (v0.6 lesson: assert error path, not "no
crash"). Confirm callback errors and UnknownTool are explicit edge cases.
"""

from __future__ import annotations

import json


from mneme import agent
from mneme.llm.client import ToolCall


# -- helpers -----------------------------------------------------------------


def _close_trace(events_path, trace_id) -> dict:
    """The post-call trace event (the one with response_hash)."""
    records = [
        json.loads(line) for line in
        events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    closes = [
        r for r in records
        if r.get("id") == trace_id and "response_hash" in r
    ]
    assert len(closes) == 1, f"expected 1 close-trace, got {len(closes)}"
    return closes[0]


def _events(events_path) -> list[dict]:
    return [
        json.loads(line) for line in
        events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _shell_call(call_id="t1", command="echo hi") -> ToolCall:
    return ToolCall(id=call_id, name="shell", arguments={"command": command})


def _agent_chat_indices(fake) -> list[int]:
    """Indices of chat() calls made by the agent loop (skip ingest's
    concept-extraction call, which uses a system prompt starting with the
    STRICT JSON marker)."""
    out = []
    for i, msgs in enumerate(fake.messages_seen):
        system = (msgs[0].get("content") if msgs else "") or ""
        if "STRICT JSON" in system:
            continue
        out.append(i)
    return out


def _agent_chat_count(fake) -> int:
    return len(_agent_chat_indices(fake))


def _agent_round(fake, n: int) -> list[dict]:
    """Return the round-`n` (1-indexed) messages list as the agent saw it."""
    indices = _agent_chat_indices(fake)
    return fake.messages_seen[indices[n - 1]]


def _agent_tools_seen(fake) -> list:
    indices = _agent_chat_indices(fake)
    return [fake.tools_seen[i] for i in indices]


def _accept(name, args):  # noqa: ARG001
    return True


def _reject(name, args):  # noqa: ARG001
    return False


# -- B1: no-tool happy path is byte-identical to v0.7 single-call -----------


def test_no_confirm_cb_keeps_single_call_v07_contract(cx, fake_llm):
    """B1: when confirm_cb=None, the agent loop must NOT declare tools; the
    one agent-loop chat call must be made with tools=None. This preserves the
    v0.7 contract used by /chat HTTP endpoint and pre-v0.8 callers.
    (ingest's separate concept-extraction chat call is filtered out — see
    _agent_chat_indices.)"""
    reply = agent.respond("hello", "turn1", cx)
    assert reply.text == "noted"
    assert _agent_chat_count(fake_llm) == 1
    assert _agent_tools_seen(fake_llm) == [None]


def test_no_confirm_cb_close_trace_records_iters_one_and_no_tool_counts(
    cx, fake_llm, tmp_path,
):
    """B1 (continued): close-trace must still carry iters=1, tool_calls=0,
    tool_rejects=0 — v0.7 turns must look like one-iter agent turns in stats."""
    reply = agent.respond("hello", "turn1", cx)
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["iters"] == 1
    assert rec["tool_calls"] == 0
    assert rec["tool_rejects"] == 0


# -- B2: one tool call + final reply (canonical happy path) ------------------


def test_one_tool_call_then_final_reply(cx, fake_llm, tmp_path, monkeypatch):
    """B2: the model asks for a tool on round 1, agent runs it, model returns
    final text on round 2. Assert iters=2, tool_calls=1, tool_rejects=0, and
    that the *final* reply text is the round-2 text (NOT the round-1 text,
    which the loop must NOT surface as the final reply)."""
    # Stub shell.execute so we don't shell out from the test runner.
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        # Round 1: ask for shell
        {"text": "I'll run that.", "tool_calls": [_shell_call("c1", "echo hi")],
         "stop_reason": "tool_use"},
        # Round 2: final text — what the user should see
        {"text": "final answer", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("do it", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "final answer"
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["iters"] == 2
    assert rec["tool_calls"] == 1
    assert rec["tool_rejects"] == 0


# -- B3: zero-tool-call first round (model never calls a tool) ---------------


def test_no_tool_calls_at_all_records_iters_one(cx, fake_llm, tmp_path):
    """B3: a confirm_cb is provided but the model never asks for a tool. Loop
    should exit on round 1 with iters=1, tool_calls=0. (Differs from B1: this
    one DOES declare tools to the provider; just none were used.)"""
    fake_llm.tool_call_script = [
        {"text": "no need for tools", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("no tools please", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "no need for tools"
    # The agent-loop chat call MUST have carried a tools list.
    assert _agent_tools_seen(fake_llm)[0] is not None
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["iters"] == 1
    assert rec["tool_calls"] == 0


# -- B4: iteration cap (model keeps requesting tools forever) ----------------


def test_iter_cap_aborts_with_exact_prefix(cx, fake_llm, tmp_path, monkeypatch):
    """B4: with MNEME_MAX_ITERS=3, a forever-calling model is cut off and the
    abort string starts with the dev-specified prefix
    'Agent loop hit the max iteration cap'. We assert *substring* (the prefix
    is what callers / users will see) but NOT generic 'abort'."""
    monkeypatch.setenv("MNEME_MAX_ITERS", "3")
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    # Make the model ask for a tool forever — script length > cap.
    fake_llm.tool_call_script = [
        {"text": f"call {i}", "tool_calls": [_shell_call(f"c{i}", "echo")],
         "stop_reason": "tool_use"}
        for i in range(6)
    ]
    reply = agent.respond("loop forever", "turn1", cx, confirm_cb=_accept)
    assert "Agent loop hit the max iteration cap" in reply.text
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["iters"] == 3
    # Three rounds, three tool calls scripted, all accepted.
    assert rec["tool_calls"] == 3
    assert rec["tool_rejects"] == 0


# -- B5: user rejects the tool — turn aborts, count includes the attempt ----


def test_reject_aborts_turn_and_counts_attempt(cx, fake_llm, tmp_path):
    """B5: confirm_cb returns False. The dev chose 'tool_calls counts every
    attempt including rejects', so a single reject yields tool_calls=1 AND
    tool_rejects=1. The turn ends with a canned abort message; we assert the
    user-visible string is the stock reject text (not a 'happy' final reply)."""
    fake_llm.tool_call_script = [
        {"text": "I want to run this", "tool_calls": [_shell_call("c1", "rm -rf /")],
         "stop_reason": "tool_use"},
        # Round-2 script entry MUST NOT be consumed — turn aborts on reject.
        {"text": "never reached", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("destroy", "turn1", cx, confirm_cb=_reject)
    assert reply.text == "Tool call rejected by user; turn aborted."
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["tool_calls"] == 1
    assert rec["tool_rejects"] == 1
    # Only round 1 of the *agent loop* — reject short-circuits before round 2.
    assert _agent_chat_count(fake_llm) == 1


# -- B6: tool error round-trips with is_error=True on the second call -------


def test_second_round_messages_carry_is_error_tool_result(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """B6 (the load-bearing one — v0.6 lesson): when the shell command exits
    non-zero, the tool_result block fed back to the model on round 2 must
    carry is_error=True. This is the *only* signal the model has that the
    tool failed; happy-path 'tool ran, count=1' tests would lie about this.

    Provider is ollama, so we look for role='tool' with content. The fake
    sets the default provider to 'ollama', so the result block uses the
    OpenAI/Ollama shape (role=tool). is_error doesn't ride on the wire for
    OpenAI/Ollama (no such field) but exit_code=2 in the formatted content
    proves the agent forwarded the failure faithfully."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellFail(cmd),  # exit_code=2
    )
    fake_llm.tool_call_script = [
        {"text": "I'll try", "tool_calls": [_shell_call("c1", "false")],
         "stop_reason": "tool_use"},
        {"text": "got the error, ok", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("run false", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "got the error, ok"

    # The agent loop made exactly two chat calls; inspect round 2.
    assert _agent_chat_count(fake_llm) == 2
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    # The exit_code line is how the LLM sees "this failed" on OpenAI/Ollama.
    assert "exit_code: 2" in content

    # The tool_result trace event AND the in-memory result block both carry
    # the structured is_error=True (the agent code path that builds the
    # next-round message uses this same dict).
    ev = _events(tmp_path / "events.jsonl")
    results = [e for e in ev if e.get("kind") == "tool_result"]
    assert len(results) == 1
    assert results[0]["exit_code"] == 2


def test_anthropic_tool_result_is_error_field_round_trips(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """B6 (Anthropic shape): on Anthropic the is_error field IS on the wire
    in the tool_result block. Switch provider, run the same failing-tool
    scenario, and confirm the next-round messages list carries
    is_error=True inside the tool_result content block."""
    fake_llm.config.provider = "anthropic"
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellFail(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "trying", "tool_calls": [_shell_call("c1", "false")],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("anthropic fail", "turn1", cx, confirm_cb=_accept)

    round2 = _agent_round(fake_llm, 2)
    user_msgs = [m for m in round2 if m.get("role") == "user"
                 and isinstance(m.get("content"), list)]
    # The tool_result is wrapped in a user message (Anthropic convention).
    assert user_msgs, "no anthropic tool_result user message in round 2"
    blocks = user_msgs[-1]["content"]
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True


# -- B17: audit events for accept and reject decisions -----------------------


def test_audit_event_logs_accepted_decision(cx, fake_llm, tmp_path, monkeypatch):
    """B17a: every tool call emits a tool_audit event with decision='accepted'
    when confirm_cb returns True. Provides the auditable breadcrumb required
    by principle 5 — the user can later prove a tool ran with their consent."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "echo hi")],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("audit accept", "turn1", cx, confirm_cb=_accept)
    ev = _events(tmp_path / "events.jsonl")
    audits = [e for e in ev if e.get("kind") == "tool_audit"]
    decisions = [a["decision"] for a in audits]
    # Pending breadcrumb is written first, then 'accepted' once the user OKs.
    assert "pending" in decisions
    assert "accepted" in decisions
    accepted = [a for a in audits if a["decision"] == "accepted"]
    assert accepted[0]["tool"] == "shell"
    assert accepted[0]["tool_call_id"] == "c1"


def test_audit_event_logs_rejected_decision(cx, fake_llm, tmp_path):
    """B17b: a rejection emits a tool_audit with decision='rejected'. There
    must be NO 'accepted' record for the same tool_call_id (auditor must be
    able to prove the call never ran)."""
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "rm -rf /")],
         "stop_reason": "tool_use"},
    ]
    agent.respond("audit reject", "turn1", cx, confirm_cb=_reject)
    ev = _events(tmp_path / "events.jsonl")
    audits = [e for e in ev if e.get("kind") == "tool_audit"]
    decisions = [a["decision"] for a in audits]
    assert "rejected" in decisions
    assert "accepted" not in decisions


# -- B18: tool loop is read-only with respect to the LLM client config ------


def test_agent_loop_does_not_mutate_client_config(cx, fake_llm, monkeypatch):
    """B18: the loop must not poke at client.config. Snapshot before/after.
    (v0.7 lesson: read-only contracts get violated when nobody asserts them.)
    """
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "echo")],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    before = dict(vars(fake_llm.config))
    agent.respond("readonly check", "turn1", cx, confirm_cb=_accept)
    after = dict(vars(fake_llm.config))
    assert before == after


# -- B19: MNEME_MAX_ITERS env override is honored at call time --------------


def test_max_iters_env_is_resolved_at_call_time_not_import(
    cx, fake_llm, monkeypatch,
):
    """B19: a test sets MNEME_MAX_ITERS *after* the module imported. The cap
    must come from os.environ at call time, not from a cached constant. We
    set cap=2 and prove the loop stopped at 2."""
    monkeypatch.setenv("MNEME_MAX_ITERS", "2")
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": f"call {i}", "tool_calls": [_shell_call(f"c{i}", "echo")],
         "stop_reason": "tool_use"}
        for i in range(5)
    ]
    reply = agent.respond("cap=2 test", "turn1", cx, confirm_cb=_accept)
    assert "max iteration cap (2)" in reply.text


def test_max_iters_invalid_env_falls_back_to_default(cx, fake_llm, monkeypatch):
    """B19b: a non-int env value must not crash; default applies."""
    monkeypatch.setenv("MNEME_MAX_ITERS", "not-a-number")
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    # Default is 6; provide 7 round-1 entries so cap fires.
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call(f"c{i}", "echo")],
         "stop_reason": "tool_use"}
        for i in range(7)
    ]
    reply = agent.respond("invalid env", "turn1", cx, confirm_cb=_accept)
    assert "max iteration cap (6)" in reply.text


# -- B23: shell.execute raising RuntimeError surfaces as a tool error -------


def test_shell_execute_runtime_error_surfaces_as_shellerror_prefix(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """B23: shell.execute is documented as 'never raises' but defend in depth.
    Monkeypatch it to raise RuntimeError; the tool_result fed back to the LLM
    must start with 'ShellError: RuntimeError:' (preserve the *type*, lesson
    from v0.6: don't lump unrelated failures into one generic message).
    """
    def boom(cmd, **kw):  # noqa: ARG001
        raise RuntimeError("kaboom")

    monkeypatch.setattr("mneme.agent.shell_tool.execute", boom)
    fake_llm.tool_call_script = [
        {"text": "trying", "tool_calls": [_shell_call("c1", "true")],
         "stop_reason": "tool_use"},
        {"text": "ok handled it", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("trigger raise", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "ok handled it"

    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    # tool_calls=1 (the attempt counts even when execute exploded);
    # tool_rejects=0 (the user did not reject); iters=2 (round 1 + final).
    assert rec["tool_calls"] == 1
    assert rec["tool_rejects"] == 0
    assert rec["iters"] == 2

    # The error TYPE must travel into the next-round messages.
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("ShellError: RuntimeError:")


# -- B24: multi-iter with one reject in the middle --------------------------


def test_reject_mid_loop_aborts_with_running_counts(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """B24: round 1 tool accepted+ran; round 2 tool rejected. Final counts:
    tool_calls=2 (both attempts), tool_rejects=1, iters=2 (rejection ends
    on the round it happened, before a 3rd LLM call)."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "r1", "tool_calls": [_shell_call("c1", "echo r1")],
         "stop_reason": "tool_use"},
        {"text": "r2", "tool_calls": [_shell_call("c2", "rm -rf /")],
         "stop_reason": "tool_use"},
    ]
    calls = {"n": 0}

    def confirm(name, args):  # noqa: ARG001
        calls["n"] += 1
        # Accept the first, reject the second.
        return calls["n"] == 1

    reply = agent.respond("two-step", "turn1", cx, confirm_cb=confirm)
    assert reply.text == "Tool call rejected by user; turn aborted."
    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["tool_calls"] == 2
    assert rec["tool_rejects"] == 1
    assert rec["iters"] == 2


# -- B25: on_intermediate_text fires for intermediate text but not empty -----


def test_intermediate_text_callback_fires_before_confirm(
    cx, fake_llm, monkeypatch,
):
    """B25: intermediate assistant text must reach the UI *before* the confirm
    prompt — so the user reads 'I'll do X' first, then decides y/N. Spy on
    invocation order: record `intermediate_count` snapshot at confirm time
    and verify intermediate text was already delivered."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "I'll run echo first", "tool_calls": [_shell_call("c1", "echo hi")],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    seen_intermediate: list[str] = []
    seen_when_confirm_called: list[int] = []

    def on_inter(text):
        seen_intermediate.append(text)

    def confirm(name, args):  # noqa: ARG001
        # When confirm is invoked, intermediate must already have been called.
        seen_when_confirm_called.append(len(seen_intermediate))
        return True

    agent.respond(
        "show before confirm", "turn1", cx,
        confirm_cb=confirm, on_intermediate_text=on_inter,
    )
    assert seen_intermediate == ["I'll run echo first"]
    assert seen_when_confirm_called == [1]


def test_intermediate_text_callback_not_invoked_when_text_is_empty(
    cx, fake_llm, monkeypatch,
):
    """B25b: when the model's intermediate text is empty (or whitespace-only
    in the trivial sense — the dev's code skips on `not asst.text`), the
    callback must NOT be called. Avoids printing blank ' … ' lines."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "echo")],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    seen: list[str] = []
    agent.respond(
        "no intermediate text", "turn1", cx,
        confirm_cb=_accept,
        on_intermediate_text=lambda t: seen.append(t),
    )
    assert seen == []


# -- B26: confirm_cb itself raises -> reject + audit names the exception ----


def test_confirm_callback_raising_logs_specific_error_type(
    cx, fake_llm, tmp_path,
):
    """B26: if confirm_cb raises, the tool is treated as rejected AND the
    audit log records the exception type by name (not 'callback failed').
    v0.6 lesson again: preserve diagnostic info; users with broken UIs need
    to find ZeroDivisionError in their log, not 'something went wrong'."""
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "echo")],
         "stop_reason": "tool_use"},
    ]

    def boom(name, args):  # noqa: ARG001
        return 1 / 0  # ZeroDivisionError

    reply = agent.respond("bad confirm cb", "turn1", cx, confirm_cb=boom)
    # Treated as a rejection: turn aborts.
    assert reply.text == "Tool call rejected by user; turn aborted."

    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    assert rec["tool_rejects"] == 1

    # The audit event must name the exception type so the user can debug.
    ev = _events(tmp_path / "events.jsonl")
    rejects = [e for e in ev if e.get("kind") == "tool_audit"
               and e.get("decision") == "rejected"]
    assert any(
        r.get("reason", "").startswith("confirm_cb ZeroDivisionError:")
        for r in rejects
    )


# -- B27: unknown tool name comes back as a tool error, not a crash ---------


def test_unknown_tool_name_returns_unknowntool_block(
    cx, fake_llm, tmp_path,
):
    """B27: M1 has exactly one tool ('shell'). If the LLM hallucinates
    another name, the loop must surface 'UnknownTool:' in the tool_result
    rather than crashing — so the model can recover on the next round."""
    fake_llm.tool_call_script = [
        {"text": "trying ghost tool",
         "tool_calls": [ToolCall(id="x", name="not_shell", arguments={})],
         "stop_reason": "tool_use"},
        {"text": "oh well, here's text", "tool_calls": [],
         "stop_reason": "end_turn"},
    ]
    reply = agent.respond("unknown tool", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "oh well, here's text"

    rec = _close_trace(tmp_path / "events.jsonl", reply.trace_id)
    # Counted as a call (it was attempted), not a reject (user said yes).
    assert rec["tool_calls"] == 1
    assert rec["tool_rejects"] == 0

    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("UnknownTool:")


# -- B7 integration smoke (the proper unit lives in test_shell_tool.py) -----


def test_shell_argument_error_when_command_not_string(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """B7-int: if the LLM passes a non-string command (e.g. dict), surface
    ArgumentError without invoking shell.execute. (Pure unit in
    test_shell_tool.py exercises shell.execute directly.)"""
    called = {"n": 0}

    def must_not_run(cmd, **kw):  # noqa: ARG001
        called["n"] += 1
        raise AssertionError("shell.execute must not be called for bad args")

    # Actually wire the guard — without this monkeypatch the `called` counter
    # is dead code (lead e2e finding).
    monkeypatch.setattr("mneme.agent.shell_tool.execute", must_not_run)
    fake_llm.tool_call_script = [
        {"text": "",
         "tool_calls": [ToolCall(id="c1", name="shell",
                                 arguments={"command": {"bad": "type"}})],
         "stop_reason": "tool_use"},
        {"text": "noted error", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("bad args", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "noted error"

    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("ArgumentError:")
    assert called["n"] == 0  # execute never reached


# -- B18-bis: events.jsonl never contains stdout/stderr CONTENT (only bytes)


def test_events_jsonl_redacts_stdout_content_even_for_5kb_outputs(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """Privacy invariant: kind=tool_result events must record stdout_bytes
    metadata only, never the raw stdout string. Lead e2e caught that the
    previous B18 test was actually `client.config` immutability — the
    no-leak invariant was unverified. Mutation guard for any future
    refactor that accidentally inlines stdout into the audit event."""
    distinctive = "X" * 5_000   # 5 KB of a recognizable character

    class _BigStdout:
        def __init__(self, command):
            self.command = command
            self.exit_code = 0
            self.stdout = distinctive
            self.stderr = ""
            self.stdout_bytes = 5_000
            self.stderr_bytes = 0
            self.truncated = False
            self.duration_ms = 1

    monkeypatch.setattr("mneme.agent.shell_tool.execute",
                        lambda cmd, **kw: _BigStdout(cmd))
    fake_llm.tool_call_script = [
        {"text": "",
         "tool_calls": [ToolCall(id="b1", name="shell",
                                 arguments={"command": "big"})],
         "stop_reason": "tool_use"},
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("read big", "turn1", cx, confirm_cb=_accept)

    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    # No chunk of distinctive content leaks. Even a 200-char run would be a
    # privacy fail — pin a strict substring.
    assert "X" * 200 not in raw, "stdout content leaked into events.jsonl"
    # Metadata IS recorded (this is the positive half of the assertion).
    tool_result_events = [
        json.loads(line) for line in raw.splitlines() if line.strip()
        and '"kind": "tool_result"' in line and "stdout_bytes" in line
    ]
    assert len(tool_result_events) == 1
    assert tool_result_events[0]["stdout_bytes"] == 5_000


# -- mini shell-result stubs used by the tests above -----------------------


class _ShellOK:
    """Minimal stub mirroring ShellResult shape used by agent code paths."""
    def __init__(self, command: str):
        self.command = command
        self.exit_code = 0
        self.stdout = "ok\n"
        self.stderr = ""
        self.stdout_bytes = 3
        self.stderr_bytes = 0
        self.truncated = False
        self.duration_ms = 1


class _ShellFail:
    """ShellResult with exit_code=2 so the agent must flag is_error=True."""
    def __init__(self, command: str):
        self.command = command
        self.exit_code = 2
        self.stdout = ""
        self.stderr = "boom\n"
        self.stdout_bytes = 0
        self.stderr_bytes = 5
        self.truncated = False
        self.duration_ms = 1


# ============================================================================
# v0.9 / M2 integration tests — registry rewire + allowed_tools filter
# ============================================================================
#
# These tests pin the agent.py refactor that replaced v0.8's hardcoded
# `shell.SCHEMA` + dispatch with `tool_registry.schemas_for_provider` and
# `tool_registry.execute`. The contract:
#
#   - confirm_cb=None -> tools=None on the wire (v0.7 behavior preserved)
#   - confirm_cb set, allowed_tools=None -> ALL registered tools' schemas
#   - allowed_tools=[]  -> tools=None on the wire (no-tools fallback)
#   - allowed_tools=[names] -> exactly that subset's schemas, in caller order


def _agent_messages_seen(fake):
    """The messages list the FAKE saw on each agent-loop chat() call."""
    idx = _agent_chat_indices(fake)
    return [fake.messages_seen[i] for i in idx]


def test_D1_registry_schemas_passed_to_llm_when_tools_enabled(cx, fake_llm):
    """D1: with confirm_cb set and no allowed_tools restriction, the LLM
    receives every registered tool's schema (4 in v0.9: file_read,
    python_exec, shell, web_fetch). Sorted alphabetical."""
    fake_llm.tool_call_script = [
        {"text": "no tools used", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=_accept)
    tools = _agent_tools_seen(fake_llm)[0]
    assert tools is not None
    # Each schema is the Ollama/OpenAI shape (because FakeLLM defaults
    # provider='ollama'); pull the names out.
    names = [t["function"]["name"] for t in tools]
    assert "shell" in names
    assert "file_read" in names
    assert "web_fetch" in names
    assert "python_exec" in names
    assert names == sorted(names)


def test_D2_allowed_tools_subset_filters_schemas_in_order(cx, fake_llm):
    """D2: allowed_tools=['file_read', 'shell'] -> exactly those two
    schemas, in the order the caller passed."""
    fake_llm.tool_call_script = [
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=_accept,
                  allowed_tools=["file_read", "shell"])
    tools = _agent_tools_seen(fake_llm)[0]
    names = [t["function"]["name"] for t in tools]
    assert names == ["file_read", "shell"]


def test_D3_allowed_tools_empty_list_degrades_to_no_tools_path(cx, fake_llm):
    """D3: allowed_tools=[] is the explicit 'no tools' sentinel. The agent
    loop must NOT call the LLM with an empty tools[] payload (some providers
    reject that). Instead, fall back to the v0.7 single-call no-tools path:
    tools=None on the wire and the chat method is the legacy plain-text
    branch (not the AssistantMessage branch)."""
    fake_llm.tool_call_script = [
        {"text": "would call tool",
         "tool_calls": [ToolCall(id="t1", name="shell", arguments={"command": "ls"})],
         "stop_reason": "tool_use"},
    ]
    reply = agent.respond("hi", "turn1", cx, confirm_cb=_accept,
                          allowed_tools=[])
    # Single agent-loop chat call, tools=None, no tool round-trip.
    tools_seen = _agent_tools_seen(fake_llm)
    assert tools_seen == [None]
    # The model's scripted tool call was NEVER consumed (no AssistantMessage path).
    assert len(fake_llm.tool_call_script) == 1
    # The reply path was the plain-text branch, so reply.text == self.reply default.
    assert reply.text == "noted"


def test_D4_allowed_tools_unknown_name_silently_filtered_in_agent_path(
    cx, fake_llm,
):
    """D4: orchestrator decision (R-Reg-X also pinned in registry tests).
    allowed_tools=['nope', 'shell'] -> tools list contains only 'shell'.
    Unknown names fall out silently. The agent loop is the caller of
    `schemas_for_provider`, so this test pins end-to-end intent."""
    fake_llm.tool_call_script = [
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=_accept,
                  allowed_tools=["nope_does_not_exist", "shell"])
    tools = _agent_tools_seen(fake_llm)[0]
    names = [t["function"]["name"] for t in tools]
    assert names == ["shell"]


def test_D5_registry_dispatch_for_file_read_works_in_agent_loop(
    cx, fake_llm, tmp_path,
):
    """D5: agent loop dispatches a `file_read` tool call via registry.execute,
    not a hardcoded if-name=='shell' chain. Prove with a real file under
    sandbox HOME (tmp_path-based file)."""
    # Use the home directory because file_read's sandbox includes Path.home().
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     dir=tmp_path, encoding="utf-8") as fh:
        fh.write("hello from disk")
        path = fh.name

    # tmp_path is under cwd... no, only if pytest happened to chdir there.
    # Just make sure path is under tmp_path which is the CWD sandbox if we
    # chdir. Actually `agent.respond` doesn't change cwd, so use the path
    # directly; sandbox includes CWD = where pytest was run, which contains
    # tmp_path. Either way: the file is created under tmp_path which equals
    # the SQLite db dir, and tmp_path is under the actual cwd (pytest tmpdir).
    # file_read sandbox: cwd is allowed -> tmp_path is allowed.

    fake_llm.tool_call_script = [
        {"text": "reading", "tool_calls": [
            ToolCall(id="f1", name="file_read", arguments={"path": path}),
        ], "stop_reason": "tool_use"},
        {"text": "got it", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("read it", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "got it"

    # The tool_result on round 2 carries the file contents.
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "hello from disk"


def test_D6_registry_dispatch_for_python_exec_subprocess_call(
    cx, fake_llm, monkeypatch,
):
    """D6: same as D5 but for python_exec. Use a monkeypatched
    subprocess.run so we don't actually spawn python."""
    from mneme.tools import python_exec as pe_mod

    class _Done:
        stdout = "hello-from-py\n"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(pe_mod.subprocess, "run", lambda *a, **kw: _Done())

    fake_llm.tool_call_script = [
        {"text": "running", "tool_calls": [
            ToolCall(id="p1", name="python_exec",
                     arguments={"code": "print('hello-from-py')"}),
        ], "stop_reason": "tool_use"},
        {"text": "done py", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    reply = agent.respond("run py", "turn1", cx, confirm_cb=_accept)
    assert reply.text == "done py"
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "exit_code: 0" in tool_msgs[0]["content"]
    assert "stdout:\nhello-from-py" in tool_msgs[0]["content"]


def test_D7_argument_error_with_json_schema_vocabulary_in_agent_path(
    cx, fake_llm,
):
    """D7 reconciled: an LLM passing wrong types reaches the registry's
    type validation. The tool_result fed back to the model must use
    JSON-schema vocabulary ('got object', 'got integer', ...).
    Lead must-fix #1: literal-substring assert."""
    fake_llm.tool_call_script = [
        {"text": "trying",
         "tool_calls": [ToolCall(id="c1", name="shell",
                                 arguments={"command": {"k": "v"}})],
         "stop_reason": "tool_use"},
        {"text": "noted error", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("bad type", "turn1", cx, confirm_cb=_accept)
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert (
        "ArgumentError: tool 'shell' parameter 'command' must be string, "
        "got object"
    ) in tool_msgs[0]["content"]


def test_D8_provider_anthropic_gets_anthropic_shape_schemas(cx, fake_llm):
    """D8: provider='anthropic' gets {name, description, input_schema}
    schemas at top-level (not OpenAI's wrapped {type: function, function}).
    """
    fake_llm.config.provider = "anthropic"
    fake_llm.tool_call_script = [
        {"text": "done", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=_accept,
                  allowed_tools=["shell"])
    tools = _agent_tools_seen(fake_llm)[0]
    assert len(tools) == 1
    # Anthropic shape: name + description + input_schema at top level.
    assert set(tools[0].keys()) == {"name", "description", "input_schema"}


def test_D9_tool_result_audit_event_carries_tool_specific_fields(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """D9: the registry returns ToolResult with `audit` dict; agent.py
    merges those fields into the tool_result trace event. For shell:
    audit includes exit_code, stdout_bytes, stderr_bytes, truncated,
    duration_ms. The trace event must carry these as top-level fields."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("c1", "echo")],
         "stop_reason": "tool_use"},
        {"text": "ok", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("audit fields", "turn1", cx, confirm_cb=_accept)
    ev = _events(tmp_path / "events.jsonl")
    results = [e for e in ev if e.get("kind") == "tool_result"]
    assert len(results) == 1
    rec = results[0]
    # Audit-derived fields are merged flat into the event.
    assert rec["exit_code"] == 0
    assert "stdout_bytes" in rec
    assert "stderr_bytes" in rec
    assert "duration_ms" in rec
    assert rec["truncated"] is False


def test_D10_unknown_tool_via_registry_returns_unknown_tool_block(
    cx, fake_llm, tmp_path,
):
    """D10: same as B27 but pinned for v0.9 registry path: registry.execute
    returns ToolResult(content='UnknownTool: ...', is_error=True). The
    tool_result trace event includes error='UnknownTool' and exit_code=None."""
    fake_llm.tool_call_script = [
        {"text": "ghost",
         "tool_calls": [ToolCall(id="x", name="ghost_tool_no_such",
                                 arguments={})],
         "stop_reason": "tool_use"},
        {"text": "ok", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("ghost", "turn1", cx, confirm_cb=_accept)
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert tool_msgs[0]["content"].startswith("UnknownTool:")
    ev = _events(tmp_path / "events.jsonl")
    results = [e for e in ev if e.get("kind") == "tool_result"]
    assert results[0]["error"] == "UnknownTool"
    assert results[0]["exit_code"] is None


def test_D11_respond_stream_with_confirm_cb_uses_loop(cx, fake_llm):
    """D11: respond_stream with confirm_cb set drives the agent loop and
    yields the final reply as a single chunk (M1 spec).
    """
    fake_llm.tool_call_script = [
        {"text": "final via stream",
         "tool_calls": [], "stop_reason": "end_turn"},
    ]
    chunks = list(agent.respond_stream("hi", "turn1", cx, confirm_cb=_accept))
    assert "".join(chunks) == "final via stream"


def test_D12_respond_stream_allowed_tools_empty_falls_back_to_v07_stream(
    cx, fake_llm,
):
    """D12: respond_stream with confirm_cb set BUT allowed_tools=[] —
    the loop's use_tools guard short-circuits to no-tools; the no-tool
    branch is the single-call path that yields plain text chunks."""
    fake_llm.reply = "stream no tools"
    chunks = list(agent.respond_stream(
        "hi", "turn1", cx, confirm_cb=_accept, allowed_tools=[],
    ))
    # Joined yields == the canned reply.
    assert "".join(chunks) == "stream no tools"


# -- R-Agt-1..7: registry-aware ratchets ----------------------------------


def test_R_Agt_1_no_hardcoded_shell_schema_import_in_agent(cx, fake_llm):
    """R-Agt-1: the agent module must NOT import shell.SCHEMA (it was
    deleted in v0.9 step 4). Just verify the attribute is gone."""
    from mneme.tools import shell as shell_mod
    assert not hasattr(shell_mod, "SCHEMA")


def test_R_Agt_2_agent_uses_registry_module(cx, fake_llm):
    """R-Agt-2: agent.py imports `tool_registry` and calls .execute() /
    .schemas_for_provider(). Quick lint."""
    from mneme import agent as agent_mod
    src = open(agent_mod.__file__, encoding="utf-8").read()
    assert "tool_registry.execute(" in src
    assert "tool_registry.schemas_for_provider(" in src


def test_R_Agt_3_close_trace_records_tool_call_id_consistently(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """R-Agt-3: tool_audit (pending + accepted) and tool_result events
    must all carry the same tool_call_id for one call, so forensic
    grep-by-id yields the full lifecycle."""
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [_shell_call("traceme", "echo")],
         "stop_reason": "tool_use"},
        {"text": "ok", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("audit chain", "turn1", cx, confirm_cb=_accept)
    ev = _events(tmp_path / "events.jsonl")
    ids = {
        e.get("tool_call_id") for e in ev
        if e.get("kind") in ("tool_audit", "tool_result")
    }
    assert ids == {"traceme"}


def test_R_Agt_4_python_exec_via_registry_records_exit_code(
    cx, fake_llm, tmp_path, monkeypatch,
):
    """R-Agt-4: python_exec's audit dict has 'exit_code', so the trace
    event must carry exit_code (not None) on a successful run."""
    from mneme.tools import python_exec as pe_mod

    class _Done:
        stdout = "x\n"
        stderr = ""
        returncode = 0
    monkeypatch.setattr(pe_mod.subprocess, "run", lambda *a, **kw: _Done())

    fake_llm.tool_call_script = [
        {"text": "", "tool_calls": [
            ToolCall(id="p1", name="python_exec", arguments={"code": "print('x')"})],
         "stop_reason": "tool_use"},
        {"text": "ok", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("pyex", "turn1", cx, confirm_cb=_accept)
    ev = _events(tmp_path / "events.jsonl")
    results = [e for e in ev if e.get("kind") == "tool_result"
               and e.get("tool") == "python_exec"]
    assert len(results) == 1
    assert results[0]["exit_code"] == 0


def test_R_Agt_5_web_fetch_via_registry_returns_body_to_llm(
    cx, fake_llm, mock_httpx, mock_getaddrinfo,
):
    """R-Agt-5: web_fetch path works through the registry. The body
    arrives in the round-2 tool_result content."""
    mock_getaddrinfo.set("example.com", "93.184.216.34")

    def handler(req):
        import httpx as _hx
        return _hx.Response(200, text="page body",
                            headers={"content-type": "text/plain"})

    mock_httpx.set_handler(handler)
    fake_llm.tool_call_script = [
        {"text": "fetching", "tool_calls": [
            ToolCall(id="w1", name="web_fetch",
                     arguments={"url": "http://example.com/"})],
         "stop_reason": "tool_use"},
        {"text": "got it", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("fetch it", "turn1", cx, confirm_cb=_accept)
    round2 = _agent_round(fake_llm, 2)
    tool_msgs = [m for m in round2 if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "page body"


def test_R_Agt_6_argument_error_does_not_invoke_underlying_tool_fn(
    cx, fake_llm, monkeypatch,
):
    """R-Agt-6: when registry rejects bad-type args, the @tool-decorated fn
    body is never called. We monkeypatch shell.shell to count calls."""
    from mneme.tools import shell as shell_mod
    called = {"n": 0}

    real_shell = shell_mod.shell

    def counting_shell(*a, **kw):
        called["n"] += 1
        return real_shell(*a, **kw)

    monkeypatch.setattr(shell_mod, "shell", counting_shell)
    fake_llm.tool_call_script = [
        {"text": "trying",
         "tool_calls": [ToolCall(id="c1", name="shell",
                                 arguments={"command": 42})],
         "stop_reason": "tool_use"},
        {"text": "noted error", "tool_calls": [], "stop_reason": "end_turn"},
    ]
    agent.respond("int command", "turn1", cx, confirm_cb=_accept)
    # The tool body should not have been invoked.
    assert called["n"] == 0


def test_R_Agt_7_iter_cap_still_works_after_v09_rewire(
    cx, fake_llm, monkeypatch,
):
    """R-Agt-7: the v0.9 rewire must not break the iter cap (B19's
    structural contract). Cap at 2, forever-tool-calling model, observe abort."""
    monkeypatch.setenv("MNEME_MAX_ITERS", "2")
    monkeypatch.setattr(
        "mneme.agent.shell_tool.execute",
        lambda cmd, **kw: _ShellOK(cmd),
    )
    fake_llm.tool_call_script = [
        {"text": f"r{i}", "tool_calls": [_shell_call(f"c{i}", "echo")],
         "stop_reason": "tool_use"}
        for i in range(5)
    ]
    reply = agent.respond("cap", "turn1", cx, confirm_cb=_accept)
    assert "max iteration cap (2)" in reply.text
