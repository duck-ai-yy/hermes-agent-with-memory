"""B22: HTTP /chat must NEVER expose tool calls.

server.py explicitly passes confirm_cb=None on every respond() call because
there's no interactive UI to confirm a tool. If a future refactor wires a
confirm_cb (or silently lets tools through), this test fires. The mutation
'pass real tools' would let the LLM trigger a tool round-trip via HTTP,
which has no safety prompt — exactly what M1 is designed to prevent.

We don't spin up an actual FastAPI TestClient because the /chat handler
uses asyncio.to_thread + a sync sqlite path with a fixed DB_PATH; that's
infra noise. Instead we exercise the same code path by calling respond()
the way server.py does, and assert the tool-loop never engaged.
"""

from __future__ import annotations

from mneme import agent
from mneme.llm.client import ToolCall


def _accept_anything(name, args):  # noqa: ARG001
    return True


def _agent_chat_calls(fake) -> int:
    """Count agent-loop (non-concept-extraction) chat() calls. Concept
    extraction uses a 'STRICT JSON' system prompt."""
    return sum(
        1 for msgs in fake.messages_seen
        if msgs and "STRICT JSON" not in (msgs[0].get("content") or "")
    )


def test_respond_with_confirm_cb_none_passes_no_tools_to_llm(cx, fake_llm):
    """Mirror server.py:54 — respond(..., confirm_cb=None). FakeLLM scripts a
    tool call but the agent must NOT trigger it because tools=None was
    declared. The script sits unused; only one chat() call happens, with
    tools kwarg None."""
    fake_llm.tool_call_script = [
        {"text": "would call tool",
         "tool_calls": [ToolCall(id="t1", name="shell",
                                 arguments={"command": "rm -rf /"})],
         "stop_reason": "tool_use"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=None)
    # Exactly one agent-loop chat call.
    assert _agent_chat_calls(fake_llm) == 1
    # That call passed tools=None (degraded path). Filter to agent calls
    # the same way: skip the concept-extraction calls.
    agent_tools = [
        t for t, msgs in zip(fake_llm.tools_seen, fake_llm.messages_seen)
        if msgs and "STRICT JSON" not in (msgs[0].get("content") or "")
    ]
    assert agent_tools == [None], (
        f"server's confirm_cb=None path leaked tools to the LLM: {agent_tools}"
    )
    # The scripted tool call sat unused — no tool_audit / tool_result events.
    ep = cx.execute("SELECT 1").fetchone()    # cx is valid
    assert ep is not None
    # The strongest assertion: chat_calls counts both concept-extraction
    # calls AND the single agent loop call, but the tool_call_script wasn't
    # consumed. Pin len.
    assert len(fake_llm.tool_call_script) == 1, (
        "agent should not have popped from tool_call_script when confirm_cb=None"
    )


def test_respond_with_confirm_cb_none_does_not_write_tool_audit_events(
    cx, fake_llm, tmp_path,
):
    """Read-only contract for server.py: an HTTP /chat call writes ingest +
    trace events but NEVER a tool_audit or tool_result event. We snapshot
    events.jsonl after the call and grep for the tool-related kinds."""
    fake_llm.tool_call_script = [
        {"text": "model wants to call shell",
         "tool_calls": [ToolCall(id="t1", name="shell",
                                 arguments={"command": "ls"})],
         "stop_reason": "tool_use"},
    ]
    agent.respond("hi", "turn1", cx, confirm_cb=None)
    text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "tool_audit" not in text
    assert "tool_result" not in text
