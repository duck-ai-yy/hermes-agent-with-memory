"""B8 / B9 / B10 / B11: provider-protocol normalization for tool calling.

Three providers, three wire shapes. The agent loop never branches by
provider because client.chat() flattens them all into AssistantMessage +
ToolCall(arguments: dict). These tests pin the four physical hardwalls
recorded in docs/knowledge/provider-tool-calling.md:

  HW1. Ollama arguments are dict (not JSON string)
  HW2. Anthropic tool_result re-feed uses user-role + content array
  HW3. Ollama has no tool_call_id on the wire — synthesize one
  HW4. Streaming tool calls are out of M1 scope (skip)

B8-B10 check the parse direction (response -> AssistantMessage).
B11 checks the round-trip direction (AssistantMessage + tool_result ->
messages list shape) via _append_assistant_and_tool_result, which is the
function the agent loop uses to build the SECOND-round messages.
"""

from __future__ import annotations

import json

import httpx

from mneme.agent import _append_assistant_and_tool_result
from mneme.llm.client import (AssistantMessage, LLMClient, LLMConfig,
                              ToolCall)


def _client_with(handler, **cfg_kwargs) -> LLMClient:
    cfg = LLMConfig(**cfg_kwargs)
    client = LLMClient(cfg)
    client._http = httpx.Client(
        base_url=cfg.base_url, transport=httpx.MockTransport(handler),
    )
    return client


# -- B8: Anthropic tool_use → AssistantMessage with toolu_ id + dict args ----


def test_anthropic_tool_use_block_parses_into_assistant_message():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/messages"
        body = json.loads(req.content)
        assert body["tools"], "tools must be forwarded to the provider"
        return httpx.Response(200, json={
            "content": [
                {"type": "text", "text": "I will check the file."},
                {"type": "tool_use",
                 "id": "toolu_abc123",
                 "name": "shell",
                 "input": {"command": "cat pyproject.toml"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 34},
        })

    client = _client_with(handler, provider="anthropic", chat_model="claude-x")
    msg = client.chat(
        [{"role": "user", "content": "what's the version?"}],
        stream=False,
        tools=[{"name": "shell", "description": "x", "input_schema": {}}],
    )
    assert isinstance(msg, AssistantMessage)
    assert msg.text == "I will check the file."
    assert msg.stop_reason == "tool_use"
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "toolu_abc123"          # provider-supplied id preserved
    assert tc.name == "shell"
    # HW1 (mirror): arguments is dict regardless of provider
    assert tc.arguments == {"command": "cat pyproject.toml"}
    assert isinstance(tc.arguments, dict)


def test_anthropic_text_only_response_yields_zero_tool_calls():
    """If the model returns only text blocks the loop must terminate —
    pinning empty tool_calls so a mutation 'always emit one tool_call' fails."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "done."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        })

    client = _client_with(handler, provider="anthropic", chat_model="claude-x")
    msg = client.chat(
        [{"role": "user", "content": "x"}], stream=False, tools=[{"name": "shell"}],
    )
    assert msg.tool_calls == []
    assert msg.stop_reason == "end_turn"


# -- B9: OpenAI tool_calls — arguments is a JSON string, must parse to dict --


def test_openai_tool_calls_arguments_parsed_from_json_string():
    """HW: OpenAI sends arguments as a JSON-encoded string. The parser must
    json.loads it so the agent layer never sees a str."""
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        body = json.loads(req.content)
        assert body["tools"]
        return httpx.Response(200, json={
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "ls -la"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        })

    client = _client_with(handler, provider="openai", chat_model="gpt-4o-mini",
                          base_url="https://api.openai.com")
    msg = client.chat(
        [{"role": "user", "content": "list files"}], stream=False,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert isinstance(msg, AssistantMessage)
    assert msg.stop_reason == "tool_calls"
    tc = msg.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "shell"
    # Critical assertion: dict, NOT the original string.
    assert tc.arguments == {"command": "ls -la"}
    assert isinstance(tc.arguments, dict)


def test_openai_malformed_arguments_string_falls_back_to_empty_dict():
    """Mutation guard: if a provider sends broken JSON, the agent layer must
    still get a dict (empty), not crash. The tool itself will surface the
    missing-arg error."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_x", "type": "function",
                        "function": {"name": "shell", "arguments": "{not json"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })

    client = _client_with(handler, provider="openai", chat_model="gpt-4o-mini",
                          base_url="https://api.openai.com")
    msg = client.chat(
        [{"role": "user", "content": "x"}], stream=False,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert msg.tool_calls[0].arguments == {}    # not str, not None — empty dict


# -- B10: Ollama tool_calls — arguments already dict, id synthesized ---------


def test_ollama_tool_calls_pass_dict_args_through_and_synthesize_id():
    """HW1 + HW3: Ollama ships arguments as object (not string) and has no
    tool_call_id. Parser must (a) not double-decode, (b) synthesize an id
    so the rest of the codebase can assume id is always populated."""
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/chat"
        body = json.loads(req.content)
        assert body["tools"]
        return httpx.Response(200, json={
            "message": {
                "content": "checking",
                "tool_calls": [{
                    "function": {
                        "name": "shell",
                        "arguments": {"command": "pwd"},   # dict, not str!
                    },
                }],
            },
            "prompt_eval_count": 3, "eval_count": 4, "done": True,
        })

    client = _client_with(handler, provider="ollama", chat_model="qwen2.5:7b")
    msg = client.chat(
        [{"role": "user", "content": "pwd"}], stream=False,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert msg.stop_reason == "tool_use"       # synthesized from presence
    tc = msg.tool_calls[0]
    assert tc.name == "shell"
    assert tc.arguments == {"command": "pwd"}
    assert isinstance(tc.arguments, dict)
    # HW3: id was missing on the wire, must be present and ULID-shaped
    assert tc.id.startswith("toolu_")
    assert len(tc.id) == len("toolu_") + 26    # ULID is 26 chars


def test_ollama_text_only_response_signals_end_turn():
    """No tool_calls → stop_reason='end_turn' regardless of Ollama silence
    on stop_reason. Mutation guard for the synthesized stop_reason logic."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": "done"}, "done": True,
            "prompt_eval_count": 1, "eval_count": 2,
        })

    client = _client_with(handler, provider="ollama", chat_model="x")
    msg = client.chat(
        [{"role": "user", "content": "x"}], stream=False,
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert msg.tool_calls == []
    assert msg.stop_reason == "end_turn"


# -- B11: round-trip messages list shape for the SECOND round ----------------
# These exercise _append_assistant_and_tool_result directly — the function
# the agent loop uses to build messages for the next chat() call. We're
# pinning the wire-shape requirements of all three providers.


def _make_asst() -> AssistantMessage:
    return AssistantMessage(
        text="I will run a command.",
        tool_calls=[ToolCall(id="t1", name="shell", arguments={"command": "ls"})],
        stop_reason="tool_use",
    )


def _make_result(is_error: bool = False) -> dict:
    return {
        "tool_call_id": "t1", "tool_name": "shell",
        "content": "exit_code: 0\nstdout:\nfoo\n",
        "is_error": is_error,
    }


def test_anthropic_round_trip_uses_user_role_with_tool_result_block():
    """HW2: Anthropic does NOT use role=tool. It uses role=user with a
    content array whose first block is type=tool_result + tool_use_id."""
    messages: list[dict] = [{"role": "user", "content": "hi"}]
    _append_assistant_and_tool_result(messages, "anthropic", _make_asst(),
                                      [_make_result()])
    # Second-to-last: assistant turn with original blocks.
    asst = messages[-2]
    assert asst["role"] == "assistant"
    assert isinstance(asst["content"], list)
    # The tool_use block must reference the original id; this is what
    # Anthropic uses to pair with our tool_result.
    tool_use_blocks = [b for b in asst["content"] if b.get("type") == "tool_use"]
    assert tool_use_blocks[0]["id"] == "t1"
    # Last: user-role wrapper with tool_result inside.
    last = messages[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    block = last["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"            # paired by id, not name
    assert block["is_error"] is False


def test_anthropic_is_error_propagates_through_round_trip():
    """Mutation guard for the second-round shape: if the agent fed
    is_error=False even when the tool actually failed, the model would never
    learn the command broke. B6's spy assertion lives here at the wire level."""
    messages: list[dict] = []
    _append_assistant_and_tool_result(messages, "anthropic", _make_asst(),
                                      [_make_result(is_error=True)])
    block = messages[-1]["content"][0]
    assert block["is_error"] is True


def test_openai_round_trip_uses_role_tool_with_tool_call_id():
    """OpenAI shape: assistant message echoes tool_calls (arguments
    re-encoded as JSON string), then role=tool message with tool_call_id."""
    messages: list[dict] = []
    _append_assistant_and_tool_result(messages, "openai", _make_asst(),
                                      [_make_result()])
    asst = messages[-2]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "t1"
    # Arguments re-encoded as JSON string for OpenAI's wire format.
    args_str = asst["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args_str, str)
    assert json.loads(args_str) == {"command": "ls"}
    last = messages[-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "t1"
    # OpenAI doesn't have an is_error field; the agent surfaces error state
    # via the content. The presence of tool_call_id is the pairing key.


def test_ollama_round_trip_uses_role_tool_with_tool_name_only():
    """HW3: Ollama pairs results by tool_name, not tool_call_id. Mutation
    guard: a refactor that adds tool_call_id to Ollama's tool message would
    pass parts of B11 but might break a strict Ollama server."""
    messages: list[dict] = []
    _append_assistant_and_tool_result(messages, "ollama", _make_asst(),
                                      [_make_result()])
    asst = messages[-2]
    assert asst["role"] == "assistant"
    # Ollama assistant: tool_calls echoed with arguments as DICT (not str).
    args = asst["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args == {"command": "ls"}
    last = messages[-1]
    assert last["role"] == "tool"
    assert last["tool_name"] == "shell"
    # Pinning the absence of tool_call_id: Ollama errors if echoed.
    assert "tool_call_id" not in last
