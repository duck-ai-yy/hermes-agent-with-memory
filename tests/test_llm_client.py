"""LLMClient adapter tests — verify the HTTP shapes without burning API credits.

Each test installs an httpx MockTransport that asserts request shape and
returns a canned response. This catches schema mistakes in the three
provider branches (ollama / openai / anthropic) without the live network.
"""

from __future__ import annotations

import json

import httpx

from mneme.llm.client import LLMClient, LLMConfig


def _client_with(handler, **cfg_kwargs) -> LLMClient:
    cfg = LLMConfig(**cfg_kwargs)
    client = LLMClient(cfg)
    client._http = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    return client


def test_ollama_chat_non_stream_round_trip():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ollama said hi"}, "done": True})

    client = _client_with(handler, provider="ollama")
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "ollama said hi"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:7b"
    assert seen["body"]["stream"] is False


def test_openai_chat_uses_v1_endpoint_and_bearer_auth():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "openai said hi"}}]}
        )

    client = _client_with(handler, provider="openai", api_key="sk-test", base_url="https://x")
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "openai said hi"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"


def test_anthropic_chat_splits_system_and_uses_messages_endpoint():
    """Anthropic's /v1/messages takes `system` as a separate top-level field,
    not a message with role=system. Catching that conversion is the point."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["x-api-key"] = request.headers.get("x-api-key")
        seen["anthropic-version"] = request.headers.get("anthropic-version")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "claude said hi"}],
                  "usage": {"input_tokens": 12, "output_tokens": 4}},
        )

    client = _client_with(
        handler, provider="anthropic", api_key="sk-ant-test",
        base_url="https://api.anthropic.com", chat_model="claude-haiku-4-5",
    )
    reply = client.chat(
        [{"role": "system", "content": "be terse"},
         {"role": "user", "content": "hi"}],
        stream=False,
    )

    assert reply == "claude said hi"
    assert seen["path"] == "/v1/messages"
    assert seen["x-api-key"] == "sk-ant-test"
    assert seen["anthropic-version"] == "2023-06-01"
    # The system message must have been lifted out into the top-level field,
    # and the messages array must contain only the user turn.
    assert seen["body"]["system"] == "be terse"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["body"]["max_tokens"] == 1024
    assert seen["body"]["model"] == "claude-haiku-4-5"
