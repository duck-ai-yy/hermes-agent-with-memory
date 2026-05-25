"""LLMClient adapter tests — verify the HTTP shapes without burning API credits.

Each test installs an httpx MockTransport that asserts request shape and
returns a canned response. This catches schema mistakes in the three
provider branches (ollama / openai / anthropic) without the live network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from mneme.llm.client import BudgetExceeded, LLMClient, LLMConfig


def _client_with(handler, embed_handler=None, **cfg_kwargs) -> LLMClient:
    cfg = LLMConfig(**cfg_kwargs)
    client = LLMClient(cfg)
    client._http = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    client._embed_http = httpx.Client(
        base_url=cfg.effective_embed_base_url,
        transport=httpx.MockTransport(embed_handler or handler),
    )
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


def test_openai_embed_requests_768_dimensions_to_fit_schema():
    """Our vec_slices table is FLOAT[768]; OpenAI text-embedding-3-small is
    native 1536-d. Asking for `dimensions: 768` keeps the schema stable."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 768}]})

    client = _client_with(
        handler, provider="openai", api_key="sk-test", base_url="https://x",
        embed_model="text-embedding-3-small",
    )
    blob = client.embed("hello")
    assert len(blob) == 768 * 4  # float32
    assert seen["path"] == "/v1/embeddings"
    assert seen["body"]["dimensions"] == 768
    assert seen["body"]["model"] == "text-embedding-3-small"


def test_embed_provider_can_differ_from_chat_provider():
    """Anthropic chat + OpenAI embed — the canonical decoupled setup."""
    chat_hits, embed_hits = [], []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        chat_hits.append(request.url.host)
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "ok"}]},
        )

    def embed_handler(request: httpx.Request) -> httpx.Response:
        embed_hits.append(request.url.host)
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 768}]})

    client = _client_with(
        chat_handler, embed_handler=embed_handler,
        provider="anthropic", base_url="https://api.anthropic.com",
        api_key="sk-ant-test", chat_model="claude-haiku-4-5",
        embed_provider="openai", embed_base_url="https://api.openai.com",
        embed_api_key="sk-test", embed_model="text-embedding-3-small",
    )
    client.chat([{"role": "user", "content": "hi"}], stream=False)
    client.embed("hello")
    assert chat_hits == ["api.anthropic.com"]
    assert embed_hits == ["api.openai.com"]


def test_ollama_non_stream_records_real_usage():
    """`prompt_eval_count` / `eval_count` from Ollama land on client.last_usage."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": "hi"}, "done": True,
            "prompt_eval_count": 42, "eval_count": 7,
        })

    client = _client_with(handler, provider="ollama")
    client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert client.last_usage is not None
    assert client.last_usage.prompt_tokens == 42
    assert client.last_usage.completion_tokens == 7
    assert client.last_usage.total_tokens == 49


def test_openai_non_stream_records_real_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
        })

    client = _client_with(handler, provider="openai", api_key="sk-test", base_url="https://x")
    client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert client.last_usage.prompt_tokens == 100
    assert client.last_usage.completion_tokens == 30


def test_ollama_stream_yields_chunks_and_records_usage_from_final_frame():
    """Ollama streams NDJSON; the `done:true` frame carries token counts."""
    body = (
        b'{"message":{"content":"hel"},"done":false}\n'
        b'{"message":{"content":"lo"},"done":false}\n'
        b'{"message":{"content":""},"done":true,'
        b'"prompt_eval_count":12,"eval_count":5}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client_with(handler, provider="ollama")
    chunks = list(client.chat([{"role": "user", "content": "hi"}], stream=True))
    assert "".join(chunks) == "hello"
    assert client.last_usage.prompt_tokens == 12
    assert client.last_usage.completion_tokens == 5


def test_openai_stream_requests_usage_and_captures_it():
    """We send `stream_options.include_usage` so OpenAI emits a final usage chunk."""
    seen: dict = {}
    body = (
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2,'
        b'"total_tokens":13}}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=body)

    client = _client_with(handler, provider="openai", api_key="sk", base_url="https://x")
    chunks = list(client.chat([{"role": "user", "content": "hi"}], stream=True))
    assert "".join(chunks) == "hello"
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert client.last_usage.prompt_tokens == 11
    assert client.last_usage.completion_tokens == 2


def test_anthropic_stream_combines_input_and_output_tokens():
    """input_tokens from message_start + output_tokens from message_delta."""
    body = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":20,'
        b'"output_tokens":1}}}\n\n'
        b'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
        b'data: {"type":"message_delta","usage":{"output_tokens":8}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client_with(
        handler, provider="anthropic", api_key="sk-ant",
        base_url="https://api.anthropic.com", chat_model="claude-haiku-4-5",
    )
    chunks = list(client.chat([{"role": "user", "content": "hi"}], stream=True))
    assert "".join(chunks) == "hi"
    assert client.last_usage.prompt_tokens == 20
    assert client.last_usage.completion_tokens == 8


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
    # The mocked response includes usage; the client should surface it.
    assert client.last_usage.prompt_tokens == 12
    assert client.last_usage.completion_tokens == 4


# -- Budget hard-wall -------------------------------------------------------

def _write_events(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_budget_blocks_cloud_call_when_today_exhausted(tmp_path):
    """Pre-call check raises before any HTTP traffic when over budget."""
    ep = tmp_path / "events.jsonl"
    _write_events(ep, [
        {"ts": int(time.time()), "kind": "trace", "id": "T1",
         "provider": "openai", "total_tokens": 1500},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called when budget is exhausted")

    client = _client_with(
        handler, provider="openai", api_key="sk", base_url="https://x",
        events_path=ep, daily_token_budget=1000,
    )
    with pytest.raises(BudgetExceeded) as exc:
        client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert exc.value.used == 1500
    assert exc.value.budget == 1000


def test_budget_ignores_ollama_spend(tmp_path):
    """Local Ollama tokens never count toward the cloud budget."""
    ep = tmp_path / "events.jsonl"
    _write_events(ep, [
        {"ts": int(time.time()), "kind": "trace", "id": "T1",
         "provider": "ollama", "total_tokens": 99_999},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = _client_with(
        handler, provider="openai", api_key="sk", base_url="https://x",
        events_path=ep, daily_token_budget=1000,
    )
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "ok"


def test_budget_ignores_yesterday_spend(tmp_path):
    """Spend from before today's local midnight does not carry over."""
    ep = tmp_path / "events.jsonl"
    _write_events(ep, [
        {"ts": int(time.time()) - 86400 * 2, "kind": "trace", "id": "T1",
         "provider": "openai", "total_tokens": 9999},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}], "usage": {},
        })

    client = _client_with(
        handler, provider="openai", api_key="sk", base_url="https://x",
        events_path=ep, daily_token_budget=1000,
    )
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "ok"


def test_budget_never_blocks_local_ollama_calls(tmp_path):
    """Ollama is free; even with budget=1 and huge prior spend it must work."""
    ep = tmp_path / "events.jsonl"
    _write_events(ep, [
        {"ts": int(time.time()), "kind": "trace", "id": "T1",
         "provider": "openai", "total_tokens": 999_999},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    client = _client_with(
        handler, provider="ollama",
        events_path=ep, daily_token_budget=1,
    )
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "ok"


def test_budget_zero_means_unlimited(tmp_path):
    """budget <= 0 is the explicit 'no cap' value; no events lookup happens."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}], "usage": {},
        })

    client = _client_with(
        handler, provider="openai", api_key="sk", base_url="https://x",
        # No events_path, no budget — must not raise looking for it.
    )
    reply = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply == "ok"
