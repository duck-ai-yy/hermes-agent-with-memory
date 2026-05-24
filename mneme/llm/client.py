"""Singleton LLM client over httpx.

Default provider is local Ollama (PRINCIPLES.md principle 4). A cloud provider
is opt-in and, when used, must emit an audit event before the call.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from ..trace import events


@dataclass(frozen=True)
class Usage:
    """Real token counts returned by the provider, not estimates."""
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetExceeded(Exception):
    """Raised before a cloud chat call when today's spend would exceed the
    configured daily token budget. Ollama (local, free) is never blocked."""

    def __init__(self, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"daily token budget exhausted: {used:,}/{budget:,}")


@dataclass(frozen=True)
class LLMConfig:
    # Chat provider.
    provider: str = "ollama"            # "ollama" | "openai" | "anthropic"
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:7b"
    api_key: str | None = None          # required only for cloud chat

    # Embed provider — may differ from chat (e.g. Anthropic for chat,
    # OpenAI for embed, since Anthropic has no embeddings endpoint).
    # Each `embed_*` field falls back to its chat counterpart when None.
    embed_model: str = "nomic-embed-text"
    embed_provider: str | None = None   # None -> reuse `provider`
    embed_base_url: str | None = None   # None -> reuse `base_url`
    embed_api_key: str | None = None    # None -> reuse `api_key`
    embed_via: str = "provider"         # "provider" | "hash" — hash = local
                                        # stdlib fallback for providers
                                        # without embeddings (DeepSeek,
                                        # Anthropic); degrades recall to
                                        # exact-text matches only.

    events_path: Path | None = None     # where audit events are written

    # Cloud-spend hard wall. <= 0 means unlimited. Checked AGAINST today's
    # `total_tokens` summed from trace events with a non-Ollama provider, so
    # an exhausted budget blocks the next cloud call but never local Ollama.
    daily_token_budget: int = 0

    @property
    def effective_embed_provider(self) -> str:
        return self.embed_provider or self.provider

    @property
    def effective_embed_base_url(self) -> str:
        return self.embed_base_url or self.base_url

    @property
    def effective_embed_api_key(self) -> str | None:
        return self.embed_api_key or self.api_key


_INSTANCE: "LLMClient | None" = None


def configure(config: LLMConfig) -> None:
    """Install the process-wide LLM client. Last call wins."""
    global _INSTANCE
    _INSTANCE = LLMClient(config)


def get_client() -> "LLMClient":
    """Return the singleton, lazily creating a default if not configured."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LLMClient(LLMConfig())
    return _INSTANCE


def _pack(vec: list[float]) -> bytes:
    """Serialize floats to the little-endian float32 blob sqlite-vec wants."""
    return struct.pack(f"<{len(vec)}f", *vec)


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._http = httpx.Client(base_url=config.base_url, timeout=120.0)
        # Separate embed client so chat and embed can hit different hosts
        # (Anthropic chat + OpenAI embed is the canonical case).
        self._embed_http = httpx.Client(
            base_url=config.effective_embed_base_url, timeout=120.0
        )
        # Populated after every chat call (None until the first call).
        # For streams, only set once the iterator is exhausted.
        self.last_usage: Usage | None = None

    def _is_cloud(self, provider: str) -> bool:
        return provider != "ollama"

    def _audit(self, endpoint: str, payload_chars: int) -> None:
        """Record an outbound cloud call before it happens (principle 4)."""
        provider = (self.config.effective_embed_provider
                    if endpoint == "embed" else self.config.provider)
        if self._is_cloud(provider) and self.config.events_path is not None:
            events.append(
                self.config.events_path, kind="audit",
                provider=provider, endpoint=endpoint,
                model=self.config.chat_model if endpoint == "chat" else self.config.embed_model,
                est_tokens=payload_chars // 4,
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def _embed_headers(self) -> dict[str, str]:
        key = self.config.effective_embed_api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _check_budget(self) -> None:
        """Raise BudgetExceeded if today's cloud spend has hit the cap.

        Soft boundary — checked before each call, not against the imminent
        call's tokens (we can't know them yet). One call may push us over,
        but no further calls go through until the next local-time midnight.
        """
        budget = self.config.daily_token_budget
        if budget <= 0 or self.config.events_path is None:
            return
        used = events.sum_cloud_tokens_since(
            self.config.events_path, events.today_start_ts()
        )
        if used >= budget:
            raise BudgetExceeded(used, budget)

    def chat(self, messages: list[dict], *, stream: bool = True):
        """Chat completion. Returns a token iterator if stream else the full string.

        `self.last_usage` is populated with real token counts once the call
        completes (immediately for non-stream, after iterator exhaustion for
        stream). It is None if the provider did not return usage.
        """
        if self._is_cloud(self.config.provider):
            self._check_budget()
        self._audit("chat", sum(len(m.get("content", "")) for m in messages))
        self.last_usage = None
        if self.config.provider == "anthropic":
            return self._chat_anthropic(messages, stream)
        payload = {"model": self.config.chat_model, "messages": messages, "stream": stream}
        if self.config.provider == "ollama":
            if not stream:
                r = self._http.post("/api/chat", json=payload)
                r.raise_for_status()
                body = r.json()
                self.last_usage = _usage_from_ollama(body)
                return body["message"]["content"]
            return self._stream_ollama(payload)
        if not stream:
            r = self._http.post("/v1/chat/completions", json=payload, headers=self._headers())
            r.raise_for_status()
            body = r.json()
            self.last_usage = _usage_from_openai(body)
            return body["choices"][0]["message"]["content"]
        return self._stream_openai(payload)

    def embed(self, text: str) -> bytes:
        """Raw embedding call (no cache — caching lives in memory/embed.py)."""
        if self.config.embed_via == "hash":
            # Deterministic local fallback for providers without embeddings.
            rng = random.Random(hashlib.sha256(text.encode()).digest())
            return _pack([rng.random() for _ in range(768)])
        self._audit("embed", len(text))
        provider = self.config.effective_embed_provider
        body: dict = {"model": self.config.embed_model, "input": text}
        if provider == "ollama":
            r = self._embed_http.post("/api/embed", json=body)
            r.raise_for_status()
            return _pack(r.json()["embeddings"][0])
        # OpenAI-compatible. `dimensions: 768` truncates 1536-d models like
        # text-embedding-3-small to fit our vec_slices FLOAT[768] schema.
        body["dimensions"] = 768
        r = self._embed_http.post("/v1/embeddings", json=body, headers=self._embed_headers())
        r.raise_for_status()
        return _pack(r.json()["data"][0]["embedding"])

    def _stream_ollama(self, payload: dict) -> Iterator[str]:
        with self._http.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    self.last_usage = _usage_from_ollama(obj)
                    break

    def _stream_openai(self, payload: dict) -> Iterator[str]:
        # `stream_options.include_usage` makes OpenAI emit a final chunk with
        # `usage` populated; without it the stream gives no token counts.
        payload = {**payload, "stream_options": {"include_usage": True}}
        with self._http.stream(
            "POST", "/v1/chat/completions", json=payload, headers=self._headers()
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                if obj.get("usage"):
                    self.last_usage = _usage_from_openai(obj)

    # -- Anthropic --------------------------------------------------------
    # /v1/messages takes `system` as a top-level string, not a message role;
    # response shape is content[0].text, not choices[0].message.content.
    def _anthropic_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": "2023-06-01",
        }

    def _anthropic_body(self, messages: list[dict], stream: bool) -> dict:
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        rest = [m for m in messages if m.get("role") != "system"]
        body: dict = {
            "model": self.config.chat_model,
            "max_tokens": 1024,
            "messages": rest,
            "stream": stream,
        }
        if system:
            body["system"] = system
        return body

    def _chat_anthropic(self, messages: list[dict], stream: bool):
        body = self._anthropic_body(messages, stream)
        headers = self._anthropic_headers()
        if not stream:
            r = self._http.post("/v1/messages", json=body, headers=headers)
            r.raise_for_status()
            payload = r.json()
            self.last_usage = _usage_from_anthropic(payload)
            return payload["content"][0]["text"]
        return self._stream_anthropic(body, headers)

    def _stream_anthropic(self, body: dict, headers: dict) -> Iterator[str]:
        # message_start carries input_tokens; message_delta carries the final
        # output_tokens. Combine them at the end.
        input_tokens = 0
        with self._http.stream("POST", "/v1/messages", json=body, headers=headers) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                obj = json.loads(line[6:])
                t = obj.get("type")
                if t == "content_block_delta":
                    delta = obj.get("delta", {}).get("text", "")
                    if delta:
                        yield delta
                elif t == "message_start":
                    input_tokens = (
                        obj.get("message", {}).get("usage", {}).get("input_tokens", 0)
                    )
                elif t == "message_delta":
                    output_tokens = obj.get("usage", {}).get("output_tokens", 0)
                    self.last_usage = Usage(input_tokens, output_tokens)


def _usage_from_ollama(obj: dict) -> Usage:
    return Usage(obj.get("prompt_eval_count", 0), obj.get("eval_count", 0))


def _usage_from_openai(obj: dict) -> Usage:
    u = obj.get("usage") or {}
    return Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def _usage_from_anthropic(obj: dict) -> Usage:
    u = obj.get("usage") or {}
    return Usage(u.get("input_tokens", 0), u.get("output_tokens", 0))
