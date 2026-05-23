"""Singleton LLM client over httpx.

Default provider is local Ollama (PRINCIPLES.md principle 4). A cloud provider
is opt-in and, when used, must emit an audit event before the call.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from ..trace import events


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"            # "ollama" | "openai"
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    api_key: str | None = None          # required only for cloud providers
    events_path: Path | None = None     # where audit events are written


# Process-wide singleton. `configure()` sets it once at startup (CLI / server);
# `get_client()` lazily falls back to defaults so library callers never crash.
# An explicit global beats `lru_cache(get_client)` because the latter keys on
# args — a configured call followed by a no-arg call would build a *second*
# client, silently losing the events_path wired in for cloud audit.
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


def _pack(vector: list[float]) -> bytes:
    """Serialize a float vector to the little-endian float32 blob sqlite-vec wants."""
    return struct.pack(f"<{len(vector)}f", *vector)


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._http = httpx.Client(base_url=config.base_url, timeout=120.0)

    @property
    def _is_cloud(self) -> bool:
        return self.config.provider != "ollama"

    def _audit(self, endpoint: str, payload_chars: int) -> None:
        """Record an outbound cloud call before it happens (principle 4)."""
        if self._is_cloud and self.config.events_path is not None:
            events.append(
                self.config.events_path,
                kind="audit",
                provider=self.config.provider,
                endpoint=endpoint,
                model=self.config.chat_model if endpoint == "chat" else self.config.embed_model,
                est_tokens=payload_chars // 4,
            )

    def _headers(self) -> dict[str, str]:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}

    def chat(self, messages: list[dict], *, stream: bool = True):
        """Chat completion. Returns a token iterator if stream else the full string."""
        self._audit("chat", sum(len(m.get("content", "")) for m in messages))
        if self.config.provider == "ollama":
            return self._chat_ollama(messages, stream)
        return self._chat_openai(messages, stream)

    def embed(self, text: str) -> bytes:
        """Raw embedding call (no cache — caching lives in memory/embed.py)."""
        self._audit("embed", len(text))
        if self.config.provider == "ollama":
            resp = self._http.post(
                "/api/embed", json={"model": self.config.embed_model, "input": text}
            )
            resp.raise_for_status()
            return _pack(resp.json()["embeddings"][0])
        resp = self._http.post(
            "/v1/embeddings",
            json={"model": self.config.embed_model, "input": text},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return _pack(resp.json()["data"][0]["embedding"])

    # -- Ollama ----------------------------------------------------------
    def _chat_ollama(self, messages: list[dict], stream: bool):
        payload = {"model": self.config.chat_model, "messages": messages, "stream": stream}
        if not stream:
            resp = self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        return self._stream_ollama(payload)

    def _stream_ollama(self, payload: dict) -> Iterator[str]:
        import json

        with self._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break

    # -- OpenAI-compatible cloud ----------------------------------------
    def _chat_openai(self, messages: list[dict], stream: bool):
        payload = {"model": self.config.chat_model, "messages": messages, "stream": stream}
        if not stream:
            resp = self._http.post(
                "/v1/chat/completions", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        return self._stream_openai(payload)

    def _stream_openai(self, payload: dict) -> Iterator[str]:
        import json

        with self._http.stream(
            "POST", "/v1/chat/completions", json=payload, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
