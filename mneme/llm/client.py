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
class LLMConfig:
    provider: str = "ollama"            # "ollama" | "openai"
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    api_key: str | None = None          # required only for cloud providers
    events_path: Path | None = None     # where audit events are written
    embed_via: str = "provider"         # "provider" | "hash" — hash = local stdlib
                                        # fallback for providers without embeddings
                                        # (e.g. DeepSeek); degrades recall to
                                        # exact-text matches only.


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

    @property
    def _is_cloud(self) -> bool:
        return self.config.provider != "ollama"

    def _audit(self, endpoint: str, payload_chars: int) -> None:
        """Record an outbound cloud call before it happens (principle 4)."""
        if self._is_cloud and self.config.events_path is not None:
            events.append(
                self.config.events_path, kind="audit",
                provider=self.config.provider, endpoint=endpoint,
                model=self.config.chat_model if endpoint == "chat" else self.config.embed_model,
                est_tokens=payload_chars // 4,
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def chat(self, messages: list[dict], *, stream: bool = True):
        """Chat completion. Returns a token iterator if stream else the full string."""
        self._audit("chat", sum(len(m.get("content", "")) for m in messages))
        payload = {"model": self.config.chat_model, "messages": messages, "stream": stream}
        if self.config.provider == "ollama":
            if not stream:
                r = self._http.post("/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
            return self._stream_ollama(payload)
        if not stream:
            r = self._http.post("/v1/chat/completions", json=payload, headers=self._headers())
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return self._stream_openai(payload)

    def embed(self, text: str) -> bytes:
        """Raw embedding call (no cache — caching lives in memory/embed.py)."""
        if self.config.embed_via == "hash":
            # Deterministic local fallback for providers without embeddings.
            rng = random.Random(hashlib.sha256(text.encode()).digest())
            return _pack([rng.random() for _ in range(768)])
        self._audit("embed", len(text))
        body = {"model": self.config.embed_model, "input": text}
        if self.config.provider == "ollama":
            r = self._http.post("/api/embed", json=body)
            r.raise_for_status()
            return _pack(r.json()["embeddings"][0])
        r = self._http.post("/v1/embeddings", json=body, headers=self._headers())
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
                    break

    def _stream_openai(self, payload: dict) -> Iterator[str]:
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
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
