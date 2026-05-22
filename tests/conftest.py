"""Shared fixtures. The sandbox has no Ollama, so LLM calls use a deterministic
fake: `embed` is a seeded RNG (same text -> same vector), `chat` returns canned
concept JSON or a canned reply depending on the system prompt.
"""

from __future__ import annotations

import random
import struct
from types import SimpleNamespace

import pytest

from mneme.memory import store


class FakeLLM:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chat_model="fake-chat", embed_model="fake-embed", provider="ollama"
        )
        self.concept_json = '{"nodes": [], "edges": []}'
        self.reply = "noted [^abc]"
        self.embed_calls = 0
        self.chat_calls = 0

    def chat(self, messages: list[dict], *, stream: bool = True):
        self.chat_calls += 1
        system = messages[0]["content"]
        if "STRICT JSON" in system:
            return self.concept_json
        return self.reply

    def embed(self, text: str) -> bytes:
        self.embed_calls += 1
        rng = random.Random(text)
        return struct.pack("<768f", *[rng.random() for _ in range(768)])


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("mneme.llm.client.get_client", lambda *a, **k: llm)
    return llm


@pytest.fixture
def cx(tmp_path):
    connection = store.connect(tmp_path / "db.sqlite")
    store.init_db(connection)
    yield connection
    connection.close()
