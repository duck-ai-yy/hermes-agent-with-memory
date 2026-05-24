"""Shared fixtures. The sandbox has no Ollama, so LLM calls use a deterministic
fake: `embed` is a seeded RNG (same text -> same vector), `chat` returns canned
concept JSON or a canned reply depending on the system prompt.

v0.8: `chat()` also handles a `tools` kwarg. When tools are passed, the fake
returns an `AssistantMessage` driven by `tool_call_script` (a list of canned
turn outputs popped one per call). The default script ends the turn cleanly
so older tests that didn't set the script still get sane behavior.
"""

from __future__ import annotations

import random
import struct
from types import SimpleNamespace

import pytest

from mneme.llm.client import AssistantMessage
from mneme.memory import store


class FakeLLM:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chat_model="fake-chat", embed_model="fake-embed", provider="ollama"
        )
        self.concept_json = '{"nodes": [], "edges": []}'
        self.reply = "noted"
        self.embed_calls = 0
        self.chat_calls = 0
        self.last_usage: SimpleNamespace | None = None
        # Agent-loop scripting (v0.8). Each list entry describes one chat()
        # invocation **when tools are passed**. An entry is either:
        #   - an AssistantMessage instance, or
        #   - a dict {"text": str, "tool_calls": [ToolCall...],
        #             "stop_reason": str}
        # When the script is exhausted, fake returns an end_turn message with
        # self.reply — so the loop terminates and stale scripts can't hang
        # the test runner.
        self.tool_call_script: list = []

    def chat(self, messages: list[dict], *, stream: bool = True, tools=None):
        self.chat_calls += 1
        self.last_usage = None
        if tools:
            # Tool-using path: respect the script; default to a clean end_turn.
            entry = self.tool_call_script.pop(0) if self.tool_call_script else None
            if entry is None:
                msg = AssistantMessage(
                    text=self.reply, tool_calls=[], stop_reason="end_turn",
                )
            elif isinstance(entry, AssistantMessage):
                msg = entry
            else:
                msg = AssistantMessage(
                    text=entry.get("text", ""),
                    tool_calls=entry.get("tool_calls", []),
                    stop_reason=entry.get("stop_reason", "end_turn"),
                )
            # Token usage is reported even when the model called a tool — the
            # agent loop accumulates across iterations for the close-trace.
            self.last_usage = self._fake_usage(msg.text or "tool")
            return msg
        system = messages[0]["content"]
        text = self.concept_json if "STRICT JSON" in system else self.reply
        if not stream:
            self.last_usage = self._fake_usage(text)
            return text
        return self._stream_chunks(text)

    def _stream_chunks(self, text: str):
        # Three roughly-equal chunks — enough to exercise the streaming path
        # without making test assertions chunk-boundary-sensitive.
        n = 3
        if len(text) < n:
            yield text
        else:
            step = len(text) // n
            for i in range(n):
                start = i * step
                end = start + step if i < n - 1 else len(text)
                yield text[start:end]
        self.last_usage = self._fake_usage(text)

    @staticmethod
    def _fake_usage(text: str) -> SimpleNamespace:
        p, c = 10, max(1, len(text) // 4)
        return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)

    def embed(self, text: str) -> bytes:
        self.embed_calls += 1
        rng = random.Random(text)
        return struct.pack("<768f", *[rng.random() for _ in range(768)])


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("mneme.llm.client.get_client", lambda: llm)
    return llm


@pytest.fixture
def cx(tmp_path):
    connection = store.connect(tmp_path / "db.sqlite")
    store.init_db(connection)
    yield connection
    connection.close()
