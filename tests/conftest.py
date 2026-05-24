"""Shared fixtures. The sandbox has no Ollama, so LLM calls use a deterministic
fake: `embed` is a seeded RNG (same text -> same vector), `chat` returns canned
concept JSON or a canned reply depending on the system prompt.

v0.8: `chat()` also handles a `tools` kwarg. When tools are passed, the fake
returns an `AssistantMessage` driven by `tool_call_script` (a list of canned
turn outputs popped one per call). The default script ends the turn cleanly
so older tests that didn't set the script still get sane behavior.

The fake also keeps spy fields (`messages_seen`, `stream_flags_seen`,
`tools_seen`, `usage_script`) so tests can verify per-call inputs (especially
that the SECOND-round messages list carries a `tool_result` with `is_error`
set — the v0.6 "happy-path lying" lesson applied to the agent loop).

v0.9: four new opt-in fixtures for the registry / tools test suite —
`isolated_registry`, `mock_httpx`, `mock_getaddrinfo`, `events_spy`. The
registry snapshot/restore fixture is opt-in here; test_tool_registry.py
flips it autouse at module scope so every test there starts from a clean
slate. Other files opt in only where they need it (e.g. tests that decorate
a fresh @tool fn).
"""

from __future__ import annotations

import copy
import json
import random
import struct
from types import SimpleNamespace

import httpx
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
        # Per-call usage override; one entry consumed per chat() call. A None
        # entry means "use the auto-derived fake usage" (the default). When
        # the script is exhausted falls back to auto. Used by cost / trace
        # accumulation tests that need known token counts per round.
        self.usage_script: list = []
        # Spies — captured at each chat() call so tests can assert what the
        # agent actually fed the LLM. messages is deep-copied because the
        # agent loop mutates the running list across iterations.
        self.messages_seen: list[list[dict]] = []
        self.stream_flags_seen: list[bool] = []
        self.tools_seen: list = []

    def chat(self, messages: list[dict], *, stream: bool = True, tools=None):
        self.chat_calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        self.stream_flags_seen.append(stream)
        self.tools_seen.append(copy.deepcopy(tools) if tools is not None else None)
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
            self.last_usage = self._next_usage(msg.text or "tool")
            return msg
        system = messages[0]["content"]
        text = self.concept_json if "STRICT JSON" in system else self.reply
        if not stream:
            self.last_usage = self._next_usage(text)
            return text
        return self._stream_chunks(text)

    def _next_usage(self, text: str) -> SimpleNamespace:
        """Honor `usage_script` if non-empty (None means auto); else auto."""
        if self.usage_script:
            override = self.usage_script.pop(0)
            if override is not None:
                return override
        return self._fake_usage(text)

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
        self.last_usage = self._next_usage(text)

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


# -- v0.9 / M2 fixtures -----------------------------------------------------


@pytest.fixture
def isolated_registry():
    """Snapshot/restore `mneme.tools.registry._REGISTRY` around one test.

    The registry is module-level state populated at import time by
    `@tool` decorations on each tool module. Tests that decorate fresh
    functions (B1-B11) or that mutate the registry must run with this
    fixture or they pollute every later test. Opt-in for most files;
    test_tool_registry.py flips it autouse via module-scope `pytestmark`.

    Restores by replacing the dict contents in place (not rebinding) so
    any cached reference to `_REGISTRY` still sees the original state.
    """
    from mneme.tools import registry as _reg
    snapshot = dict(_reg._REGISTRY)
    try:
        yield _reg._REGISTRY
    finally:
        _reg._REGISTRY.clear()
        _reg._REGISTRY.update(snapshot)


@pytest.fixture
def mock_httpx(monkeypatch):
    """Install a programmable transport that web_fetch's `httpx.Client` will
    use. `web_fetch` constructs its own `httpx.Client(...)` inside the tool,
    so we monkeypatch `httpx.Client.__init__` to inject our transport.

    Usage:
        def test_x(mock_httpx):
            mock_httpx.set_handler(lambda req: httpx.Response(200, text="hi"))
            ...

    The handler receives an `httpx.Request` and must return an
    `httpx.Response`. Multi-step (redirect) flows can stash a list of
    handlers via `set_handlers([h1, h2, ...])` to pop one per call.
    """
    state: dict = {"handler": None, "handlers": None, "calls": []}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["calls"].append(request)
        if state["handlers"] is not None:
            if not state["handlers"]:
                raise AssertionError("mock_httpx handler queue exhausted")
            h = state["handlers"].pop(0)
            return h(request)
        if state["handler"] is None:
            raise AssertionError(
                "mock_httpx: no handler installed; call set_handler/set_handlers"
            )
        return state["handler"](request)

    transport = httpx.MockTransport(_handler)
    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    spy = SimpleNamespace(
        set_handler=lambda h: state.update(handler=h, handlers=None),
        set_handlers=lambda hs: state.update(handlers=list(hs), handler=None),
        calls=state["calls"],
    )
    return spy


@pytest.fixture
def mock_getaddrinfo(monkeypatch):
    """Programmable socket.getaddrinfo for SSRF preflight tests.

    Set a host -> address (or list of addresses) mapping. Unknown hosts
    raise socket.gaierror so tests catch missing entries instead of
    silently hitting the real DNS resolver. Opt-in: not autouse.
    """
    import socket
    state: dict = {"mapping": {}}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in state["mapping"]:
            raise socket.gaierror(-2, f"unknown host (mock): {host}")
        addrs = state["mapping"][host]
        if isinstance(addrs, str):
            addrs = [addrs]
        # Mimic the real getaddrinfo tuple shape: (family, type, proto,
        # canonname, sockaddr=(addr, port)).
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (a, port or 0))
                for a in addrs]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    spy = SimpleNamespace(
        set=lambda host, addrs: state["mapping"].__setitem__(host, addrs),
        mapping=state["mapping"],
    )
    return spy


@pytest.fixture
def events_spy(tmp_path):
    """Callable that loads events.jsonl from tmp_path and filters records.

    Usage:
        recs = events_spy(kind="tool_result")
        recs = events_spy(predicate=lambda r: r.get("error") == "Timeout")
        recs = events_spy()      # all records
    """
    def _load(*, kind: str | None = None, predicate=None) -> list[dict]:
        ep = tmp_path / "events.jsonl"
        if not ep.exists():
            return []
        out: list[dict] = []
        for line in ep.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is not None and rec.get("kind") != kind:
                continue
            if predicate is not None and not predicate(rec):
                continue
            out.append(rec)
        return out
    return _load
