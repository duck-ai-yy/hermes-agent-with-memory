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


# -- v0.10 / session_id fixtures --------------------------------------------


# v0.9 schema, verbatim from commit 8a201c5 (the merge that capped v0.9).
# Pre-v0.10 databases lack `session_id` on `slices`; the migration in
# store._migrate_slices_session_id is what we exercise via this fixture.
_V09_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS slices (
    id          TEXT PRIMARY KEY,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    turn_id     TEXT,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slices_turn ON slices(turn_id);

CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    first_seen  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    src         TEXT NOT NULL REFERENCES nodes(id),
    dst         TEXT NOT NULL REFERENCES nodes(id),
    type        TEXT NOT NULL,
    slice_id    TEXT NOT NULL REFERENCES slices(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src   ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst   ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_slice ON edges(slice_id);

CREATE TABLE IF NOT EXISTS embeddings_cache (
    text_hash   TEXT PRIMARY KEY,
    vector      BLOB NOT NULL,
    created_at  INTEGER NOT NULL
);
"""


@pytest.fixture
def legacy_events_jsonl(tmp_path):
    """Write a pre-v0.10 events.jsonl (no `session_id` field) and return the
    path. Used by tests that must prove v0.10 readers tolerate older logs
    byte-for-byte (E5 KPI: v0.7-v0.9 baseline tests do not need re-writing)."""
    ep = tmp_path / "events.jsonl"
    now = 1_700_000_000
    records = [
        # Pre-call trace
        {"ts": now, "kind": "trace", "id": "T1", "query": "hi",
         "used_slices": [], "prompt_hash": "abc",
         "model": "gpt-4o-mini", "provider": "openai"},
        # Close trace
        {"ts": now, "kind": "trace", "id": "T1",
         "response_hash": "def", "citation_quality": "coarse",
         "provider": "openai", "prompt_tokens": 10,
         "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0001,
         "iters": 1, "tool_calls": 0, "tool_rejects": 0},
        # Ingest events (no session_id field)
        {"ts": now, "kind": "ingest", "slice_id": "S1", "role": "user",
         "turn_id": "TURN1", "nodes": 0, "edges": 0},
        {"ts": now, "kind": "ingest", "slice_id": "S2", "role": "assistant",
         "turn_id": "TURN1", "nodes": 0, "edges": 0},
    ]
    with open(ep, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return ep


@pytest.fixture
def legacy_db_with_v09_data(tmp_path):
    """Create a v0.9-shape SQLite DB (no slices.session_id), seed 3 slices
    across 2 distinct turn_ids, return the path. Caller then runs the v0.10
    migration via `store.init_db` and asserts back-fill behavior."""
    import sqlite3
    db_path = tmp_path / "legacy.sqlite"
    cx = sqlite3.connect(db_path)
    cx.executescript(_V09_SCHEMA)
    now = 1_700_000_000
    rows = [
        ("S1", "user",      "first user msg",   "TURN_A", now),
        ("S2", "assistant", "first reply",      "TURN_A", now + 1),
        ("S3", "user",      "second user msg",  "TURN_B", now + 2),
    ]
    cx.executemany(
        "INSERT INTO slices (id, role, text, turn_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    cx.commit()
    cx.close()
    return db_path


@pytest.fixture
def http_client(tmp_path, monkeypatch, fake_llm):
    """Spin up a TestClient with paths.DB_PATH / paths.EVENTS_PATH redirected
    into tmp_path. The DB is initialized via store.init_db so the v0.10
    schema is in place. fake_llm is the standard FakeLLM fixture so HTTP
    /chat goes through the deterministic stub.

    Lead must-fix #1: F-section tests use a real TestClient instead of
    bypassing FastAPI (which is what test_server_chat.py does on B22 lesson
    grounds). v0.10 session-id behavior lives on the HTTP boundary, so the
    test must exercise the boundary itself.
    """
    from fastapi.testclient import TestClient
    from mneme import paths
    from mneme.server import app as fastapi_app

    db_path = tmp_path / "db.sqlite"
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(paths, "DB_PATH", db_path)
    monkeypatch.setattr(paths, "EVENTS_PATH", events_path)

    cx = store.connect(db_path)
    store.init_db(cx)
    cx.close()

    return TestClient(fastapi_app)


# -- v0.13 / SSE streaming fixtures -----------------------------------------


@pytest.fixture
def sse_client(tmp_path, monkeypatch, fake_llm):
    """Spin up a TestClient with SSE-aware POST helper.

    Identical setup to `http_client` (DB / events.jsonl rerouted into tmp_path,
    FakeLLM injected) plus a `.post_sse(body, sid=None)` method that forces
    `Accept: text/event-stream` and parses the response back to (frames, raw).

    Returns the TestClient itself with two extra attributes attached:
      - `client.post_sse(body, sid=None) -> (list[dict], bytes)` parses each
        SSE frame to `{"event": ..., "data": ...}` (data still a string).
      - `client.parse_data(frame)` -> dict (json.loads the data string).

    raw bytes are preserved exactly as received so tests can check CRLF
    rejection, UTF-8 encoding, frame separator literals, etc. (lead must-fix:
    many SSE parsers tolerate CRLF; raw bytes are the only ground truth).
    """
    from fastapi.testclient import TestClient
    from mneme import paths
    from mneme.server import app as fastapi_app

    db_path = tmp_path / "db.sqlite"
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(paths, "DB_PATH", db_path)
    monkeypatch.setattr(paths, "EVENTS_PATH", events_path)

    cx = store.connect(db_path)
    store.init_db(cx)
    cx.close()

    client = TestClient(fastapi_app)

    def _parse_frames(raw: bytes) -> list[dict]:
        """Parse raw SSE bytes into a list of `{event, data}` dicts.

        Strictly LF-separated (`\\n\\n` between frames; `\\n` between lines
        inside a frame). The parser is intentionally rigid -- it will trip if
        a frame uses CRLF, has multiple blank lines, or violates the
        `event:` / `data:` literal prefixes. Test must verify raw bytes too.
        """
        # Split on the canonical "\n\n" separator. Don't try to be helpful.
        text = raw.decode("utf-8")
        # Drop trailing separator if present so we don't yield an empty frame.
        if text.endswith("\n\n"):
            text = text[:-2]
        frames: list[dict] = []
        for block in text.split("\n\n"):
            if not block:
                continue
            event_type: str | None = None
            data_parts: list[str] = []
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event_type = line[len("event: "):]
                elif line.startswith("data: "):
                    data_parts.append(line[len("data: "):])
                else:
                    # Unexpected line; surface via empty event marker.
                    pass
            frames.append({"event": event_type, "data": "\n".join(data_parts)})
        return frames

    def post_sse(body: dict, sid: str | None = None) -> tuple[list[dict], bytes]:
        payload = dict(body)
        if sid is not None:
            payload["session_id"] = sid
        with client.stream(
            "POST", "/chat",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = b"".join(resp.iter_bytes())
            # Attach last response so tests can introspect headers / status.
            post_sse.last_response = resp  # type: ignore[attr-defined]
            post_sse.last_status = resp.status_code  # type: ignore[attr-defined]
            post_sse.last_headers = dict(resp.headers)  # type: ignore[attr-defined]
        return _parse_frames(raw), raw

    def parse_data(frame: dict) -> dict:
        return json.loads(frame["data"])

    client.post_sse = post_sse  # type: ignore[attr-defined]
    client.parse_data = parse_data  # type: ignore[attr-defined]
    client.parse_frames = _parse_frames  # type: ignore[attr-defined]
    return client


@pytest.fixture
def disconnect_mocker(monkeypatch):
    """Make `starlette.requests.Request.is_disconnected` controllable.

    Usage:
        disconnect_mocker.disconnect_after(n_frames)   # False N times then True
        disconnect_mocker.disconnect_immediately()     # True on first check
        disconnect_mocker.never_disconnect()           # always False (default)

    The mock replaces the method body so any Request instance picks it up
    -- this means the disconnect signal is global across the test, which is
    fine because tests are isolated and each fixture-scoped TestClient
    talks to exactly one handler invocation in a deterministic order.
    """
    state = {"n_false": 0, "default": False, "calls": 0}

    async def fake_is_disconnected(self):  # noqa: ARG001
        state["calls"] += 1
        if state["n_false"] > 0:
            state["n_false"] -= 1
            return False
        return state["default"]

    import starlette.requests
    monkeypatch.setattr(
        starlette.requests.Request, "is_disconnected", fake_is_disconnected
    )

    spy = SimpleNamespace(
        disconnect_after=lambda n: state.update(n_false=n, default=True),
        disconnect_immediately=lambda: state.update(n_false=0, default=True),
        never_disconnect=lambda: state.update(n_false=0, default=False),
        state=state,
    )
    return spy


@pytest.fixture
def repl_runner(tmp_path, monkeypatch, fake_llm):
    """Drive the `mneme chat` REPL via Typer's CliRunner. Redirects
    paths.DB_PATH / paths.EVENTS_PATH into tmp_path and seeds the v0.10
    schema. fake_llm is wired so chat() returns AssistantMessage cleanly.

    Returns a callable: `repl_runner(input_lines: str) -> Result`.
    """
    from typer.testing import CliRunner
    from mneme import cli as cli_mod
    from mneme import paths
    from mneme.llm.client import AssistantMessage

    db_path = tmp_path / "db.sqlite"
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(paths, "DB_PATH", db_path)
    monkeypatch.setattr(paths, "EVENTS_PATH", events_path)

    cx = store.connect(db_path)
    store.init_db(cx)
    cx.close()

    # The CLI uses `confirm_cb=_cli_confirm_tool` (non-None) so the agent
    # loop expects AssistantMessage from chat(). Override the default
    # FakeLLM.chat behavior so a default `tools is not None` call returns
    # AssistantMessage instead of streaming text. The conftest default DOES
    # return AssistantMessage when tools is non-None (line 75) — so we're
    # fine. The CLI passes no tool calls in the default reply, so end_turn
    # cleanly. Tests that want different scripts can append to
    # fake_llm.tool_call_script after the fixture returns.

    runner = CliRunner()

    def _run(input_lines: str, *, extra_args=None):
        args = ["chat"] + (extra_args or [])
        return runner.invoke(cli_mod.app, args, input=input_lines,
                             catch_exceptions=False)

    # Expose the AssistantMessage symbol for tests that want to script.
    _run.AssistantMessage = AssistantMessage
    return _run
