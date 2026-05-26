"""FastAPI surface — secondary frontend, started on demand via `mneme serve`.

Synchronous SQLite work is wrapped in `asyncio.to_thread` so it does not block
the event loop (PRINCIPLES.md principle 3: no aiosqlite dependency).

v0.13: POST /chat gains an SSE streaming variant. Negotiation is Accept-based
(`text/event-stream` -> SSE; anything else -> the v0.10 JSON path byte-for-byte).
Helpers live in this module per PRINCIPLE 1 (no new files for ~50 LOC).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import paths
from .agent import respond
from .ids import ulid
from .memory import forget as forget_mod
from .memory import store
from .trace import events

app = FastAPI(title="Mneme", version="0.13.0")


# -- SSE helpers (v0.13) ---------------------------------------------------
#
# Pure functions, no side effects, no I/O. The streaming generator (Unit C)
# is added in step 2.


def _wants_sse(accept_header: str | None) -> bool:
    """Accept-header content negotiation for the SSE branch.

    True iff the first comma-separated token, stripped and lower-cased, is
    exactly `text/event-stream`. Every other value (None, empty, `*/*`,
    `application/json`, `text/event-stream;q=0.9`, garbage) silently degrades
    to the v0.10 JSON path. The literal rule is taken verbatim from the
    v0.13 design §2 — `accept_raw.split(",")[0].strip().lower()` — so the
    JSON path stays byte-identical for legacy callers.
    """
    if accept_header is None:
        return False
    return accept_header.split(",")[0].strip().lower() == "text/event-stream"


def _make_sse_frame(event_type: str, data: dict) -> bytes:
    """Format one SSE frame as `event: <type>\\ndata: <json>\\n\\n` bytes.

    Single-line, compact JSON (no spaces, no newlines), UTF-8 encoded, LF
    line endings (no CRLF). `ensure_ascii=False` keeps non-ASCII characters
    raw (e.g. CJK in user text) instead of `\\uXXXX`-escaping them — the
    payload is already UTF-8.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


class ChatRequest(BaseModel):
    messages: list[dict]
    session_id: str | None = None


@contextmanager
def _db():
    cx = store.connect(paths.DB_PATH)
    try:
        yield cx
    finally:
        cx.close()


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Run a chat turn from the last user message.

    v0.10: `session_id` is independent of `turn_id`. A fresh turn_id is
    minted every request (pre-v0.10 collapsed the two, which silently
    grouped every HTTP-driven turn under the same id). The session_id
    follows §7 of the design: None or whitespace-only -> mint a new ULID;
    any non-empty string is accepted silently (the server is stateless and
    never raises SessionNotFound).
    """
    user_msgs = [m for m in req.messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message in request")
    turn_id = ulid()
    raw_sess = req.session_id
    if raw_sess is None or raw_sess.strip() == "":
        session_id = ulid()
    else:
        session_id = raw_sess

    def work():
        with _db() as cx:
            # confirm_cb=None on purpose: HTTP /chat has no UI to confirm a
            # tool call interactively. The agent loop degrades to the v0.7
            # single-call path (no tools declared, no tool round-trips).
            # Server-side tool calling is a v0.13+ design problem.
            # allowed_tools=[] is the explicit no-tools opt-in: even if a
            # future change accidentally injects a confirm_cb here, the
            # empty allowlist will keep the server from declaring any tool.
            r = respond(user_msgs[-1]["content"], turn_id, cx,
                        session_id=session_id,
                        confirm_cb=None, allowed_tools=[])
            # Response field order is pinned (design §7): text, trace_id,
            # citation_quality, session_id. Insertion order matters because
            # FastAPI serializes the dict directly.
            return {
                "text": r.text,
                "trace_id": r.trace_id,
                "citation_quality": r.citation_quality,
                "session_id": session_id,
            }

    return await asyncio.to_thread(work)


@app.get("/explain/{trace_id}")
async def explain(trace_id: str) -> dict:
    try:
        return await asyncio.to_thread(events.explain, paths.EVENTS_PATH, trace_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"no trace: {trace_id}")


@app.post("/forget/{slice_id}")
async def forget(slice_id: str) -> dict:
    """Delete a memory (consent is implied by calling this endpoint)."""
    def work():
        with _db() as cx:
            r = forget_mod.forget(slice_id, consent=True, cx=cx)
            return {"slice_id": slice_id, "edges_removed": r.edges_removed,
                    "vectors_removed": r.vectors_removed}

    try:
        return await asyncio.to_thread(work)
    except KeyError:
        raise HTTPException(404, f"no such slice: {slice_id}")


@app.post("/snapshot")
async def snapshot() -> dict:
    def work():
        dest = paths.SNAPSHOTS_DIR / f"db-{int(time.time())}.sqlite"
        with _db() as cx:
            store.snapshot(cx, dest)
        return {"snapshot": str(dest)}

    return await asyncio.to_thread(work)


@app.get("/stats")
async def stats() -> dict:
    def work():
        with _db() as cx:
            counts = {t: cx.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("slices", "nodes", "edges")}
        counts["db_bytes"] = paths.DB_PATH.stat().st_size
        return counts

    return await asyncio.to_thread(work)
