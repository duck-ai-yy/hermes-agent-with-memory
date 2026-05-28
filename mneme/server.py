"""FastAPI surface — secondary frontend, started on demand via `mneme serve`.

Synchronous SQLite work is wrapped in `asyncio.to_thread` so it does not block
the event loop (PRINCIPLES.md principle 3: no aiosqlite dependency).
"""

from __future__ import annotations

import asyncio
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

app = FastAPI(title="Mneme", version="0.11.0")


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
