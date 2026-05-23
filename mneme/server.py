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

app = FastAPI(title="Mneme", version="0.1.0")


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
    """Run a chat turn from the last user message."""
    user_msgs = [m for m in req.messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message in request")
    turn_id = req.session_id or ulid()

    def work():
        with _db() as cx:
            r = respond(user_msgs[-1]["content"], turn_id, cx)
            return {"text": r.text, "trace_id": r.trace_id, "citation_quality": r.citation_quality}

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
