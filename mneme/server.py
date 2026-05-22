"""FastAPI surface — secondary frontend, started on demand via `mneme serve`.

Synchronous SQLite work is wrapped in `asyncio.to_thread` so it does not block
the event loop (PRINCIPLES.md principle 3: no aiosqlite dependency).
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import paths
from .agent import respond
from .ids import ulid
from .memory import forget as forget_mod
from .memory import store
from .trace import events

app = FastAPI(title="Mneme", version="0.0.1")


class ChatRequest(BaseModel):
    messages: list[dict]
    session_id: str | None = None


def _run_turn(text: str, turn_id: str) -> dict:
    cx = store.connect(paths.DB_PATH)
    try:
        reply = respond(text, turn_id, cx)
    finally:
        cx.close()
    return {
        "text": reply.text,
        "trace_id": reply.trace_id,
        "citation_quality": reply.citation_quality,
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Run a chat turn from the last user message."""
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(400, "no user message in request")
    turn_id = req.session_id or ulid()
    return await asyncio.to_thread(_run_turn, user_messages[-1]["content"], turn_id)


@app.get("/explain/{trace_id}")
async def explain(trace_id: str) -> dict:
    """Return the full trace record."""
    try:
        return await asyncio.to_thread(events.explain, paths.EVENTS_PATH, trace_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"no trace: {trace_id}")


def _run_forget(slice_id: str) -> dict:
    cx = store.connect(paths.DB_PATH)
    try:
        result = forget_mod.forget(slice_id, consent=True, cx=cx)
    finally:
        cx.close()
    return {
        "slice_id": slice_id,
        "edges_removed": result.edges_removed,
        "vectors_removed": result.vectors_removed,
    }


@app.post("/forget/{slice_id}")
async def forget(slice_id: str) -> dict:
    """Delete a memory (consent is implied by calling this endpoint)."""
    try:
        return await asyncio.to_thread(_run_forget, slice_id)
    except KeyError:
        raise HTTPException(404, f"no such slice: {slice_id}")


def _run_snapshot() -> dict:
    import time

    cx = store.connect(paths.DB_PATH)
    try:
        dest = paths.SNAPSHOTS_DIR / f"db-{int(time.time())}.sqlite"
        store.snapshot(cx, dest)
    finally:
        cx.close()
    return {"snapshot": str(dest)}


@app.post("/snapshot")
async def snapshot() -> dict:
    """Trigger a SQLite .backup."""
    return await asyncio.to_thread(_run_snapshot)


def _run_stats() -> dict:
    cx = store.connect(paths.DB_PATH)
    try:
        counts = {
            table: cx.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("slices", "nodes", "edges")
        }
    finally:
        cx.close()
    counts["db_bytes"] = paths.DB_PATH.stat().st_size
    return counts


@app.get("/stats")
async def stats() -> dict:
    """Return slice / node / edge counts and database size."""
    return await asyncio.to_thread(_run_stats)
