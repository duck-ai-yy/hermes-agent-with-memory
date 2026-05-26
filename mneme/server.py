"""FastAPI surface — secondary frontend, started on demand via `mneme serve`.

Synchronous SQLite work is wrapped in `asyncio.to_thread` so it does not block
the event loop (PRINCIPLES.md principle 3: no aiosqlite dependency).

v0.13: POST /chat gains an SSE streaming variant. Negotiation is Accept-based
(`text/event-stream` -> SSE; anything else -> the v0.10 JSON path byte-for-byte).
Helpers live in this module per PRINCIPLE 1 (no new files for ~50 LOC).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import paths
from .agent import respond, respond_stream
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


# SSE response headers — set together as a single dict per lead phase-1 hint:
# `Content-Type` is set implicitly via StreamingResponse(media_type=...), but
# `Cache-Control` and `X-Accel-Buffering` must travel together. The v0.15
# Feishu webhook adapter will sit behind nginx; `X-Accel-Buffering: no` is
# the only reliable way to defeat the default proxy buffering.
_SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


async def _sse_event_stream(
    user_text: str,
    turn_id: str,
    session_id: str,
    request: Request,
):
    """Drive `agent.respond_stream` and yield SSE-framed bytes.

    Sequence (happy path):
      1. yield `start` frame  ({session_id, turn_id})       -- before LLM call
      2. for each text chunk yielded by respond_stream:
           yield `delta` frame ({text})
      3. on StopIteration, read `.value` (Reply), merge close-trace fields
         from events.jsonl, yield `done` frame in the pinned insertion order.

    `respond_stream` is a sync generator; every sync call (open cx, pull
    chunk, read close-trace, close cx) is dispatched onto a single-worker
    thread executor so they all run on the same OS thread. This keeps the
    SQLite connection (which is bound to its creation thread) happy without
    relaxing the default `check_same_thread=True` everywhere else. The
    executor is owned by this generator and shut down in `finally`.
    """
    # Step 1: yield start frame BEFORE touching the LLM (design §3).
    # The data dict insertion order matches the design literal {session_id,
    # turn_id} -- session_id first, turn_id second.
    yield _make_sse_frame("start", {"session_id": session_id, "turn_id": turn_id})

    # One-worker executor pins every sync hop to the same thread so the
    # SQLite connection (created in that thread) can be reused across hops.
    loop = asyncio.get_running_loop()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def _run(fn, *args):
        return await loop.run_in_executor(pool, fn, *args)

    cx = await _run(store.connect, paths.DB_PATH)
    try:
        # Build the generator. Calling respond_stream() does NOT execute the
        # body; it returns a generator object. The body runs on first next().
        # confirm_cb=None + allowed_tools=[] mirror the v0.10 JSON path
        # exactly (§7 "tool 仍 closed"). The v0.7 streaming path is hit
        # because confirm_cb is None.
        gen = respond_stream(
            user_text, turn_id, cx,
            session_id=session_id,
            confirm_cb=None, allowed_tools=[],
        )

        reply = None
        # Use a sentinel to communicate StopIteration from the worker thread:
        # run_in_executor cannot directly propagate StopIteration.
        _DONE = object()

        def _pull_next():
            try:
                return next(gen)
            except StopIteration as e:
                _pull_next.reply = e.value  # type: ignore[attr-defined]
                return _DONE

        while True:
            chunk = await _run(_pull_next)
            if chunk is _DONE:
                reply = _pull_next.reply  # type: ignore[attr-defined]
                break
            # Step 2: one delta frame per text chunk.
            yield _make_sse_frame("delta", {"text": chunk})

        # Step 3: done frame. The close-trace event was written by
        # _close_turn inside respond_stream right before StopIteration
        # fired, so it is already on disk. Reading it back is the cheapest
        # way to honour the v0.7 boundary 1/2/5 rules (cost / tokens may
        # be omitted) without duplicating that logic on the HTTP layer.
        ep = await _run(store.events_path, cx)
        merged: dict = {}
        if ep is not None:
            try:
                merged = await _run(events.explain, ep, reply.trace_id)
            except (KeyError, FileNotFoundError):
                merged = {}

        # Pinned insertion order (design §3): trace_id, citation_quality,
        # session_id, [cost_usd], [prompt_tokens], [completion_tokens],
        # [total_tokens], iters. The optional token / cost fields are
        # only included when the close-trace event itself recorded them
        # (mirrors v0.7 boundary 1/2/5 exactly).
        done_data: dict = {
            "trace_id": reply.trace_id,
            "citation_quality": reply.citation_quality,
            "session_id": session_id,
        }
        if "cost_usd" in merged:
            done_data["cost_usd"] = merged["cost_usd"]
        if "prompt_tokens" in merged:
            done_data["prompt_tokens"] = merged["prompt_tokens"]
        if "completion_tokens" in merged:
            done_data["completion_tokens"] = merged["completion_tokens"]
        if "total_tokens" in merged:
            done_data["total_tokens"] = merged["total_tokens"]
        # iters is always recorded in the close-trace (mneme/agent.py:173).
        done_data["iters"] = merged.get("iters", 0)

        yield _make_sse_frame("done", done_data)
    finally:
        try:
            await _run(cx.close)
        finally:
            pool.shutdown(wait=False)


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
async def chat(req: ChatRequest, request: Request):
    """Run a chat turn from the last user message.

    v0.10: `session_id` is independent of `turn_id`. A fresh turn_id is
    minted every request (pre-v0.10 collapsed the two, which silently
    grouped every HTTP-driven turn under the same id). The session_id
    follows §7 of the design: None or whitespace-only -> mint a new ULID;
    any non-empty string is accepted silently (the server is stateless and
    never raises SessionNotFound).

    v0.13: when the client sends `Accept: text/event-stream`, the handler
    returns a StreamingResponse with SSE-framed events. Anything else --
    including a missing Accept header, `application/json`, `*/*`, or a
    weighted `text/event-stream;q=0.9` -- silently degrades to the v0.10
    JSON path so legacy callers keep byte-identical responses (§2).
    Pre-stream validation (no user message) still returns HTTP 400 + JSON
    even when SSE was requested -- the SSE channel is not opened (§5).
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

    if _wants_sse(request.headers.get("accept")):
        return StreamingResponse(
            _sse_event_stream(user_msgs[-1]["content"], turn_id, session_id, request),
            media_type="text/event-stream; charset=utf-8",
            headers=_SSE_HEADERS,
        )

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
