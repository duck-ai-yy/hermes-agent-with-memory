"""Chat loop: ingest -> retrieve -> build prompt -> LLM (+ tool*) -> persist.

v0.8 / M1: the single "LLM call" stage is now a loop. The model can request
a tool call; the agent runs it (after user confirmation), feeds the result
back, and asks the model again — repeating up to `max_iters` times. When the
model finally stops requesting tools we close the turn.

The five-stage flow (ingest / retrieve / build / call / persist) is
unchanged from v0.7. Only the "call" stage is now multi-step. Memory,
citation classification, trace shape, and cost accounting are preserved
byte-for-byte; cost simply accumulates across all LLM calls in the turn.

`respond` returns the full reply; `respond_stream` yields chunks and
returns the same Reply via StopIteration.value once exhausted.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterator

from . import soul
from .ids import ulid
from .llm import client as _llm
from .llm import pricing
from .llm.client import AssistantMessage, ToolCall, Usage
from .memory import ingest, retrieve, store
from .tools import shell as shell_tool
from .trace import events

# Permissive on purpose: catches fabricated tags too, not just well-formed ULIDs.
_CITATION_RE = re.compile(r"\[\^([^\]]+)\]")

# Hard ceiling on LLM <-> tool round-trips per turn. Stops a confused model
# from looping forever. Configurable via env for power users / tests.
_DEFAULT_MAX_ITERS = 6


def _max_iters() -> int:
    """Resolve max iterations at call time (so tests can `monkeypatch.setenv`)."""
    try:
        return max(1, int(os.environ.get("MNEME_MAX_ITERS", _DEFAULT_MAX_ITERS)))
    except ValueError:
        return _DEFAULT_MAX_ITERS


# Callback type. Returns True to allow the tool call, False to reject.
ConfirmCb = Callable[[str, dict], bool]


def _classify_citations(reply_text: str, slice_ids: list[str]) -> str:
    """coarse | explicit | fabricated — see PRINCIPLES.md principle 5."""
    markers = _CITATION_RE.findall(reply_text)
    if not markers:
        return "coarse"
    known = set(slice_ids)
    return "explicit" if all(m in known for m in markers) else "fabricated"


@dataclass(frozen=True)
class Reply:
    text: str
    trace_id: str
    citation_quality: str          # "explicit" | "coarse" | "fabricated"


def build_prompt(user_text: str, slices: list, blueprint: str) -> tuple[str, str]:
    """Return (stable_prefix, dynamic_suffix).

    The prefix is byte-identical across turns so it hits the prompt cache;
    retrieved slices live only in the suffix (PRINCIPLES.md principle 2).
    """
    chat = soul.load_prompt("chat")
    recall_tpl = soul.load_prompt("recall")

    stable_prefix = blueprint.strip() + "\n\n" + chat["system"].strip()
    header = chat["retrieved_context_header"].strip()
    if slices:
        body = "\n".join(
            recall_tpl["slice_line"]
            .format(id=s.id, role=s.role, created=s.created_at, text=s.text)
            .strip()
            for s in slices
        )
    else:
        body = recall_tpl["empty"].strip()
    return stable_prefix, f"{header}\n{body}\n\nUSER: {user_text}"


# -- Tool schema declarations the loop sends to the LLM --------------------

def _tool_schemas_for_provider(provider: str) -> list[dict]:
    """Wrap shell_tool.SCHEMA in the request shape each provider expects.

    Anthropic takes the schema flat under `tools=[]`; OpenAI/Ollama nest it
    under `{type:"function", function:{...}}`. See
    docs/knowledge/provider-tool-calling.md §1.
    """
    s = shell_tool.SCHEMA
    if provider == "anthropic":
        return [{
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["input_schema"],
        }]
    # OpenAI / Ollama function shape
    return [{
        "type": "function",
        "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        },
    }]


# -- Turn open / close (unchanged structurally from v0.7) ------------------

def _open_turn(user_text: str, turn_id: str, cx: sqlite3.Connection):
    """Stages 1-3: ingest, retrieve, build prompt; write the pre-call trace."""
    user_sid = ingest.save_user_message(user_text, turn_id, cx)
    # Exclude the just-ingested user slice from its own retrieval; otherwise
    # vector search returns it first and the LLM thinks the user is repeating.
    slices = retrieve.recall(user_text, cx, exclude={user_sid})
    prefix, suffix = build_prompt(user_text, slices, soul.load_blueprint())

    trace_id = ulid()
    client = _llm.get_client()
    ep = store.events_path(cx)

    # Trace BEFORE the call so a crash mid-call still leaves a record.
    if ep is not None:
        events.append(ep, kind="trace", id=trace_id, query=user_text,
                      used_slices=[s.id for s in slices],
                      prompt_hash=soul.prompt_hash(prefix + "\n" + suffix),
                      model=client.config.chat_model, provider=client.config.provider)
    return slices, prefix, suffix, trace_id, client, ep


def _close_turn(
    reply_text: str,
    slices: list,
    trace_id: str,
    client,
    ep,
    turn_id: str,
    cx,
    *,
    accumulated_usage: Usage | None,
    accumulated_cost: float,
    had_unknown_price: bool,
    iters: int,
    tool_calls_count: int,
    tool_rejects_count: int,
) -> Reply:
    """Stages 4-5: classify, persist assistant message, close the trace.

    Accumulated usage/cost across all LLM calls in the turn are passed in
    explicitly — the loop owns the running totals because a single
    `client.last_usage` only reflects the most recent call.

    Cost field rules (carried forward from v0.7, must not regress):
      - Ollama with known pricing -> cost_usd=0.0
      - cloud + known model       -> cost_usd=<sum>
      - any call hit "unknown"    -> OMIT cost_usd entirely (boundary 1/2)
      - usage never observed      -> OMIT every token + cost field (boundary 5)
    """
    citation_quality = _classify_citations(reply_text, [s.id for s in slices])
    ingest.save_assistant_message(reply_text, turn_id, cx)
    if ep is not None:
        extra: dict = {
            "iters": iters,
            "tool_calls": tool_calls_count,
            "tool_rejects": tool_rejects_count,
        }
        if accumulated_usage is not None:
            extra["prompt_tokens"] = accumulated_usage.prompt_tokens
            extra["completion_tokens"] = accumulated_usage.completion_tokens
            extra["total_tokens"] = accumulated_usage.total_tokens
            # Omit cost_usd iff ANY call in the turn was an unknown price.
            # Mirrors v0.7 boundary 1/2 — historical sums must not be
            # polluted with a partial cost for a turn whose total is unknown.
            if not had_unknown_price:
                extra["cost_usd"] = accumulated_cost
        events.append(ep, kind="trace", id=trace_id,
                      response_hash=soul.prompt_hash(reply_text),
                      citation_quality=citation_quality,
                      provider=client.config.provider, **extra)
    return Reply(reply_text, trace_id, citation_quality)


# -- The agent loop --------------------------------------------------------

def _accumulate(
    running: Usage | None, latest: Usage | None,
) -> Usage | None:
    """Sum Usage objects across iterations. None means 'never observed yet';
    once any iteration reports usage we keep summing real numbers."""
    if latest is None:
        return running
    if running is None:
        return Usage(latest.prompt_tokens, latest.completion_tokens)
    return Usage(
        running.prompt_tokens + latest.prompt_tokens,
        running.completion_tokens + latest.completion_tokens,
    )


def _price_call(client, usage: Usage) -> tuple[float, bool]:
    """Return (cost_or_zero, had_unknown). Pricing failures degrade to
    (0.0, True) — same defensive posture as v0.7's _close_turn."""
    try:
        cost = pricing.cost_usd(client.config.provider, client.config.chat_model, usage)
    except Exception:
        cost = None
    if cost is None:
        return 0.0, True
    return cost, False


def _run_tool_call(
    tc: ToolCall, ep, trace_id: str, confirm_cb: ConfirmCb,
) -> tuple[dict, bool]:
    """Run one tool call with audit + confirm. Return (tool_result_block, ran).

    `tool_result_block` is the dict to feed back to the LLM (provider-agnostic
    representation; the message-builder below wraps it per provider). `ran`
    is True iff the tool actually executed (False on rejection).
    """
    args = tc.arguments or {}
    # Audit the *pending* decision so a crash between confirm and execute
    # still leaves a breadcrumb (principle 5).
    if ep is not None:
        events.append(ep, kind="tool_audit", trace_id=trace_id,
                      tool=tc.name, decision="pending",
                      tool_call_id=tc.id, arguments=args)
    try:
        approved = bool(confirm_cb(tc.name, args))
    except Exception as exc:
        # confirm callback itself failed — log specifically, treat as reject.
        if ep is not None:
            events.append(ep, kind="tool_audit", trace_id=trace_id,
                          tool=tc.name, decision="rejected",
                          tool_call_id=tc.id,
                          reason=f"confirm_cb {type(exc).__name__}: {exc}")
        return _reject_block(tc, "user rejected (confirm callback failed)"), False

    if not approved:
        if ep is not None:
            events.append(ep, kind="tool_audit", trace_id=trace_id,
                          tool=tc.name, decision="rejected",
                          tool_call_id=tc.id)
        return _reject_block(tc, "Tool call rejected by user."), False

    if ep is not None:
        events.append(ep, kind="tool_audit", trace_id=trace_id,
                      tool=tc.name, decision="accepted",
                      tool_call_id=tc.id)

    # M1 has exactly one tool. v0.9 will look this up in a registry.
    if tc.name != "shell":
        # Surface the error type specifically (lessons/developer.md v0.6).
        content = f"UnknownTool: no tool named {tc.name!r} is registered"
        if ep is not None:
            events.append(ep, kind="tool_result", trace_id=trace_id,
                          tool=tc.name, tool_call_id=tc.id,
                          error="UnknownTool", exit_code=None)
        return {
            "tool_call_id": tc.id, "tool_name": tc.name,
            "content": content, "is_error": True,
        }, True

    command = args.get("command")
    if not isinstance(command, str):
        msg = f"ArgumentError: shell.command must be a string, got {type(command).__name__}"
        if ep is not None:
            events.append(ep, kind="tool_result", trace_id=trace_id,
                          tool=tc.name, tool_call_id=tc.id,
                          error="ArgumentError", exit_code=None)
        return {
            "tool_call_id": tc.id, "tool_name": tc.name,
            "content": msg, "is_error": True,
        }, True

    try:
        result = shell_tool.execute(command)
    except Exception as exc:
        # shell.execute is "never raises" by contract, but defend in depth:
        # any leak is reported with its exact error type, not "unknown error".
        msg = f"ShellError: {type(exc).__name__}: {exc}"
        if ep is not None:
            events.append(ep, kind="tool_result", trace_id=trace_id,
                          tool=tc.name, tool_call_id=tc.id,
                          error=type(exc).__name__, exit_code=None)
        return {
            "tool_call_id": tc.id, "tool_name": tc.name,
            "content": msg, "is_error": True,
        }, True

    content = shell_tool.format_for_llm(result)
    if ep is not None:
        events.append(ep, kind="tool_result", trace_id=trace_id,
                      tool=tc.name, tool_call_id=tc.id,
                      exit_code=result.exit_code,
                      stdout_bytes=result.stdout_bytes,
                      stderr_bytes=result.stderr_bytes,
                      truncated=result.truncated,
                      duration_ms=result.duration_ms)
    return {
        "tool_call_id": tc.id, "tool_name": tc.name,
        "content": content,
        "is_error": result.exit_code != 0,
    }, True


def _reject_block(tc: ToolCall, message: str) -> dict:
    return {
        "tool_call_id": tc.id, "tool_name": tc.name,
        "content": message, "is_error": True,
    }


def _append_assistant_and_tool_result(
    messages: list[dict], provider: str, asst: AssistantMessage, tool_results: list[dict],
) -> None:
    """Round-trip the model's tool_use response and our tool_result(s) into
    the running messages list. Provider shapes differ — see
    docs/knowledge/provider-tool-calling.md §1."""
    if provider == "anthropic":
        # Assistant message: replay the original content blocks. Order matters
        # — Anthropic rejects tool_result that references a tool_use_id not
        # present in the *immediately preceding* assistant turn.
        asst_content: list[dict] = []
        if asst.text:
            asst_content.append({"type": "text", "text": asst.text})
        for tc in asst.tool_calls:
            asst_content.append({
                "type": "tool_use", "id": tc.id, "name": tc.name,
                "input": tc.arguments,
            })
        messages.append({"role": "assistant", "content": asst_content})
        # Tool results go back as one user message containing all result blocks.
        result_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_call_id"],
                "content": r["content"],
                "is_error": r["is_error"],
            }
            for r in tool_results
        ]
        messages.append({"role": "user", "content": result_blocks})
        return

    if provider == "ollama":
        # Ollama: assistant message replays tool_calls (arguments as dict).
        messages.append({
            "role": "assistant",
            "content": asst.text or "",
            "tool_calls": [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in asst.tool_calls
            ],
        })
        # No tool_call_id on the wire — pair by tool_name.
        for r in tool_results:
            messages.append({
                "role": "tool", "content": r["content"], "tool_name": r["tool_name"],
            })
        return

    # OpenAI-compatible: assistant message echoes tool_calls (arguments as
    # JSON string), tool results use role=tool with tool_call_id.
    import json as _json
    messages.append({
        "role": "assistant",
        "content": asst.text or "",
        "tool_calls": [
            {
                "id": tc.id, "type": "function",
                "function": {"name": tc.name, "arguments": _json.dumps(tc.arguments)},
            }
            for tc in asst.tool_calls
        ],
    })
    for r in tool_results:
        messages.append({
            "role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"],
        })


def _agent_loop(
    prefix: str,
    suffix: str,
    client,
    ep,
    trace_id: str,
    confirm_cb: ConfirmCb | None,
    *,
    on_intermediate_text: Callable[[str], None] | None = None,
):
    """Drive the LLM <-> tool conversation. Return (reply_text, totals dict).

    totals = {usage, cost, had_unknown_price, iters, tool_calls, tool_rejects}.
    """
    messages: list[dict] = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": suffix},
    ]
    provider = client.config.provider
    # If no confirm callback (e.g. HTTP /chat in M1), degrade to no tools —
    # equivalent to the v0.7 single-call path, no surprises.
    use_tools = confirm_cb is not None
    tools = _tool_schemas_for_provider(provider) if use_tools else None

    running_usage: Usage | None = None
    running_cost = 0.0
    had_unknown = False
    iters = 0
    tool_calls_count = 0
    tool_rejects_count = 0
    reply_text = ""
    max_iters = _max_iters()

    while True:
        iters += 1
        if not use_tools:
            # v0.7 path: single non-stream call; agent loop is a no-op here.
            reply_text = client.chat(messages, stream=False)
            running_usage = _accumulate(running_usage, getattr(client, "last_usage", None))
            if getattr(client, "last_usage", None) is not None:
                add, unk = _price_call(client, client.last_usage)
                running_cost += add
                had_unknown = had_unknown or unk
            break

        asst: AssistantMessage = client.chat(messages, stream=False, tools=tools)
        running_usage = _accumulate(running_usage, getattr(client, "last_usage", None))
        if getattr(client, "last_usage", None) is not None:
            add, unk = _price_call(client, client.last_usage)
            running_cost += add
            had_unknown = had_unknown or unk

        # No tool calls -> this is the final assistant text. Done.
        if not asst.tool_calls:
            reply_text = asst.text
            break

        # Surface intermediate assistant text to the UI before we run the tool
        # so the user sees "I'll do X" before being asked to confirm.
        if asst.text and on_intermediate_text is not None:
            on_intermediate_text(asst.text)

        # Run each tool call (M1 typically issues one at a time, but the loop
        # is defensive: it handles a list).
        tool_results: list[dict] = []
        any_rejected = False
        for tc in asst.tool_calls:
            tool_calls_count += 1
            result_block, ran = _run_tool_call(tc, ep, trace_id, confirm_cb)
            tool_results.append(result_block)
            if not ran:
                any_rejected = True
                tool_rejects_count += 1

        if any_rejected:
            # Architect spec: rejection aborts the turn cleanly with a stock
            # message — don't feed the rejection back into the LLM, just end.
            reply_text = "Tool call rejected by user; turn aborted."
            break

        _append_assistant_and_tool_result(messages, provider, asst, tool_results)

        if iters >= max_iters:
            # Surface the iter cap explicitly. The LLM didn't get to finish,
            # but we're not going to loop forever — better honesty than hang.
            reply_text = (
                f"Agent loop hit the max iteration cap ({max_iters}). "
                f"Last assistant text: {asst.text!r}"
            )
            break

    return reply_text, {
        "usage": running_usage,
        "cost": running_cost,
        "had_unknown_price": had_unknown,
        "iters": iters,
        "tool_calls": tool_calls_count,
        "tool_rejects": tool_rejects_count,
    }


def respond(
    user_text: str,
    turn_id: str,
    cx: sqlite3.Connection,
    *,
    confirm_cb: ConfirmCb | None = None,
    on_intermediate_text: Callable[[str], None] | None = None,
) -> Reply:
    """Run one full chat turn (non-streaming).

    `confirm_cb` enables tool use: when not None, the LLM may request tool
    calls; each call goes through the callback for y/N before executing.
    When None, behavior is byte-identical to v0.7's single-call path.
    """
    slices, prefix, suffix, trace_id, client, ep = _open_turn(user_text, turn_id, cx)
    reply_text, totals = _agent_loop(
        prefix, suffix, client, ep, trace_id, confirm_cb,
        on_intermediate_text=on_intermediate_text,
    )
    return _close_turn(
        reply_text, slices, trace_id, client, ep, turn_id, cx,
        accumulated_usage=totals["usage"],
        accumulated_cost=totals["cost"],
        had_unknown_price=totals["had_unknown_price"],
        iters=totals["iters"],
        tool_calls_count=totals["tool_calls"],
        tool_rejects_count=totals["tool_rejects"],
    )


def respond_stream(
    user_text: str,
    turn_id: str,
    cx: sqlite3.Connection,
    *,
    confirm_cb: ConfirmCb | None = None,
    on_intermediate_text: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Streaming variant: yields text chunks; returns the Reply via
    StopIteration.value once the iterator is exhausted.

    When `confirm_cb` is None (the v0.7 contract), the final LLM call streams
    normally — preserving the chunk-yielding behavior 100%. When `confirm_cb`
    is set, the loop drives non-stream calls until the model stops requesting
    tools, then yields the final assistant text as a single chunk. M1 keeps
    the streaming-of-tool-args spike out of scope — see
    docs/knowledge/provider-tool-calling.md §3.
    """
    slices, prefix, suffix, trace_id, client, ep = _open_turn(user_text, turn_id, cx)

    if confirm_cb is None:
        # v0.7 streaming path, preserved verbatim except for the close_turn
        # signature change. No agent loop overhead.
        chunks: list[str] = []
        for chunk in client.chat(
            [{"role": "system", "content": prefix},
             {"role": "user", "content": suffix}],
            stream=True,
        ):
            chunks.append(chunk)
            yield chunk
        # Use the (single) last_usage from the streaming call as the totals.
        usage = getattr(client, "last_usage", None)
        cost, had_unknown = (0.0, False)
        if usage is not None:
            cost, had_unknown = _price_call(client, usage)
        return _close_turn(
            "".join(chunks), slices, trace_id, client, ep, turn_id, cx,
            accumulated_usage=usage, accumulated_cost=cost,
            had_unknown_price=had_unknown, iters=1,
            tool_calls_count=0, tool_rejects_count=0,
        )

    # Tool-enabled streaming: run the loop non-stream, yield final text once.
    reply_text, totals = _agent_loop(
        prefix, suffix, client, ep, trace_id, confirm_cb,
        on_intermediate_text=on_intermediate_text,
    )
    if reply_text:
        yield reply_text
    return _close_turn(
        reply_text, slices, trace_id, client, ep, turn_id, cx,
        accumulated_usage=totals["usage"],
        accumulated_cost=totals["cost"],
        had_unknown_price=totals["had_unknown_price"],
        iters=totals["iters"],
        tool_calls_count=totals["tool_calls"],
        tool_rejects_count=totals["tool_rejects"],
    )
