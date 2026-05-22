"""Chat loop: ingest -> retrieve -> build prompt -> LLM call -> persist.

This is the orchestration layer. The five numbered stages mirror the event
flow in README.md and docs/MEMORY.md.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import soul
from .ids import ulid
from .llm import client as _llm
from .memory import ingest, retrieve, store
from .trace import events


@dataclass(frozen=True)
class Reply:
    text: str
    trace_id: str
    citation_quality: str          # "explicit" | "coarse"


def build_prompt(user_text: str, slices: list, blueprint: str) -> tuple[str, str]:
    """Return (stable_prefix, dynamic_suffix).

    The prefix is byte-identical across turns (blueprint + chat system block)
    so it hits the prompt cache; retrieved slices live only in the suffix
    (PRINCIPLES.md principle 2).
    """
    chat = soul.load_prompt("chat")
    recall_tpl = soul.load_prompt("recall")

    stable_prefix = blueprint.strip() + "\n\n" + chat["system"].strip()

    header = chat["retrieved_context_header"].strip()
    if slices:
        lines = [
            recall_tpl["slice_line"]
            .format(id=s.id, role=s.role, created=s.created_at, text=s.text)
            .strip()
            for s in slices
        ]
        context = header + "\n" + "\n".join(lines)
    else:
        context = header + "\n" + recall_tpl["empty"].strip()

    dynamic_suffix = context + "\n\nUSER: " + user_text
    return stable_prefix, dynamic_suffix


def respond(user_text: str, turn_id: str, cx: sqlite3.Connection) -> Reply:
    """Run one full chat turn."""
    # ① ingest the user message (slice + vector + concept graph).
    ingest.save_user_message(user_text, turn_id, cx)

    # ② retrieve relevant memory.
    slices = retrieve.recall(user_text, cx)

    # ③ build the prompt: stable prefix + dynamic suffix.
    prefix, suffix = build_prompt(user_text, slices, soul.load_blueprint())
    messages = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": suffix},
    ]

    # ④ trace BEFORE the call (crash-safe), then call the LLM.
    trace_id = ulid()
    client = _llm.get_client()
    ep = store.events_path(cx)
    if ep is not None:
        events.append(
            ep,
            kind="trace",
            id=trace_id,
            query=user_text,
            used_slices=[s.id for s in slices],
            prompt_hash=soul.prompt_hash(prefix + "\n" + suffix),
            model=client.config.chat_model,
            provider=client.config.provider,
        )

    reply_text = client.chat(messages, stream=False)
    citation_quality = "explicit" if "[^" in reply_text else "coarse"

    # ⑤ persist the assistant reply; close out the trace.
    ingest.save_assistant_message(reply_text, turn_id, cx)
    if ep is not None:
        events.append(
            ep,
            kind="trace",
            id=trace_id,
            response_hash=soul.prompt_hash(reply_text),
            citation_quality=citation_quality,
        )

    return Reply(reply_text, trace_id, citation_quality)
