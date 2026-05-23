"""Chat loop: ingest -> retrieve -> build prompt -> LLM call -> persist.

The five stages mirror the event flow in README.md and docs/MEMORY.md.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from . import soul
from .ids import ulid
from .llm import client as _llm
from .memory import ingest, retrieve, store
from .trace import events

# Permissive on purpose: catches fabricated tags too, not just well-formed ULIDs.
_CITATION_RE = re.compile(r"\[\^([^\]]+)\]")


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


def respond(user_text: str, turn_id: str, cx: sqlite3.Connection) -> Reply:
    """Run one full chat turn."""
    ingest.save_user_message(user_text, turn_id, cx)
    slices = retrieve.recall(user_text, cx)
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

    reply_text = client.chat(
        [{"role": "system", "content": prefix}, {"role": "user", "content": suffix}],
        stream=False,
    )
    citation_quality = _classify_citations(reply_text, [s.id for s in slices])

    ingest.save_assistant_message(reply_text, turn_id, cx)
    if ep is not None:
        events.append(ep, kind="trace", id=trace_id,
                      response_hash=soul.prompt_hash(reply_text),
                      citation_quality=citation_quality)
    return Reply(reply_text, trace_id, citation_quality)
