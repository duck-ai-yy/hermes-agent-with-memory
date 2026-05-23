"""Single append-only event log: ~/.mneme/events.jsonl.

One file, one `kind` field — traces, ingests, forgets and audit events all go
here (PRINCIPLES.md principle 1: one log, not several). A trace event is
written BEFORE each LLM call so a crash mid-call still leaves a record
(principle 5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# kind ∈ {"trace", "ingest", "forget", "snapshot", "audit"}


def append(log_path: Path, *, kind: str, **fields: object) -> None:
    """Append one JSON line to events.jsonl. Never raises into the caller."""
    try:
        record = {"ts": int(time.time()), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        # Logging must never break the main path (principle 5: best-effort).
        pass


def explain(log_path: Path, trace_id: str) -> dict:
    """Return the merged trace record for `trace_id`.

    A turn writes two `trace` lines with the same id: one before the LLM call
    (query, used_slices, prompt_hash, model) and one after (response_hash,
    citation_quality). They are merged here, later fields winning.
    """
    merged: dict = {}
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "trace" and record.get("id") == trace_id:
                merged.update(record)
    if not merged:
        raise KeyError(trace_id)
    return merged
