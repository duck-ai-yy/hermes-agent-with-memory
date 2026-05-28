"""Token-budget-aware prompt assembly (v0.11 / M3).

Three concerns live here, on purpose:

  * Unit A — `estimate_tokens`: pure str -> int heuristic.
  * Unit B — `context_window_for` + `budget_for_retrieval`: look up the
    `(provider, model) -> context_window` from `pricing.yaml`, subtract the
    completion + tools-schema reserves to get the slice budget.
  * Unit C — `assemble_prompt` + `AssemblyMeta` + `PromptTooBig`: given a
    prefix, retrieved slices, the user text and a budget, drop low-score
    slices from the tail until the prompt fits — or raise if even the empty
    suffix overflows.

The heuristic divisor (4 chars/token) matches the `payload_chars // 4`
estimate already used in `mneme/llm/client.py` so the project speaks one
token-estimation language (PRINCIPLE 1). No tokenizer dependency is taken.

`pricing.yaml` carries both the price table (read by `pricing.py`) and the
new `context_windows:` section read here — same update cadence (vendors ship
prices and window sizes together), so co-locating cuts the maintenance tax.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import pricing

if TYPE_CHECKING:  # pragma: no cover — types only
    from ..memory.retrieve import Slice

# -- Module constants (pinned by design §3 / §4 / §8) ---------------------

_FALLBACK_WINDOW = 8192
_COMPLETION_RESERVE_FLOOR = 1024
# Integer divisor: window // 4 == 25% — matches the design's default ratio.
_COMPLETION_RESERVE_RATIO_DEFAULT_INV = 4
_TOOLS_SCHEMA_RESERVE = 1024


# -- Unit A: estimate_tokens ----------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic character -> token count.

    Formula `len(text) // 4 + 1`:
      - `// 4` matches the `payload_chars // 4` estimate in `client.py:161`
        so one number means one thing project-wide (PRINCIPLE 1).
      - `+ 1` lifts the empty string to 1 so budget arithmetic stays away
        from 0 (an "empty suffix" still consumes a token slot in practice).

    No model-specific branch (e.g. EN vs CN): the assembly strategy drops
    whole slices, not bytes, so a ~15% estimation error only shifts the
    drop count by 0-1. v0.12 can swap in a real tokenizer behind this same
    signature if a real miss ever shows up.
    """
    return len(text) // 4 + 1


# -- Unit B (upper half): context_window_for ------------------------------

# (provider, model) tuples we have already warned about — module-level so
# the warning fires once per process per unknown pair (PRINCIPLE 1: same
# shape as pricing.py's one-shot stderr).
_warned: set[tuple[str, str]] = set()
# A sentinel so the "missing context_windows section" warning also fires
# at most once, regardless of how many lookups hit it.
_warned_missing_section = False


def context_window_for(provider: str, model: str) -> int:
    """Look up the context window (in tokens) for `(provider, model)`.

    - Ollama is always returned as the fallback constant: a user's modelfile
      `num_ctx` is not introspectable from here, so we stay conservative
      (design §3) — silent, no warning.
    - A hit in `pricing.yaml`'s `context_windows:` section returns that int.
    - Anything else (unknown provider, unknown model, or a malformed
      `context_windows:` section) returns `_FALLBACK_WINDOW` and emits a
      one-shot stderr warning naming the actual layer that failed (the
      `pricing.py:_load_table` lesson, applied to this loader too).
    """
    global _warned_missing_section

    if provider == "ollama":
        # Conservative default; intentional and silent (design §3).
        return _FALLBACK_WINDOW

    table = pricing._load_table()
    windows = table.get("context_windows")
    if not isinstance(windows, dict):
        if not _warned_missing_section:
            print(
                "mneme: pricing.yaml missing context_windows section, "
                "falling back to 8192 tokens",
                file=sys.stderr,
            )
            _warned_missing_section = True
        return _FALLBACK_WINDOW

    provider_windows = windows.get(provider)
    if isinstance(provider_windows, dict):
        value = provider_windows.get(model)
        if isinstance(value, int):
            return value

    key = (provider, model)
    if key not in _warned:
        print(
            f"mneme: unknown context window for {provider}/{model}, "
            "falling back to 8192 tokens",
            file=sys.stderr,
        )
        _warned.add(key)
    return _FALLBACK_WINDOW


# -- Unit B (lower half): budget_for_retrieval ----------------------------

# Per-bad-value warn-once cache for MNEME_CONTEXT_BUDGET_RATIO, same shape
# as _warned above (set[str], one entry per offending value).
_warned_ratio: set[str] = set()


def _reserve_ratio_inv() -> int:
    """Resolve the completion-reserve ratio's integer inverse at call time.

    Returns the "N" in `window // N` (so the default 0.25 means N == 4).
    Reads `MNEME_CONTEXT_BUDGET_RATIO` from the environment on each call —
    mirrors `agent._max_iters()`'s style so monkeypatching env vars in
    tests "just works" without process restart.

    Invalid values (non-numeric, <= 0.0, >= 1.0) silently fall back to the
    default with a one-shot stderr warning naming the bad value.
    """
    raw = os.environ.get("MNEME_CONTEXT_BUDGET_RATIO")
    if raw is None:
        return _COMPLETION_RESERVE_RATIO_DEFAULT_INV
    try:
        ratio = float(raw)
    except ValueError:
        if raw not in _warned_ratio:
            print(
                f"mneme: invalid MNEME_CONTEXT_BUDGET_RATIO={raw}, "
                "falling back to 0.25",
                file=sys.stderr,
            )
            _warned_ratio.add(raw)
        return _COMPLETION_RESERVE_RATIO_DEFAULT_INV
    if ratio <= 0.0 or ratio >= 1.0:
        if raw not in _warned_ratio:
            print(
                f"mneme: invalid MNEME_CONTEXT_BUDGET_RATIO={raw}, "
                "falling back to 0.25",
                file=sys.stderr,
            )
            _warned_ratio.add(raw)
        return _COMPLETION_RESERVE_RATIO_DEFAULT_INV
    # Convert ratio to integer divisor: `window // (1/ratio)` == `window * ratio`
    # in spirit, kept as integer division to match the design's `// 4` pin.
    return max(1, int(round(1.0 / ratio)))


def budget_for_retrieval(provider: str, model: str) -> int:
    """Tokens available for prefix + suffix after reserving completion + tools.

    `max(_COMPLETION_RESERVE_FLOOR, window // N)` keeps very small windows
    (like ollama's 8192 conservative default) from collapsing the reserve
    to a useless few tokens; the 1024 floor matches the tools-schema
    reserve so both halves of the "fixed overhead" have the same minimum.
    Returns `max(0, ...)` so a pathological config (window <= reserves)
    surfaces as budget=0 — which `assemble_prompt` then turns into the
    `PromptTooBig` raise for any non-empty prompt (design §4).
    """
    window = context_window_for(provider, model)
    inv = _reserve_ratio_inv()
    completion_reserve = max(_COMPLETION_RESERVE_FLOOR, window // inv)
    return max(0, window - completion_reserve - _TOOLS_SCHEMA_RESERVE)


# -- Unit C: assemble_prompt + AssemblyMeta + PromptTooBig ----------------

@dataclass(frozen=True)
class AssemblyMeta:
    """Forensic record of how `assemble_prompt` shaped the final prompt.

    `estimated_tokens` is the heuristic (`estimate_tokens`) total over the
    final prefix + suffix — NOT the provider's authoritative token count,
    which only arrives in the close-trace via usage. `dropped_slice_ids`
    preserves drop order (tail-first) so a reviewer can replay the
    decision; `kept_slice_ids` preserves the order the surviving slices
    appear in the suffix (== the input order with dropped IDs removed).
    """

    estimated_tokens: int
    kept_slice_ids: list[str]
    dropped_slice_ids: list[str]
    original_slice_count: int


class PromptTooBig(Exception):
    """Raised when even an empty-suffix prompt overflows the budget.

    Carries the forensic triple (`estimated`, `budget`, `dropped_count`)
    as instance attributes so a CLI / HTTP caller can recover the same
    numbers without going through events.jsonl — `except PromptTooBig as
    exc: exc.estimated` works directly (design §4 / lead D9).
    """

    def __init__(self, estimated: int, budget: int, dropped_count: int) -> None:
        self.estimated = estimated
        self.budget = budget
        self.dropped_count = dropped_count
        super().__init__(
            f"prompt exceeds budget after dropping all {dropped_count} "
            f"retrieved slices: estimated={estimated} budget={budget}"
        )


def assemble_prompt(
    prefix: str,
    slices: list["Slice"],
    user_text: str,
    budget: int,
) -> tuple[str, str, AssemblyMeta]:
    """Return `(prefix, suffix, meta)` after enforcing the token budget.

    Strategy: rebuild `suffix` via `agent.build_prompt(user_text, kept, ...)`
    so the suffix format stays single-sourced (recall.yaml's `slice_line`
    / `empty` templates own it — PRINCIPLE 1, no double formatting).
    Drop slices from the tail (`retrieve.recall` returns them score DESC,
    so the tail is the lowest-relevance entry) until `estimated_tokens
    (prefix) + estimated_tokens(suffix) <= budget`. If we run out of
    slices and the prompt still doesn't fit, raise `PromptTooBig`.

    `prefix` and `user_text` are never modified or truncated — the
    stable prefix preserves the LLM provider's prompt cache hit
    (PRINCIPLE 2) and the user's query is ground truth (PRINCIPLE 5,
    silent truncation would let the model answer the wrong question).
    """
    # Lazy-import to avoid the agent <-> llm import cycle: agent.py
    # already pulls in llm.client / llm.pricing at module import, so
    # llm.budget cannot pull in agent at module load.
    from .. import soul
    from ..agent import build_prompt

    blueprint = soul.load_blueprint()
    original_count = len(slices)
    kept = list(slices)

    # Initial build with all slices.
    _, suffix = build_prompt(user_text, kept, blueprint)
    estimated = estimate_tokens(prefix) + estimate_tokens(suffix)

    dropped_ids: list[str] = []
    # Drop from the tail one at a time, rebuilding suffix each round.
    while estimated > budget and kept:
        dropped_ids.append(kept[-1].id)
        kept = kept[:-1]
        _, suffix = build_prompt(user_text, kept, blueprint)
        estimated = estimate_tokens(prefix) + estimate_tokens(suffix)

    if estimated > budget:
        # Even with no retrieved slices the prefix + user_text overflows;
        # there is nothing more to drop without lying to the model.
        raise PromptTooBig(
            estimated=estimated, budget=budget, dropped_count=len(dropped_ids),
        )

    return prefix, suffix, AssemblyMeta(
        estimated_tokens=estimated,
        kept_slice_ids=[s.id for s in kept],
        dropped_slice_ids=dropped_ids,
        original_slice_count=original_count,
    )
