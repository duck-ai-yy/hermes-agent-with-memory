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

import sys

from . import pricing

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
