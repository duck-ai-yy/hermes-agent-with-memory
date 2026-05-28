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
