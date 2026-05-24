"""Cost calculation for chat completions.

The price table ships with the package as `pricing.yaml`. Ollama is local and
hard-coded to $0.0 regardless of model name. Anything else unknown returns
`None` — callers must distinguish `0.0` (truly free) from `None` (don't know).

Historical `cost_usd` written to events.jsonl is summed as-is by `stats`; we
never re-price old turns from this table, so a price change here only affects
new traces.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import yaml

from .client import Usage

_TABLE_PATH = Path(__file__).resolve().parent / "pricing.yaml"


@functools.cache
def _load_table() -> dict:
    """Parse pricing.yaml once per process.

    Returns {} (not None) on any failure so callers see "unknown model" rather
    than crashing. A one-shot stderr warning names the actual failure so the
    user isn't sent debugging the wrong layer (see docs/lessons/developer.md).
    """
    try:
        text = _TABLE_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except FileNotFoundError as exc:
        print(f"mneme: pricing.yaml unreadable: missing ({exc})", file=sys.stderr)
        return {}
    except OSError as exc:
        print(f"mneme: pricing.yaml unreadable: {exc}", file=sys.stderr)
        return {}
    except yaml.YAMLError as exc:
        print(f"mneme: pricing.yaml unreadable: yaml parse error: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(
            f"mneme: pricing.yaml unreadable: top-level not a mapping (got {type(data).__name__})",
            file=sys.stderr,
        )
        return {}
    return data


def cost_usd(provider: str, model: str, usage: Usage) -> float | None:
    """Compute USD cost for one chat call. None = unknown, 0.0 = free."""
    if provider == "ollama":
        return 0.0
    table = _load_table()
    prices = table.get(provider, {}).get(model)
    if not isinstance(prices, dict):
        return None
    try:
        in_price = float(prices["input"])
        out_price = float(prices["output"])
    except (KeyError, TypeError, ValueError):
        return None
    return (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1_000_000
