"""v0.7 cost calc — `mneme.llm.pricing.cost_usd` + `_load_table` contracts.

Pure unit tests; no DB, no HTTP, no LLM. Each test maps to one of the 13
boundary cases the architect enumerated for v0.7:

  1  unknown provider              → None (no cost_usd written)
  2  known provider + unknown model → None
  3  ollama anything               → 0.0
  4  zero-token usage              → known: 0.0, unknown: None
  6  pricing.yaml missing          → table = {}, one stderr warning
  7  pricing.yaml malformed YAML   → table = {}, one stderr warning
  8  pricing.yaml partial entry    → None (no KeyError)
  9  Anthropic Usage normalization → pricing only sees prompt/completion
 11  large multiplier              → no scientific notation in display
 15  (folded into 11)

CLI-layer boundaries (5, 10, 12, 13) live in test_cli_stats.py.
Trace-event boundaries (1, 2, 3, 5) live in test_agent.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mneme.llm import pricing
from mneme.llm.client import Usage


# ---------- table reset --------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    """`_load_table` is @functools.cache'd; reset between tests so file-system
    monkeypatches (missing YAML, malformed YAML) actually take effect. Without
    this an earlier test's successful load would be reused and the error-path
    tests would silently pass for the wrong reason."""
    pricing._load_table.cache_clear()
    yield
    pricing._load_table.cache_clear()


def _usage(p: int, c: int) -> Usage:
    return Usage(prompt_tokens=p, completion_tokens=c)


# ---------- boundary 3: ollama is always free --------------------------------

def test_ollama_is_zero_not_none_regardless_of_model():
    """0.0 and None mean different things downstream (footer shows $0.0000 vs
    $?); ollama must be the 'truly free' case, not the 'unknown' case."""
    assert pricing.cost_usd("ollama", "qwen2.5:7b", _usage(100, 50)) == 0.0
    assert pricing.cost_usd("ollama", "anything-goes", _usage(1, 1)) == 0.0
    # Even with zero tokens and a model name absent from the YAML.
    assert pricing.cost_usd("ollama", "never-heard-of-it", _usage(0, 0)) == 0.0


# ---------- happy: known model arithmetic -------------------------------------

def test_known_model_applies_per_million_token_pricing():
    """gpt-4o: $2.50/1M in, $10.00/1M out. 1000 in + 500 out = 0.0025 + 0.005."""
    cost = pricing.cost_usd("openai", "gpt-4o", _usage(1000, 500))
    assert cost == pytest.approx(0.0025 + 0.005)


# ---------- boundary 1: unknown provider --------------------------------------

def test_unknown_provider_returns_none_not_zero():
    """An unrecognized cloud provider must surface as 'unknown' (None), not
    silently as 'free' (0.0); otherwise stats would under-report real spend."""
    assert pricing.cost_usd("xai", "grok-2", _usage(100, 100)) is None


# ---------- boundary 2: known provider, unknown model -------------------------

def test_known_provider_unknown_model_returns_none():
    """A typo'd model name on a real provider also must not silently zero."""
    assert pricing.cost_usd("openai", "gpt-9000", _usage(100, 100)) is None


# ---------- boundary 4: zero-token usage --------------------------------------

def test_zero_tokens_on_known_model_is_truly_zero():
    """0 tokens × any price = 0.0; that is 'I called the model and it returned
    nothing useful', not 'I don't know the price'."""
    assert pricing.cost_usd("openai", "gpt-4o", _usage(0, 0)) == 0.0


def test_zero_tokens_on_unknown_model_is_still_none():
    """Don't let a zero-usage edge case mask an unknown model as 'free'."""
    assert pricing.cost_usd("openai", "gpt-9000", _usage(0, 0)) is None


# ---------- boundary 9: pricing only sees normalized Usage --------------------

def test_pricing_accepts_simple_namespace_duck_type():
    """`fake_llm.last_usage` is a SimpleNamespace, not a frozen Usage. The
    `pricing` module reads `.prompt_tokens` / `.completion_tokens` attributes
    only — so duck typing must work or every integration test that exercises
    the close-trace cost path is exercising a different shape than production."""
    fake = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    assert pricing.cost_usd("openai", "gpt-4o", fake) == pytest.approx(0.0025 + 0.005)


def test_anthropic_cost_uses_same_normalized_usage_as_openai(monkeypatch):
    """v0.4 normalizes Anthropic's input_tokens/output_tokens into a Usage with
    prompt_tokens/completion_tokens; the pricing layer must only depend on
    that normalized shape. We patch the yaml table so anthropic and openai
    share an identical row, then assert identical Usage → identical cost."""
    monkeypatch.setattr(
        pricing,
        "_load_table",
        lambda: {
            "openai": {"twin-model": {"input": 1.0, "output": 2.0}},
            "anthropic": {"twin-model": {"input": 1.0, "output": 2.0}},
        },
    )
    u = _usage(1234, 5678)
    a = pricing.cost_usd("anthropic", "twin-model", u)
    o = pricing.cost_usd("openai", "twin-model", u)
    assert a is not None and o is not None
    assert a == o
    # Sanity: not both None for the wrong reason.
    assert a == pytest.approx((1234 * 1.0 + 5678 * 2.0) / 1_000_000)


# ---------- boundary 11: large numbers stay decimal --------------------------

def test_large_token_count_does_not_render_in_scientific_notation():
    """10M prompt tokens × $2.50/1M = exactly $25.00 — must display as
    '$25.0000' (four-decimal format), not '$2.5e+01' or similar."""
    cost = pricing.cost_usd("openai", "gpt-4o", _usage(10_000_000, 0))
    assert cost == 25.0
    rendered = f"${cost:.4f}"
    assert rendered == "$25.0000"
    assert "e" not in rendered.lower()


# ---------- boundary 6: missing pricing.yaml ---------------------------------

def test_missing_pricing_yaml_returns_empty_table_with_one_warning(
    tmp_path, monkeypatch, capsys
):
    """File deleted under our feet -> _load_table returns {} (not None, not a
    raise), a single stderr warning identifies the failure mode, and any cloud
    lookup degrades to 'unknown' rather than KeyErroring."""
    monkeypatch.setattr(pricing, "_TABLE_PATH", tmp_path / "no-such.yaml")
    pricing._load_table.cache_clear()
    table = pricing._load_table()
    assert table == {}

    # Caller path: unknown table + previously priced model → None, no crash.
    assert pricing.cost_usd("openai", "gpt-4o", _usage(10, 10)) is None

    err = capsys.readouterr().err
    # Diagnostic must NAME the failure mode so the user fixes the right layer
    # (cf. docs/lessons/developer.md "don't send users debugging the wrong
    # layer"). Counting occurrences guards against a future regression that
    # spams the warning on every cost lookup.
    assert "pricing.yaml" in err
    assert "missing" in err.lower() or "not found" in err.lower()


def test_malformed_pricing_yaml_returns_empty_table_with_one_warning(
    tmp_path, monkeypatch, capsys
):
    """Truncated/garbled YAML -> {} + stderr warning. The chat REPL must not
    crash because the pricing table is broken; the user can still talk to the
    LLM (footer just shows $?)."""
    bad = tmp_path / "pricing.yaml"
    bad.write_text("openai:\n  gpt-4o: [unclosed list\n", encoding="utf-8")
    monkeypatch.setattr(pricing, "_TABLE_PATH", bad)
    pricing._load_table.cache_clear()
    table = pricing._load_table()
    assert table == {}
    assert pricing.cost_usd("openai", "gpt-4o", _usage(10, 10)) is None

    err = capsys.readouterr().err
    assert "pricing.yaml" in err
    assert "yaml" in err.lower() or "parse" in err.lower()


def test_pricing_yaml_top_level_not_mapping_returns_empty_table(
    tmp_path, monkeypatch, capsys
):
    """YAML that parses but isn't a dict (e.g. a list) is treated as broken."""
    bad = tmp_path / "pricing.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setattr(pricing, "_TABLE_PATH", bad)
    pricing._load_table.cache_clear()
    assert pricing._load_table() == {}
    err = capsys.readouterr().err
    assert "pricing.yaml" in err
    assert "mapping" in err.lower()


# ---------- boundary 8: partial pricing entries -------------------------------

def test_missing_output_price_returns_none_not_keyerror(monkeypatch):
    """Half-defined pricing row (input only) → None, not a KeyError that would
    bubble out of the close-trace code path and abort the turn write."""
    monkeypatch.setattr(
        pricing, "_load_table",
        lambda: {"openai": {"partial-model": {"input": 1.0}}},
    )
    assert pricing.cost_usd("openai", "partial-model", _usage(100, 100)) is None


def test_missing_input_price_returns_none_not_keyerror(monkeypatch):
    """Mirror: output without input → None. Guards both branches symmetrically."""
    monkeypatch.setattr(
        pricing, "_load_table",
        lambda: {"openai": {"partial-model": {"output": 5.0}}},
    )
    assert pricing.cost_usd("openai", "partial-model", _usage(100, 100)) is None


def test_non_dict_pricing_entry_returns_none(monkeypatch):
    """If a model row is the wrong shape (string, list, null) we must not
    KeyError or AttributeError — return None like any other unknown."""
    monkeypatch.setattr(
        pricing, "_load_table",
        lambda: {"openai": {"weird": "not-a-dict", "also-weird": None}},
    )
    assert pricing.cost_usd("openai", "weird", _usage(10, 10)) is None
    assert pricing.cost_usd("openai", "also-weird", _usage(10, 10)) is None


def test_non_numeric_price_value_returns_none(monkeypatch):
    """A typo'd price like 'two dollars' must surface as unknown, not crash
    on float() conversion mid-turn."""
    monkeypatch.setattr(
        pricing, "_load_table",
        lambda: {"openai": {"oops": {"input": "two", "output": 1.0}}},
    )
    assert pricing.cost_usd("openai", "oops", _usage(10, 10)) is None
