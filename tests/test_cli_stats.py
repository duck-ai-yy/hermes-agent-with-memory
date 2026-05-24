"""v0.7 cost display — CLI surface for `_cost_footer`, `_today_cost_breakdown`,
`mneme stats`, `mneme explain`.

Splits cleanly from test_search_cli.py: this file owns the cost/billing column
of the CLI — different feature, different surface area. The architect's 13
boundaries that involve user-visible bytes (5, 10, 12, 13) live here; the
pricing-arithmetic boundaries are in test_pricing.py.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mneme import cli, paths
from mneme.llm import pricing
from mneme.memory import store


# ---------- fixtures ---------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    pricing._load_table.cache_clear()
    yield
    pricing._load_table.cache_clear()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home_paths(tmp_path, monkeypatch):
    """Redirect ~/.mneme/{db,events} into tmp. Returns (db_path, events_path)."""
    db_path = tmp_path / "db.sqlite"
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(paths, "DB_PATH", db_path)
    monkeypatch.setattr(paths, "EVENTS_PATH", events_path)
    cx = store.connect(db_path)
    store.init_db(cx)
    cx.close()
    return db_path, events_path


def _client(provider="openai", chat_model="gpt-4o"):
    """Mimic the duck-typed shape `_cost_footer` reads off the LLM client."""
    return SimpleNamespace(config=SimpleNamespace(provider=provider, chat_model=chat_model))


def _usage(p=1000, c=500):
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def _write_events(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------- _cost_footer: format & boundary 12 -------------------------------

def test_cost_footer_unknown_model_renders_question_mark():
    """Boundary 1/2/12: unknown -> `· $?` so the user knows it's missing data,
    not zero spend. A space-dot-space prefix matches the tokens segment that
    precedes it."""
    out = cli._cost_footer(_client(provider="xai", chat_model="grok"), _usage())
    assert out == " · $?"


def test_cost_footer_ollama_renders_four_zero_decimals():
    """Boundary 3 + 12: ollama -> `· $0.0000`. NOT `$?` (we know it's free)
    and NOT `$0` (must keep the 4-decimal grid for column alignment)."""
    out = cli._cost_footer(_client(provider="ollama", chat_model="qwen2.5"), _usage())
    assert out == " · $0.0000"


def test_cost_footer_sub_cent_cost_renders_less_than_marker():
    """Boundary 12: 0 < cost < $0.0001 must render `· <$0.0001`, not
    `$0.0000` (which would look indistinguishable from ollama-free)."""
    # gpt-4o-mini: 0.15/1M in, 0.60/1M out. 100 in + 50 out = 0.000015 + 0.00003 = 4.5e-5
    out = cli._cost_footer(_client(chat_model="gpt-4o-mini"), _usage(p=100, c=50))
    assert out == " · <$0.0001"


def test_cost_footer_normal_cost_uses_four_decimals():
    """Boundary 12: priced normal turn -> `· $X.XXXX`. 1000 in + 500 out on
    gpt-4o = 2500/1e6 + 5000/1e6 = 0.0075 → '$0.0075'."""
    out = cli._cost_footer(_client(chat_model="gpt-4o"), _usage(p=1000, c=500))
    assert out == " · $0.0075"


def test_cost_footer_degrades_to_unknown_when_pricing_raises(monkeypatch):
    """Boundary 6/7 + 12: a broken pricing path must NEVER break the REPL
    footer; it must silently degrade to `· $?` (same visual signal as
    'unknown model'). We force a raise so we can verify the except branch."""
    def boom(*a, **kw):
        raise RuntimeError("synthetic pricing failure")
    monkeypatch.setattr(cli.pricing, "cost_usd", boom)
    out = cli._cost_footer(_client(), _usage())
    assert out == " · $?"


# ---------- _today_cost_breakdown: boundary 10 -------------------------------

def test_today_cost_breakdown_sums_priced_counts_unpriced_ignores_yesterday(
    tmp_path,
):
    """Boundary 10 in one shot: 3 close-traces today (2 priced, 1 unpriced)
    + 1 priced yesterday. Expect cost = today's two priced sums, unpriced=1,
    yesterday excluded entirely. Also seed a pre-call trace row (no
    total_tokens) for both today and yesterday — must be ignored."""
    ep = tmp_path / "events.jsonl"
    now = int(time.time())
    today_start = now - 60  # within today's window for the test
    yesterday = now - 86400 * 2

    _write_events(ep, [
        # Pre-call trace today: no total_tokens, must be ignored entirely.
        {"ts": now, "kind": "trace", "id": "PRE", "query": "x",
         "provider": "openai"},
        # Today, priced ($0.0075 + $0.0001).
        {"ts": now, "kind": "trace", "id": "A",
         "provider": "openai", "total_tokens": 1500,
         "prompt_tokens": 1000, "completion_tokens": 500, "cost_usd": 0.0075},
        {"ts": now, "kind": "trace", "id": "B",
         "provider": "openai", "total_tokens": 100,
         "prompt_tokens": 50, "completion_tokens": 50, "cost_usd": 0.0001},
        # Today, unpriced (unknown model — no cost_usd).
        {"ts": now, "kind": "trace", "id": "C",
         "provider": "xai", "total_tokens": 200,
         "prompt_tokens": 100, "completion_tokens": 100},
        # Yesterday: must not contribute to either total or counts.
        {"ts": yesterday, "kind": "trace", "id": "D",
         "provider": "openai", "total_tokens": 5000,
         "prompt_tokens": 4000, "completion_tokens": 1000, "cost_usd": 9.99},
        # Yesterday pre-call: also ignored.
        {"ts": yesterday, "kind": "trace", "id": "D", "query": "y",
         "provider": "openai"},
    ])

    cost, priced, unpriced = cli._today_cost_breakdown(ep, today_start)
    assert cost == pytest.approx(0.0076)
    assert priced == 2
    assert unpriced == 1


def test_today_cost_breakdown_skips_precall_traces_with_no_total_tokens(tmp_path):
    """Pre-call trace events share the trace id with their close-trace but
    have no `total_tokens` — they must not inflate the unpriced count, which
    would mis-report 'N unknown models' to the user."""
    ep = tmp_path / "events.jsonl"
    now = int(time.time())
    _write_events(ep, [
        {"ts": now, "kind": "trace", "id": "T1",
         "query": "q", "provider": "openai"},  # pre-call: skipped
        {"ts": now, "kind": "trace", "id": "T1",
         "provider": "openai", "total_tokens": 100, "cost_usd": 0.0001},  # close
    ])
    cost, priced, unpriced = cli._today_cost_breakdown(ep, now - 60)
    assert (cost, priced, unpriced) == (pytest.approx(0.0001), 1, 0)


def test_today_cost_breakdown_on_missing_events_file_is_zeros(tmp_path):
    """Fresh user, never chatted — `stats` must not error on a missing log."""
    ep = tmp_path / "never-existed.jsonl"
    assert cli._today_cost_breakdown(ep, 0) == (0.0, 0, 0)


# ---------- `mneme stats` end-to-end -----------------------------------------

def test_stats_prints_today_cost_breakdown_line(runner, home_paths):
    """Wire end-to-end: cost line is in stdout with the documented format
    `today cost: $X.XXXX (P priced, U unpriced)`. We don't assert the rest of
    the stats output (already covered elsewhere)."""
    _, ep = home_paths
    now = int(time.time())
    _write_events(ep, [
        {"ts": now, "kind": "trace", "id": "A",
         "provider": "openai", "total_tokens": 1500, "cost_usd": 0.0075},
        {"ts": now, "kind": "trace", "id": "B",
         "provider": "xai", "total_tokens": 200},  # unpriced
    ])
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0, result.stdout
    assert "today cost: $0.0075 (1 priced, 1 unpriced)" in result.stdout


def test_stats_on_fresh_install_shows_zero_cost_line(runner, home_paths):
    """Boundary 5 + 10 corner: no traces yet → the cost line must still
    render, with zeros. Empty footer would silently hide the feature."""
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0, result.stdout
    assert "today cost: $0.0000 (0 priced, 0 unpriced)" in result.stdout


# ---------- `mneme explain` shows cost_usd: boundary 13 ----------------------

def test_explain_includes_cost_usd_field_when_logged(runner, home_paths):
    """Boundary 13: explain merges all trace lines for an id, so cost_usd
    (written by _close_turn) must appear in its dump. This guards the user's
    audit trail — 'how much did THAT turn cost?' must be answerable."""
    _, ep = home_paths
    now = int(time.time())
    _write_events(ep, [
        # Pre-call row.
        {"ts": now, "kind": "trace", "id": "TID-001", "query": "q",
         "provider": "openai", "model": "gpt-4o"},
        # Close row with cost.
        {"ts": now, "kind": "trace", "id": "TID-001",
         "provider": "openai", "total_tokens": 1500,
         "prompt_tokens": 1000, "completion_tokens": 500,
         "response_hash": "abc", "citation_quality": "coarse",
         "cost_usd": 0.0075},
    ])
    result = runner.invoke(cli.app, ["explain", "TID-001"])
    assert result.exit_code == 0, result.stdout
    assert "cost_usd" in result.stdout
    assert "0.0075" in result.stdout


def test_explain_does_not_invent_cost_usd_for_unpriced_trace(runner, home_paths):
    """Boundary 13 negative: unpriced (unknown model) turn must NOT show a
    spurious cost_usd line in explain — a fabricated zero here would mislead
    cost audits. Confirms the agent-layer 'omit on None' choice round-trips
    through events.explain unchanged."""
    _, ep = home_paths
    now = int(time.time())
    _write_events(ep, [
        {"ts": now, "kind": "trace", "id": "TID-002", "query": "q",
         "provider": "xai", "model": "grok"},
        {"ts": now, "kind": "trace", "id": "TID-002",
         "provider": "xai", "total_tokens": 200,
         "prompt_tokens": 100, "completion_tokens": 100,
         "response_hash": "def", "citation_quality": "coarse"},
    ])
    result = runner.invoke(cli.app, ["explain", "TID-002"])
    assert result.exit_code == 0, result.stdout
    assert "cost_usd" not in result.stdout


# ---------- price-drift invariant (test lead caveat) -------------------------

def test_today_cost_breakdown_uses_logged_cost_not_current_price_table(tmp_path, monkeypatch):
    """Architect #10 sub-invariant: the daily cost sum reads `cost_usd`
    directly from each trace event — it must NOT re-price by calling
    `pricing.cost_usd(provider, model, usage)` against the current table.
    This protects historical spend from price-table drift (someone updates
    pricing.yaml; yesterday's recorded cost stays the same).
    """
    ep = tmp_path / "events.jsonl"
    now = int(time.time())
    ep.write_text(
        json.dumps({
            "ts": now, "kind": "trace", "id": "T1",
            "provider": "openai", "model": "gpt-4o",
            "prompt_tokens": 1000, "completion_tokens": 500,
            "total_tokens": 1500,
            "cost_usd": 0.0075,
        }) + "\n"
    )
    # If anyone ever "optimizes" stats to re-price from the current table,
    # this monkeypatch returns 10× the logged value and the assertion fires.
    monkeypatch.setattr(pricing, "cost_usd", lambda *a, **kw: 0.075)

    cost, priced, unpriced = cli._today_cost_breakdown(ep, 0)
    assert cost == 0.0075   # logged value, not the 10× monkeypatched price
    assert priced == 1
    assert unpriced == 0
