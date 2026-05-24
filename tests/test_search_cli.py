"""v0.6 `mneme search` — CLI surface + Slice.score contract.

Two layers covered here:

* unit  — `_format_hit` is a pure function (no DB, no LLM); its formatting
          rules (truncation, newline normalization, width clamp) live here.
* int.  — the `search` Typer command, exercised through CliRunner with the
          existing fake LLM, against a temp ~/.mneme/db. We assert exit code
          and stdout substrings, never full string equality (the timestamp
          column would make that brittle).

Why a new file rather than appending to test_memory.py: the search command is
its own surface area (a frontend), and conflating it with memory-layer tests
would muddy what each file is responsible for.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mneme import cli, paths
from mneme.memory import ingest, retrieve, store
from mneme.memory.retrieve import Slice


# ---------- fixtures ---------------------------------------------------------

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home_db(tmp_path, monkeypatch, fake_llm):
    """Redirect ~/.mneme/db to a temp file and yield an opened connection.

    `mneme search` calls `paths.DB_PATH.exists()` then `store.connect(paths.DB_PATH)`,
    so patching the module attribute (read at call-time) is sufficient.
    """
    db_path = tmp_path / "db.sqlite"
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(paths, "DB_PATH", db_path)
    monkeypatch.setattr(paths, "EVENTS_PATH", events_path)
    cx = store.connect(db_path)
    store.init_db(cx)
    yield cx
    cx.close()


# ---------- _format_hit: pure unit tests -------------------------------------

def _slice(text="hello", score=0.5, sid="01ABC", role="user", created=0):
    return Slice(sid, role, text, created, score)


def test_format_hit_truncates_long_text_with_ellipsis():
    s = _slice(text="x" * 200)
    out = cli._format_hit(s, width=50, full=False)
    assert "x" * 50 + "…" in out
    assert "x" * 51 not in out  # exactly width chars before the ellipsis


def test_format_hit_full_flag_disables_truncation():
    s = _slice(text="y" * 200)
    out = cli._format_hit(s, width=50, full=True)
    assert "y" * 200 in out


def test_format_hit_normalizes_embedded_newlines():
    """Newlines would wreck the one-line-per-hit grid; they become ' ⏎ '."""
    s = _slice(text="line one\nline two")
    out = cli._format_hit(s, width=100, full=False)
    assert "\n" not in out
    assert "line one ⏎ line two" in out


def test_format_hit_clamps_zero_width_to_one():
    """width=0 would yield text[:0] = '' which is useless; spec says clamp to 1."""
    s = _slice(text="abcdef")
    out = cli._format_hit(s, width=0, full=False)
    assert "a…" in out  # first char + ellipsis, not just "…" alone
    # And does not raise on negative either.
    cli._format_hit(s, width=-5, full=False)


def test_format_hit_renders_score_to_three_decimals():
    """Score column is fixed-precision so the grid stays aligned."""
    out = cli._format_hit(_slice(score=0.123456), width=20, full=False)
    assert "0.123" in out and "0.1234" not in out


# ---------- search command: happy + zero-result ------------------------------

def test_search_returns_hits_with_header_and_footer(runner, home_db, fake_llm):
    ingest.save_user_message("the capital of memory is recall", "turn1", home_db)
    home_db.commit()

    result = runner.invoke(cli.app, ["search", "the capital of memory is recall"])
    assert result.exit_code == 0
    assert "score" in result.stdout  # header
    assert "the capital of memory is recall" in result.stdout
    assert "hits" in result.stdout    # footer summary line


def test_search_on_empty_db_prints_no_hits_and_exits_zero(runner, home_db):
    """Graceful: nothing indexed yet is a normal state, not an error."""
    result = runner.invoke(cli.app, ["search", "anything"])
    assert result.exit_code == 0
    assert "no hits" in result.stdout
    assert 'query="anything"' in result.stdout


# ---------- search command: error paths --------------------------------------

def test_search_without_init_errors_clearly(runner, tmp_path, monkeypatch, fake_llm):
    """No db file → friendly message + exit 1. The user gets actionable text."""
    monkeypatch.setattr(paths, "DB_PATH", tmp_path / "does-not-exist.sqlite")
    monkeypatch.setattr(paths, "EVENTS_PATH", tmp_path / "events.jsonl")
    result = runner.invoke(cli.app, ["search", "x"])
    assert result.exit_code == 1
    assert "mneme init" in result.stdout


def test_search_missing_query_argument_fails(runner, home_db):
    """Typer must reject the call when the required positional is absent."""
    result = runner.invoke(cli.app, ["search"])
    assert result.exit_code != 0


def test_search_non_integer_k_fails(runner, home_db):
    """`-k notanumber` is a usage error, not a silent fallback."""
    result = runner.invoke(cli.app, ["search", "x", "-k", "notanumber"])
    assert result.exit_code != 0


def test_search_empty_query_is_graceful_not_a_crash(runner, home_db, fake_llm):
    """A purely-whitespace query short-circuits before embedding (would be
    a useless embed call) and prints a clear message, exit 0."""
    result = runner.invoke(cli.app, ["search", "   "])
    assert result.exit_code == 0
    assert "empty query" in result.stdout


def test_search_reports_embed_provider_failure_with_exit_1(runner, home_db, fake_llm, monkeypatch):
    """If `recall` (i.e. the embedder) raises, the command must NOT crash with
    a traceback — it surfaces a red one-liner and exits 1.
    """
    def boom(*a, **kw):
        raise RuntimeError("connection refused")
    # cli.py does `from .memory import retrieve`, so patching `cli.retrieve`
    # is what the command actually resolves at call time.
    monkeypatch.setattr(cli.retrieve, "recall", boom)

    result = runner.invoke(cli.app, ["search", "x"])
    assert result.exit_code == 1
    assert "embed provider unreachable" in result.stdout
    assert "connection refused" in result.stdout


# ---------- search command: -k boundaries ------------------------------------

def test_search_k_zero_is_graceful(runner, home_db, fake_llm):
    """`-k 0` is a degenerate but legal request — vector index returns nothing,
    we expect the no-hits message, exit 0. (No crash, no traceback.)
    """
    ingest.save_user_message("seed", "turn1", home_db)
    home_db.commit()
    result = runner.invoke(cli.app, ["search", "seed", "-k", "0"])
    assert result.exit_code == 0
    assert "no hits" in result.stdout


def test_search_k_at_sqlite_vec_max_works(runner, home_db, fake_llm):
    """4096 is the documented sqlite-vec ceiling — at the ceiling we must still
    return cleanly. This pins the boundary so a future bump in the limit, or a
    regression below it, is caught immediately.
    """
    ingest.save_user_message("only row", "turn1", home_db)
    home_db.commit()
    result = runner.invoke(cli.app, ["search", "only row", "-k", "4096"])
    assert result.exit_code == 0
    assert "only row" in result.stdout


@pytest.mark.xfail(
    reason=(
        "BUG v0.6: -k > 4096 raises sqlite3.OperationalError from sqlite-vec "
        "('k value in knn query too large'), which the `except Exception` "
        "block mislabels as 'embed provider unreachable' and exits 1. "
        "Expected: clamp k to the index ceiling and succeed, OR fail with a "
        "k-specific message ('-k must be ≤ 4096')."
    ),
    strict=True,
)
def test_search_k_huge_does_not_crash(runner, home_db, fake_llm):
    """`-k` far exceeding the vector-index ceiling must clamp gracefully.

    A naive user typing `-k 10000` should not see a misleading
    'embed provider unreachable' error — the embedder is fine; the index
    rejected the query parameter.
    """
    ingest.save_user_message("only row", "turn1", home_db)
    home_db.commit()
    result = runner.invoke(cli.app, ["search", "only row", "-k", "10000"])
    assert result.exit_code == 0
    assert "only row" in result.stdout


# ---------- search command: special characters & determinism -----------------

def test_search_handles_special_characters_in_query(runner, home_db, fake_llm):
    """Quotes, SQL meta-chars, unicode — must not break parsing or SQL."""
    ingest.save_user_message("Tauri's API; SELECT * FROM 你好", "turn1", home_db)
    home_db.commit()
    weird = "Tauri's API; SELECT * FROM 你好"
    result = runner.invoke(cli.app, ["search", weird])
    assert result.exit_code == 0


def test_search_same_query_twice_is_byte_identical(runner, home_db, fake_llm):
    """Cache-hit invariant (PRINCIPLES.md #2): identical query => identical
    output bytes, modulo nothing. The timestamp column comes from `created_at`
    of the slice (not wall clock), so this IS deterministic.
    """
    ingest.save_user_message("stable result fixture", "turn1", home_db)
    ingest.save_user_message("another slice", "turn1", home_db)
    home_db.commit()

    a = runner.invoke(cli.app, ["search", "stable result fixture"])
    b = runner.invoke(cli.app, ["search", "stable result fixture"])
    assert a.exit_code == 0 and b.exit_code == 0
    assert a.stdout == b.stdout


# ---------- Slice.score contract --------------------------------------------

def test_slice_score_default_is_zero_keeps_old_callers_working():
    """`Slice.score` is a new field. Existing tests / agent code construct
    Slice without it — the default-zero keeps them green. This guards against
    someone removing the default in a future refactor.
    """
    s = Slice("id1", "user", "t", 0)
    assert s.score == 0.0


def test_recall_fills_score_and_is_monotonically_non_increasing(cx, fake_llm):
    """recall() must populate `score` (not leave it at 0.0) and rows must come
    back sorted highest→lowest. This is the contract `_print_hits` relies on
    to render meaningful output."""
    ingest.save_user_message("alpha bravo charlie", "turn1", cx)
    ingest.save_user_message("delta echo foxtrot", "turn1", cx)
    ingest.save_user_message("golf hotel india", "turn1", cx)

    hits = retrieve.recall("alpha bravo charlie", cx, k=10)
    assert hits, "expected non-empty recall to validate score field"
    assert any(h.score > 0 for h in hits), "score should be populated, not all zero"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), f"score must be non-increasing, got {scores}"
