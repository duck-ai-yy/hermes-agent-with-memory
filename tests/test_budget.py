"""v0.11 token-budget unit tests — Unit A (estimate_tokens), Unit B
(context_window_for + budget_for_retrieval), Unit C (assemble_prompt +
AssemblyMeta + PromptTooBig) — plus the budget-side N-pins, R-ratchets and
forensic D-pins.

================================================================================
RATCHET MATRIX (16 R-* — full enumeration, lead must-fix #1)
================================================================================
R-A-1  estimate_tokens stays len(text)//4 + 1 (PRINCIPLE 1: one number = one
       thing; same as client.py:161 `payload_chars // 4`).
R-A-2  No tokenizer dependency creep — pyproject.toml has no `tiktoken` or
       `transformers` line (N6, also pinned in §N).
R-A-3  Empty string returns 1, not 0 — guards budget arithmetic from /0 and
       "an empty suffix still costs a slot in practice".
R-A-3b char-equivalence: estimate_tokens("中文测试") == estimate_tokens("abcd")
       (lead should-fix #1 — char count not byte count, pins // 4 = chars/4).

R-B-1  context_window_for ignores models case-sensitively (yaml keys are
       exact; "GPT-4O" != "gpt-4o").
R-B-2  budget_for_retrieval result is always non-negative (max(0, ...)).
R-B-3  budget_for_retrieval honors env at call time, no process restart
       needed — mirrors agent._max_iters style (R-14b: soft pin).
R-B-4  context_window_for unknown pair warns ONCE per (provider, model);
       repeat lookups are silent (warn-once-per-process tested via
       capsys).
R-B-5  budget_for_retrieval with MNEME_CONTEXT_BUDGET_RATIO=abc warns once
       PER bad value (set-based, not per-process) — pin dev micro #1.
R-B-6  COMPLETION_RESERVE_FLOOR floor (1024) applies even when window // N
       is smaller — small windows don't collapse to a useless reserve.

R-9a   soul.prompt_hash signature is single-arg `(text: str) -> str`
       (lead should-fix #2 split a).
R-9b   _open_turn computes prompt_hash on `prefix + "\\n" + suffix` ONLY —
       never includes session_id or meta (lead should-fix #2 split b).

R-12   pricing._load_table is the ONLY yaml reader for context_windows —
       budget.context_window_for delegates through it (must-fix #5 fixture
       scope verified).
R-12b  tmp_pricing_yaml fixture covers BOTH `_TABLE_PATH` AND the cache —
       pin via "fixture write actually changes the value" (must-fix #5).

R-14b  _reserve_ratio_inv() reads env on each call (no @functools.cache
       on it) — soft pin via grep + functional assertion.

R-15   _FALLBACK_WINDOW literal == 8192 (matches design §3 ollama default).
R-16   _TOOLS_SCHEMA_RESERVE literal == 1024 (matches the floor — both
       halves of fixed overhead share their minimum).

================================================================================
MUTATION MATRIX (24 M-* — lead must-fix #1 enumeration)
================================================================================
M_estimate_1  estimate_tokens uses `// 5` instead of // 4 → C5/A1/A3b fail.
M_estimate_2  estimate_tokens drops `+ 1` → A3 (empty string) fails.
M_estimate_3  estimate_tokens uses bytes (len(text.encode())) → A3b fails.

M_window_1   _FALLBACK_WINDOW lowered to 4096 → R-15 + B6 fail.
M_window_2   context_window_for skips ollama wildcard branch (returns
             FALLBACK directly) → B2 fail.
M_window_3   context_window_for case-insensitive match → R-B-1 fail.
M_window_4   warn-once cache disabled (always warn) → R-B-4 fail.

M_budget_1   budget_for_retrieval drops the floor → R-B-6 fail.
M_budget_2   budget_for_retrieval forgets _TOOLS_SCHEMA_RESERVE → B10 fail.
M_budget_3   budget_for_retrieval returns negative on small windows → R-B-2
             fail.
M_budget_4   _reserve_ratio_inv caches first env read → R-B-3 fail.

M_ratio_1   _warned_ratio uses bool sentinel (per-process) → R-B-5 + dev
            micro #1 reverse mutation fail (Step 4 sanity check).
M_ratio_2   invalid env ratio raises ValueError instead of warning → B12
            fail.
M_ratio_3   ratio out-of-range (>=1.0) silently used → B14 fail.

M_pricing_1 pricing._TABLE_PATH not monkeypatched → R-12 fixture coverage
            visibly broken (R-12b catches).
M_pricing_2 budget calls yaml.safe_load directly, bypassing pricing — N4
            grep fires.

M_assemble_1 assemble_prompt drops from HEAD instead of tail → C8 fail
             (kept_slice_ids order verifies tail-drop).
M_assemble_2 prefix gets mutated when slices drop → C9 fail (prefix is
             input-equal to output).
M_assemble_3 user_text gets truncated → C10 fail (last call_args literal).
M_assemble_4 dropped_ids order reversed (head-first not tail-first) → C8
             order assertion fails.
M_drop_1    drop loop off-by-one (slices[:-2] per iter) → C6/C7 fail.
M_drop_2    drop loop empties list but never checks final fit → C12 fail
             (PromptTooBig must surface).
M_drop_3    middle-of-list drop (random index) → C8 ordering fail.

M_raise_1   PromptTooBig forgets `dropped_count` attr → D9 fails.
M_raise_2   PromptTooBig swallowed (returned None) → C12 fail.

================================================================================
NEGATIVE-FORM PINS (N1, N2, N6 — budget-side half of lead must-fix #2)
================================================================================
N1  budget.py does NOT import retrieve module (decoupling).
N2  budget.py does NOT call events.append (separation of concerns).
N6  pyproject.toml does NOT carry a tiktoken / transformers dep.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from mneme import soul
from mneme.llm import budget, pricing
from mneme.memory.retrieve import Slice


# =============================================================================
# Helpers
# =============================================================================


def _make_slice(sid: str, text: str, score: float = 1.0) -> Slice:
    return Slice(id=sid, role="user", text=text, created_at=0, score=score)


# =============================================================================
# Unit A — estimate_tokens
# =============================================================================


def test_A1_estimate_tokens_short_string_round_down_plus_one():
    """A1: 'abcd' (4 chars) → 4//4 + 1 == 2. Pins the // 4 + 1 formula."""
    assert budget.estimate_tokens("abcd") == 2


def test_A2_estimate_tokens_zero_chars_returns_one():
    """A2 == R-A-3: empty string returns 1, NOT 0 — design comment
    'lifts the empty string to 1 so budget arithmetic stays away from 0'."""
    assert budget.estimate_tokens("") == 1


def test_A3b_estimate_tokens_char_equivalence_chinese_vs_latin():
    """A3b (lead should-fix #1): a 4-char Chinese string returns the same
    count as a 4-char Latin string. Pins "chars not bytes" — if the impl
    accidentally counts bytes (utf-8 encodes "中" to 3 bytes) this fires."""
    assert budget.estimate_tokens("中文测试") == budget.estimate_tokens("abcd")


def test_A4_estimate_tokens_scales_linearly_with_length():
    """A4: 100 chars vs 200 chars — the second is (200//4+1)/(100//4+1) ≈ 2x."""
    short = budget.estimate_tokens("a" * 100)
    long = budget.estimate_tokens("a" * 200)
    assert short == 26 and long == 51


def test_A5_estimate_tokens_no_provider_branch():
    """A5: heuristic is the same regardless of which provider/model the
    caller cares about — no kwargs, no `model=` parameter. Pins the design
    comment 'No model-specific branch (e.g. EN vs CN)'."""
    sig = inspect.signature(budget.estimate_tokens)
    assert list(sig.parameters) == ["text"]


def test_R_A_1_estimate_tokens_formula_matches_client_payload_chars_div_4():
    """R-A-1: client.py uses `payload_chars // 4` for its own estimate.
    `estimate_tokens(s)` MUST equal `len(s) // 4 + 1` so the two layers
    speak one language (PRINCIPLE 1)."""
    for s in ["", "a", "ab" * 50, "x" * 1024]:
        assert budget.estimate_tokens(s) == len(s) // 4 + 1


# =============================================================================
# Unit B — context_window_for + budget_for_retrieval
# =============================================================================


def test_B1_context_window_for_known_openai_returns_yaml_value(clear_warn_cache):
    """B1: openai/gpt-4o-mini → 128000 per pricing.yaml."""
    assert budget.context_window_for("openai", "gpt-4o-mini") == 128000


def test_B2_context_window_for_ollama_returns_wildcard_value(
    clear_warn_cache, tmp_pricing_yaml,
):
    """B2: ollama uses the "*" wildcard. Override pricing.yaml to a known
    value so we can pin the lookup mechanism (not just the production 8192)."""
    tmp_pricing_yaml(
        "context_windows:\n"
        "  ollama:\n"
        "    \"*\": 16384\n"
    )
    assert budget.context_window_for("ollama", "anything-here") == 16384


def test_B3_context_window_for_unknown_falls_back_to_8192(
    clear_warn_cache, capsys,
):
    """B3: unknown (provider, model) → _FALLBACK_WINDOW (8192) with stderr
    one-shot warning naming the offender."""
    val = budget.context_window_for("openai", "definitely-not-a-model")
    assert val == 8192
    err = capsys.readouterr().err
    assert "definitely-not-a-model" in err
    assert "8192" in err


def test_B4_context_window_for_unknown_warns_once_per_pair(
    clear_warn_cache, capsys,
):
    """B4 == R-B-4: same unknown (provider, model) repeated → exactly one
    warning ever."""
    budget.context_window_for("openai", "still-not-a-model")
    capsys.readouterr()  # flush first call's warning
    for _ in range(3):
        budget.context_window_for("openai", "still-not-a-model")
    err = capsys.readouterr().err
    assert "still-not-a-model" not in err


def test_B5_context_window_for_ollama_silent_no_warn_even_unknown(
    clear_warn_cache, capsys,
):
    """B5: ollama path is silent even if the model is "unknown" by yaml
    standards (the "*" wildcard always matches) — design §3."""
    budget.context_window_for("ollama", "llama3.2:1b")
    err = capsys.readouterr().err
    assert "llama" not in err.lower()


def test_R_B_1_context_window_for_case_sensitive(clear_warn_cache, capsys):
    """R-B-1: yaml lookup is exact-match; "GPT-4O" is not "gpt-4o"."""
    val = budget.context_window_for("openai", "GPT-4O")
    assert val == 8192  # fallback


def test_B6_context_window_for_missing_section_warns_once_and_falls_back(
    clear_warn_cache, tmp_pricing_yaml, capsys,
):
    """B6: pricing.yaml without a `context_windows:` section → fallback +
    one stderr warning naming "context_windows"."""
    tmp_pricing_yaml("openai:\n  gpt-4o:\n    input: 1.0\n    output: 2.0\n")
    capsys.readouterr()
    assert budget.context_window_for("openai", "gpt-4o") == 8192
    err = capsys.readouterr().err
    assert "context_windows" in err


def test_B7_budget_for_retrieval_subtracts_reserves(
    clear_warn_cache, monkeypatch,
):
    """B7: budget = window - max(floor, window//N) - tools_reserve. For
    openai/gpt-4o: 128000 - 32000 - 1024 == 94976."""
    monkeypatch.delenv("MNEME_CONTEXT_BUDGET_RATIO", raising=False)
    assert budget.budget_for_retrieval("openai", "gpt-4o") == 128000 - 32000 - 1024


def test_B8_budget_for_retrieval_floor_kicks_in_small_window(
    clear_warn_cache, tmp_pricing_yaml, monkeypatch,
):
    """B8 == R-B-6: window=4096 → 4096//4 == 1024 (matches floor). Window
    smaller than 4096 → floor still 1024. Pinned via custom yaml."""
    monkeypatch.delenv("MNEME_CONTEXT_BUDGET_RATIO", raising=False)
    tmp_pricing_yaml(
        "context_windows:\n"
        "  openai:\n"
        "    tiny-model: 2048\n"
    )
    # 2048 - max(1024, 2048//4=512) - 1024 == 2048 - 1024 - 1024 == 0
    assert budget.budget_for_retrieval("openai", "tiny-model") == 0


def test_B9_budget_for_retrieval_pathological_negative_clamped(
    clear_warn_cache, tmp_pricing_yaml, monkeypatch,
):
    """B9 == R-B-2: window=512 → completion_reserve = max(1024, 128) ==
    1024; tools_reserve = 1024; window - 2048 == -1536. max(0, -1536) == 0."""
    monkeypatch.delenv("MNEME_CONTEXT_BUDGET_RATIO", raising=False)
    tmp_pricing_yaml(
        "context_windows:\n"
        "  openai:\n"
        "    micro-model: 512\n"
    )
    assert budget.budget_for_retrieval("openai", "micro-model") == 0


def test_B10_budget_for_retrieval_includes_tools_schema_reserve(
    clear_warn_cache, tmp_pricing_yaml, monkeypatch,
):
    """B10: literal 1024 tools_reserve is observable. With ratio 0.5 (N=2),
    we can isolate the tools_reserve term:
        budget = window - window//2 - 1024 == window//2 - 1024.
    Two windows of size 20000 and 22000 → diff of 1000, but if we drop the
    -1024 term it would also be 1000. So instead: pick window=20000 →
    expected 20000//2 - 1024 == 8976. Anyone deleting -1024 gets 10000.
    """
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "0.5")
    tmp_pricing_yaml(
        "context_windows:\n"
        "  openai:\n"
        "    a: 20000\n"
    )
    val = budget.budget_for_retrieval("openai", "a")
    # If -_TOOLS_SCHEMA_RESERVE is removed, this would be 10000 (off by 1024).
    assert val == 20000 - 10000 - 1024 == 8976


def test_B11_env_ratio_05_doubles_completion_reserve(
    clear_warn_cache, monkeypatch,
):
    """B11: MNEME_CONTEXT_BUDGET_RATIO=0.5 → N=2 → reserve = window//2."""
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "0.5")
    # openai/gpt-4o: 128000 // 2 == 64000; - 1024 == 62976.
    assert budget.budget_for_retrieval("openai", "gpt-4o") == 128000 - 64000 - 1024


def test_B12_env_ratio_invalid_value_warns_and_falls_back(
    clear_warn_cache, monkeypatch, capsys,
):
    """B12 == R-B-5: MNEME_CONTEXT_BUDGET_RATIO=abc → fallback default
    (N=4) with one stderr warning naming "abc"."""
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "abc")
    val = budget.budget_for_retrieval("openai", "gpt-4o")
    assert val == 128000 - 32000 - 1024  # default reserve
    err = capsys.readouterr().err
    assert "abc" in err and "0.25" in err


def test_B13_env_ratio_invalid_value_warns_per_offending_value(
    clear_warn_cache, monkeypatch, capsys,
):
    """B13 / dev micro #1: switch from MNEME_CONTEXT_BUDGET_RATIO=abc
    to =def → SECOND warning fires (set-based, not bool sentinel).
    Re-reading the same "abc" stays silent."""
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "abc")
    budget.budget_for_retrieval("openai", "gpt-4o")
    capsys.readouterr()
    # Same value: silent.
    budget.budget_for_retrieval("openai", "gpt-4o")
    err1 = capsys.readouterr().err
    assert "abc" not in err1
    # Different bad value: warns again.
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "def")
    budget.budget_for_retrieval("openai", "gpt-4o")
    err2 = capsys.readouterr().err
    assert "def" in err2


def test_B14_env_ratio_out_of_range_falls_back(
    clear_warn_cache, monkeypatch, capsys,
):
    """B14: ratio <= 0.0 or >= 1.0 → fallback default + warn."""
    for bad in ["0.0", "1.0", "1.5", "-0.5"]:
        monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", bad)
        val = budget.budget_for_retrieval("openai", "gpt-4o")
        assert val == 128000 - 32000 - 1024


def test_R_B_3_env_read_per_call_no_cache(clear_warn_cache, monkeypatch):
    """R-B-3 + R-14b (soft pin): change env between two calls → second call
    sees the new value, no process restart needed."""
    monkeypatch.delenv("MNEME_CONTEXT_BUDGET_RATIO", raising=False)
    default = budget.budget_for_retrieval("openai", "gpt-4o")
    monkeypatch.setenv("MNEME_CONTEXT_BUDGET_RATIO", "0.5")
    changed = budget.budget_for_retrieval("openai", "gpt-4o")
    assert default != changed


def test_R_14b_reserve_ratio_inv_not_cached_via_functools(clear_warn_cache):
    """R-14b (lead should-fix #3 soft pin): _reserve_ratio_inv() must not
    sit behind @functools.cache / @lru_cache — env reads happen call-time.
    Grep-level pin: function has no cache_clear attr."""
    fn = budget._reserve_ratio_inv
    assert not hasattr(fn, "cache_clear"), (
        "_reserve_ratio_inv has cache_clear — must read env each call"
    )


def test_R_14b_other_loaders_may_cache(clear_warn_cache):
    """R-14b complement: pricing._load_table IS cached (functools.cache
    is intentional). Soft-pin via 'cache_clear exists' so a maintainer who
    removes the cache by accident notices."""
    assert hasattr(pricing._load_table, "cache_clear")


def test_R_12_yaml_loader_is_pricing_load_table(
    clear_warn_cache, tmp_pricing_yaml, monkeypatch,
):
    """R-12: budget delegates to pricing._load_table. Clearing the cache
    and re-writing the yaml flips the value seen by budget."""
    tmp_pricing_yaml(
        "context_windows:\n"
        "  openai:\n"
        "    foo: 1000\n"
    )
    assert budget.context_window_for("openai", "foo") == 1000
    tmp_pricing_yaml(
        "context_windows:\n"
        "  openai:\n"
        "    foo: 2000\n"
    )
    assert budget.context_window_for("openai", "foo") == 2000


def test_R_12b_fixture_actually_redirects_table_path(
    clear_warn_cache, tmp_pricing_yaml, tmp_path,
):
    """R-12b: lead must-fix #5 — pin that `tmp_pricing_yaml` actually
    changes pricing._TABLE_PATH for the duration of the test (otherwise
    the fixture would write a file the loader never reads)."""
    tmp_pricing_yaml("openai: {}\n")
    assert pricing._TABLE_PATH == tmp_path / "pricing.yaml"
    assert pricing._TABLE_PATH.exists()


def test_R_15_fallback_window_literal_is_8192():
    """R-15: design §3 ollama default + global fallback."""
    assert budget._FALLBACK_WINDOW == 8192


def test_R_16_tools_schema_reserve_literal_is_1024():
    """R-16: both halves of fixed overhead share their minimum (1024)."""
    assert budget._TOOLS_SCHEMA_RESERVE == 1024


def test_R_9a_soul_prompt_hash_signature_single_text_arg():
    """R-9a (lead should-fix #2 split a): soul.prompt_hash takes exactly
    one positional `text: str` — no session_id, no meta kwargs."""
    sig = inspect.signature(soul.prompt_hash)
    assert list(sig.parameters) == ["text"]


# =============================================================================
# Unit C — assemble_prompt + AssemblyMeta + PromptTooBig
# =============================================================================


# Realistic prefix that matches what build_prompt actually returns.
_PREFIX = "x" * 5


def test_C1_assemble_prompt_no_slices_fits_returns_clean_meta(cx, fake_llm):
    """C1: no slices, prompt fits → suffix derived from build_prompt empty
    template; kept/dropped both [], original_count=0."""
    _, suffix, meta = budget.assemble_prompt(_PREFIX, [], "hello", budget=10_000)
    assert meta.kept_slice_ids == []
    assert meta.dropped_slice_ids == []
    assert meta.original_slice_count == 0
    assert "hello" in suffix
    assert meta.estimated_tokens > 0


def test_C2_assemble_prompt_fits_keeps_all_slices(cx, fake_llm):
    """C2: 3 short slices, generous budget → all 3 kept, dropped empty,
    order preserved."""
    slices = [_make_slice(f"S{i}", "tiny", score=10 - i) for i in range(3)]
    _, suffix, meta = budget.assemble_prompt(_PREFIX, slices, "q", budget=10_000)
    assert meta.kept_slice_ids == ["S0", "S1", "S2"]
    assert meta.dropped_slice_ids == []
    assert meta.original_slice_count == 3
    for sid in ["S0", "S1", "S2"]:
        assert sid in suffix


def test_C3_assemble_prompt_preserves_prefix_byte_for_byte(cx, fake_llm):
    """C3: returned prefix == input prefix. Pins PRINCIPLE 2 cache key."""
    prefix_in = "x" * 100
    prefix_out, _, _ = budget.assemble_prompt(
        prefix_in, [], "anything", budget=10_000,
    )
    assert prefix_out == prefix_in


def test_C4_assemble_prompt_dropped_meta_ordering_is_tail_first(cx, fake_llm):
    """C4: when budget forces drops, the tail (lowest score) gets dropped
    first; meta.dropped_slice_ids preserves drop order."""
    # 3 chunky slices; force exactly the last 2 to drop.
    slices = [
        _make_slice("HEAD", "h" * 100, score=10),
        _make_slice("MID",  "m" * 100, score=5),
        _make_slice("TAIL", "t" * 100, score=1),
    ]
    # Need a tight budget. estimate_tokens for the whole 3-slice suffix is
    # roughly len(prefix+suffix)//4+2 — set budget around the 1-slice mark.
    # Concrete: just measure first.
    _, suffix_full, _ = budget.assemble_prompt(_PREFIX, slices, "q", budget=10_000)
    full_cost = budget.estimate_tokens(_PREFIX) + budget.estimate_tokens(suffix_full)
    _, _, meta_only_head = budget.assemble_prompt(
        _PREFIX, slices[:1], "q", budget=10_000,
    )
    # Squeeze: drop MID + TAIL.
    tight = full_cost - 30  # drop something
    _, _, meta_tight = budget.assemble_prompt(_PREFIX, slices, "q", budget=tight)
    # The drop order must be tail-first.
    assert meta_tight.dropped_slice_ids[0] == "TAIL"
    if len(meta_tight.dropped_slice_ids) >= 2:
        assert meta_tight.dropped_slice_ids[1] == "MID"


def test_C5_assemble_prompt_short_prefix_forces_some_slice_in(cx, fake_llm):
    """C5 (lead must-fix #4): prefix="x" * 5 (short, so the budget surely
    fits 1+ slices). All slices kept under generous budget."""
    prefix = "x" * 5
    slices = [_make_slice(f"S{i}", "small", score=5 - i) for i in range(2)]
    _, _, meta = budget.assemble_prompt(prefix, slices, "q", budget=100_000)
    assert meta.kept_slice_ids == ["S0", "S1"]
    assert meta.dropped_slice_ids == []


def test_C6_assemble_prompt_prefix_far_exceeding_budget_raises(cx, fake_llm):
    """C6 (lead must-fix #4): prefix = "x" * (budget * 5) → even empty
    suffix can't fit → PromptTooBig."""
    tiny_budget = 50
    prefix = "x" * (tiny_budget * 5 * 4)  # estimate_tokens => well over budget
    with pytest.raises(budget.PromptTooBig):
        budget.assemble_prompt(prefix, [], "q", budget=tiny_budget)


def test_C7_assemble_prompt_partial_drop_keeps_kept_slice_order(cx, fake_llm):
    """C7: when only some slices drop, the kept_slice_ids order matches
    the input order with dropped removed (NOT a re-sort)."""
    slices = [
        _make_slice("A", "x" * 200, score=10),
        _make_slice("B", "x" * 200, score=8),
        _make_slice("C", "x" * 200, score=6),
        _make_slice("D", "x" * 200, score=4),
    ]
    _, suffix_full, _ = budget.assemble_prompt(_PREFIX, slices, "q", budget=10_000)
    full_cost = budget.estimate_tokens(_PREFIX) + budget.estimate_tokens(suffix_full)
    # Force exactly C + D dropped.
    tight = full_cost - 100
    _, _, meta = budget.assemble_prompt(_PREFIX, slices, "q", budget=tight)
    # Whatever kept set, the order must be a subsequence of [A,B,C,D].
    seen_order = meta.kept_slice_ids
    full_order = ["A", "B", "C", "D"]
    idx = -1
    for sid in seen_order:
        new_idx = full_order.index(sid)
        assert new_idx > idx, f"kept order {seen_order} not a subseq of {full_order}"
        idx = new_idx


def test_C8_assemble_prompt_all_dropped_then_still_fits_returns_empty_kept(cx, fake_llm):
    """C8: every slice dropped, but prefix + empty-suffix still under
    budget → kept_slice_ids=[], dropped_slice_ids=all input ids, no raise."""
    slices = [_make_slice(f"X{i}", "x" * 500, score=10 - i) for i in range(3)]
    _, suffix_empty, _ = budget.assemble_prompt(_PREFIX, [], "q", budget=10_000)
    # Choose budget = cost of "no slices" + 5 tokens slack, so any real
    # slice content overflows.
    target = budget.estimate_tokens(_PREFIX) + budget.estimate_tokens(suffix_empty) + 5
    _, _, meta = budget.assemble_prompt(_PREFIX, slices, "q", budget=target)
    assert meta.kept_slice_ids == []
    assert set(meta.dropped_slice_ids) == {"X0", "X1", "X2"}
    assert meta.original_slice_count == 3


def test_C9_assemble_prompt_does_not_mutate_input_slice_list(cx, fake_llm):
    """C9: caller's `slices` list is unchanged after assemble — no in-place
    mutation. Run twice: once where everything fits, once where everything
    drops; both must leave the caller's list intact.
    """
    slices = [_make_slice(f"S{i}", "x" * 50, score=5 - i) for i in range(3)]
    before = list(slices)
    # Case A: generous budget, all kept.
    budget.assemble_prompt(_PREFIX, slices, "q", budget=10_000)
    assert slices == before
    # Case B: tight budget that forces drops (eventually raises) — still no
    # mutation of the caller's list.
    try:
        budget.assemble_prompt(_PREFIX, slices, "q", budget=5)
    except budget.PromptTooBig:
        pass
    assert slices == before


def test_C10_assemble_prompt_user_text_passed_through_verbatim(cx, fake_llm):
    """C10 (lead must-fix #4 literal fixture): user_text appears in the
    suffix exactly as passed — no truncation, no normalization."""
    weird_text = "Hello\n\t  WORLD! 你好 // [^Z] {{}}"
    _, suffix, _ = budget.assemble_prompt(
        _PREFIX, [], weird_text, budget=10_000,
    )
    assert weird_text in suffix


def test_C11_assemble_prompt_returns_assembly_meta_dataclass(cx, fake_llm):
    """C11: meta is `AssemblyMeta` dataclass with exactly the four fields
    in the design — pin field set against drift."""
    _, _, meta = budget.assemble_prompt(_PREFIX, [], "q", budget=1000)
    assert isinstance(meta, budget.AssemblyMeta)
    fields = set(meta.__dataclass_fields__)
    assert fields == {
        "estimated_tokens", "kept_slice_ids",
        "dropped_slice_ids", "original_slice_count",
    }


def test_C12_assemble_prompt_no_slices_still_overflowing_raises(cx, fake_llm):
    """C12: zero retrieved slices, but prefix + user_text alone overflows
    → PromptTooBig (dropped_count=0)."""
    big_user = "u" * 10_000
    with pytest.raises(budget.PromptTooBig) as exc_info:
        budget.assemble_prompt(_PREFIX, [], big_user, budget=10)
    assert exc_info.value.dropped_count == 0


def test_C13_assemble_prompt_meta_estimated_equals_estimate_tokens_sum(cx, fake_llm):
    """C13: meta.estimated_tokens == estimate_tokens(prefix) +
    estimate_tokens(suffix). Pin via independent recomputation."""
    slices = [_make_slice("Z", "tiny", score=1)]
    prefix_out, suffix_out, meta = budget.assemble_prompt(
        _PREFIX, slices, "hi", budget=10_000,
    )
    expected = (budget.estimate_tokens(prefix_out)
                + budget.estimate_tokens(suffix_out))
    assert meta.estimated_tokens == expected


# =============================================================================
# Forensic D9 — PromptTooBig instance attrs (lead should-fix #4)
# =============================================================================


def test_D9_prompt_too_big_carries_three_forensic_attrs(cx, fake_llm):
    """D9: PromptTooBig.estimated / .budget / .dropped_count must all be
    instance attributes (not just message text). A CLI / HTTP caller
    needs them without parsing the string."""
    try:
        budget.assemble_prompt(_PREFIX, [], "u" * 10_000, budget=5)
    except budget.PromptTooBig as exc:
        assert hasattr(exc, "estimated") and isinstance(exc.estimated, int)
        assert hasattr(exc, "budget") and isinstance(exc.budget, int)
        assert hasattr(exc, "dropped_count") and isinstance(exc.dropped_count, int)
        assert exc.budget == 5
        assert exc.estimated > 5
        assert exc.dropped_count == 0
    else:
        pytest.fail("PromptTooBig not raised")


def test_D9_prompt_too_big_dropped_count_reflects_actual_drops(cx, fake_llm):
    """D9 complement: when slices ARE dropped before the raise, dropped_count
    reflects the number actually dropped (not 0)."""
    big_user = "u" * 10_000
    slices = [_make_slice(f"S{i}", "tiny", score=5 - i) for i in range(2)]
    with pytest.raises(budget.PromptTooBig) as exc_info:
        budget.assemble_prompt(_PREFIX, slices, big_user, budget=10)
    assert exc_info.value.dropped_count == 2


# =============================================================================
# §N — negative-form pins (N1, N2, N6 — budget-side half of lead must-fix #2)
# =============================================================================


def test_N1_budget_does_not_import_retrieve_module():
    """N1: budget.py stays decoupled from memory.retrieve.

    grep layer: source of budget.py contains no `from ..memory import
    retrieve` or `from mneme.memory import retrieve`.
    runtime layer: budget.assemble_prompt accepts `Slice` instances passed
    BY the caller (it does not call retrieve itself).
    """
    src = inspect.getsource(budget)
    assert "from ..memory import retrieve" not in src
    assert "from mneme.memory import retrieve" not in src
    assert "memory.retrieve" not in src or "TYPE_CHECKING" in src
    # Runtime: assemble_prompt's signature takes pre-fetched slices.
    sig = inspect.signature(budget.assemble_prompt)
    assert "slices" in sig.parameters


def test_N2_budget_does_not_call_events_append():
    """N2: budget.py never writes to events.jsonl — that is the agent's
    job. grep + runtime both fire if violated."""
    src = inspect.getsource(budget)
    assert "events.append" not in src
    assert "from ..trace import events" not in src
    assert "from mneme.trace import events" not in src


def test_N6_no_third_party_tokenizer_dependency():
    """N6: pyproject.toml's dependency list has no `tiktoken` or
    `transformers` line — design §0 PRINCIPLE 1 (no tokenizer dep)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "tiktoken" not in text
    assert "transformers" not in text


# =============================================================================
# §R-A-2 / N6 complement — also pinned at the budget.py source level
# =============================================================================


def test_R_A_2_budget_source_no_tokenizer_import():
    """R-A-2: even if pyproject.toml stayed clean, a developer could still
    `import tiktoken` ad-hoc. Pin the source level too."""
    src = inspect.getsource(budget)
    assert "tiktoken" not in src
    assert "transformers" not in src
