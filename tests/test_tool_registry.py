"""v0.9 / M2 — registry public API + @tool decorator + execute() negatives.

Sections:
  A1-A8: schema generation happy path (single source of truth for tool name,
         description, type mapping, defaults vs required).
  B1-B11: @tool error paths — duplicates, missing hints, *args/**kwargs,
          Optional/Union, types.UnionType, missing docstring, missing Args
          block warn, no `Args:` block at all + zero-arg func.
  E1-E8: schemas_for_provider + execute() error contract + JSON-type
         vocabulary (D7) + the "got X" error message (D7 reconciled).
  R-Reg-1..7: ratchet pins — full-string equality on unknown-provider error
              (R-Reg-6 reconciled), silent filter on allow=[unknown] (intent
              contract), idempotence of names() ordering, schema dict is
              the new contract, no global side-effect on TypeError raises,
              events.append crash does not break registration (R-Reg-7).

NOTE: `isolated_registry` is autouse at module scope (Lead must-fix #3).
Without it, B1 through B11 (and every @tool exercise) would leave state
in `_REGISTRY` that bleeds into later test files.
"""

from __future__ import annotations

from typing import Optional, Union

import pytest

from mneme.tools import registry
from mneme.tools.registry import ToolResult, tool

# Autouse the snapshot/restore for every test in this file.
pytestmark = pytest.mark.usefixtures("isolated_registry")


# --- helpers --------------------------------------------------------------


def _ok(text: str = "ok") -> ToolResult:
    return ToolResult(content=text, is_error=False, audit={})


# --- A1-A8: schema generation happy path ---------------------------------


def test_A1_decorator_registers_by_function_name():
    @tool
    def my_unique_tool(x: str) -> ToolResult:
        """Greet.

        Args:
            x: a name.
        """
        return _ok()

    entry = registry.get("my_unique_tool")
    assert entry is not None
    assert entry.name == "my_unique_tool"


def test_A2_schema_top_level_keys_are_name_description_input_schema():
    @tool
    def t_a2(x: str) -> ToolResult:
        """Top-level shape pin.

        Args:
            x: thing.
        """
        return _ok()

    schema = registry.get("t_a2").schema
    assert set(schema.keys()) == {"name", "description", "input_schema"}
    assert schema["name"] == "t_a2"
    assert schema["description"] == "Top-level shape pin."


def test_A3_input_schema_is_json_schema_object_with_props_and_required():
    @tool
    def t_a3(req_one: str, opt_one: int = 5) -> ToolResult:
        """req + opt.

        Args:
            req_one: required.
            opt_one: optional.
        """
        return _ok()

    s = registry.get("t_a3").schema["input_schema"]
    assert s["type"] == "object"
    assert set(s["properties"].keys()) == {"req_one", "opt_one"}
    # required is exactly the args without defaults — in declaration order.
    assert s["required"] == ["req_one"]


def test_A4_default_value_is_emitted_in_property():
    @tool
    def t_a4(timeout: float = 10.0) -> ToolResult:
        """has default.

        Args:
            timeout: secs.
        """
        return _ok()

    props = registry.get("t_a4").schema["input_schema"]["properties"]
    assert props["timeout"]["default"] == 10.0
    assert "default" not in props.get("timeout_required", {})


def test_A5_type_map_covers_str_int_float_bool_list_dict():
    @tool
    def t_a5(a: str, b: int, c: float, d: bool, e: list, f: dict) -> ToolResult:
        """six types.

        Args:
            a: s.
            b: i.
            c: f.
            d: bo.
            e: l.
            f: d.
        """
        return _ok()

    props = registry.get("t_a5").schema["input_schema"]["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert props["c"]["type"] == "number"
    assert props["d"]["type"] == "boolean"
    assert props["e"]["type"] == "array"
    assert props["f"]["type"] == "object"


def test_A6_parametrized_list_and_dict_generics_resolve_to_array_object():
    @tool
    def t_a6(xs: list[str], ys: dict[str, int]) -> ToolResult:
        """generic.

        Args:
            xs: list.
            ys: dict.
        """
        return _ok()

    props = registry.get("t_a6").schema["input_schema"]["properties"]
    assert props["xs"]["type"] == "array"
    assert props["ys"]["type"] == "object"


def test_A7_description_is_first_non_empty_line_of_docstring():
    @tool
    def t_a7(x: str) -> ToolResult:
        """First line is the description.

        And lots more detail here that should NOT end up as the description.

        Args:
            x: x.
        """
        return _ok()

    assert registry.get("t_a7").schema["description"] == \
        "First line is the description."


def test_A8_param_descriptions_come_from_args_block():
    @tool
    def t_a8(a: str, b: int) -> ToolResult:
        """A8.

        Args:
            a: alpha description here.
            b: beta description here.
        """
        return _ok()

    props = registry.get("t_a8").schema["input_schema"]["properties"]
    assert props["a"]["description"] == "alpha description here."
    assert props["b"]["description"] == "beta description here."


# --- B1-B11: decorator-error negatives ------------------------------------


def test_B1_duplicate_name_raises_valueerror():
    @tool
    def dup_tool(x: str) -> ToolResult:
        """one.

        Args:
            x: x.
        """
        return _ok()

    with pytest.raises(ValueError, match="duplicate name"):
        @tool
        def dup_tool(x: str) -> ToolResult:  # noqa: F811
            """two.

            Args:
                x: x.
            """
            return _ok()


def test_B2_missing_type_hint_raises_typeerror():
    with pytest.raises(TypeError, match="no type hint"):
        @tool
        def t_b2(x) -> ToolResult:  # noqa: ANN001
            """no hint.

            Args:
                x: x.
            """
            return _ok()


class _CustomType:
    """Module-level class for B3 (typing.get_type_hints resolves strings)."""


def test_B3_unsupported_type_raises_typeerror():
    with pytest.raises(TypeError, match="unsupported type"):
        @tool
        def t_b3(x: _CustomType) -> ToolResult:
            """custom.

            Args:
                x: x.
            """
            return _ok()


def test_B4_var_positional_args_raises_typeerror():
    with pytest.raises(TypeError, match=r"\*args"):
        @tool
        def t_b4(*args: str) -> ToolResult:
            """no var positional.

            Args:
                args: nope.
            """
            return _ok()


def test_B5_var_keyword_kwargs_raises_typeerror():
    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        @tool
        def t_b5(**kwargs: str) -> ToolResult:
            """no var keyword.

            Args:
                kwargs: nope.
            """
            return _ok()


def test_B6_missing_docstring_raises_valueerror():
    with pytest.raises(ValueError, match="must have a docstring"):
        @tool
        def t_b6(x: str) -> ToolResult:
            return _ok()


def test_B7_optional_int_via_typing_Union_raises_typeerror():
    """Lead clarified (orchestrator decision): Optional[int] is rejected at
    the decorator. B8 in plan is testing the TypeError raise, not how an
    Optional ends up in the JSON schema (it doesn't — it crashes)."""
    with pytest.raises(TypeError, match="unsupported type"):
        @tool
        def t_b7(x: Optional[int] = None) -> ToolResult:
            """opt.

            Args:
                x: x.
            """
            return _ok()


def test_B8_pep604_union_int_pipe_none_raises_typeerror():
    """PEP-604 `int | None` is a `types.UnionType` (different code path than
    typing.Union). Registry must reject both."""
    with pytest.raises(TypeError, match="unsupported type"):
        @tool
        def t_b8(x: int | None = None) -> ToolResult:
            """pipe.

            Args:
                x: x.
            """
            return _ok()


def test_B9_typing_Union_two_concrete_types_raises_typeerror():
    with pytest.raises(TypeError, match="unsupported type"):
        @tool
        def t_b9(x: Union[int, str] = 0) -> ToolResult:
            """un.

            Args:
                x: x.
            """
            return _ok()


def test_B10_missing_args_block_with_parameters_still_registers():
    """A docstring with no `Args:` block should still register the tool but
    leave parameter descriptions empty (best-effort warn is logged inside
    events.jsonl when EVENTS_PATH exists — separate ratchet R-Reg-7)."""
    @tool
    def t_b10(x: str) -> ToolResult:
        """just a description, no Args."""
        return _ok()

    assert registry.get("t_b10") is not None
    props = registry.get("t_b10").schema["input_schema"]["properties"]
    assert props["x"]["description"] == ""


def test_B11_zero_arg_func_registers_with_empty_props_and_required():
    @tool
    def t_b11() -> ToolResult:
        """zero args."""
        return _ok()

    s = registry.get("t_b11").schema["input_schema"]
    assert s["properties"] == {}
    assert s["required"] == []


# --- E1-E8: provider schemas + execute() contract -------------------------


def test_E1_schemas_for_provider_anthropic_shape():
    """Anthropic puts `name` + `description` + `input_schema` at top level."""
    schemas = registry.schemas_for_provider("anthropic", allow=["shell"])
    assert len(schemas) == 1
    assert set(schemas[0].keys()) == {"name", "description", "input_schema"}
    assert schemas[0]["name"] == "shell"


def test_E2_schemas_for_provider_openai_shape():
    """OpenAI wraps in `{type: function, function: {...}}`. `parameters` not
    `input_schema`. See docs/knowledge/provider-tool-calling.md §1."""
    schemas = registry.schemas_for_provider("openai", allow=["shell"])
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert "function" in schemas[0]
    assert set(schemas[0]["function"].keys()) == {"name", "description", "parameters"}


def test_E3_schemas_for_provider_ollama_shape_matches_openai():
    """Ollama uses the OpenAI shape verbatim."""
    s_open = registry.schemas_for_provider("openai", allow=["shell"])
    s_oll = registry.schemas_for_provider("ollama", allow=["shell"])
    assert s_open == s_oll


def test_E4_schemas_for_provider_allow_none_returns_all_sorted():
    """allow=None returns every registered tool, names sorted alphabetically."""
    all_names = registry.names()
    # The four built-in tools must be present after `import mneme.tools`.
    for expected in ("file_read", "python_exec", "shell", "web_fetch"):
        assert expected in all_names
    schemas = registry.schemas_for_provider("anthropic", allow=None)
    got = [s["name"] for s in schemas]
    assert got == sorted(got)


def test_E5_schemas_for_provider_allow_empty_returns_empty_list():
    """allow=[] is the explicit "no tools" sentinel — distinguishes from
    allow=None (= every tool). Server uses this on the HTTP /chat path."""
    assert registry.schemas_for_provider("anthropic", allow=[]) == []
    assert registry.schemas_for_provider("openai", allow=[]) == []


def test_E6_execute_unknown_tool_returns_unknown_tool_error_block():
    """Never raises. The content prefix is checked by the agent loop
    so the LLM sees a parseable error."""
    tr = registry.execute("nope_no_such_tool", {})
    assert tr.is_error is True
    assert "UnknownTool" in tr.content
    assert "nope_no_such_tool" in tr.content
    assert tr.audit["error"] == "UnknownTool"


def test_E7_execute_missing_required_arg_returns_argument_error():
    tr = registry.execute("shell", {})   # `command` is required
    assert tr.is_error is True
    assert tr.audit["error"] == "ArgumentError"
    assert "command" in tr.content
    assert "is required" in tr.content


def test_E8_execute_extra_undeclared_arg_is_rejected_not_silently_ignored():
    """v0.9 contract pin — registry.py line ~302 says "silently ignored is
    too forgiving"."""
    tr = registry.execute("shell", {"command": "echo ok", "bogus": "x"})
    assert tr.is_error is True
    assert tr.audit["error"] == "ArgumentError"
    assert "bogus" in tr.content
    assert "is not declared" in tr.content


# --- R-Reg-1: get(unknown) returns None, doesn't raise -------------------


def test_R_Reg_1_get_unknown_returns_none():
    assert registry.get("definitely_not_a_tool") is None


# --- R-Reg-2: names() is sorted ------------------------------------------


def test_R_Reg_2_names_returns_sorted_list():
    out = registry.names()
    assert out == sorted(out)


# --- R-Reg-3: ToolEntry.schema dict is the v0.9 contract (no SCHEMA literal)


def test_R_Reg_3_tool_entry_has_dict_schema_field():
    entry = registry.get("shell")
    assert entry is not None
    assert isinstance(entry.schema, dict)
    # No leftover `SCHEMA` literal on the shell module (v0.8 -> v0.9 cleanup).
    from mneme.tools import shell as shell_mod
    assert not hasattr(shell_mod, "SCHEMA")


# --- R-Reg-4: TypeError during decoration leaves _REGISTRY untouched -----


def test_R_Reg_4_failed_decoration_does_not_leave_partial_entry():
    before = set(registry._REGISTRY.keys())
    with pytest.raises(TypeError):
        @tool
        def t_partial(x: int | None = None) -> ToolResult:
            """should fail.

            Args:
                x: x.
            """
            return _ok()
    after = set(registry._REGISTRY.keys())
    assert before == after, "failed @tool left a partial entry in _REGISTRY"


# --- R-Reg-5: allow filter keeps schema order matching `allow` ordering ---


def test_R_Reg_5_allow_filter_preserves_caller_order():
    """When `allow` is a list, schemas come out in the order the caller
    asked for — orchestrator decision: not always alphabetical (that's
    only when allow=None)."""
    schemas = registry.schemas_for_provider(
        "anthropic", allow=["web_fetch", "shell", "file_read"],
    )
    got = [s["name"] for s in schemas]
    assert got == ["web_fetch", "shell", "file_read"]


# --- R-Reg-6 (RECONCILED): unknown provider raises full-string ValueError -


def test_R_Reg_6_unknown_provider_raises_valueerror_full_string():
    """Lead must-fix #2: the literal message including the alphabetical
    provider list is pinned. Don't substring-match; full-string equality."""
    with pytest.raises(ValueError) as excinfo:
        registry.schemas_for_provider("banana", allow=None)
    assert str(excinfo.value) == (
        "schemas_for_provider: unknown provider 'banana'; "
        "expected one of: anthropic, openai, ollama"
    )


# --- R-Reg-7 (nice-to-have): events.append failure does not kill register -


def test_R_Reg_7_events_append_failure_does_not_break_registration(monkeypatch):
    """Lead nice-to-have R-Reg-7: registry.py wraps the tool_registry_warn
    log in a best-effort try/except. Force events.append to raise — the
    @tool MUST still register."""
    # Point EVENTS_PATH to a file that exists so the warn path is exercised,
    # then break events.append.
    from mneme import paths as _paths
    from mneme.trace import events as _events

    # Make a fake existing file so the `ep.exists()` branch is taken.
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(b"")
    tmp.close()

    from pathlib import Path as _P
    monkeypatch.setattr(_paths, "EVENTS_PATH", _P(tmp.name))

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_events, "append", boom)

    @tool
    def t_r_reg_7(x: str) -> ToolResult:
        """no args block — triggers warn path."""
        return _ok()

    # Despite events.append exploding, the tool registered.
    assert registry.get("t_r_reg_7") is not None


# --- R-Reg-X (orchestrator decision): allow=[unknown] silently filters ----


def test_R_Reg_X_allow_unknown_name_silently_filters_to_empty():
    """Orchestrator decision: this is intentional. The agent loop passes
    `allow=[user_subset]` and an unknown name (typo, removed tool) must NOT
    blow up — it just falls out of the returned list. Pin as intent so a
    future refactor that tries to "harden" this into a raise gets caught."""
    schemas = registry.schemas_for_provider(
        "anthropic", allow=["definitely_no_such_tool"],
    )
    assert schemas == []


# --- D7 ratchet (RECONCILED): "got <json-type>" wording -------------------


def test_D7_argument_error_uses_json_schema_type_vocabulary_object():
    tr = registry.execute("shell", {"command": {"k": "v"}})
    assert tr.is_error is True
    assert (
        "ArgumentError: tool 'shell' parameter 'command' must be string, "
        "got object"
    ) in tr.content


def test_D7_argument_error_uses_json_schema_type_vocabulary_integer():
    tr = registry.execute("shell", {"command": 42})
    assert tr.is_error is True
    assert (
        "ArgumentError: tool 'shell' parameter 'command' must be string, "
        "got integer"
    ) in tr.content


def test_D7_argument_error_uses_json_schema_type_vocabulary_array():
    tr = registry.execute("shell", {"command": ["echo", "hi"]})
    assert tr.is_error is True
    assert (
        "ArgumentError: tool 'shell' parameter 'command' must be string, "
        "got array"
    ) in tr.content


def test_D7_argument_error_boolean_not_lumped_with_integer():
    """Special-case in _value_matches: bool is NOT an integer for arg
    validation — Python's `isinstance(True, int)` is True, but the LLM
    should see `boolean` not `integer` in the error."""
    @tool
    def t_d7_bool(n: int) -> ToolResult:
        """want int.

        Args:
            n: n.
        """
        return _ok()
    tr = registry.execute("t_d7_bool", {"n": True})
    assert tr.is_error is True
    assert (
        "ArgumentError: tool 't_d7_bool' parameter 'n' must be integer, "
        "got boolean"
    ) in tr.content


# --- Execute happy path: defaults are applied from Python signature -------


def test_execute_uses_python_default_when_arg_omitted():
    """If the caller omits an optional arg, Python's call semantics apply
    the default — execute() does NOT need to fish defaults out of the
    schema. Verified by calling python_exec with no timeout arg."""
    # We don't actually want to spawn python here; install a stub.
    captured: dict = {}

    @tool
    def t_default(x: int = 7) -> ToolResult:
        """def.

        Args:
            x: x.
        """
        captured["x"] = x
        return _ok()

    tr = registry.execute("t_default", {})
    assert tr.is_error is False
    assert captured["x"] == 7


def test_execute_catches_tool_body_exception_and_reports_type():
    """If a tool body raises (against its 'never raises' contract),
    execute() catches and reports the type — not crash."""
    @tool
    def t_explode(x: str) -> ToolResult:
        """boom.

        Args:
            x: x.
        """
        raise RuntimeError("internal blow-up")

    tr = registry.execute("t_explode", {"x": "hi"})
    assert tr.is_error is True
    assert "RuntimeError" in tr.content
    assert "internal blow-up" in tr.content
    assert tr.audit["error"] == "RuntimeError"
