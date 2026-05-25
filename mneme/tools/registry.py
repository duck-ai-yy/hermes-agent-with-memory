"""Tool registry — `@tool` decorator + central lookup.

v0.9 / M2: replace v0.8's hardcoded `shell.SCHEMA` + agent dispatch with a
single source of truth. Each tool decorates a plain Python function; the
decorator inspects type hints + docstring to build the JSON schema sent to
the LLM, registers a `ToolEntry` keyed by function name, and the agent
loop calls `execute(name, args)` to run it.

PRINCIPLES.md principle 1: no ToolProvider base class, no env-var
allowlist, no future-MCP hook. Just a dict + a decorator + one execute().
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable


# --- Public dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """Result of one tool invocation.

    `content` is the LLM-facing string (already formatted, already clipped).
    `audit` is a per-tool dict of structured fields the agent layer merges
    into the `tool_result` event — shape is each tool's own business.
    """
    content: str
    is_error: bool
    audit: dict


@dataclass(frozen=True)
class ToolEntry:
    """One registered tool. `format_result` is currently unused (each tool
    formats inline) but kept on the dataclass so future tools that want to
    return a structured value + late-format have a slot."""
    name: str
    fn: Callable[..., ToolResult]
    schema: dict
    format_result: Callable[[Any], str] | None = None


# --- Registry state --------------------------------------------------------

_REGISTRY: dict[str, ToolEntry] = {}


# --- Type mapping (Spike 1) ------------------------------------------------

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON-schema type string.

    v0.9 rejects Optional / Union (Spike 1 — no v0.9 tool needs them).
    """
    # typing.get_origin handles list[str], dict[str, int], etc.
    origin = typing.get_origin(annotation)
    if origin is not None:
        # Optional[X] / X | None / Union[A, B] -> origin is Union or types.UnionType
        if origin is typing.Union:
            raise TypeError(f"unsupported type {annotation!r}")
        # Generic containers: list[...], dict[...]
        if origin in _TYPE_MAP:
            return _TYPE_MAP[origin]
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    # types.UnionType (PEP 604 `X | Y`) — not a typing.Union origin
    import types as _types
    if isinstance(annotation, _types.UnionType):
        raise TypeError(f"unsupported type {annotation!r}")
    raise TypeError(f"unsupported type {annotation!r}")


# --- Docstring parsing -----------------------------------------------------


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (description, {param_name: description}).

    Format: first non-empty line is the description. An optional `Args:`
    block has lines like `    name: text`; continuation lines (more indented
    than the param line) append to the most recent param description.

    No Args block -> empty dict (caller may log a tool_registry_warn).
    """
    lines = doc.splitlines()
    # First non-empty line
    description = ""
    for ln in lines:
        s = ln.strip()
        if s:
            description = s
            break

    # Find Args: block
    params: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    arg_line_indent: int | None = None
    for ln in lines:
        stripped = ln.strip()
        if not in_args:
            if stripped.lower().startswith("args:"):
                in_args = True
            continue
        if not stripped:
            # blank line ends the Args block conservatively when followed by
            # an unindented section — but a blank line *within* args is fine.
            # Keep iterating; only an obvious new section header ends it.
            continue
        # Determine if this is a new param ("name: text") or continuation.
        indent = len(ln) - len(ln.lstrip())
        if arg_line_indent is None:
            arg_line_indent = indent
        # If indent dropped below the first arg-line indent, we've left the block.
        if indent < arg_line_indent and stripped and not stripped.startswith(("-", "*")):
            in_args = False
            current_name = None
            continue
        # New param line? Pattern: `name: text` at the arg-line indent.
        if indent == arg_line_indent and ":" in stripped:
            name, _, text = stripped.partition(":")
            name = name.strip()
            text = text.strip()
            # Sanity: a real param name is an identifier.
            if name.isidentifier():
                params[name] = text
                current_name = name
                continue
        # Continuation: append (more indented than the arg line) to current.
        if current_name is not None and indent > arg_line_indent:
            existing = params.get(current_name, "")
            sep = " " if existing else ""
            params[current_name] = f"{existing}{sep}{stripped}"
    return description, params


# --- The decorator ---------------------------------------------------------


def tool(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Register `fn` as a tool. fn.__name__ is the tool name.

    Errors raise at import time (not silently swallowed):
      - duplicate name        -> ValueError
      - missing type hint     -> TypeError
      - unsupported type      -> TypeError
      - missing docstring     -> ValueError
      - *args / **kwargs      -> TypeError
    """
    name = fn.__name__
    if name in _REGISTRY:
        raise ValueError(f"@tool: duplicate name {name!r} (already registered)")

    doc = inspect.getdoc(fn)
    if not doc:
        raise ValueError(f"@tool {name!r}: function must have a docstring")

    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    description, param_docs = _parse_docstring(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"@tool {name!r}: *args/**kwargs not supported")
        if pname not in hints:
            raise TypeError(f"@tool {name!r}: parameter {pname!r} has no type hint")
        try:
            jtype = _json_type(hints[pname])
        except TypeError:
            raise TypeError(
                f"@tool {name!r}: parameter {pname!r} has unsupported type {hints[pname]!r}"
            ) from None
        prop: dict = {"type": jtype, "description": param_docs.get(pname, "")}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = param.default
        properties[pname] = prop

    schema = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    _REGISTRY[name] = ToolEntry(name=name, fn=fn, schema=schema)

    # If the docstring had no `Args:` block but the function takes parameters,
    # emit a best-effort warning event so reviewers can see the missing block.
    if param_docs == {} and len(properties) > 0:
        try:
            from .. import paths as _paths
            from ..trace import events as _events
            ep = getattr(_paths, "EVENTS_PATH", None)
            if ep is not None and ep.exists():
                _events.append(ep, kind="tool_registry_warn", tool=name,
                               reason="missing_args_block")
        except Exception:
            # best-effort only — registry must not blow up at import time
            pass

    return fn


# --- Public API ------------------------------------------------------------


def get(name: str) -> ToolEntry | None:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return sorted(_REGISTRY.keys())


def schemas_for_provider(
    provider: str, allow: list[str] | None = None,
) -> list[dict]:
    """Return tool schemas in the wire shape each provider expects.

    `allow` filters the registered tools by name; None = all, [] = none.
    See docs/knowledge/provider-tool-calling.md §1.
    """
    selected: list[ToolEntry]
    if allow is None:
        selected = [_REGISTRY[n] for n in sorted(_REGISTRY.keys())]
    else:
        selected = [_REGISTRY[n] for n in allow if n in _REGISTRY]

    if provider == "anthropic":
        return [{
            "name": t.schema["name"],
            "description": t.schema["description"],
            "input_schema": t.schema["input_schema"],
        } for t in selected]
    if provider in ("openai", "ollama"):
        return [{
            "type": "function",
            "function": {
                "name": t.schema["name"],
                "description": t.schema["description"],
                "parameters": t.schema["input_schema"],
            },
        } for t in selected]
    raise ValueError(
        f"schemas_for_provider: unknown provider {provider!r}; "
        f"expected one of: anthropic, openai, ollama"
    )


def execute(name: str, args: dict) -> ToolResult:
    """Dispatch by name. NEVER raises.

    - UnknownTool: name not in registry
    - ArgumentError: missing required arg or wrong type
    Any exception escaping the tool body is caught and wrapped — but each
    tool is also expected to be "never raises" by its own contract.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        msg = f"UnknownTool: no tool named {name!r} is registered"
        return ToolResult(content=msg, is_error=True,
                          audit={"error": "UnknownTool"})

    schema_props = entry.schema["input_schema"]["properties"]
    required = entry.schema["input_schema"]["required"]

    # Validate required + types against schema, then call by **kwargs.
    call_kwargs: dict = {}
    for pname in required:
        if pname not in args:
            msg = (f"ArgumentError: tool {name!r} parameter {pname!r} "
                   f"is required")
            return ToolResult(content=msg, is_error=True,
                              audit={"error": "ArgumentError"})

    for pname, value in args.items():
        if pname not in schema_props:
            # extra args silently ignored is too forgiving — pin as error.
            msg = (f"ArgumentError: tool {name!r} parameter {pname!r} "
                   f"is not declared")
            return ToolResult(content=msg, is_error=True,
                              audit={"error": "ArgumentError"})
        expected = schema_props[pname]["type"]
        if not _value_matches(value, expected):
            actual = _json_type_of(value)
            msg = (f"ArgumentError: tool {name!r} parameter {pname!r} "
                   f"must be {expected}, got {actual}")
            return ToolResult(content=msg, is_error=True,
                              audit={"error": "ArgumentError"})
        call_kwargs[pname] = value

    # Defaults are applied by Python's call semantics; just call.
    try:
        return entry.fn(**call_kwargs)
    except Exception as exc:
        # Tools must be "never raises"; if one leaks, surface the type.
        msg = f"{type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"error": type(exc).__name__})


_JSON_TYPE_PY = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_type_of(value: Any) -> str:
    """Python value -> JSON-schema type name. The error message uses this so
    the LLM sees the same vocabulary as the schema (`string`/`object` etc.),
    not Python repr (`str`/`dict`). See docs/lessons/architect.md v0.8 lesson
    on pinning user-visible literals."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _value_matches(value: Any, json_type: str) -> bool:
    """JSON-schema-style runtime type check.

    Special-case bool vs integer: in JSON, `true` is NOT an integer (Python's
    `isinstance(True, int)` is True, but for tool args we want strict).
    """
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _JSON_TYPE_PY.get(json_type)
    if py is None:
        return True  # unknown — be lenient
    return isinstance(value, py)
