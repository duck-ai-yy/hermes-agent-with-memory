"""Command-line interface — the primary frontend.

  mneme init | chat | explain | forget | snapshot | blueprint | stats | serve

In-process and ephemeral: a command runs and exits. Only `serve` is long-lived.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import httpx
import typer

from . import paths
from .agent import respond_stream
from .ids import ulid
from .llm import client as _llm
from .llm import pricing
from .llm.client import BudgetExceeded, LLMConfig, configure
from .memory import forget as forget_mod
from .memory import retrieve, store
from .trace import events

app = typer.Typer(add_completion=False, help="Local-first chat agent with memory.")


_ENV_FIELDS = {
    "MNEME_PROVIDER": "provider",
    "MNEME_BASE_URL": "base_url",
    "MNEME_CHAT_MODEL": "chat_model",
    "MNEME_API_KEY": "api_key",
    "MNEME_EMBED_MODEL": "embed_model",
    "MNEME_EMBED_PROVIDER": "embed_provider",
    "MNEME_EMBED_BASE_URL": "embed_base_url",
    "MNEME_EMBED_API_KEY": "embed_api_key",
    "MNEME_EMBED_VIA": "embed_via",
}


def _configure_llm() -> None:
    """Build the singleton LLM client; env vars override Ollama defaults."""
    overrides = {f: v for env, f in _ENV_FIELDS.items() if (v := os.environ.get(env))}
    configure(LLMConfig(
        events_path=paths.EVENTS_PATH,
        daily_token_budget=_budget_from_env(),
        **overrides,
    ))


def _budget_from_env() -> int:
    """Parse MNEME_DAILY_TOKEN_BUDGET; non-int or unset means unlimited."""
    try:
        return int(os.environ.get("MNEME_DAILY_TOKEN_BUDGET", "0"))
    except ValueError:
        return 0


def _open_db():
    if not paths.DB_PATH.exists():
        typer.secho("no database — run `mneme init` first", fg=typer.colors.RED)
        raise typer.Exit(1)
    return store.connect(paths.DB_PATH)


@app.command()
def init() -> None:
    """Create ~/.mneme/ (db, blueprint, prompts, events) and detect Ollama."""
    paths.HOME.mkdir(parents=True, exist_ok=True)
    paths.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    cx = store.connect(paths.DB_PATH)
    store.init_db(cx)
    cx.close()
    typer.echo(f"database  {paths.DB_PATH}")

    if not paths.BLUEPRINT_PATH.exists():
        paths.BLUEPRINT_PATH.write_text(
            paths.EXAMPLE_BLUEPRINT.read_text(encoding="utf-8"), encoding="utf-8"
        )
    typer.echo(f"blueprint {paths.BLUEPRINT_PATH}")

    seeded = 0
    for src in paths.EXAMPLE_PROMPTS_DIR.glob("*.yaml"):
        dst = paths.PROMPTS_DIR / src.name
        if not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            seeded += 1
    typer.echo(f"prompts   {paths.PROMPTS_DIR} ({seeded} seeded)")

    paths.EVENTS_PATH.touch(exist_ok=True)
    typer.echo(f"events    {paths.EVENTS_PATH}")

    cfg = LLMConfig()
    try:
        httpx.get(cfg.base_url, timeout=2.0)
        typer.secho(f"ollama    reachable at {cfg.base_url}", fg=typer.colors.GREEN)
    except Exception:
        typer.secho(
            f"ollama    NOT reachable at {cfg.base_url} — start it before `mneme chat`",
            fg=typer.colors.YELLOW,
        )


@app.command()
def chat() -> None:
    """Interactive REPL. Slash commands: /explain, /forget <id>, /quit."""
    _configure_llm()
    cx = _open_db()
    turn_id = ulid()
    last_trace: str | None = None
    typer.echo("mneme chat — /quit to exit, /forget <id>, /explain [<id>]")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line.startswith("/forget "):
            _forget_interactive(line.split(maxsplit=1)[1].strip(), cx)
            continue
        if line.startswith("/explain"):
            parts = line.split(maxsplit=1)
            tid = parts[1].strip() if len(parts) > 1 else last_trace
            if tid:
                _print_trace(tid)
            else:
                typer.echo("nothing to explain yet")
            continue

        try:
            reply = _stream_to_stdout(line, turn_id, cx)
        except BudgetExceeded as exc:
            typer.secho(f"\nbudget: {exc} — set MNEME_DAILY_TOKEN_BUDGET higher "
                        "or switch to local Ollama", fg=typer.colors.YELLOW)
            continue
        except Exception as exc:  # provider down, etc.
            typer.secho(f"\nerror: {exc}", fg=typer.colors.RED)
            continue
        last_trace = reply.trace_id
        client = _llm.get_client()
        usage = getattr(client, "last_usage", None)
        tokens_part = f" · tokens: {usage.total_tokens}" if usage else ""
        cost_part = _cost_footer(client, usage) if usage else ""
        # `iters` was added to the close-trace in v0.8; surface it in the
        # footer so the user knows how many LLM calls happened this turn.
        iters_part = _iters_footer(reply.trace_id)
        typer.secho(
            f"       [trace {reply.trace_id} · citations: "
            f"{reply.citation_quality}{iters_part}{tokens_part}{cost_part}]",
            fg=typer.colors.BRIGHT_BLACK,
        )
    cx.close()


def _iters_footer(trace_id: str) -> str:
    """Read iters from the just-written close-trace event. Best effort —
    if the events file is unreadable the footer simply omits the field."""
    try:
        record = events.explain(paths.EVENTS_PATH, trace_id)
    except (KeyError, FileNotFoundError, OSError):
        return ""
    iters = record.get("iters")
    return f" · iters: {iters}" if iters else ""


def _cli_confirm_tool(name: str, args: dict) -> bool:
    """Confirmation callback handed to the agent loop.

    Renders the proposed command short-form, then prompts y/N (default N).
    Keeping the prompt one-shot: the model already showed any "I'm going
    to..." text via the intermediate-text callback; this prompt is purely
    the safety gate.
    """
    if name == "shell":
        cmd = args.get("command", "")
        typer.secho(f"       [tool] shell: {cmd}", fg=typer.colors.CYAN)
    else:
        typer.secho(f"       [tool] {name}: {args}", fg=typer.colors.CYAN)
    return typer.confirm("       run this command?", default=False)


def _print_intermediate(text: str) -> None:
    """Render an intermediate assistant message (between tool calls).

    Visually distinct from the final `mneme>` reply so the user can tell
    "what the agent is about to do" from "what the agent finally said".
    The sentinel format is `       …` (5 leading spaces + ellipsis) on the
    first line, matching the indent of the footer line.
    """
    if not text.strip():
        return
    typer.secho(f"       … {text}", fg=typer.colors.BRIGHT_BLACK)


def _cost_footer(client, usage) -> str:
    """Format the ` · $X.XXXX` suffix for the REPL footer.

    Returns `· $?` when the model isn't priced and `· <$0.0001` for sub-cent
    spend — the user picks "unknown" vs "essentially free" at a glance.
    Pricing failures degrade silently to `· $?`; they must not break the REPL.
    """
    try:
        cost = pricing.cost_usd(client.config.provider, client.config.chat_model, usage)
    except Exception:
        cost = None
    if cost is None:
        return " · $?"
    if 0 < cost < 0.0001:
        return " · <$0.0001"
    return f" · ${cost:.4f}"


def _stream_to_stdout(user_text: str, turn_id: str, cx):
    """Drive `respond_stream`, printing chunks live; return the final Reply.

    With v0.8 the agent may run a tool mid-turn. Intermediate assistant text
    ("I'll do X first…") prints on its own indented line BEFORE the confirm
    prompt; the `mneme> ` header is deferred until the final text starts
    streaming so it stays adjacent to the last reply (no awkward empty
    `mneme> ` followed by the tool prompt).
    """
    header_printed = False

    def ensure_header() -> None:
        nonlocal header_printed
        if not header_printed:
            sys.stdout.write("mneme> ")
            sys.stdout.flush()
            header_printed = True

    gen = respond_stream(
        user_text, turn_id, cx,
        confirm_cb=_cli_confirm_tool,
        on_intermediate_text=_print_intermediate,
        # CLI exposes every registered tool — None lets the registry
        # decide (v0.9 ships shell / file_read / web_fetch / python_exec).
        allowed_tools=None,
    )
    reply = None
    while True:
        try:
            chunk = next(gen)
        except StopIteration as stop:
            reply = stop.value
            break
        ensure_header()
        sys.stdout.write(chunk)
        sys.stdout.flush()
    # If the turn produced no streamed chunks (e.g. rejection-only path),
    # still print an empty `mneme>` line so the footer hangs off something.
    ensure_header()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return reply


@app.command()
def explain(trace_id: str) -> None:
    """Dump a trace: prompt, used slices/edges, model, citation quality."""
    _print_trace(trace_id)


@app.command()
def forget(slice_id: str) -> None:
    """Delete a memory after an explicit confirmation prompt."""
    cx = _open_db()
    _forget_interactive(slice_id, cx)
    cx.close()


@app.command()
def snapshot() -> None:
    """SQLite .backup into ~/.mneme/snapshots/."""
    cx = _open_db()
    dest = paths.SNAPSHOTS_DIR / f"db-{int(time.time())}.sqlite"
    store.snapshot(cx, dest)
    cx.close()
    typer.echo(f"snapshot  {dest}")


@app.command()
def blueprint() -> None:
    """Open blueprint.md in $EDITOR."""
    if not paths.BLUEPRINT_PATH.exists():
        typer.secho("no blueprint — run `mneme init` first", fg=typer.colors.RED)
        raise typer.Exit(1)
    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(paths.BLUEPRINT_PATH)], check=False)


@app.command()
def stats() -> None:
    """Print slice / node / edge counts, database size, and lifetime tokens."""
    cx = _open_db()
    for table in ("slices", "nodes", "edges"):
        count = cx.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        typer.echo(f"{table:8} {count}")
    cx.close()
    size = paths.DB_PATH.stat().st_size
    typer.echo(f"db size  {size / 1024:.1f} KiB")
    totals = _token_totals(paths.EVENTS_PATH)
    typer.echo(
        f"tokens   in: {totals['prompt']:,}  out: {totals['completion']:,}  "
        f"total: {totals['total']:,}"
    )
    today_start = events.today_start_ts()
    today = events.sum_cloud_tokens_since(paths.EVENTS_PATH, today_start)
    budget = _budget_from_env()
    if budget > 0:
        typer.echo(f"today    {today:,} / {budget:,} cloud tokens")
    else:
        typer.echo(f"today    {today:,} cloud tokens (no budget)")
    cost, priced, unpriced = _today_cost_breakdown(paths.EVENTS_PATH, today_start)
    typer.echo(
        f"today cost: ${cost:.4f} ({priced} priced, {unpriced} unpriced)"
    )


def _token_totals(events_path) -> dict:
    """Sum real token counts across all trace events. Missing fields = 0."""
    totals = {"prompt": 0, "completion": 0, "total": 0}
    if not events_path.exists():
        return totals
    with open(events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "trace":
                continue
            totals["prompt"] += rec.get("prompt_tokens", 0)
            totals["completion"] += rec.get("completion_tokens", 0)
            totals["total"] += rec.get("total_tokens", 0)
    return totals


def _today_cost_breakdown(events_path, since_ts: int) -> tuple[float, int, int]:
    """Sum today's logged cost_usd and count priced vs unpriced turn-close traces.

    A "priced" trace = the turn-close event carried `cost_usd` (known model).
    "Unpriced" = the turn-close event had `total_tokens` but no `cost_usd`
    (unknown-model path). We sum what was *logged*, never re-pricing from the
    current table — past turns keep their original cost even if prices drift.
    """
    cost = 0.0
    priced = 0
    unpriced = 0
    if not events_path.exists():
        return cost, priced, unpriced
    with open(events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "trace" or rec.get("ts", 0) < since_ts:
                continue
            # Only turn-close events carry total_tokens; the pre-call trace
            # row for the same id has neither tokens nor cost.
            if "total_tokens" not in rec:
                continue
            if "cost_usd" in rec:
                cost += rec["cost_usd"]
                priced += 1
            else:
                unpriced += 1
    return cost, priced, unpriced


@app.command()
def search(
    query: str,
    k: int = typer.Option(10, "-k", "--k"),
    hops: int = 2,
    width: int = 100,
    full: bool = False,
) -> None:
    """Vector + graph search over long-term memory.

    Read-only: never writes slices, vec_slices, or events. May write to
    embeddings_cache for a novel query — that is the v0.3 cache contract,
    not a v0.6 side effect.
    """
    if not query.strip():
        typer.echo("empty query")
        return
    _configure_llm()
    cx = _open_db()
    try:
        hits = retrieve.recall(query, cx, k=k, hops=hops)
    except Exception as exc:
        typer.secho(f"embed provider unreachable: {exc}", fg=typer.colors.RED)
        cx.close()
        raise typer.Exit(1)
    cx.close()
    _print_hits(hits, query, k, hops, width, full)


def _print_hits(hits, query: str, k: int, hops: int, width: int, full: bool) -> None:
    if not hits:
        typer.echo(f'no hits · query="{query}" · k={k}')
        return
    typer.echo(f"{'score':<6} {'id':<26} {'role':<10} {'when':<16} text")
    for s in hits:
        typer.echo(_format_hit(s, width, full))
    typer.echo(f'{len(hits)} hits · query="{query}" · k={k} · hops={hops}')


def _format_hit(s, width: int, full: bool) -> str:
    width = max(1, width)
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.created_at))
    text = s.text.replace("\n", " ⏎ ")
    if not full and len(text) > width:
        text = text[:width] + "…"
    return f"{s.score:.3f}  {s.id:<26} {s.role:<10} {when}  {text}"


@app.command()
def serve(port: int = 7890) -> None:
    """Start the FastAPI server (see server.py)."""
    import uvicorn

    _configure_llm()
    uvicorn.run("mneme.server:app", host="127.0.0.1", port=port)


def _forget_interactive(slice_id: str, cx) -> None:
    if not typer.confirm(f"permanently delete slice {slice_id}?"):
        typer.echo("cancelled")
        return
    try:
        result = forget_mod.forget(slice_id, consent=True, cx=cx)
    except KeyError:
        typer.secho(f"no such slice: {slice_id}", fg=typer.colors.RED)
        return
    typer.echo(
        f"forgot {slice_id} — {result.edges_removed} edge(s), "
        f"{result.vectors_removed} vector(s) removed"
    )


def _print_trace(trace_id: str) -> None:
    try:
        record = events.explain(paths.EVENTS_PATH, trace_id)
    except (KeyError, FileNotFoundError):
        typer.secho(f"no trace: {trace_id}", fg=typer.colors.RED)
        return
    for key, value in record.items():
        typer.echo(f"{key:18} {value}")


if __name__ == "__main__":
    app()
