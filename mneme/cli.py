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
        usage = getattr(_llm.get_client(), "last_usage", None)
        tokens_part = f" · tokens: {usage.total_tokens}" if usage else ""
        typer.secho(
            f"       [trace {reply.trace_id} · citations: "
            f"{reply.citation_quality}{tokens_part}]",
            fg=typer.colors.BRIGHT_BLACK,
        )
    cx.close()


def _stream_to_stdout(user_text: str, turn_id: str, cx):
    """Drive `respond_stream`, printing chunks live; return the final Reply."""
    sys.stdout.write("mneme> ")
    sys.stdout.flush()
    gen = respond_stream(user_text, turn_id, cx)
    reply = None
    while True:
        try:
            chunk = next(gen)
        except StopIteration as stop:
            reply = stop.value
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()
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
    today = events.sum_cloud_tokens_since(paths.EVENTS_PATH, events.today_start_ts())
    budget = _budget_from_env()
    if budget > 0:
        typer.echo(f"today    {today:,} / {budget:,} cloud tokens")
    else:
        typer.echo(f"today    {today:,} cloud tokens (no budget)")


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
