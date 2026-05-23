"""Soul layer: the agent's identity, loaded from user-local text files.

`blueprint.md` goes verbatim into the stable prompt prefix; `prompts/*.yaml`
are the externalized prompt templates. Identity lives in `~/.mneme/`, NOT in
the repo — the repo only ships `examples/` templates used as a one-time seed
by `mneme init`. Diff your identity in your own private repo, if you want;
the public code never carries it (PRINCIPLES.md principle 4).

This module only *loads* — it never mutates identity.
"""

from __future__ import annotations

import functools
import hashlib

import yaml

from . import paths


@functools.cache
def load_prompt(name: str) -> dict:
    """Load and parse a prompt template by name.

    Prefers the runtime copy at `~/.mneme/prompts/{name}.yaml`; falls back to
    the repo's `examples/prompts/{name}.yaml` so fresh checkouts and tests
    work before `mneme init` has seeded the runtime location.

    Cached because prompts are immutable within a process (no hot-reload per
    PRINCIPLE 2: stable prefix preserves cache hits).
    """
    for candidate in (
        paths.PROMPTS_DIR / f"{name}.yaml",
        paths.EXAMPLE_PROMPTS_DIR / f"{name}.yaml",
    ):
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no prompt {name!r} found; run `mneme init`")


def load_blueprint() -> str:
    """Return the system blueprint text.

    Prefers the user-editable runtime copy (`~/.mneme/blueprint.md`, written
    by `mneme init`); falls back to the repo example so fresh checkouts and
    tests work pre-init.
    """
    for candidate in (paths.BLUEPRINT_PATH, paths.EXAMPLE_BLUEPRINT):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("no blueprint.md found; run `mneme init`")


def prompt_hash(text: str) -> str:
    """Stable 16-hex-char fingerprint of an assembled prompt or response."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
