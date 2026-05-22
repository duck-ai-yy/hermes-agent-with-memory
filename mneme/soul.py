"""Soul layer: the agent's identity, loaded from version-controlled text.

`blueprint.md` goes verbatim into the stable prompt prefix; `prompts/*.yaml`
are the externalized prompt templates. Keeping identity in files rather than
in `.py` is what makes it diffable and recoverable (PRINCIPLES.md principle 5).

This module only *loads* — it never mutates identity.
"""

from __future__ import annotations

import hashlib

import yaml

from . import paths


def load_prompt(name: str) -> dict:
    """Load and parse `prompts/{name}.yaml`."""
    return yaml.safe_load((paths.PROMPTS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def load_blueprint() -> str:
    """Return the system blueprint text.

    Prefers the user-editable runtime copy (`~/.mneme/blueprint.md`, written by
    `mneme init`); falls back to the repo copy so the agent works pre-init.
    """
    for candidate in (paths.BLUEPRINT_PATH, paths.PROJECT_BLUEPRINT):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("no blueprint.md found; run `mneme init`")


def prompt_hash(text: str) -> str:
    """Stable 16-hex-char fingerprint of an assembled prompt or response."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
