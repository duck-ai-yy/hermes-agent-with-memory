"""Filesystem locations. One module so frontends and the runtime agree on
where things live, instead of hardcoding paths everywhere.

Identity (prompts + blueprint) is **not** in the repo — only template copies
are. `mneme init` seeds the runtime location from the templates; from then on
the user owns identity privately under `~/.mneme/`.
"""

from __future__ import annotations

from pathlib import Path

# Runtime data — created by `mneme init`, excluded from git.
HOME = Path.home() / ".mneme"
DB_PATH = HOME / "db.sqlite"
EVENTS_PATH = HOME / "events.jsonl"
BLUEPRINT_PATH = HOME / "blueprint.md"
PROMPTS_DIR = HOME / "prompts"
SNAPSHOTS_DIR = HOME / "snapshots"

# Templates shipped with the repo — read-only fallback when the runtime copy
# isn't seeded yet (fresh checkout, tests). Real users edit the runtime copies.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
EXAMPLE_PROMPTS_DIR = EXAMPLES_DIR / "prompts"
EXAMPLE_BLUEPRINT = EXAMPLES_DIR / "blueprint.md"
