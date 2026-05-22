"""Filesystem locations. One module so the two frontends (cli, server) and the
runtime agree on where things live, instead of hardcoding `~/.mneme/` widely.

Runtime data lives under ~/.mneme/ (never in the repo). The version-controlled
identity — prompts/ and blueprint.md — lives in the repo root next to this
package (`pip install -e .`).
"""

from __future__ import annotations

from pathlib import Path

# Runtime data — created by `mneme init`, excluded from git.
HOME = Path.home() / ".mneme"
DB_PATH = HOME / "db.sqlite"
EVENTS_PATH = HOME / "events.jsonl"
BLUEPRINT_PATH = HOME / "blueprint.md"
SNAPSHOTS_DIR = HOME / "snapshots"

# Version-controlled identity, shipped with the repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROJECT_BLUEPRINT = PROJECT_ROOT / "blueprint.md"
