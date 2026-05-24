"""Tools the agent can call.

v0.9 / M2: each tool module registers itself with `registry` via the
`@tool` decorator at import time. Importing this package guarantees all
ships-with-mneme tools are loaded; the agent loop talks only to
`registry.execute` / `registry.schemas_for_provider`.
"""

# Importing the modules triggers their @tool decorators.
from . import shell  # noqa: F401
