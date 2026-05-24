"""Tools the agent can call.

M1 ships one tool — `shell` — wired into the agent loop directly. v0.9 / M2
will introduce a `@tool` decorator and registry; until then we keep this
package deliberately bare (PRINCIPLES.md principle 1).
"""
