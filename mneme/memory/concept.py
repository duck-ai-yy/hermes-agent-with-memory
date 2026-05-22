"""Concept extraction: text -> {nodes, edges} via one LLM call.

Uses prompts/concept_extract.yaml. Edge/node kinds are a CLOSED set; anything
outside it is dropped (a free-form type space would break retrieval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm import client as _llm
from ..soul import load_prompt


class ConceptExtractionFailed(Exception):
    """Raised when the LLM output cannot be parsed or violates the schema."""


@dataclass(frozen=True)
class Node:
    name: str
    kind: str


@dataclass(frozen=True)
class Edge:
    src: str
    type: str
    dst: str


@dataclass(frozen=True)
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def _parse_json(raw: str) -> dict:
    """Parse strict JSON, tolerating a ```json fenced wrapper."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConceptExtractionFailed(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConceptExtractionFailed("top-level JSON is not an object")
    return data


def extract(text: str) -> Graph:
    """Extract a concept graph from `text`."""
    spec = load_prompt("concept_extract")
    allowed_kinds = set(spec["allowed_node_kinds"])
    allowed_types = set(spec["allowed_edge_types"])

    messages = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": text},
    ]
    try:
        raw = _llm.get_client().chat(messages, stream=False)
    except Exception as exc:  # network / provider failure
        raise ConceptExtractionFailed(f"LLM call failed: {exc}") from exc

    data = _parse_json(raw)

    nodes = [
        Node(n["name"], n["kind"])
        for n in data.get("nodes", [])
        if isinstance(n, dict) and n.get("name") and n.get("kind") in allowed_kinds
    ]
    names = {n.name for n in nodes}
    edges = [
        Edge(e["src"], e["type"], e["dst"])
        for e in data.get("edges", [])
        if isinstance(e, dict)
        and e.get("type") in allowed_types
        and e.get("src") in names
        and e.get("dst") in names
    ]
    return Graph(nodes, edges)
