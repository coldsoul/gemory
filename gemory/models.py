"""Data model: Node and Edge dataclasses for the memory graph.

All timestamps are ISO-8601 UTC strings.
The graph is DIRECTED (networkx.DiGraph, treated as a DAG).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Represents one node in the memory graph.

    Embeddings are NOT stored on the node — they live in a sidecar
    file (embeddings.json) keyed by node id.
    """

    id: str
    content: str
    confidence: float
    provenance: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    level: int = 0  # Computed/derived, not hand-set. PoC leaves are all 0.
    kind: str = "fact"                    # "fact" (leaf, extracted) or "abstraction" (dreamer-created)
    label: str = ""                       # Short theme label for abstraction nodes; empty for facts
    abstraction_kind: str = ""            # "" (not an abstraction), "topic" (level-1 topic), "theme" (higher-level)


@dataclass
class Edge:
    """A directed relationship between two nodes.

    Relation values:
    - "related": symmetric association (store-time edge creation)
    - "parent_of": hierarchy edge (abstraction -> member)
    Other values (child_of, contradicts) are reserved for later.
    """

    source: str
    target: str
    weight: float
    relation: str = "related"
