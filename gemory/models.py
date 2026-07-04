"""Data model: Node and Edge dataclasses for the memory graph.

All timestamps are ISO-8601 UTC strings.
The graph is DIRECTED (networkx.DiGraph, treated as a DAG).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Represents one atomic fact in the memory graph.

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


@dataclass
class Edge:
    """A directed relationship between two nodes.

    PoC uses exactly "related" as the relation value.
    Other values (parent_of, child_of, contradicts) are reserved for later.
    """

    source: str
    target: str
    weight: float
    relation: str = "related"
