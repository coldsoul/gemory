"""Graph-diff helper: snapshot -> act -> diff, for deterministic store assertions."""

from dataclasses import dataclass, field
from typing import Any

from gemory.graph import GraphStore


@dataclass
class GraphSnapshot:
    """Immutable snapshot of a GraphStore at a point in time."""

    node_contents: dict[str, str]
    node_confidences: dict[str, float]
    node_provenance: dict[str, list[dict]]
    edge_set: set[tuple[str, str, str]]  # (source, target, relation)
    embedding_count: int


@dataclass
class GraphDiff:
    """Delta between two graph snapshots."""

    nodes_added: list[tuple[str, str]] = field(default_factory=list)       # (id, content)
    nodes_removed: list[tuple[str, str]] = field(default_factory=list)     # (id, content)
    provenance_added: list[tuple[str, str]] = field(default_factory=list)  # (node_id, source_id)
    confidence_changes: list[tuple[str, float, float]] = field(default_factory=list)  # (id, old, new)
    edges_added: list[tuple[str, str, str]] = field(default_factory=list)  # (source, target, relation)
    edges_removed: list[tuple[str, str, str]] = field(default_factory=list)  # (source, target, relation)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.nodes_added,
            self.nodes_removed,
            self.provenance_added,
            self.confidence_changes,
            self.edges_added,
            self.edges_removed,
        ])

    def summary(self) -> str:
        lines: list[str] = []
        if self.nodes_added:
            lines.append(f"Nodes added ({len(self.nodes_added)}):")
            for nid, content in self.nodes_added:
                lines.append(f"  + {nid[:8]}: {content[:60]}")
        if self.nodes_removed:
            lines.append(f"Nodes removed ({len(self.nodes_removed)}):")
            for nid, content in self.nodes_removed:
                lines.append(f"  - {nid[:8]}: {content[:60]}")
        if self.provenance_added:
            lines.append(f"Provenance added ({len(self.provenance_added)}):")
            for nid, src in self.provenance_added:
                lines.append(f"  ~ {nid[:8]} source={src}")
        if self.confidence_changes:
            lines.append(f"Confidence changes ({len(self.confidence_changes)}):")
            for nid, old, new in self.confidence_changes:
                lines.append(f"  ~ {nid[:8]}: {old:.1f} -> {new:.1f}")
        if self.edges_added:
            lines.append(f"Edges added ({len(self.edges_added)}):")
            for s, t, r in self.edges_added:
                lines.append(f"  + {s[:8]} -> {t[:8]} [{r}]")
        if self.edges_removed:
            lines.append(f"Edges removed ({len(self.edges_removed)}):")
            for s, t, r in self.edges_removed:
                lines.append(f"  - {s[:8]} -> {t[:8]} [{r}]")
        if not lines:
            lines.append("(no changes)")
        return "\n".join(lines)


def snapshot(graph: GraphStore) -> GraphSnapshot:
    """Capture the current state of *graph* as an immutable snapshot."""
    nodes = graph.all_nodes()

    node_contents = {n.id: n.content for n in nodes}
    node_confidences = {n.id: n.confidence for n in nodes}
    node_provenance = {n.id: list(n.provenance) for n in nodes}

    edge_set: set[tuple[str, str, str]] = set()
    for n in nodes:
        for neighbor in graph.get_neighbors(n.id):
            # GraphStore edges carry weight + relation, but for diff purposes we
            # only track the existence of a directed edge with its relation.
            edge_set.add((n.id, neighbor, "related"))

    embedding_count = (
        len(graph._embeddings) if hasattr(graph, "_embeddings") else 0
    )

    return GraphSnapshot(
        node_contents=node_contents,
        node_confidences=node_confidences,
        node_provenance=node_provenance,
        edge_set=edge_set,
        embedding_count=embedding_count,
    )


def diff(before: GraphSnapshot, after: GraphSnapshot) -> GraphDiff:
    """Compute the delta between two snapshots."""
    d = GraphDiff()

    # Nodes added
    for nid, content in after.node_contents.items():
        if nid not in before.node_contents:
            d.nodes_added.append((nid, content))

    # Nodes removed
    for nid, content in before.node_contents.items():
        if nid not in after.node_contents:
            d.nodes_removed.append((nid, content))

    # Provenance added (only for nodes present in both)
    for nid in after.node_provenance:
        if nid in before.node_provenance:
            before_srcs = {p["source_id"] for p in before.node_provenance[nid]}
            for p in after.node_provenance[nid]:
                if p["source_id"] not in before_srcs:
                    d.provenance_added.append((nid, p["source_id"]))

    # Confidence changes
    for nid in after.node_confidences:
        if nid in before.node_confidences:
            old_c = before.node_confidences[nid]
            new_c = after.node_confidences[nid]
            if abs(old_c - new_c) > 1e-9:
                d.confidence_changes.append((nid, old_c, new_c))

    # Edges added / removed
    for edge in after.edge_set:
        if edge not in before.edge_set:
            d.edges_added.append(edge)
    for edge in before.edge_set:
        if edge not in after.edge_set:
            d.edges_removed.append(edge)

    return d
