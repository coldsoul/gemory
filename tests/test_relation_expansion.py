"""Tests for relation expansion in traversal recall."""

import pytest

from src.graph import GraphStore
from src.recall import traverse_recall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def expansion_graph(tmp_graph_path):
    """Build a graph with relates_to edges for testing expansion."""
    store = GraphStore(tmp_graph_path)

    # Root A (topic) with 2 facts.
    root_a = store.add_node(
        content="Topic A", embedding=[1.0, 0.0],
        provenance={"source_id": "ra", "label": "Topic A", "timestamp": ""},
        kind="abstraction", label="Topic A",
        summary="Summary A.", reach=2,
    )
    store.set_node_attr(root_a, "level", 1)
    for i in range(2):
        fid = store.add_node(
            content=f"Fact A{i}", embedding=[float(i + 1), 0.0],
            provenance={"source_id": f"fa{i}", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(root_a, fid)

    # Root B (topic) with 1 fact + stated relates_to to Root A.
    root_b = store.add_node(
        content="Topic B", embedding=[0.0, 1.0],
        provenance={"source_id": "rb", "label": "Topic B", "timestamp": ""},
        kind="abstraction", label="Topic B",
        summary="Summary B.", reach=1,
    )
    store.set_node_attr(root_b, "level", 1)
    fid = store.add_node(
        content="Fact B0", embedding=[0.0, 1.0],
        provenance={"source_id": "fb0", "label": "", "timestamp": ""},
    )
    store.add_parent_edge(root_b, fid)
    store._graph.add_edge(
        root_b, root_a,
        relation="relates_to", provenance="stated", origin_fact="fb0",
    )

    return store, root_a, root_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRelationExpansion:
    """Relation expansion one-hop from kept branches."""

    def test_expansion_includes_related_node(self, expansion_graph, monkeypatch):
        """When traversal keeps root A, expansion pulls in root B via relates_to."""
        store, root_a, root_b = expansion_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"]],
        )

        text, metrics = traverse_recall(
            "test", store, relation_expansion=True,
        )
        assert "Related context" in text
        assert "Topic B" in text
        assert "Fact B0" in text

    def test_no_expansion_when_flag_off(self, expansion_graph, monkeypatch):
        """With relation_expansion=False, no related context appears."""
        store, root_a, root_b = expansion_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"]],
        )

        text, metrics = traverse_recall(
            "test", store, relation_expansion=False,
        )
        assert "Related context" not in text

    def test_pruner_unchanged_by_expansion(self, expansion_graph, monkeypatch):
        """Expansion flag does not change which branches were pruned."""
        store, root_a, root_b = expansion_graph

        def logging_prune(q, c):
            return [item["id"] for item in c]

        monkeypatch.setattr("src.llm.prune_branches", logging_prune)

        _, metrics_on = traverse_recall("test", store, relation_expansion=True)
        pruned_on = metrics_on["branches_pruned"]

        _, metrics_off = traverse_recall(
            "test", store, relation_expansion=False,
        )
        pruned_off = metrics_off["branches_pruned"]

        assert pruned_on == pruned_off, (
            f"Prune count changed: {pruned_on} vs {pruned_off}"
        )

    def test_derived_edges_not_followed(self, tmp_graph_path, monkeypatch):
        """Derived relates_to edges are NOT followed."""
        store = GraphStore(tmp_graph_path)

        root_a = store.add_node(
            content="A", embedding=[1.0, 0.0],
            provenance={"source_id": "ra", "label": "A", "timestamp": ""},
            kind="abstraction", label="A", summary="A.", reach=1,
        )
        store.set_node_attr(root_a, "level", 1)
        fid_a = store.add_node(
            content="Fact A", embedding=[1.0, 0.0],
            provenance={"source_id": "fa", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(root_a, fid_a)

        root_b = store.add_node(
            content="B", embedding=[0.0, 1.0],
            provenance={"source_id": "rb", "label": "B", "timestamp": ""},
            kind="abstraction", label="B", summary="B.", reach=1,
        )
        store.set_node_attr(root_b, "level", 1)
        fid_b = store.add_node(
            content="Fact B", embedding=[0.0, 1.0],
            provenance={"source_id": "fb", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(root_b, fid_b)

        # DERIVED relates_to edge — must NOT be followed.
        store._graph.add_edge(
            root_a, root_b,
            relation="relates_to", provenance="derived", origin_fact="",
        )

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"]],
        )

        text, _ = traverse_recall("test", store, relation_expansion=True)
        assert "Related context" not in text

    def test_one_hop_not_transitive(self, tmp_graph_path, monkeypatch):
        """A ->rel-> B ->rel-> C: expansion from A reaches B only, not C."""
        store = GraphStore(tmp_graph_path)

        nodes = {}
        for name in ["A", "B", "C"]:
            n = store.add_node(
                content=name, embedding=[1.0, 0.0],
                provenance={
                    "source_id": name, "label": name, "timestamp": "",
                },
                kind="abstraction", label=name, summary=f"Summary {name}.",
                reach=1,
            )
            store.set_node_attr(n, "level", 1)
            f = store.add_node(
                content=f"Fact {name}", embedding=[1.0, 0.0],
                provenance={
                    "source_id": f"f{name}", "label": "", "timestamp": "",
                },
            )
            store.add_parent_edge(n, f)
            nodes[name] = n

        # A ->rel-> B (stated)
        store._graph.add_edge(
            nodes["A"], nodes["B"],
            relation="relates_to", provenance="stated", origin_fact="fA",
        )
        # B ->rel-> C (stated)
        store._graph.add_edge(
            nodes["B"], nodes["C"],
            relation="relates_to", provenance="stated", origin_fact="fB",
        )

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"]],
        )

        text, _ = traverse_recall("test", store, relation_expansion=True)
        assert "Summary B" in text  # B should appear (1 hop)
        assert "Summary C" not in text  # C should NOT appear (2 hops)
