"""Tests for relates_to edges — creation, access, reach exclusion."""

import pytest

from src.graph import GraphStore
from src.reach import compute_reach


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def relation_graph(tmp_graph_path):
    """A fresh GraphStore for relation-edge tests."""
    store = GraphStore(tmp_graph_path)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRelatesToEdge:
    """relates_to edge creation, idempotency, filtering, and reach exclusion."""

    def test_add_relates_to_creates_edge(self, relation_graph):
        store = relation_graph
        a = store.add_node(
            "A", [1.0, 0.0],
            {"source_id": "s1", "label": "", "timestamp": ""},
        )
        b = store.add_node(
            "B", [0.0, 1.0],
            {"source_id": "s2", "label": "", "timestamp": ""},
        )

        store.add_relates_to_edge(a, b, origin_fact="test")

        edges = store.get_edges_by_relation("relates_to")
        assert len(edges) == 1
        assert edges[0][0] == a
        assert edges[0][1] == b
        assert edges[0][2]["provenance"] == "stated"
        assert edges[0][2]["origin_fact"] == "test"

    def test_add_relates_to_is_idempotent(self, relation_graph):
        store = relation_graph
        a = store.add_node(
            "A", [1.0, 0.0],
            {"source_id": "s1", "label": "", "timestamp": ""},
        )
        b = store.add_node(
            "B", [0.0, 1.0],
            {"source_id": "s2", "label": "", "timestamp": ""},
        )

        store.add_relates_to_edge(a, b)
        store.add_relates_to_edge(a, b)  # second call
        store.add_relates_to_edge(a, b)  # third call

        edges = store.get_edges_by_relation("relates_to")
        assert len(edges) == 1  # No duplicates

    def test_get_edges_by_relation_filters_correctly(self, relation_graph):
        store = relation_graph
        a = store.add_node(
            "A", [1.0, 0.0],
            {"source_id": "s1", "label": "", "timestamp": ""},
        )
        b = store.add_node(
            "B", [0.0, 1.0],
            {"source_id": "s2", "label": "", "timestamp": ""},
        )
        c = store.add_node(
            "C", [0.0, 0.0],
            {"source_id": "s3", "label": "", "timestamp": ""},
        )

        store.add_parent_edge(a, b)
        store.add_relates_to_edge(b, c)

        parent_edges = store.get_edges_by_relation("parent_of")
        relate_edges = store.get_edges_by_relation("relates_to")

        assert len(parent_edges) == 1
        assert len(relate_edges) == 1
        assert parent_edges[0][0] == a
        assert relate_edges[0][0] == b

    def test_reach_excludes_relates_to(self, relation_graph):
        """relates_to edges must NEVER contribute to reach."""
        store = relation_graph

        a = store.add_node(
            "Topic A", [1.0, 0.0],
            {"source_id": "t1", "label": "Topic A", "timestamp": ""},
            kind="abstraction",
        )
        store.set_node_attr(a, "level", 1)

        fact_ids = []
        for i in range(3):
            fid = store.add_node(
                f"Fact {i}", [float(i), 0.0],
                {"source_id": f"f{i}", "label": "", "timestamp": ""},
            )
            store.add_parent_edge(a, fid)
            fact_ids.append(fid)

        b = store.add_node(
            "Topic B", [0.0, 1.0],
            {"source_id": "t2", "label": "Topic B", "timestamp": ""},
            kind="abstraction",
        )
        store.set_node_attr(b, "level", 1)
        store.add_relates_to_edge(a, b)

        # B's reach should be 0 (relates_to is not containment).
        assert compute_reach(store, [b]) == 0

        # A's reach should be 3 (only parent_of children).
        assert compute_reach(store, [a]) == 3

    def test_get_parents_excludes_relates_to(self, relation_graph):
        """get_parents only returns parent_of parents, not relates_to."""
        store = relation_graph

        child = store.add_node(
            "Child", [1.0, 0.0],
            {"source_id": "c1", "label": "", "timestamp": ""},
        )
        parent = store.add_node(
            "Parent", [0.9, 0.1],
            {"source_id": "p1", "label": "", "timestamp": ""},
            kind="abstraction",
        )
        store.set_node_attr(parent, "level", 1)
        relater = store.add_node(
            "Relater", [0.0, 1.0],
            {"source_id": "r1", "label": "", "timestamp": ""},
        )

        store.add_parent_edge(parent, child)
        store.add_relates_to_edge(relater, child)

        parents = store.get_parents(child)
        assert len(parents) == 1
        assert parents[0] == parent  # Only parent_of parent, not relater
