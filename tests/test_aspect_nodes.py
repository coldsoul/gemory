"""Tests for aspect nodes — downward consolidation."""

import pytest

from src import config as cfg
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aspect_graph(tmp_graph_path):
    """Build a graph with one over-large topic (25 children > MAX_NODE_CHILDREN)
    and one thin topic (5 children, below threshold)."""
    store = GraphStore(tmp_graph_path)

    # Topic A: 25 children (exceeds MAX_NODE_CHILDREN=20).
    topic_a = store.add_node(
        content="Topic A", embedding=[1.0, 0.0],
        provenance={
            "source_id": "ta", "label": "Topic A", "timestamp": "",
        },
        kind="abstraction", label="Topic A", summary="Summary A.",
        reach=25,
    )
    store.set_node_attr(topic_a, "level", 1)
    for i in range(25):
        fid = store.add_node(
            content=f"Fact A{i}", embedding=[float(i + 1), 0.0],
            provenance={"source_id": f"fa{i}", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(topic_a, fid)

    # Topic B: 5 children (below threshold -- should not be split).
    topic_b = store.add_node(
        content="Topic B", embedding=[0.0, 1.0],
        provenance={
            "source_id": "tb", "label": "Topic B", "timestamp": "",
        },
        kind="abstraction", label="Topic B", summary="Summary B.",
        reach=5,
    )
    store.set_node_attr(topic_b, "level", 1)
    for i in range(5):
        fid = store.add_node(
            content=f"Fact B{i}", embedding=[0.0, float(i + 1)],
            provenance={"source_id": f"fb{i}", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(topic_b, fid)

    return store, topic_a, topic_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindOverlargeNodes:
    """``_find_overlarge_nodes`` identifies nodes with too many children."""

    def test_identifies_overlarge_node(self, aspect_graph):
        store, topic_a, topic_b = aspect_graph
        from src.dreamer import _find_overlarge_nodes
        candidates = _find_overlarge_nodes(store)
        assert topic_a in candidates

    def test_thin_node_not_identified(self, aspect_graph):
        store, topic_a, topic_b = aspect_graph
        from src.dreamer import _find_overlarge_nodes
        candidates = _find_overlarge_nodes(store)
        assert topic_b not in candidates

    def test_empty_graph_no_candidates(self, tmp_graph_path):
        store = GraphStore(tmp_graph_path)
        from src.dreamer import _find_overlarge_nodes
        candidates = _find_overlarge_nodes(store)
        assert candidates == []
