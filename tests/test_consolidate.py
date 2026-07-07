"""Tests for consolidate.py — cluster_layer and summarize_layer input contracts."""

import pytest

from src.consolidate import cluster_layer, summarize_layer
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gestalt_graph(tmp_graph_path):
    """Build a graph with facts and an abstraction node for testing."""
    store = GraphStore(tmp_graph_path)

    # 4 facts that form a clear cluster
    facts = [
        "The user uses uv for package management.",
        "The user prefers Python for backend work.",
        "The user works with networkx.",
        "The user develops on a VPS.",
    ]
    fact_ids = []
    for f in facts:
        fid = store.add_node(
            content=f, embedding=[1.0, 0.0],
            provenance={"source_id": "test", "label": "", "timestamp": ""},
        )
        fact_ids.append(fid)

    # An abstraction node at level 1 with a summary
    abs_id = store.add_node(
        content="The user uses various coding tools and environments.",
        embedding=[0.9, 0.1],
        provenance={
            "source_id": "dreamer:test", "label": "Coding Tools", "timestamp": "",
        },
        kind="abstraction", label="Coding Tools", reach=4,
    )
    store.set_node_attr(abs_id, "level", 1)
    for fid in fact_ids:
        store.add_parent_edge(abs_id, fid)

    return store, fact_ids, abs_id


# ---------------------------------------------------------------------------
# Tests: cluster_layer
# ---------------------------------------------------------------------------

class TestClusterLayer:
    """``cluster_layer`` wraps ``cluster_nodes`` with gestalt contract."""

    def test_cluster_layer_uses_node_ids(self, gestalt_graph):
        """cluster_layer accepts node IDs and returns clusters."""
        store, fact_ids, _ = gestalt_graph
        clusters = cluster_layer(store, fact_ids)
        # All 4 facts have similar embeddings -> should be clustered
        assert len(clusters) > 0

    def test_cluster_layer_empty_input(self, gestalt_graph):
        """Empty node list returns empty result."""
        store, _, _ = gestalt_graph
        clusters = cluster_layer(store, [])
        assert clusters == []


# ---------------------------------------------------------------------------
# Tests: summarize_layer
# ---------------------------------------------------------------------------

class TestSummarizeLayer:
    """``summarize_layer`` reads from graph, not from pre-extracted strings."""

    def test_summarize_layer_input_is_children_content(
        self, gestalt_graph, monkeypatch,
    ):
        """summarize_layer receives children content, not full transitive facts."""
        store, fact_ids, abs_id = gestalt_graph

        # Capture what summarize_cluster receives
        received = []

        def capture_summarize(contents):
            received.append(contents)
            return {"label": "Test", "summary": "A test summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", capture_summarize)

        # Summarize over the abstraction node's children (facts)
        cluster = set(fact_ids)
        result = summarize_layer(store, cluster)

        assert len(received) == 1
        # It should receive the fact texts (children content)
        contents = received[0]
        assert len(contents) == 4
        assert any("The user uses uv" in c for c in contents)
        assert result["label"] == "Test"
        assert result["summary"] == "A test summary."

    def test_summarize_layer_compound_upward(self, tmp_graph_path, monkeypatch):
        """Above L1, summarize gets children's summaries, not raw facts."""
        store = GraphStore(tmp_graph_path)

        # Create a fact and a level-1 abstraction
        f1 = store.add_node(
            content="Fact 1 detail", embedding=[1.0, 0.0],
            provenance={"source_id": "s1", "label": "", "timestamp": ""},
        )
        a1 = store.add_node(
            content="Topic A summary", embedding=[0.9, 0.1],
            provenance={
                "source_id": "d:test", "label": "Topic A", "timestamp": "",
            },
            kind="abstraction", label="Topic A", reach=1,
        )
        store.set_node_attr(a1, "level", 1)
        store.add_parent_edge(a1, f1)

        # Capture summarize_cluster
        received = []

        def capture(contents):
            received.append(contents)
            return {"label": "Theme", "summary": "Higher summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", capture)

        # Now summarize over the level-1 abstraction (to create level-2 theme)
        result = summarize_layer(store, {a1})

        assert len(received) == 1
        # Should receive the abstraction's content (its summary), not the raw fact
        assert "Topic A summary" in received[0][0]
