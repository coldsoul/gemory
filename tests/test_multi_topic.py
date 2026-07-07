"""Tests for multi-topic fact storage — multiple parent_of edges, idempotency."""

import pytest

from src.extractor import store_facts
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_topic_graph(tmp_graph_path):
    """A fresh GraphStore for multi-topic tests."""
    store = GraphStore(tmp_graph_path)
    return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embed(monkeypatch, stub):
    """Patch ``llm.embed`` in both extractor (for store_facts) and topics
    (for resolve_topic)."""
    monkeypatch.setattr("src.llm.embed", stub.embed)
    monkeypatch.setattr("src.topics.embed", stub.embed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiTopicLinking:
    """Multiple topic edges per fact."""

    def test_two_topics_create_two_edges(self, multi_topic_graph, monkeypatch):
        """A fact with two topics gets two parent_of edges, one per topic."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{
            "fact": "The user uses systemd-run for the collector.",
            "topics": ["Sofia transit project", "user infrastructure knowledge"],
        }]
        result = store_facts(facts, "test_source", "test_label", multi_topic_graph)

        assert result["new_nodes"] == 1
        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        assert len(fact_nodes) == 1
        fact_node = fact_nodes[0]

        # Should have 2 parent_of edges (from 2 topic nodes).
        parents = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents) == 2, (
            f"Expected 2 topic parents, got {len(parents)}"
        )

        # Both topic parent nodes should be abstraction nodes.
        for pid in parents:
            tp = multi_topic_graph.get_node(pid)
            assert tp.kind == "abstraction"

    def test_single_topic_creates_one_edge(self, multi_topic_graph, monkeypatch):
        """A fact with one topic gets exactly one parent_of edge."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{"fact": "The user prefers Python.", "topics": ["user profile"]}]
        store_facts(facts, "test_source", "test_label", multi_topic_graph)

        fact_node = multi_topic_graph.all_nodes()[0]
        parents = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents) == 1

    def test_re_store_does_not_duplicate_topic_edges(
        self, multi_topic_graph, monkeypatch,
    ):
        """Storing the same fact twice does not duplicate topic edges (idempotent)."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{
            "fact": "The user uses systemd-run for the collector.",
            "topics": ["Sofia transit project", "user infrastructure knowledge"],
        }]

        # First store.
        store_facts(facts, "test_source", "test_label", multi_topic_graph)
        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        first_parents = multi_topic_graph.get_parents(fact_node.id)

        # Second store — same fact, same source -> should be skipped.
        result2 = store_facts(facts, "test_source", "test_label", multi_topic_graph)
        assert result2["skipped"] == 1

        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        second_parents = multi_topic_graph.get_parents(fact_node.id)
        assert len(second_parents) == len(first_parents)

    def test_corroboration_preserves_topic_links(
        self, multi_topic_graph, monkeypatch,
    ):
        """Corroborating a fact from a different source preserves topic links."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{
            "fact": "The user uses systemd-run for the collector.",
            "topics": ["Sofia transit project", "user infrastructure knowledge"],
        }]

        # First store with source_1.
        store_facts(facts, "source_1", "test_label", multi_topic_graph)
        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        parents_before = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents_before) == 2

        # Second store with source_2 -> corroborate (new source_id).
        result2 = store_facts(facts, "source_2", "test_label", multi_topic_graph)
        assert result2["corroborated"] == 1

        # Still only 1 fact node (no new fact node created).
        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        assert len(fact_nodes) == 1

        fact_node = fact_nodes[0]
        parents_after = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents_after) == len(parents_before)

        # Confidence should have been bumped.
        assert fact_node.confidence == 2.0

    def test_empty_topics_list_no_edges(self, multi_topic_graph, monkeypatch):
        """A fact with empty topics list gets no topic parent edges."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{"fact": "The user likes trees.", "topics": []}]
        store_facts(facts, "test_source", "test_label", multi_topic_graph)

        fact_node = multi_topic_graph.all_nodes()[0]
        parents = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents) == 0

    def test_dual_topic_fact_can_roll_up_to_single_theme(
        self, multi_topic_graph, monkeypatch,
    ):
        """A fact with two topics: both topics become parents, graph is valid."""
        from tests.stubs import HashStub

        stub = HashStub(dim=1536)
        _patch_embed(monkeypatch, stub)

        facts = [{
            "fact": "The user uses systemd-run.",
            "topics": ["Topic A", "Topic B"],
        }]
        store_facts(facts, "test_source", "test_label", multi_topic_graph)

        fact_nodes = [n for n in multi_topic_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        parents = multi_topic_graph.get_parents(fact_node.id)
        assert len(parents) == 2

        # Both topics should have the fact as their child.
        for pid in parents:
            children = multi_topic_graph.get_children(pid)
            assert fact_node.id in children
