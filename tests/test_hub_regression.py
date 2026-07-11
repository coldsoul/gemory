"""Regression: ownership facts file under the THING, not user profile."""

import pytest

from src.extractor import store_facts
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _patch_all(monkeypatch, stub):
    """Patch ``llm.embed`` in extract, topics, and llm modules."""
    monkeypatch.setattr("src.llm.embed", stub.embed)
    monkeypatch.setattr("src.topics.embed", stub.embed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOwnershipScoping:
    """Ownership facts file under the THING with a relates_to edge."""

    @pytest.fixture
    def hub_graph(self, tmp_graph_path, monkeypatch):
        from tests.stubs import HashStub
        store = GraphStore(tmp_graph_path)
        stub = HashStub(dim=1536)
        _patch_all(monkeypatch, stub)
        return store

    def test_ownership_fact_files_under_thing(self, hub_graph):
        """'The user has a project called Gemory' -> filed under Gemory topic."""
        facts = [{
            "fact": "The user has a project called Gemory.",
            "topics": ["Gemory"],
            "relates": [{"from": "user profile", "to": "Gemory"}],
        }]
        result = store_facts(facts, "test_source", "test", hub_graph)
        assert result["new_nodes"] == 1

        fact_nodes = [n for n in hub_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        parents = hub_graph.get_parents(fact_node.id)
        assert len(parents) == 1

        parent = hub_graph.get_node(parents[0])
        assert "Gemory" in parent.content or "Gemory" in parent.label

    def test_ownership_fact_creates_relates_to_edge(self, hub_graph):
        """Ownership facts create relates_to edges."""
        facts = [{
            "fact": "The user has a project called Gemory.",
            "topics": ["Gemory"],
            "relates": [{"from": "user profile", "to": "Gemory"}],
        }]
        store_facts(facts, "test_source", "test", hub_graph)

        relate_edges = hub_graph.get_edges_by_relation("relates_to")
        assert len(relate_edges) == 1, (
            f"Expected 1 relates_to edge, got {len(relate_edges)}"
        )

    def test_ownership_not_multi_topic(self, hub_graph):
        """Ownership fact has ONE topic + relation, not two topics."""
        facts = [{
            "fact": "The user created MS Navigator.",
            "topics": ["MS Navigator"],
            "relates": [{"from": "user profile", "to": "MS Navigator"}],
        }]
        store_facts(facts, "test_source", "test", hub_graph)

        fact_nodes = [n for n in hub_graph.all_nodes() if n.kind == "fact"]
        fact_node = fact_nodes[0]
        parents = hub_graph.get_parents(fact_node.id)
        assert len(parents) == 1, (
            f"Ownership fact should have 1 topic parent, got {len(parents)}. "
            f"Multi-topic on ownership recreates the hub."
        )

    def test_profile_stays_clean(self, hub_graph):
        """After storing several ownership facts, each filed under its thing,
        user profile is NOT linked as a parent to them."""
        facts = [
            {"fact": "The user has a project called Gemory.",
             "topics": ["Gemory"],
             "relates": [{"from": "user profile", "to": "Gemory"}]},
            {"fact": "The user created MS Navigator.",
             "topics": ["MS Navigator"],
             "relates": [{"from": "user profile", "to": "MS Navigator"}]},
            {"fact": "The user owns a bald cypress.",
             "topics": ["bald cypress tree"],
             "relates": [{"from": "user profile", "to": "bald cypress tree"}]},
        ]
        store_facts(facts, "test_source", "test", hub_graph)

        # Find the user-profile topic.
        abstraction_nodes = [
            n for n in hub_graph.all_nodes()
            if n.kind == "abstraction"
        ]
        up_nodes = [
            n for n in abstraction_nodes
            if "user profile" in n.content.lower()
            or "user profile" in n.label.lower()
        ]

        if up_nodes:
            up = up_nodes[0]
            children = hub_graph.get_children(up.id)
            assert len(children) == 0, (
                f"User profile has {len(children)} children -- "
                f"ownership facts must be filed under the thing, not user profile"
            )

        # But there should be relates_to edges from user profile to each thing.
        relate_edges = hub_graph.get_edges_by_relation("relates_to")
        assert len(relate_edges) == 3, (
            f"Expected 3 relates_to edges, got {len(relate_edges)}"
        )
