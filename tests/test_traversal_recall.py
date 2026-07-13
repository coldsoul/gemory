"""Tests for traversal recall — grouped output, budget, pruning."""

import logging

import pytest

from src.graph import GraphStore
from src.recall import traverse_recall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def traversal_graph(tmp_graph_path):
    """Build a small 2-level graph with 2 roots and facts under them."""
    store = GraphStore(tmp_graph_path)

    # Root A: a topic with 3 facts
    root_a = store.add_node(
        content="Topic A", embedding=[1.0, 0.0],
        provenance={"source_id": "ra", "label": "Topic A", "timestamp": ""},
        kind="abstraction", label="Topic A",
        summary="Summary of Topic A.", reach=3,
    )
    store.set_node_attr(root_a, "level", 1)
    for i in range(3):
        fid = store.add_node(
            content=f"Fact A{i}", embedding=[float(i + 1), 0.0],
            provenance={"source_id": f"fa{i}", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(root_a, fid)

    # Root B: a topic with 2 facts
    root_b = store.add_node(
        content="Topic B", embedding=[0.0, 1.0],
        provenance={"source_id": "rb", "label": "Topic B", "timestamp": ""},
        kind="abstraction", label="Topic B",
        summary="Summary of Topic B.", reach=2,
    )
    store.set_node_attr(root_b, "level", 1)
    for i in range(2):
        fid = store.add_node(
            content=f"Fact B{i}", embedding=[0.0, float(i + 1)],
            provenance={"source_id": f"fb{i}", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(root_b, fid)

    return store, root_a, root_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTraversalRecall:
    """Grouped-output format, budget, and pruning behavior."""

    def test_returns_grouped_by_branch(self, traversal_graph, monkeypatch):
        """Output is grouped by branch with summaries, not a flat list."""
        store, root_a, root_b = traversal_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"], c[1]["id"]],
        )

        text, metrics = traverse_recall("test query", store, top_k=20)

        assert "## Topic A" in text
        assert "## Topic B" in text
        assert "(Summary of Topic A.)" in text
        assert "(Summary of Topic B.)" in text
        assert "Fact A0" in text
        assert "Fact B0" in text
        assert "[#1" not in text

    def test_prune_keeps_selected_branch_only(self, traversal_graph, monkeypatch):
        """Keeping only root A means root B facts do not appear."""
        store, root_a, root_b = traversal_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [c[0]["id"]],  # Keep only root A
        )

        text, metrics = traverse_recall("test", store, top_k=20)
        assert "Fact A0" in text
        assert "Fact A1" in text
        assert "Fact A2" in text
        assert "Fact B0" not in text

    def test_discarded_branch_returns_no_facts(self, traversal_graph, monkeypatch):
        """A discarded (total-pruned) branch contributes zero facts."""
        store, root_a, root_b = traversal_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [],
        )

        text, metrics = traverse_recall("test", store, top_k=20)
        assert "No matching facts" in text or metrics["branches_pruned"] > 0

    def test_budget_triggers_deeper_descent(self, traversal_graph, monkeypatch):
        """With a tight budget, pruning is called again to descend deeper."""
        store, root_a, root_b = traversal_graph

        call_count = [0]

        def counting_prune(q, c):
            call_count[0] += 1
            return [item["id"] for item in c]

        monkeypatch.setattr("src.llm.prune_branches", counting_prune)

        text, metrics = traverse_recall("test", store, top_k=1)
        assert call_count[0] >= 1

    def test_budget_exhaustion_logged(self, traversal_graph, monkeypatch, caplog):
        """When budget is exceeded, a loud warning is logged."""
        store, root_a, root_b = traversal_graph

        monkeypatch.setattr(
            "src.llm.prune_branches",
            lambda q, c: [item["id"] for item in c],
        )

        caplog.set_level(logging.WARNING)
        text, metrics = traverse_recall("test", store, top_k=1)
        assert metrics["budget_exceeded"] is True
        assert any("BUDGET EXCEEDED" in r.message for r in caplog.records)
