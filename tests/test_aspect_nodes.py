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


class TestDownwardSplit:
    """``_split_node`` splits over-large nodes into aspect sub-nodes."""

    def test_overlarge_node_is_split(self, aspect_graph, monkeypatch):
        """A node with 25 children gets split into aspects."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group of facts."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        created = _split_node(store, topic_a, "test_run")
        assert len(created) > 0

        children_after = store.get_children(topic_a)
        assert len(children_after) > 0

    def test_thin_node_not_split(self, aspect_graph, monkeypatch):
        """A node with 5 children is not split."""
        store, topic_a, topic_b = aspect_graph

        from src.dreamer import _split_node
        created = _split_node(store, topic_b, "test_run")
        assert created == []

    def test_leftovers_stay_as_direct_children(self, tmp_graph_path, monkeypatch):
        """Children that don't cluster stay as direct children (not re-parented
        into a misc aspect)."""
        store = GraphStore(tmp_graph_path)

        parent = store.add_node(
            content="Parent", embedding=[1.0, 0.0],
            provenance={
                "source_id": "p", "label": "Parent", "timestamp": "",
            },
            kind="abstraction", label="Parent", reach=15,
        )
        store.set_node_attr(parent, "level", 1)

        # 5 similar children (will cluster)
        for i in range(5):
            fid = store.add_node(
                content=f"Similar {i}", embedding=[1.0, float(i) * 0.1],
                provenance={"source_id": f"s{i}", "label": "", "timestamp": ""},
            )
            store.add_parent_edge(parent, fid)

        # 10 far-apart children (will not cluster with the similar group)
        for i in range(10):
            fid = store.add_node(
                content=f"Far {i}", embedding=[0.0, 1.0, float(i)],
                provenance={"source_id": f"f{i}", "label": "", "timestamp": ""},
            )
            store.add_parent_edge(parent, fid)

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        created = _split_node(store, parent, "test_run")

        # At least some far children should remain as direct children
        # (not all 15 children got re-parented under aspects)
        children_after = store.get_children(parent)
        # The parent should still have the far children as direct children
        # plus possibly aspect children
        far_direct = [c for c in children_after
                      if "Far" in store.get_node(c).content]
        assert len(far_direct) > 0

    def test_no_misc_aspect_created(self, tmp_graph_path, monkeypatch):
        """A non-theme summarizer result vetoes the aspect creation."""
        store = GraphStore(tmp_graph_path)

        parent = store.add_node(
            content="Parent", embedding=[1.0, 0.0],
            provenance={
                "source_id": "p", "label": "Parent", "timestamp": "",
            },
            kind="abstraction", label="Parent", reach=25,
        )
        store.set_node_attr(parent, "level", 1)
        for i in range(25):
            fid = store.add_node(
                content=f"Fact {i}", embedding=[float(i + 1), float(i + 1)],
                provenance={"source_id": f"f{i}", "label": "", "timestamp": ""},
            )
            store.add_parent_edge(parent, fid)

        # Stub summarizer to return a non-theme result.
        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {
                "label": "Miscellaneous facts",
                "summary": "No common theme emerged.",
            },
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        created = _split_node(store, parent, "test_run")
        assert created == []  # Vetoed


class TestReachAndLevel:
    """Reach and level recomputation after split."""

    def test_split_preserves_reach(self, aspect_graph, monkeypatch):
        """Original node's reach unchanged after split (same leaves)."""
        store, topic_a, topic_b = aspect_graph
        from src.reach import compute_reach

        reach_before = compute_reach(store, [topic_a])
        assert reach_before == 25

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        _split_node(store, topic_a, "test_run")

        reach_after = compute_reach(store, [topic_a])
        assert reach_after == 25  # Same leaves, unchanged

    def test_aspect_level_is_computed(self, aspect_graph, monkeypatch):
        """Aspect's level = 1 + max(child levels)."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        _split_node(store, topic_a, "test_run")

        for child_id in store.get_children(topic_a):
            child = store.get_node(child_id)
            if child.kind == "abstraction":
                assert child.level >= 1
                for grandchild in store.get_children(child_id):
                    assert store.get_node(grandchild).level == 0


class TestIdempotency:
    """Re-running _split_node does not duplicate aspects."""

    def test_rerun_does_not_duplicate_aspects(self, aspect_graph, monkeypatch):
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        first_run = _split_node(store, topic_a, "run1")
        second_run = _split_node(store, topic_a, "run2")

        assert len(second_run) == 0  # No new aspects on re-run

    def test_already_split_node_not_re_split(self, aspect_graph, monkeypatch):
        """A node whose children are within threshold after split is not re-split."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _find_overlarge_nodes, _split_node
        _split_node(store, topic_a, "run1")

        oversize_after = _find_overlarge_nodes(store)
        assert topic_a not in oversize_after


class TestLevelRecomputing:
    def test_level_recomputed_after_splits(self, aspect_graph, monkeypatch):
        """After splits, parent level updated to 1 + max(child)."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        _split_node(store, topic_a, "test_run")

        topic_node = store.get_node(topic_a)
        assert topic_node.level >= 2


class TestNoOrphanEdges:
    def test_reparent_moves_not_copies(self, aspect_graph, monkeypatch):
        """After re-parenting, child has exactly one parent_of parent."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Group", "summary": "A group."},
        )
        monkeypatch.setattr("src.dreamer.embed", lambda x: [1.0, 0.0])

        from src.dreamer import _split_node
        _split_node(store, topic_a, "test_run")

        for child_id in store.get_children(topic_a):
            child = store.get_node(child_id)
            if child.kind == "abstraction":
                for grandchild in store.get_children(child_id):
                    parents = store.get_parents(grandchild)
                    assert len(parents) == 1, (
                        f"Fact {grandchild} has {len(parents)} parents -- "
                        f"should have exactly 1 after re-parenting"
                    )


class TestIdempotencyFullRun:
    """Running the full dreamer twice produces identical graph structure."""

    def test_double_run_is_noop(self, aspect_graph, monkeypatch):
        """Node count, edge count identical after second run."""
        store, topic_a, topic_b = aspect_graph

        monkeypatch.setattr(
            "src.consolidate.summarize_cluster",
            lambda facts, **kw: {"label": "Aspect", "summary": "An aspect."},
        )
        monkeypatch.setattr("src.llm.embed", lambda x: [1.0, 0.0])
        # Prevent new abstractions in upward pass
        monkeypatch.setattr(
            "src.dreamer.cluster_layer",
            lambda graph, node_ids, method="hybrid", seed=42, context=None: [],
        )

        from src.dreamer import _split_node
        from src.reach import backfill_reach
        # Run once
        for nid in [topic_a]:
            _split_node(store, nid, "run1")
        backfill_reach(store)
        
        node_count_1 = len(store.all_nodes())
        edge_count_1 = sum(1 for _ in store.get_edges_by_relation("parent_of"))

        # Run again — should be no-op
        for nid in [topic_a]:
            _split_node(store, nid, "run2")
        backfill_reach(store)

        node_count_2 = len(store.all_nodes())
        edge_count_2 = sum(1 for _ in store.get_edges_by_relation("parent_of"))

        assert node_count_1 == node_count_2, (
            f"Node count changed: {node_count_1} → {node_count_2}"
        )
        assert edge_count_1 == edge_count_2, (
            f"Edge count changed: {edge_count_1} → {edge_count_2}"
        )
