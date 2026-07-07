"""Tests for reach computation — transitive leaf count."""

import pytest

from src.graph import GraphStore
from src.reach import compute_reach, update_reach


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reach_graph(tmp_graph_path):
    """Build a small hierarchy for testing reach.

    * 6 level-0 fact nodes (F0-F5).
    * Topic A (level 1) with children F0, F1, F2.
    * Topic B (level 1) with children F3, F4, F5.
    """
    store = GraphStore(tmp_graph_path)
    facts = []
    for i in range(6):
        fid = store.add_node(
            content=f"Fact {i}",
            embedding=[1.0, 0.0],
            provenance={"source_id": f"src{i}", "label": "", "timestamp": ""},
        )
        facts.append(fid)

    topic_a = store.add_node(
        content="Topic A", embedding=[1.0, 0.0],
        provenance={"source_id": "dreamer:test", "label": "Topic A", "timestamp": ""},
        kind="abstraction",
    )
    store.set_node_attr(topic_a, "level", 1)
    for fid in facts[:3]:
        store.add_parent_edge(topic_a, fid)

    topic_b = store.add_node(
        content="Topic B", embedding=[0.0, 1.0],
        provenance={"source_id": "dreamer:test", "label": "Topic B", "timestamp": ""},
        kind="abstraction",
    )
    store.set_node_attr(topic_b, "level", 1)
    for fid in facts[3:]:
        store.add_parent_edge(topic_b, fid)

    return store, facts, topic_a, topic_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReach:
    """``compute_reach`` and ``update_reach`` correctness."""

    def test_reach_of_single_fact_is_one(self, reach_graph):
        store, facts, _, _ = reach_graph
        assert compute_reach(store, [facts[0]]) == 1

    def test_reach_of_topic_with_three_children_is_three(self, reach_graph):
        store, _, topic_a, _ = reach_graph
        assert compute_reach(store, [topic_a]) == 3

    def test_reach_of_multiple_nodes_is_union(self, reach_graph):
        store, _, topic_a, topic_b = reach_graph
        assert compute_reach(store, [topic_a, topic_b]) == 6

    def test_reach_of_empty_list_is_zero(self, reach_graph):
        store, *_ = reach_graph
        assert compute_reach(store, []) == 0

    def test_reach_avoids_double_counting(self, reach_graph):
        store, facts, _, _ = reach_graph
        # Same fact id listed twice -> still 1
        assert compute_reach(store, [facts[0], facts[0]]) == 1

    def test_update_reach_stores_value(self, reach_graph):
        store, _, topic_a, _ = reach_graph
        r = update_reach(store, topic_a)
        assert r == 3
        assert store.get_node(topic_a).reach == 3

    def test_thin_but_coherent_case(self, tmp_graph_path):
        """Three 2-fact topics have combined reach 6 >= MIN_REACH.

        Each topic individually: reach = 2 (< 5, not worth abstracting).
        Combined across 3 topics: reach = 6 (>= 5, worth a theme).
        """
        from src import config

        store = GraphStore(tmp_graph_path)
        topics = []
        for t in range(3):
            tid = store.add_node(
                content=f"Topic {t}", embedding=[1.0, 0.0],
                provenance={"source_id": f"t{t}", "label": "", "timestamp": ""},
                kind="abstraction",
            )
            store.set_node_attr(tid, "level", 1)
            for f in range(2):
                fid = store.add_node(
                    content=f"Fact {t}-{f}", embedding=[1.0, 0.0],
                    provenance={"source_id": f"f{t}{f}", "label": "", "timestamp": ""},
                )
                store.add_parent_edge(tid, fid)
            topics.append(tid)

        # Each topic individually: reach = 2 (< MIN_REACH=5)
        for tid in topics:
            assert compute_reach(store, [tid]) == 2

        # Combined: reach = 6 (>= MIN_REACH=5, worth a theme!)
        assert compute_reach(store, topics) >= config.MIN_REACH
        assert compute_reach(store, topics) == 6
