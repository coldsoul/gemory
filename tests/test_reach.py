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

    def test_multi_parent_reach_is_distinct_union(self, tmp_graph_path):
        """A fact with two topic parents contributes 1 to combined reach, not 2.

        Setup:
        - 1 fact node (F1)
        - 2 topic nodes (T1, T2) — both parent_of F1
        - 1 theme node (Th1) — parent_of both T1 and T2

        Expected: reach(Th1) = 1, reach(T1) = 1, reach(T2) = 1
        Union reach of {T1, T2} = 1 (not 2)
        """
        store = GraphStore(tmp_graph_path)

        # Create fact F1.
        f1 = store.add_node(
            content="The user uses systemd-run for the collector.",
            embedding=[1.0, 0.0],
            provenance={"source_id": "src1", "label": "", "timestamp": ""},
        )

        # Create topic T1, parent_of F1.
        t1 = store.add_node(
            content="Sofia transit project",
            embedding=[0.9, 0.1],
            provenance={
                "source_id": "topic-registry", "label": "Sofia transit project",
                "timestamp": "",
            },
            kind="abstraction", label="Sofia transit project",
            abstraction_kind="topic",
        )
        store.set_node_attr(t1, "level", 1)
        store.add_parent_edge(t1, f1)

        # Create topic T2, parent_of F1 (same fact, second parent).
        t2 = store.add_node(
            content="user infrastructure knowledge",
            embedding=[0.8, 0.2],
            provenance={
                "source_id": "topic-registry", "label": "user infrastructure knowledge",
                "timestamp": "",
            },
            kind="abstraction", label="user infrastructure knowledge",
            abstraction_kind="topic",
        )
        store.set_node_attr(t2, "level", 1)
        store.add_parent_edge(t2, f1)

        # Verify individual topic reach.
        assert compute_reach(store, [t1]) == 1  # one fact child
        assert compute_reach(store, [t2]) == 1  # same fact

        # VERIFY: union reach of both topics = 1 (not 2!)
        union_reach = compute_reach(store, [t1, t2])
        assert union_reach == 1, (
            f"Multi-parent reach must be distinct union. "
            f"Expected 1 leaf, got {union_reach}. "
            f"A naive sum would give 2, which is wrong."
        )

        # Now add a second fact under T2 only.
        f2 = store.add_node(
            content="Another systemd fact.",
            embedding=[0.5, 0.5],
            provenance={"source_id": "src2", "label": "", "timestamp": ""},
        )
        store.add_parent_edge(t2, f2)

        # T1 still has 1 child (F1), T2 now has 2 (F1, F2).
        assert compute_reach(store, [t1]) == 1
        assert compute_reach(store, [t2]) == 2

        # Union of T1+T2 = distinct {F1, F2} = 2.
        assert compute_reach(store, [t1, t2]) == 2

        # Now create a theme Th1 parent_of both topics.
        th1 = store.add_node(
            content="User projects theme",
            embedding=[0.7, 0.3],
            provenance={
                "source_id": "dreamer:test", "label": "Projects", "timestamp": "",
            },
            kind="abstraction", label="Projects",
        )
        store.set_node_attr(th1, "level", 2)
        store.add_parent_edge(th1, t1)
        store.add_parent_edge(th1, t2)

        # Theme reach = distinct facts under both topics = {F1, F2} = 2.
        # NOT 3 (which would be 1+2 if counting non-distinct).
        actual = compute_reach(store, [th1])
        assert actual == 2, (
            f"Theme reach must be distinct union of all leaf descendants. "
            f"Expected 2 distinct leaves, got {actual}"
        )
