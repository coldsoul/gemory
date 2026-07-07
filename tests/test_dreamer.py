"""Tests for the Gemory dreamer consolidation process.

All LLM calls are stubbed — no real API requests are made.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gemory import config
from gemory.cluster import cluster_nodes
from gemory.graph import GraphStore
from tests.stubs import LookupStub, vectors_with_cosines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_summarize(label: str = "Test Theme", summary: str = "A test summary."):
    """Return a factory that produces a fixed summarization result."""
    return lambda facts: {"label": label, "summary": summary}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_graph(tmp_path):
    """Build a graph with 8 nodes — one clear cluster of 4, one of 3, one lone.

    * Cluster 1: 4 nodes with mutual cosine ~0.88
    * Cluster 2: 3 nodes with mutual cosine ~0.88
    * Lone: 1 node far from everyone (cosine ~0.30)
    """
    memory_path = str(tmp_path / "memory.json")
    store = GraphStore(memory_path)

    n = 8
    gram = np.full((n, n), 0.30)
    np.fill_diagonal(gram, 1.0)

    # Cluster 1 (indices 0-3)
    for i in range(4):
        for j in range(4):
            if i != j:
                gram[i, j] = 0.88
    # Cluster 2 (indices 4-6)
    for i in range(4, 7):
        for j in range(4, 7):
            if i != j:
                gram[i, j] = 0.88

    gram = (gram + gram.T) / 2
    np.fill_diagonal(gram, 1.0)

    vecs = vectors_with_cosines(gram)
    facts = [
        "The user uses uv for package management.",
        "The user prefers Python for backend work.",
        "The user works with networkx for graph processing.",
        "The user develops on a VPS.",
        "The user flies FPV drones on weekends.",
        "The user builds custom drone frames.",
        "The user uses Betaflight firmware.",
        "The user drinks oat milk lattes.",
    ]

    for i, fact in enumerate(facts):
        store.add_node(
            content=fact,
            embedding=vecs[i].tolist(),
            provenance={"source_id": f"test_{i}", "label": "", "timestamp": ""},
        )

    store.save()
    return store, memory_path


@pytest.fixture
def small_graph(tmp_path):
    """Graph where each cluster would have only 2 nodes (< MIN_CLUSTER_SIZE)."""
    store = GraphStore(str(tmp_path / "small.json"))
    gram = np.full((4, 4), 0.30)
    np.fill_diagonal(gram, 1.0)
    # Two pairs, each at 0.88 — but clusters of 2 are below threshold
    gram[0, 1] = gram[1, 0] = 0.88
    gram[2, 3] = gram[3, 2] = 0.88

    vecs = vectors_with_cosines(gram)
    for i, fact in enumerate(["A1", "A2", "B1", "B2"]):
        store.add_node(fact, vecs[i].tolist(), {"source_id": f"s{i}"})
    return store, str(tmp_path / "small.json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicAbstraction:
    """One obvious cluster -> exactly one abstraction node."""

    def test_basic_abstraction(self, populated_graph, monkeypatch):
        store, memory_path = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from gemory import dreamer as dr

        # Stub LLM calls
        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        # Run consolidation
        abstractions = dr._consolidate_level(
            store, all_ids, "test-run", [], 1,
        )

        # The fixture has 2 clusters (4 nodes + 3 nodes), both above threshold
        assert len(abstractions) == 2

        for ab in abstractions:
            abs_id = ab["id"]
            abs_node = store.get_node(abs_id)

            assert abs_node.kind == "abstraction"
            assert abs_node.label == "Test Theme"
            assert abs_node.level == 1

            # parent_of edges to member facts
            children = store.get_children(abs_id)
            member_ids = ab["member_ids"]
            assert len(children) == len(member_ids)
            assert len(member_ids) >= 3  # each cluster >= MIN_CLUSTER_SIZE

            # Embedding should be stored
            assert abs_id in store._embeddings


class TestMinSizeRespected:
    """Clusters below MIN_CLUSTER_SIZE produce no abstractions."""

    def test_min_size_respected(self, small_graph, monkeypatch):
        store, _ = small_graph
        all_ids = [n.id for n in store.all_nodes()]

        from gemory import dreamer as dr

        stub = LookupStub({}, dim=3)
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub.embed)

        abstractions = dr._consolidate_level(store, all_ids, "test-run", [], 1)
        assert abstractions == []


class TestConsolidationIdempotency:
    """Running consolidation twice should produce no new changes."""

    def test_consolidation_idempotency(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from gemory import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        # First run
        r1 = dr._consolidate_level(store, all_ids, "run1", [], 1)
        assert len(r1) == 2  # 2 clusters in fixture

        n_before = len(store.all_nodes())

        # Second run — existing abstractions list now contains r1
        r2 = dr._consolidate_level(store, all_ids, "run2", r1, 1)
        # The overlap check should skip the existing abstraction
        assert len(r2) == 2  # returns the existing abstraction ids
        n_after = len(store.all_nodes())

        # No new nodes were added on second run
        assert n_after == n_before


class TestAbstractionProvenance:
    """Abstraction nodes have correct provenance metadata."""

    def test_abstraction_provenance(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from gemory import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        abstractions = dr._consolidate_level(store, all_ids, "run-abc", [], 1)
        assert len(abstractions) >= 1

        # Check each abstraction's provenance
        for ab in abstractions:
            abs_node = store.get_node(ab["id"])
            assert len(abs_node.provenance) == 1
            prov = abs_node.provenance[0]
            assert prov["source_id"].startswith("dreamer:")
            assert "run-abc" in prov["source_id"]
            assert "member_ids" in prov
            assert len(prov["member_ids"]) >= 3
            assert abs_node.kind == "abstraction"


class TestAbstractionLevel:
    """Abstraction level is one above the highest child level."""

    def test_abstraction_level_correct(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from gemory import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        abstractions = dr._consolidate_level(store, all_ids, "run1", [], 1)
        assert len(abstractions) >= 1

        for ab in abstractions:
            abs_node = store.get_node(ab["id"])
            # All children are at level 0, so abstraction is at level 1
            assert abs_node.level == 1

            # Individual children are at level 0
            for mid in ab["member_ids"]:
                child = store.get_node(mid)
                assert child.level == 0


class TestDryRunWritesNothing:
    """Dry-run mode must NOT modify the graph file on disk."""

    def test_dry_run_writes_nothing(self, populated_graph, monkeypatch):
        store, memory_path = populated_graph
        store.save()  # ensure file is written

        from gemory import dreamer as dr

        # Read before
        with open(memory_path) as f:
            before = f.read()

        # Stub LLM
        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        # Also stub sys.argv for main()
        monkeypatch.setattr(
            sys, "argv",
            ["dreamer.py", "--memory-path", memory_path],
        )
        monkeypatch.setattr(dr, "sys", sys)  # ensure dreamer uses patched sys.argv

        dr.main()

        # Read after
        with open(memory_path) as f:
            after = f.read()

        assert before == after


class TestApplyWritesAndBacksUp:
    """Apply mode must modify the graph and create backup files."""

    def test_apply_writes_and_backs_up(
        self, populated_graph, monkeypatch, tmp_path,
    ):
        store, memory_path = populated_graph
        # Add a topic node with enough children so level-1 enrichment runs.
        topic = store.add_node(
            "Test topic", [1.0, 0.0, 0.0], {"source_id": "topics"},
            kind="abstraction", label="Test topic", abstraction_kind="topic",
        )
        store.set_node_attr(topic, "level", 1)
        # Link the 4 cluster-1 nodes as children
        all_nodes = store.all_nodes()
        linked = 0
        for n in all_nodes:
            if n.level == 0 and linked < 4:
                store.add_parent_edge(topic, n.id)
                linked += 1
        store.save()

        from gemory import dreamer as dr

        stub_embed = LookupStub(
            {"Auto Theme": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr(dr, "summarize_cluster", _stub_summarize(
            label="Auto Theme", summary="Enriched summary.",
        ))
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        monkeypatch.setattr(
            sys, "argv",
            ["dreamer.py", "--apply", "--memory-path", memory_path],
        )
        monkeypatch.setattr(dr, "sys", sys)

        dr.main()

        # Check backup files exist
        dir_entries = os.listdir(tmp_path)
        bak_files = [f for f in dir_entries if f.endswith(".bak")]
        assert len(bak_files) >= 1

        # Graph file should still exist
        assert os.path.exists(memory_path)

        # Load and verify the topic was enriched
        store2 = GraphStore(memory_path)
        store2.load()
        nodes = store2.all_nodes()
        abs_nodes = [n for n in nodes if n.kind == "abstraction"]
        assert len(abs_nodes) >= 1


class TestRecursiveConsolidation:
    """Multiple levels of hierarchy: topics -> themes."""

    def test_recursive_consolidation(self, monkeypatch, tmp_path):
        from gemory import dreamer as dr
        from gemory.graph import GraphStore
        import numpy as np
        from tests.stubs import vectors_with_cosines

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create 9 fact nodes.
        f_ids = []
        for i in range(9):
            nid = store.add_node(
                f"fact {i}", [1.0, 0.0, 0.0], {"source_id": "s"},
            )
            f_ids.append(nid)

        # Create 3 topic nodes with embeddings that cluster together at level 2
        # (mutual cosine 0.88 > CLUSTER_SIM_THRESHOLD).
        gram = np.array([[1.0, 0.88, 0.88],
                         [0.88, 1.0, 0.88],
                         [0.88, 0.88, 1.0]])
        tvecs = vectors_with_cosines(gram, dim=4)
        topics = []
        for i in range(3):
            tid = store.add_node(
                f"Topic {i}", tvecs[i].tolist(), {"source_id": "topics"},
                kind="abstraction", label=f"Topic {i}",
                abstraction_kind="topic",
            )
            store.set_node_attr(tid, "level", 1)
            # Link 3 facts to each topic
            for j in range(3):
                store.add_parent_edge(tid, f_ids[i * 3 + j])
            topics.append(tid)

        # Stub summarizer and embed.
        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Enriched summary."}

        monkeypatch.setattr(dr, "summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # Level 1: enrich topics.
        level1_abs = dr._consolidate_level_1(store, "run1")
        assert len(level1_abs) == 3

        # Level 2: cluster topics -> 1 theme.
        level2_abs = dr._consolidate_level(
            store, [a["id"] for a in level1_abs], "run1", level1_abs, 2,
        )
        assert len(level2_abs) >= 1

        level2_node = store.get_node(level2_abs[0]["id"])
        assert level2_node.level == 2
        assert level2_node.kind == "abstraction"
        assert level2_node.abstraction_kind == "theme"

        # Theme should be parent of the topics.
        for ab in level1_abs:
            parents = store.get_parents(ab["id"])
            assert level2_abs[0]["id"] in parents


class TestTopicGroupsFarApartFacts:
    """Facts far apart in embedding but sharing a topic are grouped by topic."""

    def test_topic_groups_far_apart_facts(self, monkeypatch, tmp_path) -> None:
        from gemory import dreamer as dr
        from gemory.graph import GraphStore
        from gemory.topics import resolve_topic

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create 6 facts: 3 about "Gemory" (far-apart embeddings), 3 about
        # "FPV drones" (far-apart embeddings).
        facts = [
            ("Gemory fact 1", [1.0, 0.0, 0.0, 0.0]),
            ("Gemory fact 2", [0.3, 0.9, 0.0, 0.0]),
            ("Gemory fact 3", [0.0, 0.5, 0.8, 0.0]),
            ("FPV fact 1",    [0.0, 0.0, 0.0, 1.0]),
            ("FPV fact 2",    [0.0, 0.0, 0.3, 0.9]),
            ("FPV fact 3",    [0.3, 0.0, 0.0, 0.5]),
        ]
        f_ids = []
        for content, emb in facts:
            nid = store.add_node(
                content, emb, {"source_id": "s"},
            )
            f_ids.append(nid)

        # Stub topic embed so resolve_topic can create topic nodes.
        topic_stub = LookupStub(
            {"Gemory": [1.0, 0.0, 0.0, 0.0],
             "FPV drones": [0.0, 1.0, 0.0, 0.0]},
            dim=4,
        )
        monkeypatch.setattr("gemory.topics.embed", topic_stub.embed)

        topic_gemory = resolve_topic(store, "Gemory")
        topic_fpv = resolve_topic(store, "FPV drones")
        assert topic_gemory is not None
        assert topic_fpv is not None

        # Link facts to their topics.
        for i in range(3):
            store.add_parent_edge(topic_gemory, f_ids[i])
        for i in range(3, 6):
            store.add_parent_edge(topic_fpv, f_ids[i])

        # Stub dreamer's summarize_cluster.
        call_log: list[list[str]] = []

        def mock_summarize(facts_list):
            call_log.append(facts_list)
            return {"label": "Auto Theme", "summary": "Enriched summary."}

        monkeypatch.setattr(dr, "summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", topic_stub.embed)

        # Run level-1 enrichment.
        enriched = dr._consolidate_level_1(store, "run1")

        assert len(enriched) == 2, (
            f"Expected 2 enriched topics, got {len(enriched)}"
        )
        assert len(call_log) == 2, (
            f"Expected 2 summarize_cluster calls, got {len(call_log)}"
        )

        # Verify each topic's content was enriched with a summary.
        gemory_node = store.get_node(topic_gemory)
        assert "Enriched" in gemory_node.content
        fpv_node = store.get_node(topic_fpv)
        assert "Enriched" in fpv_node.content


class TestLevel2ThemeGrouping:
    """Similar topics cluster into themes at level 2."""

    def test_level2_theme_grouping(self, monkeypatch, tmp_path) -> None:
        from gemory import dreamer as dr
        from gemory.graph import GraphStore
        import numpy as np
        from tests.stubs import vectors_with_cosines

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create fact nodes (7 per topic, not all need to be used).
        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(18)]

        # Create 6 topic nodes: Group A (t0-t2, mutual cosine 0.88),
        # Group B (t3-t5, mutual cosine 0.88), groups are far apart.
        gram = np.zeros((6, 6))
        # Group A: indices 0-2
        for i in range(3):
            for j in range(3):
                gram[i, j] = 0.88 if i != j else 1.0
        # Group B: indices 3-5
        for i in range(3, 6):
            for j in range(3, 6):
                gram[i, j] = 0.88 if i != j else 1.0
        # Cross-group: far apart
        for i in range(3):
            for j in range(3, 6):
                gram[i, j] = gram[j, i] = 0.30

        tvecs = vectors_with_cosines(gram, dim=6)
        topics = []
        for i in range(6):
            tid = store.add_node(
                f"Topic {i}", tvecs[i].tolist(), {"source_id": "topics"},
                kind="abstraction", label=f"Topic {i}",
                abstraction_kind="topic",
            )
            store.set_node_attr(tid, "level", 1)
            for j in range(3):
                store.add_parent_edge(tid, f_ids[i * 3 + j])
            topics.append(tid)

        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Theme summary."}

        monkeypatch.setattr(dr, "summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # Level 1: enrich topics.
        level1 = dr._consolidate_level_1(store, "run1")
        assert len(level1) == 6

        # Level 2: cluster topics -> themes.
        level2 = dr._consolidate_level(
            store, [a["id"] for a in level1], "run1", level1, 2,
        )
        assert len(level2) == 2, f"Expected 2 themes, got {len(level2)}"

        for theme_info in level2:
            theme_node = store.get_node(theme_info["id"])
            assert theme_node.level == 2
            assert theme_node.kind == "abstraction"
            assert theme_node.abstraction_kind == "theme"

        # Each theme has parent_of to its constituent topics.
        for theme_info in level2:
            children = store.get_children(theme_info["id"])
            assert len(children) >= 1
            for cid in children:
                parent_of = cid in topics


class TestIdempotencyPreservesTopics:
    """Re-running the dreamer does not duplicate topics or themes."""

    def test_idempotency_preserves_topics(self, monkeypatch, tmp_path) -> None:
        from gemory import dreamer as dr
        from gemory.graph import GraphStore
        import numpy as np
        from tests.stubs import vectors_with_cosines

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create 9 fact nodes.
        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(9)]

        # Create 3 topic nodes with clustering embeddings.
        gram = np.array([[1.0, 0.88, 0.88],
                         [0.88, 1.0, 0.88],
                         [0.88, 0.88, 1.0]])
        tvecs = vectors_with_cosines(gram, dim=4)
        for i in range(3):
            tid = store.add_node(
                f"Topic {i}", tvecs[i].tolist(), {"source_id": "topics"},
                kind="abstraction", label=f"Topic {i}",
                abstraction_kind="topic",
            )
            store.set_node_attr(tid, "level", 1)
            for j in range(3):
                store.add_parent_edge(tid, f_ids[i * 3 + j])

        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Theme summary."}

        monkeypatch.setattr(dr, "summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # First run.
        level1 = dr._consolidate_level_1(store, "run1")
        assert len(level1) == 3
        level2 = dr._consolidate_level(
            store, [a["id"] for a in level1], "run1", level1, 2,
        )
        assert len(level2) == 1
        n_after_first = len(store.all_nodes())

        # Second run — should not create new nodes.
        level1_2 = dr._consolidate_level_1(store, "run2")
        assert len(level1_2) == 3  # same 3 topics
        level2_2 = dr._consolidate_level(
            store, [a["id"] for a in level1_2], "run2", level1_2, 2,
        )
        # Already-parented topics are skipped at level 2, so no new themes.
        assert len(level2_2) == 0
        n_after_second = len(store.all_nodes())
        assert n_after_second == n_after_first
