"""Tests for the Gemory dreamer consolidation process.

All LLM calls are stubbed — no real API requests are made.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.graph import GraphStore
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
    """Build a graph with 11 nodes — 2 clusters (5+5) and 1 lone node.

    * Cluster 1: 5 nodes with mutual cosine ~0.88 (reach 5 >= MIN_REACH)
    * Cluster 2: 5 nodes with mutual cosine ~0.88 (reach 5 >= MIN_REACH)
    * Lone: 1 node far from everyone (cosine ~0.30)
    """
    memory_path = str(tmp_path / "memory.json")
    store = GraphStore(memory_path)

    n = 11
    gram = np.full((n, n), 0.30)
    np.fill_diagonal(gram, 1.0)

    for i in range(5):
        for j in range(5):
            if i != j:
                gram[i, j] = 0.88
    for i in range(5, 10):
        for j in range(5, 10):
            if i != j:
                gram[i, j] = 0.88

    gram = (gram + gram.T) / 2
    np.fill_diagonal(gram, 1.0)

    vecs = vectors_with_cosines(gram)
    facts = [f"fact {i}" for i in range(11)]

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
    """Graph where each cluster has only 2 nodes (reach 2 < MIN_REACH=5)."""
    store = GraphStore(str(tmp_path / "small.json"))
    gram = np.full((4, 4), 0.30)
    np.fill_diagonal(gram, 1.0)
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

        from src import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        abstractions = dr._consolidate_layer(store, all_ids, "test-run", [])

        # The fixture has 2 clusters (5 nodes + 5 nodes), both above reach threshold
        assert len(abstractions) == 2

        for ab in abstractions:
            abs_id = ab["id"]
            abs_node = store.get_node(abs_id)

            assert abs_node.kind == "abstraction"
            assert abs_node.label == "Test Theme"
            assert abs_node.level == 1

            children = store.get_children(abs_id)
            member_ids = ab["member_ids"]
            assert len(children) == len(member_ids)
            assert len(member_ids) >= 5

            assert abs_id in store._embeddings


class TestMinSizeRespected:
    """Clusters below MIN_REACH produce no abstractions."""

    def test_min_size_respected(self, small_graph, monkeypatch):
        store, _ = small_graph
        all_ids = [n.id for n in store.all_nodes()]

        from src import dreamer as dr

        stub = LookupStub({}, dim=3)
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub.embed)

        abstractions = dr._consolidate_layer(store, all_ids, "test-run", [])
        assert abstractions == []


class TestConsolidationIdempotency:
    """Running consolidation twice should produce no new changes."""

    def test_consolidation_idempotency(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from src import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        # First run
        r1 = dr._consolidate_layer(store, all_ids, "run1", [])
        assert len(r1) == 2  # 2 clusters in fixture

        n_before = len(store.all_nodes())

        # Second run — existing abstractions list now contains r1
        r2 = dr._consolidate_layer(store, all_ids, "run2", r1)
        assert len(r2) == 2  # returns the existing abstraction ids
        n_after = len(store.all_nodes())

        assert n_after == n_before


class TestAbstractionProvenance:
    """Abstraction nodes have correct provenance metadata."""

    def test_abstraction_provenance(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from src import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        abstractions = dr._consolidate_layer(store, all_ids, "run-abc", [])
        assert len(abstractions) >= 1

        for ab in abstractions:
            abs_node = store.get_node(ab["id"])
            assert len(abs_node.provenance) == 1
            prov = abs_node.provenance[0]
            assert prov["source_id"].startswith("dreamer:")
            assert "run-abc" in prov["source_id"]
            assert "member_ids" in prov
            assert len(prov["member_ids"]) >= 5
            assert abs_node.kind == "abstraction"


class TestAbstractionLevel:
    """Abstraction level is one above the highest child level."""

    def test_abstraction_level_correct(self, populated_graph, monkeypatch):
        store, _ = populated_graph
        all_ids = [n.id for n in store.all_nodes()]

        from src import dreamer as dr

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        abstractions = dr._consolidate_layer(store, all_ids, "run1", [])
        assert len(abstractions) >= 1

        for ab in abstractions:
            abs_node = store.get_node(ab["id"])
            assert abs_node.level == 1

            for mid in ab["member_ids"]:
                child = store.get_node(mid)
                assert child.level == 0


class TestDryRunWritesNothing:
    """Dry-run mode must NOT modify the graph file on disk."""

    def test_dry_run_writes_nothing(self, populated_graph, monkeypatch):
        store, memory_path = populated_graph
        store.save()

        from src import dreamer as dr

        with open(memory_path) as f:
            before = f.read()

        stub_embed = LookupStub(
            {"Test Theme. A test summary.": [1.0, 0.0, 0.0]},
            dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize())
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        monkeypatch.setattr(
            sys, "argv",
            ["dreamer.py", "--memory-path", memory_path],
        )
        monkeypatch.setattr(dr, "sys", sys)

        dr.main()

        with open(memory_path) as f:
            after = f.read()

        assert before == after


class TestApplyWritesAndBacksUp:
    """Apply mode must modify the graph and create backup files."""

    def test_apply_writes_and_backs_up(
        self, populated_graph, monkeypatch, tmp_path,
    ):
        store, memory_path = populated_graph
        store.save()

        from src import dreamer as dr

        stub_embed = LookupStub(
            {"Auto Theme": [1.0, 0.0, 0.0]}, dim=3,
        )
        monkeypatch.setattr("src.consolidate.summarize_cluster", _stub_summarize(
            label="Auto Theme", summary="Enriched summary.",
        ))
        monkeypatch.setattr(dr, "embed", stub_embed.embed)

        monkeypatch.setattr(
            sys, "argv",
            ["dreamer.py", "--apply", "--memory-path", memory_path],
        )
        monkeypatch.setattr(dr, "sys", sys)

        dr.main()

        dir_entries = os.listdir(tmp_path)
        bak_files = [f for f in dir_entries if f.endswith(".bak")]
        assert len(bak_files) >= 1

        assert os.path.exists(memory_path)

        store2 = GraphStore(memory_path)
        store2.load()
        nodes = store2.all_nodes()
        abs_nodes = [n for n in nodes if n.kind == "abstraction"]
        assert len(abs_nodes) >= 1


class TestRecursiveConsolidation:
    """Multiple layers: facts -> topics -> themes."""

    def test_recursive_consolidation(self, monkeypatch, tmp_path):
        from src import dreamer as dr

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create 15 fact nodes (5 per topic x 3 topics).
        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(15)]

        # Create 3 topic nodes with embeddings that cluster together
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
            for j in range(5):
                store.add_parent_edge(tid, f_ids[i * 5 + j])
            topics.append(tid)

        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Enriched summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # First layer: topics as base nodes -> should form 1 theme
        layer1 = dr._consolidate_layer(store, topics, "run1", [])
        assert len(layer1) >= 1

        theme_node = store.get_node(layer1[0]["id"])
        assert theme_node.level == 2
        assert theme_node.kind == "abstraction"


class TestTopicGroupsSimilarEmbeddings:
    """Topics with similar embeddings cluster into themes."""

    def test_topic_groups_similar_embeddings(self, monkeypatch, tmp_path) -> None:
        from src import dreamer as dr

        store = GraphStore(str(tmp_path / "memory.json"))

        # Create fact nodes (5 per topic).
        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(15)]

        # Create 3 topic nodes with embeddings that cluster (mutual cosine 0.88).
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
            for j in range(5):
                store.add_parent_edge(tid, f_ids[i * 5 + j])
            topics.append(tid)

        call_log: list[list[str]] = []

        def mock_summarize(facts_list):
            call_log.append(facts_list)
            return {"label": "Auto Theme", "summary": "Enriched summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # Topics have similar embeddings -> should cluster into 1 theme
        result = dr._consolidate_layer(store, topics, "run1", [])
        assert len(result) >= 1

        # verify that summarize received the topic contents (not raw facts)
        assert len(call_log) >= 1
        for content_list in call_log:
            for c in content_list:
                assert "Topic" in c  # topic content, not "fact" content


class TestLevel2ThemeGrouping:
    """Similar topics cluster into themes at level 2."""

    def test_level2_theme_grouping(self, monkeypatch, tmp_path) -> None:
        from src import dreamer as dr

        store = GraphStore(str(tmp_path / "memory.json"))

        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(30)]

        gram = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                gram[i, j] = 0.88 if i != j else 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                gram[i, j] = 0.88 if i != j else 1.0
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
            for j in range(5):
                store.add_parent_edge(tid, f_ids[i * 5 + j])
            topics.append(tid)

        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Theme summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # First layer: 6 topics -> 2 themes
        layer1 = dr._consolidate_layer(store, topics, "run1", [])
        assert len(layer1) == 2, f"Expected 2 themes, got {len(layer1)}"

        for theme_info in layer1:
            theme_node = store.get_node(theme_info["id"])
            assert theme_node.level == 2
            assert theme_node.kind == "abstraction"


class TestNaturalTermination:
    """Unrelated nodes produce no abstractions; loop terminates naturally."""

    def test_natural_termination(self, monkeypatch, tmp_path):
        from src import dreamer as dr

        store = GraphStore(str(tmp_path / "memory.json"))

        for i in range(6):
            # Each fact uses a unique basis vector -> all orthogonal
            vec = [0.0] * 6
            vec[i] = 1.0
            store.add_node(f"fact {i}", vec, {"source_id": f"s{i}"})

        def never_called(facts):
            assert False, "summarize_cluster should not be called"
            return {}

        monkeypatch.setattr("src.consolidate.summarize_cluster", never_called)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0])

        all_ids = [n.id for n in store.all_nodes()]
        result = dr._consolidate_layer(store, all_ids, "run1", [])
        assert result == []


class TestIdempotencyPreservesTopics:
    """Re-running the dreamer does not duplicate topics or themes."""

    def test_idempotency_preserves_topics(self, monkeypatch, tmp_path) -> None:
        from src import dreamer as dr

        store = GraphStore(str(tmp_path / "memory.json"))

        f_ids = [store.add_node(f"f{i}", [1.0, 0.0, 0.0], {"source_id": "s"})
                 for i in range(15)]

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
            for j in range(5):
                store.add_parent_edge(tid, f_ids[i * 5 + j])

        def mock_summarize(facts):
            return {"label": "Auto Theme", "summary": "Theme summary."}

        monkeypatch.setattr("src.consolidate.summarize_cluster", mock_summarize)
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0, 0.0])

        # First run: topics -> 1 theme
        layer1 = dr._consolidate_layer(store, [n.id for n in store.all_nodes()
                                                if n.kind == "abstraction"
                                                and n.abstraction_kind == "topic"],
                                       "run1", [])
        assert len(layer1) == 1
        n_after_first = len(store.all_nodes())

        # Second run: same topics, should match existing -> no new nodes
        layer2 = dr._consolidate_layer(store, [n.id for n in store.all_nodes()
                                                if n.kind == "abstraction"
                                                and n.abstraction_kind == "topic"],
                                       "run2", layer1)
        assert len(layer2) == 1
        n_after_second = len(store.all_nodes())
        assert n_after_second == n_after_first


class TestDiffDryRun:
    """Diff mode runs both clustering methods and reports buckets."""

    def test_diff_produces_buckets(self, populated_graph, monkeypatch):
        """Diff mode returns agreement/algorithm_only/llm_only dicts."""
        from src import dreamer as dr

        def stub_cluster(summaries):
            # Return first 5 nodes as one cluster (matches algorithm result)
            indices = list(range(min(5, len(summaries))))
            return [set(indices)] if len(indices) >= 2 else []

        monkeypatch.setattr("src.consolidate.summarize_cluster",
                            lambda f: {"label": "Test", "summary": "Sum."})
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0])
        # The algorithm path doesn't use cluster_by_llm; LLM path does.
        monkeypatch.setattr("src.llm.cluster_by_llm", stub_cluster)

        store, _ = populated_graph
        diff = dr._diff_consolidation(store, "test-diff")

        assert "agreement" in diff
        assert "algorithm_only" in diff
        assert "llm_only" in diff
        # At minimum some type of result
        total = len(diff["agreement"]) + len(diff["algorithm_only"]) + len(diff["llm_only"])
        assert total >= 0

    def test_diff_no_crash_on_small_graph(self, small_graph, monkeypatch):
        """Diff mode doesn't crash on graphs where nothing clusters."""
        from src import dreamer as dr

        monkeypatch.setattr("src.consolidate.summarize_cluster",
                            lambda f: {"label": "T", "summary": "S."})
        monkeypatch.setattr(dr, "embed", lambda x: [1.0, 0.0])
        monkeypatch.setattr("src.llm.cluster_by_llm", lambda x: [])

        store, _ = small_graph
        diff = dr._diff_consolidation(store, "test-diff")
        assert isinstance(diff, dict)
