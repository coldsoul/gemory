"""Tests for hybrid clustering (algorithm + LLM methods)."""

import pytest

from src.consolidate import cluster_layer
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hybrid_graph(tmp_graph_path):
    """Build a graph where algorithm finds one cluster but misses another.

    * 3 nodes with similar embeddings (algorithm will cluster them).
    * 2 nodes far from the first cluster but conceptually related (LLM can
      find them).
    """
    store = GraphStore(tmp_graph_path)
    ids: dict[str, str] = {}

    # 3 nodes with similar embeddings (algorithm cluster).
    for name in ["a", "b", "c"]:
        ids[name] = store.add_node(
            content=f"Fact {name}", embedding=[1.0, 0.1, 0.0],
            provenance={"source_id": name, "label": "", "timestamp": ""},
            label=f"Label {name}",
        )
    # 2 nodes far from the first cluster but with similar labels/summaries
    # (LLM should group them).
    for name in ["d", "e"]:
        ids[name] = store.add_node(
            content=f"Fact {name}", embedding=[0.0, 0.0, 1.0],
            provenance={"source_id": name, "label": "", "timestamp": ""},
            label=f"Label {name}",
        )

    return store, ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlgorithmClustering:
    """``method="algorithm"`` uses cosine-based Louvain clustering."""

    def test_algorithm_finds_similar_cluster(self, hybrid_graph):
        store, ids = hybrid_graph
        clusters = cluster_layer(store, list(ids.values()), method="algorithm")
        assert len(clusters) >= 1  # At least one cluster found

    def test_algorithm_empty_input(self, hybrid_graph):
        store, _ = hybrid_graph
        clusters = cluster_layer(store, [], method="algorithm")
        assert clusters == []


class TestLLMClustering:
    """``method="llm"`` delegates to ``cluster_by_llm``."""

    def test_llm_method_calls_cluster_by_llm(self, hybrid_graph, monkeypatch):
        store, ids = hybrid_graph
        called: list = []

        def stub_cluster(summaries, **kw):
            called.append(summaries)
            return [{0, 1}]  # Cluster first two nodes

        monkeypatch.setattr("src.llm.cluster_by_llm", stub_cluster)

        clusters = cluster_layer(
            store, list(ids.values())[:3], method="llm",
        )
        assert len(called) == 1
        assert len(clusters) >= 1


class TestHybridClustering:
    """``method="hybrid"`` combines algorithm + LLM."""

    def test_hybrid_combines_both_methods(self, hybrid_graph, monkeypatch):
        store, ids = hybrid_graph
        all_ids = list(ids.values())

        llm_called: list = []

        def stub_cluster(summaries, **kw):
            llm_called.append(summaries)
            # Cluster whatever leftovers the LLM receives
            return [{0, 1}] if len(summaries) >= 2 else []

        monkeypatch.setattr("src.llm.cluster_by_llm", stub_cluster)

        clusters = cluster_layer(store, all_ids, method="hybrid")
        total_in_clusters = sum(len(c) for c in clusters)
        assert total_in_clusters >= 2

    def test_hybrid_no_leftovers_skips_llm(self, hybrid_graph, monkeypatch):
        """If algorithm clusters everything, LLM is not called."""
        store, ids = hybrid_graph
        # Only use the 3 similar nodes (algorithm will cluster all).
        similar_ids = [ids["a"], ids["b"], ids["c"]]

        llm_called: list = []
        monkeypatch.setattr(
            "src.llm.cluster_by_llm",
            lambda x, **kw: llm_called.append(x) or [],
        )
        clusters = cluster_layer(store, similar_ids, method="hybrid")
        assert len(clusters) >= 1
        # LLM should NOT be called because algorithm covered everything.
        assert len(llm_called) == 0

    def test_unknown_method_raises(self, hybrid_graph):
        store, ids = hybrid_graph
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_layer(store, list(ids.values()), method="nonexistent")
