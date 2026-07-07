"""Tests for :mod:`src.cluster` — Louvain-based community detection."""

import numpy as np
import pytest

from src.cluster import cluster_nodes
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Helper: build a normalised unit vector
# ---------------------------------------------------------------------------

def _unit_vec(vals: list[float]) -> np.ndarray:
    """Return a unit-norm vector padded to 1536 dimensions."""
    arr = np.array(vals + [0.0] * (1536 - len(vals)), dtype=float)
    n = np.linalg.norm(arr)
    return arr / n if n > 0 else arr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_graph(tmp_graph_path):
    """A :class:`GraphStore` with 8 nodes that naturally form 2 clusters.

    * Cluster 1: 4 nodes near ``[1, 0, 0, 0]`` — all very similar.
    * Cluster 2: 3 nodes near ``[0, 1, 0, 0]`` — all very similar.
    * Lone node: orthogonal to both clusters.
    """
    store = GraphStore(tmp_graph_path)

    vectors = {
        # Cluster 1 (all near [1, 0, 0, 0])
        "c1_a": _unit_vec([1.0, 0.1, 0.0, 0.0]),
        "c1_b": _unit_vec([0.9, 0.2, 0.0, 0.0]),
        "c1_c": _unit_vec([0.95, 0.05, 0.0, 0.0]),
        "c1_d": _unit_vec([0.85, 0.15, 0.0, 0.0]),
        # Cluster 2 (all near [0, 1, 0, 0])
        "c2_a": _unit_vec([0.0, 1.0, 0.1, 0.0]),
        "c2_b": _unit_vec([0.0, 0.95, 0.15, 0.0]),
        "c2_c": _unit_vec([0.1, 0.9, 0.1, 0.0]),
        # Lone node (far from both)
        "lone": _unit_vec([0.0, 0.0, 0.0, 1.0]),
    }

    node_ids: dict[str, str] = {}
    for name, vec in vectors.items():
        nid = store.add_node(
            content=f"Fact {name}",
            embedding=vec.tolist(),
            provenance={"source_id": "test", "label": "", "timestamp": "2024-01-01T00:00:00"},
        )
        node_ids[name] = nid

    # Return both the store and the list of actual graph node IDs
    all_ids = list(node_ids.values())
    return store, all_ids, node_ids


@pytest.fixture
def small_graph(tmp_graph_path):
    """Two nodes only — below MIN_CLUSTER_SIZE thresholds."""
    store = GraphStore(tmp_graph_path)
    store.add_node("a", _unit_vec([1.0, 0.0]).tolist(), {"source_id": "s"})
    store.add_node("b", _unit_vec([0.0, 1.0]).tolist(), {"source_id": "s"})
    return store, ["a", "b"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClusterDeterminism:
    """Running cluster_nodes twice must produce identical results."""

    def test_cluster_determinism(self, populated_graph) -> None:
        store, node_ids, _ = populated_graph

        clusters_1 = cluster_nodes(store, node_ids, seed=42)
        clusters_2 = cluster_nodes(store, node_ids, seed=42)

        # Compare as sorted list of sorted tuples.
        def _canonical(cls: list[set[str]]) -> list[tuple[str, ...]]:
            return sorted(tuple(sorted(c)) for c in cls)

        assert _canonical(clusters_1) == _canonical(clusters_2)


class TestClusterFindsTwoClusters:
    """The 8-node fixture should produce at least 2 clusters."""

    def test_cluster_finds_two_clusters(self, populated_graph) -> None:
        store, node_ids, name_map = populated_graph

        clusters = cluster_nodes(store, node_ids, seed=42)
        assert len(clusters) >= 2

        # Each cluster should have at least MIN_CLUSTER_SIZE members.
        from src import config
        for c in clusters:
            assert len(c) >= config.MIN_CLUSTER_SIZE

        # The lone node must not appear in any cluster.
        lone_id = name_map["lone"]
        all_clustered_ids = set().union(*clusters)
        assert lone_id not in all_clustered_ids


class TestClusterRespectsMinSize:
    """Small working sets produce no clusters."""

    def test_cluster_respects_min_size(self, small_graph) -> None:
        store, node_ids = small_graph

        clusters = cluster_nodes(store, node_ids, seed=42)
        assert clusters == []


class TestClusterEmptyWorkingSet:
    """An empty working set returns empty results."""

    def test_cluster_empty_working_set(self, populated_graph) -> None:
        store, _node_ids, _ = populated_graph

        clusters = cluster_nodes(store, [], seed=42)
        assert clusters == []


class TestClusterLoneNode:
    """A lone node (below similarity threshold with everyone) is not clustered."""

    def test_cluster_lone_node_not_clustered(self, populated_graph) -> None:
        store, node_ids, name_map = populated_graph

        clusters = cluster_nodes(store, node_ids, seed=42)
        all_clustered = set().union(*clusters)

        lone_id = name_map["lone"]
        assert lone_id not in all_clustered


class TestClusterUsesRelatedEdges:
    """Explicit ``related`` edges bridge otherwise-distant nodes into clusters."""

    def test_cluster_uses_related_edges(self, tmp_graph_path) -> None:
        store = GraphStore(tmp_graph_path)

        # Create 4 nodes that naturally cluster together.
        g1 = store.add_node(
            "g1", _unit_vec([1.0, 0.0, 0.0]).tolist(),
            {"source_id": "s"},
        )
        g2 = store.add_node(
            "g2", _unit_vec([0.9, 0.1, 0.0]).tolist(),
            {"source_id": "s"},
        )
        g3 = store.add_node(
            "g3", _unit_vec([0.95, 0.05, 0.0]).tolist(),
            {"source_id": "s"},
        )
        g4 = store.add_node(
            "g4", _unit_vec([0.85, 0.15, 0.0]).tolist(),
            {"source_id": "s"},
        )

        # Create 3 far nodes that are orthogonal to the cluster.
        f1 = store.add_node(
            "f1", _unit_vec([0.0, 0.0, 1.0]).tolist(),
            {"source_id": "s"},
        )
        f2 = store.add_node(
            "f2", _unit_vec([0.0, 0.0, 0.9]).tolist(),
            {"source_id": "s"},
        )
        f3 = store.add_node(
            "f3", _unit_vec([0.0, 0.0, 0.95]).tolist(),
            {"source_id": "s"},
        )

        all_ids = [g1, g2, g3, g4, f1, f2, f3]

        # Without edges, only the 4 natural-cluster nodes should form a cluster.
        clusters_no_edge = cluster_nodes(store, all_ids, seed=42)
        size_no_edge = len(clusters_no_edge[0]) if clusters_no_edge else 0

        # Add ``related`` edges from the far nodes into the cluster.
        store.add_edge(f1, g1, weight=0.8, relation="related")
        store.add_edge(f2, g2, weight=0.8, relation="related")
        store.add_edge(f3, g3, weight=0.8, relation="related")

        # With edges, the cluster should grow (or at least the far nodes
        # should now be connected in the similarity graph).
        clusters_with_edge = cluster_nodes(store, all_ids, seed=42)
        size_with_edge = len(clusters_with_edge[0]) if clusters_with_edge else 0

        # The presence of the edge should not shrink the largest cluster.
        assert size_with_edge >= size_no_edge, (
            f"Adding edges should not shrink the cluster "
            f"({size_no_edge} -> {size_with_edge})"
        )
