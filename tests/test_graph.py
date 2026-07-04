"""Unit tests for :mod:`gemory.graph` (GraphStore).

Tests exercise the public API only -- no direct ``networkx`` or ``numpy``
imports in this file.
"""

import json
import math
import os
from dataclasses import asdict

import pytest

from gemory.graph import GraphStore
from gemory.models import Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_node(
    store: GraphStore,
    content: str,
    embedding: list[float],
    source_id: str = "test_src",
) -> str:
    """Convenience: add a node with a single-source provenance."""
    return store.add_node(content, embedding, {"source_id": source_id})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_store(tmp_path) -> GraphStore:
    """GraphStore pointed at a temporary directory."""
    return GraphStore(str(tmp_path / "memory.json"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddAndGetNode:
    """Add a node then retrieve it -- verify every field."""

    def test_add_and_get_node(self, graph_store: GraphStore) -> None:
        embedding = [0.1, 0.2, 0.3]
        provenance = {"source_id": "src_a", "detail": "test"}
        node_id = graph_store.add_node("Hello world", embedding, provenance)

        node = graph_store.get_node(node_id)

        assert node.id == node_id
        assert node.content == "Hello world"
        assert node.confidence == 1.0  # CONFIDENCE_BASE
        assert node.provenance == [provenance]
        assert node.level == 0
        assert isinstance(node.created_at, str) and len(node.created_at) > 0
        assert isinstance(node.updated_at, str) and len(node.updated_at) > 0
        assert node.created_at == node.updated_at  # freshly created

    def test_get_node_missing_raises(self, graph_store: GraphStore) -> None:
        with pytest.raises(KeyError):
            graph_store.get_node("nonexistent-id")


class TestAddEdge:
    """Add edges and verify the successor relationship."""

    def test_add_edge(self, graph_store: GraphStore) -> None:
        id_a = _add_node(graph_store, "Node A", [1.0, 0.0])
        id_b = _add_node(graph_store, "Node B", [0.0, 1.0])

        graph_store.add_edge(id_a, id_b, weight=0.75, relation="related")

        neighbors = graph_store.get_neighbors(id_a)
        assert id_b in neighbors
        assert len(neighbors) == 1

    def test_add_edge_default_relation(self, graph_store: GraphStore) -> None:
        id_a = _add_node(graph_store, "A", [1.0, 0.0])
        id_b = _add_node(graph_store, "B", [0.0, 1.0])

        graph_store.add_edge(id_a, id_b, weight=0.5)

        neighbors = graph_store.get_neighbors(id_a)
        assert id_b in neighbors


class TestSaveLoadRoundtrip:
    """Save then load into a fresh GraphStore -- everything must match."""

    def test_save_load_roundtrip(self, graph_store: GraphStore, tmp_path) -> None:
        # --- populate ---
        emb_a = [0.1, 0.2, 0.3]
        emb_b = [0.4, 0.5, 0.6]
        emb_c = [0.7, 0.8, 0.9]

        id_a = graph_store.add_node("Fact A", emb_a, {"source_id": "s1"})
        id_b = graph_store.add_node("Fact B", emb_b, {"source_id": "s2"})
        id_c = graph_store.add_node("Fact C", emb_c, {"source_id": "s3"})

        graph_store.add_edge(id_a, id_b, weight=0.8)
        graph_store.add_edge(id_b, id_c, weight=0.6, relation="supports")

        # --- save ---
        graph_store.save()

        # --- load into a brand-new store ---
        store2 = GraphStore(str(tmp_path / "memory.json"))
        store2.load()

        # --- verify node count ---
        assert len(store2.all_nodes()) == 3

        # --- verify each node's fields ---
        for node_id, expected_emb in [
            (id_a, emb_a),
            (id_b, emb_b),
            (id_c, emb_c),
        ]:
            orig = graph_store.get_node(node_id)
            loaded = store2.get_node(node_id)

            assert loaded.id == orig.id
            assert loaded.content == orig.content
            assert loaded.confidence == orig.confidence
            assert loaded.provenance == orig.provenance
            assert loaded.level == orig.level
            assert loaded.created_at == orig.created_at
            assert loaded.updated_at == orig.updated_at

            # Embedding round-trip (sidecar check)
            nid, sim = store2.find_similar(expected_emb, top_k=1)[0]
            assert nid == node_id
            assert sim == pytest.approx(1.0)

        # --- verify edges via get_neighbors ---
        assert store2.get_neighbors(id_a) == [id_b]
        assert store2.get_neighbors(id_b) == [id_c]
        assert store2.get_neighbors(id_c) == []

        # --- verify roots ---
        roots = store2.get_roots()
        assert id_a in roots
        assert id_c not in roots
        assert id_b not in roots


class TestFindSimilar:
    """End-to-end similarity search."""

    def test_find_similar_ordering(self, graph_store: GraphStore) -> None:
        # 2D vectors for deterministic maths.
        id_a = _add_node(graph_store, "A", [1.0, 0.0, 0.0], "s1")
        id_b = _add_node(graph_store, "B", [0.0, 1.0, 0.0], "s2")
        id_c = _add_node(graph_store, "C", [0.0, 0.0, 1.0], "s3")
        id_d = _add_node(graph_store, "D", [0.8, 0.6, 0.0], "s4")

        query = [1.0, 0.0, 0.0]
        results = graph_store.find_similar(query, top_k=10)

        # All four should appear, sorted descending.
        assert len(results) == 4
        sims = [sim for _, sim in results]
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))

        ids = [nid for nid, _ in results]
        assert ids[0] == id_a  # closest (cosine=1.0)

    def test_find_similar_threshold(self, graph_store: GraphStore) -> None:
        id_a = _add_node(graph_store, "A", [1.0, 0.0, 0.0], "s1")
        id_d = _add_node(graph_store, "D", [0.8, 0.6, 0.0], "s4")
        _add_node(graph_store, "B", [0.0, 1.0, 0.0], "s2")
        _add_node(graph_store, "C", [0.0, 0.0, 1.0], "s3")

        query = [1.0, 0.0, 0.0]
        results = graph_store.find_similar(query, threshold=0.5, top_k=10)

        # Only A (1.0) and D (0.8) should survive the threshold.
        assert len(results) == 2
        assert results[0][0] == id_a
        assert results[1][0] == id_d

    def test_find_similar_empty(self, graph_store: GraphStore) -> None:
        """Search on an empty store returns an empty list."""
        assert graph_store.find_similar([1.0, 0.0]) == []

    def test_find_similar_top_k(self, graph_store: GraphStore) -> None:
        for i in range(5):
            v = [float(v) for v in [1 if i == j else 0 for j in range(5)]]
            _add_node(graph_store, f"N{i}", v, f"s{i}")

        query = [1.0, 0.0, 0.0, 0.0, 0.0]
        results = graph_store.find_similar(query, top_k=3)
        assert len(results) == 3


class TestFindSimilarCosineMath:
    """Verify actual cosine similarity values are correct."""

    def test_cosine_math(self, graph_store: GraphStore) -> None:
        id_v1 = _add_node(graph_store, "v1", [1.0, 0.0], "s1")
        id_v2 = _add_node(graph_store, "v2", [0.0, 1.0], "s2")
        id_v3 = _add_node(graph_store, "v3", [1.0, 1.0], "s3")

        query = [1.0, 0.0]

        results = dict(graph_store.find_similar(query, top_k=10))

        # Same vector -> cosine = 1.0
        assert results[id_v1] == pytest.approx(1.0)

        # Orthogonal -> cosine = 0.0
        assert results[id_v2] == pytest.approx(0.0)

        # [1,1] has norm sqrt(2), so dot([1,0],[1,1]) / sqrt(2) = 1/sqrt(2)
        expected = 1.0 / math.sqrt(2)
        assert results[id_v3] == pytest.approx(expected, rel=1e-6)


class TestBumpConfidence:
    """Corroboration logic."""

    def test_bump_new_source(self, graph_store: GraphStore) -> None:
        node_id = _add_node(graph_store, "fact", [0.1, 0.2], "src1")
        orig_updated = graph_store.get_node(node_id).updated_at

        result = graph_store.bump_confidence(
            node_id, {"source_id": "src2", "detail": "second source"}
        )
        assert result is True

        node = graph_store.get_node(node_id)
        assert node.confidence == 2.0  # base 1.0 + increment 1.0
        assert len(node.provenance) == 2
        assert node.provenance[0]["source_id"] == "src1"
        assert node.provenance[1]["source_id"] == "src2"
        assert node.updated_at != orig_updated

    def test_bump_idempotent(self, graph_store: GraphStore) -> None:
        node_id = _add_node(graph_store, "fact", [0.1, 0.2], "src1")
        graph_store.bump_confidence(node_id, {"source_id": "src2"})

        # Bump again with the same source_id.
        result = graph_store.bump_confidence(
            node_id, {"source_id": "src2", "detail": "duplicate"}
        )
        assert result is False

        node = graph_store.get_node(node_id)
        assert node.confidence == 2.0  # still 2.0, not 3.0
        assert len(node.provenance) == 2  # not duplicated
        # Order should be preserved.
        assert node.provenance[0]["source_id"] == "src1"
        assert node.provenance[1]["source_id"] == "src2"

    def test_bump_preserves_provenance_order(self, graph_store: GraphStore) -> None:
        node_id = _add_node(graph_store, "fact", [0.1, 0.2], "src1")
        graph_store.bump_confidence(node_id, {"source_id": "src2"})
        graph_store.bump_confidence(node_id, {"source_id": "src3"})
        graph_store.bump_confidence(node_id, {"source_id": "src4"})

        node = graph_store.get_node(node_id)
        expected = [
            {"source_id": "src1"},
            {"source_id": "src2"},
            {"source_id": "src3"},
            {"source_id": "src4"},
        ]
        assert node.provenance == expected


class TestGetNeighbors:
    """Successor lookups."""

    def test_get_neighbors(self, graph_store: GraphStore) -> None:
        id_a = _add_node(graph_store, "A", [1.0, 0.0], "s1")
        id_b = _add_node(graph_store, "B", [0.0, 1.0], "s2")
        id_c = _add_node(graph_store, "C", [0.5, 0.5], "s3")

        graph_store.add_edge(id_a, id_b, weight=0.9)
        graph_store.add_edge(id_a, id_c, weight=0.8)
        graph_store.add_edge(id_b, id_c, weight=0.7)

        assert set(graph_store.get_neighbors(id_a)) == {id_b, id_c}
        assert graph_store.get_neighbors(id_b) == [id_c]
        assert graph_store.get_neighbors(id_c) == []

    def test_get_neighbors_missing_node(self, graph_store: GraphStore) -> None:
        with pytest.raises(Exception):
            graph_store.get_neighbors("does-not-exist")


class TestGetRoots:
    """Nodes with no incoming edges."""

    def test_get_roots(self, graph_store: GraphStore) -> None:
        id_a = _add_node(graph_store, "A", [1.0, 0.0], "s1")
        id_b = _add_node(graph_store, "B", [0.0, 1.0], "s2")
        id_c = _add_node(graph_store, "C", [0.5, 0.5], "s3")

        graph_store.add_edge(id_a, id_b, weight=0.8)

        roots = graph_store.get_roots()
        assert id_a in roots
        assert id_b not in roots
        assert id_c in roots

    def test_get_roots_empty_graph(self, graph_store: GraphStore) -> None:
        assert graph_store.get_roots() == []


class TestAllNodes:
    """``all_nodes()`` returns every node."""

    def test_all_nodes(self, graph_store: GraphStore) -> None:
        assert graph_store.all_nodes() == []

        id_a = _add_node(graph_store, "A", [1.0, 0.0], "s1")
        id_b = _add_node(graph_store, "B", [0.0, 1.0], "s2")

        nodes = graph_store.all_nodes()
        assert len(nodes) == 2
        ids = {n.id for n in nodes}
        assert ids == {id_a, id_b}

    def test_all_nodes_field_values(self, graph_store: GraphStore) -> None:
        provenance = {"source_id": "s1"}
        node_id = graph_store.add_node("test", [0.5, 0.5], provenance)

        nodes = graph_store.all_nodes()
        node = next(n for n in nodes if n.id == node_id)
        assert node.content == "test"
        assert node.confidence == 1.0
        assert node.provenance == [provenance]
        assert node.level == 0


class TestLoadMissingFiles:
    """Loading when files don't exist = empty graph, no error."""

    def test_load_missing_files(self, tmp_path) -> None:
        path = str(tmp_path / "nonexistent" / "memory.json")
        store = GraphStore(path)
        store.load()  # should not raise

        assert store.all_nodes() == []
        assert store.get_roots() == []
        assert store.find_similar([1.0, 0.0]) == []


class TestLoadOrphanedEmbedding:
    """Embeddings without a matching node are silently ignored."""

    def test_load_orphaned_embedding(self, graph_store: GraphStore, tmp_path) -> None:
        # Save a normal graph.
        node_id = _add_node(graph_store, "exists", [0.1, 0.2], "s1")
        graph_store.save()

        # Manually add an orphan embedding to the sidecar.
        emb_path = os.path.join(
            os.path.dirname(str(tmp_path / "memory.json")), "embeddings.json"
        )
        with open(emb_path) as f:
            sidecar = json.load(f)
        sidecar["orphan-id"] = [0.9, 0.9]
        with open(emb_path, "w") as f:
            json.dump(sidecar, f)

        # Load into a fresh store -- should succeed silently.
        store2 = GraphStore(str(tmp_path / "memory.json"))
        store2.load()

        # Orphan embedding must not create a phantom node.
        assert len(store2.all_nodes()) == 1
        assert store2.get_node(node_id) is not None

        # The orphan embedding should NOT appear in similarity results.
        results = store2.find_similar([0.9, 0.9], top_k=10)
        ids = [nid for nid, _ in results]
        assert "orphan-id" not in ids


class TestLoadNodeWithoutEmbedding:
    """Node without a matching embedding is an error."""

    def test_load_node_without_embedding_errors(
        self, graph_store: GraphStore, tmp_path
    ) -> None:
        node_id = _add_node(graph_store, "orphan", [0.1, 0.2], "s1")
        graph_store.save()

        # Delete the sidecar file entirely.
        emb_path = os.path.join(
            os.path.dirname(str(tmp_path / "memory.json")), "embeddings.json"
        )
        os.remove(emb_path)

        store2 = GraphStore(str(tmp_path / "memory.json"))
        with pytest.raises(ValueError, match="no embedding"):
            store2.load()


class TestSaveIsAtomic:
    """Atomic write via temp file + os.replace."""

    def test_no_tmp_files_remain(self, graph_store: GraphStore, tmp_path) -> None:
        _add_node(graph_store, "data", [1.0, 0.0], "s1")
        graph_store.save()

        # No .tmp files should linger.
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_files_are_valid_json(self, graph_store: GraphStore, tmp_path) -> None:
        _add_node(graph_store, "data", [1.0, 0.0], "s1")
        _add_node(graph_store, "data2", [0.0, 1.0], "s2")
        graph_store.add_edge(
            graph_store.get_roots()[0],
            graph_store.all_nodes()[1].id,
            weight=0.5,
        )
        graph_store.save()

        # Both files must be parseable JSON.
        for fname in ["memory.json", "embeddings.json"]:
            with open(os.path.join(tmp_path, fname)) as f:
                data = json.load(f)
            assert data is not None


class TestMemoryJsonNoEmbeddings:
    """memory.json must not contain embedding vectors."""

    def test_memory_json_no_embeddings(
        self, graph_store: GraphStore, tmp_path
    ) -> None:
        _add_node(graph_store, "A", [0.123, 0.456], "s1")
        _add_node(graph_store, "B", [0.789, 0.321], "s2")
        graph_store.save()

        mem_path = os.path.join(tmp_path, "memory.json")
        with open(mem_path) as f:
            data = json.load(f)

        # Walk the JSON tree and assert no key contains "embed".
        def _check(obj, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    assert "embed" not in key.lower(), (
                        f"Found embedding-related key {key!r} at {path}"
                    )
                    _check(val, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")

        _check(data)
