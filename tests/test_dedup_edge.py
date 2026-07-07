"""Tests for dedup merging and edge creation between thresholds.

Uses vectors_with_cosines to build controlled similarity matrices.
Reads DEDUP_THRESHOLD and EDGE_THRESHOLD from config at test time.
"""

import pytest

from src import config
from src.extractor import store_facts
from src.graph import GraphStore
from tests.stubs import LookupStub, vectors_with_cosines


def _build_lookup(gram: dict[tuple[str, str], float], dim: int = 4) -> LookupStub:
    """Build a LookupStub from a dict mapping (fact_label, fact_label) -> cosine.

    The *gram* dict only needs to specify off-diagonal entries for pairs that
    matter; all unspecified pairs default to 0.0.
    """
    labels = sorted({k[0] for k in gram} | {k[1] for k in gram})
    n = len(labels)
    mat = [[1.0] * n for _ in range(n)]
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                mat[i][j] = gram.get((a, b), gram.get((b, a), 0.0))
                mat[j][i] = mat[i][j]
    vectors = vectors_with_cosines(mat, dim=dim)
    lookup = {}
    for label, vec in zip(labels, vectors):
        lookup[label] = vec.tolist()
    return LookupStub(lookup, dim=dim)


class TestMergeAboveDedup:
    """Facts above DEDUP_THRESHOLD merge into a single node."""

    def test_merge_above_dedup(self, monkeypatch, tmp_path) -> None:
        dedup = config.DEDUP_THRESHOLD  # 0.80 (from .env)
        edge = config.EDGE_THRESHOLD    # 0.75

        # A and B at similarity 0.95 (well above dedup threshold)
        stub = _build_lookup({("A", "B"): 0.95})
        import src.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", stub.embed)

        graph = GraphStore(str(tmp_path / "memory.json"))

        r1 = store_facts([{"fact": "A", "topics": []}], "src1", None, graph)
        assert r1["new_nodes"] == 1

        r2 = store_facts([{"fact": "B", "topics": []}], "src2", None, graph)
        assert r2["new_nodes"] == 0
        assert r2["corroborated"] == 1
        assert r2["skipped"] == 0

        nodes = graph.all_nodes()
        assert len(nodes) == 1  # merged into one node
        assert len(nodes[0].provenance) == 2  # both sources recorded


class TestEdgeBetweenThresholds:
    """New fact between thresholds creates a new node AND an edge."""

    def test_edge_between_thresholds(self, monkeypatch, tmp_path) -> None:
        dedup = config.DEDUP_THRESHOLD
        edge = config.EDGE_THRESHOLD

        # A and C at similarity 0.78 (between thresholds for .env config)
        stub = _build_lookup({("A", "C"): 0.78})
        import src.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", stub.embed)

        graph = GraphStore(str(tmp_path / "memory.json"))

        r1 = store_facts([{"fact": "A", "topics": []}], "src1", None, graph)
        assert r1["new_nodes"] == 1
        node_a = graph.all_nodes()[0].id

        r2 = store_facts([{"fact": "C", "topics": []}], "src2", None, graph)
        assert r2["new_nodes"] == 1
        assert r2["corroborated"] == 0
        assert r2["skipped"] == 0

        nodes = graph.all_nodes()
        assert len(nodes) == 2

        # Find the new node (the one that is NOT A)
        node_c = next(n.id for n in nodes if n.id != node_a)

        # Edge should exist from C -> A (new node to existing)
        neighbors = graph.get_neighbors(node_c)
        assert node_a in neighbors


class TestNoEdgeBelowEdge:
    """New fact below EDGE_THRESHOLD creates a new node but NO edge."""

    def test_no_edge_below_edge(self, monkeypatch, tmp_path) -> None:
        # A and D at similarity 0.50 (well below edge threshold)
        stub = _build_lookup({("A", "D"): 0.50})
        import src.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", stub.embed)

        graph = GraphStore(str(tmp_path / "memory.json"))

        r1 = store_facts([{"fact": "A", "topics": []}], "src1", None, graph)
        r2 = store_facts([{"fact": "D", "topics": []}], "src2", None, graph)
        assert r2["new_nodes"] == 1

        nodes = graph.all_nodes()
        assert len(nodes) == 2

        # No edges should exist
        for node in nodes:
            assert graph.get_neighbors(node.id) == []
