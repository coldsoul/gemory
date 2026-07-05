"""Tests for :func:`gemory.recall.recall` using stub embeddings."""

import pytest

from gemory.graph import GraphStore
from gemory.recall import recall
from tests.stubs import LookupStub


def _build_recall_lookup() -> LookupStub:
    """Return a LookupStub with 3 distinct concepts at known 3D vectors."""
    return LookupStub({
        # Three orthogonal unit vectors in 3D
        "The user likes Python":          [1.0, 0.0, 0.0],
        "The user works on a VPS":        [0.0, 1.0, 0.0],
        "The user practices bonsai":      [0.0, 0.0, 1.0],
        # Query vectors
        "Python query":                   [1.0, 0.0, 0.0],
        "VPS query":                      [0.8, 0.6, 0.0],
    }, dim=3)


class TestRecall:
    """``recall`` with stub embeddings."""

    @pytest.fixture
    def populated_graph(self, tmp_path) -> GraphStore:
        graph = GraphStore(str(tmp_path / "memory.json"))
        stub = _build_recall_lookup()
        graph.add_node("The user likes Python", stub.embed("The user likes Python"),
                       {"source_id": "s1"})
        graph.add_node("The user works on a VPS", stub.embed("The user works on a VPS"),
                       {"source_id": "s2"})
        graph.add_node("The user practices bonsai", stub.embed("The user practices bonsai"),
                       {"source_id": "s3"})
        # Bump one node for non-default confidence
        graph.bump_confidence(
            graph.all_nodes()[0].id, {"source_id": "s4"},
        )
        return graph

    def test_recall_ranks_relevant_node_highest(
        self, populated_graph, monkeypatch,
    ) -> None:
        """A query matching a specific fact should rank that fact first."""
        import gemory.recall as rec_mod
        stub = _build_recall_lookup()
        monkeypatch.setattr(rec_mod.llm, "embed", stub.embed)

        output = recall("Python query", populated_graph, top_k=3)
        lines = output.splitlines()
        rank_lines = [l for l in lines if l.startswith("[#")]
        assert len(rank_lines) == 3

        first_content = lines[lines.index(rank_lines[0]) + 1]
        assert first_content == "The user likes Python"

    def test_recall_respects_top_k(
        self, populated_graph, monkeypatch,
    ) -> None:
        import gemory.recall as rec_mod
        stub = _build_recall_lookup()
        monkeypatch.setattr(rec_mod.llm, "embed", stub.embed)

        output = recall("Python query", populated_graph, top_k=2)
        assert output.count("[#") == 2

    def test_recall_empty_graph(self, tmp_path, monkeypatch) -> None:
        import gemory.recall as rec_mod
        stub = _build_recall_lookup()
        monkeypatch.setattr(rec_mod.llm, "embed", stub.embed)

        empty = GraphStore(str(tmp_path / "empty.json"))
        output = recall("anything", empty, top_k=5)
        assert output == "Memory is empty. No facts stored yet."
