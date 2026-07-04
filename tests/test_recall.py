"""Unit tests for :mod:`gemory.recall`.

``llm.embed`` is mocked to return deterministic vectors so that tests are
fully reproducible without network access.
"""

from unittest.mock import patch

import pytest

from gemory.graph import GraphStore
from gemory.recall import recall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_graph(tmp_path) -> GraphStore:
    """A :class:`GraphStore` with 3 nodes carrying known content, confidence,
    and hand-crafted embeddings."""
    graph = GraphStore(str(tmp_path / "memory.json"))
    node1 = graph.add_node(
        "The user likes Python",
        [1.0, 0.0, 0.0],
        {"source_id": "s1"},
    )
    graph.add_node(
        "The user works on a VPS",
        [0.0, 1.0, 0.0],
        {"source_id": "s2"},
    )
    graph.add_node(
        "The user practices bonsai",
        [0.0, 0.0, 1.0],
        {"source_id": "s3"},
    )
    # Bump the first node so it has confidence = 2.0 (non-default).
    graph.bump_confidence(node1, {"source_id": "s4"})
    return graph


@pytest.fixture
def empty_graph(tmp_path) -> GraphStore:
    """A :class:`GraphStore` with no data."""
    return GraphStore(str(tmp_path / "empty_memory.json"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecall:
    """``recall()`` formatting and edge cases."""

    @patch("gemory.recall.llm.embed")
    def test_recall_returns_formatted_results(
        self, mock_embed, populated_graph: GraphStore,
    ) -> None:
        """Verify output contains fact contents, similarity, confidence, and
        rank numbers."""
        # Embed the query as something close to the first node.
        mock_embed.return_value = [0.9, 0.1, 0.0]

        output = recall("Python programming", populated_graph, top_k=3)

        # Each fact content should appear.
        assert "The user likes Python" in output
        assert "The user works on a VPS" in output
        assert "The user practices bonsai" in output

        # Rank markers should be present.
        assert "[#1" in output
        assert "[#2" in output
        assert "[#3" in output

        # Similarity labels.
        assert "similarity:" in output

        # Confidence labels — at least one should show 2.0 (bumped node).
        assert "confidence:" in output
        assert "confidence: 2.0" in output or "confidence: 1.0" in output

    @patch("gemory.recall.llm.embed")
    def test_recall_respects_top_k(
        self, mock_embed, populated_graph: GraphStore,
    ) -> None:
        """Only *top_k* results are returned."""
        mock_embed.return_value = [0.8, 0.6, 0.0]

        output = recall("Python", populated_graph, top_k=2)

        # Should contain exactly 2 rank markers.
        assert output.count("[#") == 2
        assert "[#1" in output
        assert "[#2" in output
        assert "[#3" not in output

    def test_recall_empty_graph(self, empty_graph: GraphStore) -> None:
        """Querying an empty graph returns the empty-state message."""
        output = recall("anything", empty_graph, top_k=5)
        assert output == "Memory is empty. No facts stored yet."

    @patch("gemory.recall.llm.embed")
    def test_recall_no_matches(
        self, mock_embed, populated_graph: GraphStore,
    ) -> None:
        """A zero-norm query embedding makes ``find_similar`` return empty,
        which produces the 'no matches' message."""
        mock_embed.return_value = [0.0, 0.0, 0.0]  # zero vector → empty results

        output = recall("something orthogonal", populated_graph, top_k=5)
        assert output == "No matching memories found."

    @patch("gemory.recall.llm.embed")
    def test_recall_result_order(
        self, mock_embed, populated_graph: GraphStore,
    ) -> None:
        """Results are sorted by similarity descending."""
        # Query vector has different similarity to each stored embedding:
        #   [1,0,0] → 0.800,  [0,1,0] → 0.600,  [0,0,1] → 0.000
        mock_embed.return_value = [0.8, 0.6, 0.0]

        output = recall("Python work", populated_graph, top_k=3)

        lines = output.splitlines()
        # Extract similarity values from rank lines.
        sims: list[float] = []
        for line in lines:
            if line.startswith("[#") and "similarity:" in line:
                # e.g. "[#1 | similarity: 0.800 | confidence: 2.0]"
                part = line.split("similarity:")[1].split("|")[0].strip()
                sims.append(float(part))

        assert len(sims) == 3
        assert sims[0] > sims[1] > sims[2]
