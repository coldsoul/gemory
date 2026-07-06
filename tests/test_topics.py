"""Tests for :mod:`gemory.topics` — topic registry with match-or-create."""

import numpy as np
import pytest

from gemory import config
from tests.stubs import LookupStub, vectors_with_cosines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def topic_graph(tmp_graph_path):
    """A fresh, empty GraphStore for topic tests."""
    from gemory.graph import GraphStore
    return GraphStore(tmp_graph_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolveTopic:
    """``resolve_topic`` match-or-create logic."""

    def test_creates_topic_on_first_use(self, topic_graph, monkeypatch) -> None:
        """First call with a non-empty topic creates a topic node."""
        import gemory.topics as top
        stub = LookupStub({"Gemory": [1.0, 0.0, 0.0]}, dim=3)
        monkeypatch.setattr(top, "embed", stub.embed)

        topic_id = top.resolve_topic(topic_graph, "Gemory")
        assert topic_id is not None

        node = topic_graph.get_node(topic_id)
        assert node.kind == "abstraction"
        assert node.abstraction_kind == "topic"
        assert node.label == "Gemory"
        assert node.level == 1

    def test_matches_existing_topic_above_threshold(
        self, topic_graph, monkeypatch,
    ) -> None:
        """A close variant resolves to the same existing topic node."""
        import gemory.topics as top

        # Two topic vectors at cosine 0.90 (above default TOPIC_MATCH_THRESHOLD=0.85)
        gram = np.array([[1.0, 0.90], [0.90, 1.0]])
        vecs = vectors_with_cosines(gram)
        stub = LookupStub({
            "Gemory": vecs[0].tolist(),
            "Gemory memory system": vecs[1].tolist(),
        }, dim=vecs.shape[1])
        monkeypatch.setattr(top, "embed", stub.embed)

        # Create first topic
        id1 = top.resolve_topic(topic_graph, "Gemory")
        assert id1 is not None
        assert len(topic_graph.get_topic_nodes()) == 1

        # Resolve a close variant — should match the existing one
        id2 = top.resolve_topic(topic_graph, "Gemory memory system")
        assert id2 == id1
        # No new topic node created
        assert len(topic_graph.get_topic_nodes()) == 1

    def test_creates_new_topic_below_threshold(
        self, topic_graph, monkeypatch,
    ) -> None:
        """A far-apart topic string creates a new topic node."""
        import gemory.topics as top

        # "Gemory" at basis [1,0,0], "FPV drones" at basis [0,1,0] (cosine 0)
        stub = LookupStub({
            "Gemory": [1.0, 0.0, 0.0],
            "FPV drones": [0.0, 1.0, 0.0],
        }, dim=3)
        monkeypatch.setattr(top, "embed", stub.embed)

        id1 = top.resolve_topic(topic_graph, "Gemory")
        assert id1 is not None

        id2 = top.resolve_topic(topic_graph, "FPV drones")
        assert id2 is not None
        assert id2 != id1
        assert len(topic_graph.get_topic_nodes()) == 2

    def test_empty_topic_returns_none(self, topic_graph, monkeypatch) -> None:
        import gemory.topics as top
        stub = LookupStub({}, dim=3)
        monkeypatch.setattr(top, "embed", stub.embed)

        result = top.resolve_topic(topic_graph, "")
        assert result is None
        assert len(topic_graph.get_topic_nodes()) == 0

    def test_whitespace_only_topic_returns_none(
        self, topic_graph, monkeypatch,
    ) -> None:
        import gemory.topics as top
        stub = LookupStub({}, dim=3)
        monkeypatch.setattr(top, "embed", stub.embed)

        result = top.resolve_topic(topic_graph, "   ")
        assert result is None
        assert len(topic_graph.get_topic_nodes()) == 0
