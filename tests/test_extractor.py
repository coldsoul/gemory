"""Unit tests for :mod:`gemory.extractor`.

``compute_source_id`` tests are pure function tests.
``store_facts`` tests mock ``gemory.extractor.llm.embed`` so that every fact
maps to a deterministic, hand-crafted embedding vector.
"""

import hashlib
from unittest.mock import patch

import pytest

from gemory.extractor import compute_source_id, store_facts
from gemory.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_graph(tmp_path) -> GraphStore:
    """A :class:`GraphStore` backed by a temporary directory."""
    return GraphStore(str(tmp_path / "memory.json"))


@pytest.fixture
def sample_transcript() -> str:
    """A multi-turn conversation transcript for ``compute_source_id`` tests."""
    return (
        "User: Hello, I like Python.\n"
        "Assistant: That's great! Python is a versatile language.\n"
        "User: I use it for data science mostly.\n"
        "Assistant: Awesome, what libraries do you use?\n"
        "User: Pandas and numpy are my go-to."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_side_effect(mapping: dict[str, list[float]]):
    """Return a ``side_effect`` function for ``llm.embed`` that maps fact
    strings to deterministic embedding vectors."""
    def side_effect(text: str) -> list[float]:
        return mapping[text]
    return side_effect


# ===================================================================
# compute_source_id tests
# ===================================================================

class TestComputeSourceId:
    """Stable source-id derivation from conversation transcripts."""

    def test_same_first_exchange_same_id(self) -> None:
        t1 = (
            "User: Hello\nAssistant: Hi\n"
            "User: How are you?\nAssistant: Fine, thanks"
        )
        t2 = (
            "User: Hello\nAssistant: Hi\n"
            "User: Totally different\nAssistant: Totally different"
        )
        assert compute_source_id(t1) == compute_source_id(t2)

    def test_different_first_exchange_different_id(self) -> None:
        t1 = "User: Hello\nAssistant: Hi"
        t2 = "User: Goodbye\nAssistant: Bye"
        assert compute_source_id(t1) != compute_source_id(t2)

    def test_whitespace_normalization(self) -> None:
        t1 = "User: Hello\nAssistant: Hi"
        t2 = "User:   Hello\r\nAssistant: Hi"
        assert compute_source_id(t1) == compute_source_id(t2)

    def test_boundary_collision_prevention(self) -> None:
        # Without separator "AB" + "C" would collide with "A" + "BC".
        t1 = "User: AB\nAssistant: C"
        t2 = "User: A\nAssistant: BC"
        assert compute_source_id(t1) != compute_source_id(t2)

    def test_single_turn_fallback(self) -> None:
        transcript = "User: Only me, no assistant"
        sid = compute_source_id(transcript)
        expected_digest = hashlib.sha256(
            "Only me, no assistant".encode("utf-8")
        ).hexdigest()
        assert sid == f"sha256:{expected_digest}"

    def test_no_user_message_raises(self) -> None:
        with pytest.raises(ValueError, match="no user message"):
            compute_source_id("Assistant: Hello there\nAssistant: Still here")

    def test_handling_of_role_variants(self) -> None:
        sid_human = compute_source_id(
            "Human: Hello\nAI: Hi, how can I help?"
        )
        sid_user = compute_source_id(
            "User: Hello\nAssistant: Hi, how can I help?"
        )

        # Both must parse and produce a valid sha256:... string.
        assert sid_human.startswith("sha256:")
        assert sid_user.startswith("sha256:")

        # Because the content is identical after normalisation AND the role
        # prefixes are stripped, the hashes should match.
        assert sid_human == sid_user

    def test_sample_transcript(self, sample_transcript: str) -> None:
        """Smoke test: a realistic transcript produces a valid source_id."""
        sid = compute_source_id(sample_transcript)
        assert sid.startswith("sha256:")
        assert len(sid) == len("sha256:") + 64  # 64 hex chars


# ===================================================================
# store_facts tests
# ===================================================================

class TestStoreFacts:
    """Idempotent fact storage with dedup, corroboration, and edge creation."""

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    @patch("gemory.extractor.llm.embed")
    def test_exact_rerun_idempotency(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """Same facts + same source_id on second run → all skipped."""
        mock_embed.side_effect = _embed_side_effect({"A": [1.0, 0.0, 0.0]})

        # First run
        r1 = store_facts(["A"], "src1", None, temp_graph)
        assert r1["new_nodes"] == 1
        assert r1["corroborated"] == 0
        assert r1["skipped"] == 0

        # Second run with identical source_id
        r2 = store_facts(["A"], "src1", None, temp_graph)
        assert r2["new_nodes"] == 0
        assert r2["corroborated"] == 0
        assert r2["skipped"] == 1

        assert len(temp_graph.all_nodes()) == 1

    @patch("gemory.extractor.llm.embed")
    def test_grown_transcript(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """Second run has one old (skipped) and one new fact."""
        mock_embed.side_effect = _embed_side_effect({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
        })

        # First run: only A
        r1 = store_facts(["A"], "src1", None, temp_graph)
        assert r1["new_nodes"] == 1
        node_ids_r1 = {n.id for n in temp_graph.all_nodes()}

        # Second run: A (same source → skip) + B (new)
        r2 = store_facts(["A", "B"], "src1", None, temp_graph)
        assert r2["new_nodes"] == 1
        assert r2["skipped"] == 1
        assert r2["corroborated"] == 0

        nodes = temp_graph.all_nodes()
        assert len(nodes) == 2

        # A's provenance should have exactly 1 entry (not duplicated)
        a_node = next(n for n in nodes if n.id in node_ids_r1)
        assert len(a_node.provenance) == 1
        assert a_node.provenance[0]["source_id"] == "src1"

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    @patch("gemory.extractor.llm.embed")
    def test_dedup_merges(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """A close fact (above DEDUP_THRESHOLD) corroborates the existing
        node rather than creating a new one."""
        # embedding for "X": [1,0,0]
        # embedding for "X_similar": [0.95, ~0.312, 0]  cosine = 0.95
        mock_embed.side_effect = _embed_side_effect({
            "X": [1.0, 0.0, 0.0],
            "X_similar": [0.95, 0.3122498999199199, 0.0],
        })

        # First call — creates node
        store_facts(["X"], "src1", None, temp_graph)
        assert len(temp_graph.all_nodes()) == 1

        # Second call with different source — should corroborate
        r2 = store_facts(["X_similar"], "src2", None, temp_graph)
        assert r2["new_nodes"] == 0
        assert r2["corroborated"] == 1
        assert r2["skipped"] == 0
        assert len(temp_graph.all_nodes()) == 1  # still 1 node

        # Verify confidence was bumped exactly once
        node = temp_graph.all_nodes()[0]
        assert node.confidence == 2.0  # base 1.0 + 1 increment
        assert len(node.provenance) == 2

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    @patch("gemory.extractor.llm.embed")
    def test_edge_creation_for_close_but_distinct(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """New fact similarity is above EDGE_THRESHOLD but below
        DEDUP_THRESHOLD → new node + edge created."""
        # existing: [1,0,0]
        # new:      [0.8, 0.6, 0]    cosine = 0.8
        mock_embed.side_effect = _embed_side_effect({
            "Existing": [1.0, 0.0, 0.0],
            "NewClose": [0.8, 0.6, 0.0],
        })

        # Seed one existing node
        store_facts(["Existing"], "src1", None, temp_graph)
        existing_id = temp_graph.all_nodes()[0].id

        # Store the close-but-distinct fact
        r = store_facts(["NewClose"], "src2", None, temp_graph)
        assert r["new_nodes"] == 1
        assert len(temp_graph.all_nodes()) == 2

        # Edge should exist from new node to existing node
        new_node = next(
            n for n in temp_graph.all_nodes() if n.id != existing_id
        )
        neighbors = temp_graph.get_neighbors(new_node.id)
        assert existing_id in neighbors

    @patch("gemory.extractor.llm.embed")
    def test_edge_not_created_for_distant_facts(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """New fact similarity is below EDGE_THRESHOLD → new node but NO edge."""
        mock_embed.side_effect = _embed_side_effect({
            "Existing": [1.0, 0.0, 0.0],
            "NewDistant": [0.0, 0.0, 1.0],
        })

        store_facts(["Existing"], "src1", None, temp_graph)
        existing_id = temp_graph.all_nodes()[0].id

        r = store_facts(["NewDistant"], "src2", None, temp_graph)
        assert r["new_nodes"] == 1
        assert len(temp_graph.all_nodes()) == 2

        new_node = next(
            n for n in temp_graph.all_nodes() if n.id != existing_id
        )
        assert temp_graph.get_neighbors(new_node.id) == []

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    @patch("gemory.extractor.llm.embed")
    def test_store_facts_returns_correct_summary(
        self, mock_embed, temp_graph: GraphStore,
    ) -> None:
        """Returned dict has the right counts for a mixed scenario."""
        mock_embed.side_effect = _embed_side_effect({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
        })

        # Both A and B are new
        r1 = store_facts(["A", "B"], "src1", None, temp_graph)
        assert r1 == {"facts_extracted": 2, "new_nodes": 2,
                       "corroborated": 0, "skipped": 0}

        # A is skipped (same source), B is skipped too
        r2 = store_facts(["A", "B"], "src1", None, temp_graph)
        assert r2 == {"facts_extracted": 2, "new_nodes": 0,
                       "corroborated": 0, "skipped": 2}

        # A is corroborated (new source)
        r3 = store_facts(["A"], "src2", None, temp_graph)
        assert r3 == {"facts_extracted": 1, "new_nodes": 0,
                       "corroborated": 1, "skipped": 0}

    # ------------------------------------------------------------------
    # Save behaviour
    # ------------------------------------------------------------------

    @patch("gemory.extractor.llm.embed")
    def test_store_facts_saves_at_end(
        self, mock_embed, temp_graph: GraphStore, tmp_path,
    ) -> None:
        """After ``store_facts`` completes, loading a fresh store from
        disk yields the same data — proving ``save()`` was called."""
        mock_embed.side_effect = _embed_side_effect({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
        })

        store_facts(["A", "B"], "src1", None, temp_graph)

        # Load into a brand-new store at the same path
        store2 = GraphStore(str(tmp_path / "memory.json"))
        store2.load()

        assert len(store2.all_nodes()) == 2
        assert store2.find_similar([1.0, 0.0, 0.0], top_k=1)
