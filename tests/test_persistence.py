"""Tests for graph persistence: round-trip, pretty-printing, orphan/tolerance."""

import json
import os
from pathlib import Path

import pytest

from src.extractor import store_facts
from src.graph import GraphStore
from tests.conftest import make_lookup_stub
from tests.stubs import LookupStub

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected(path: str) -> list[str]:
    with open(FIXTURE_DIR / path) as f:
        data = json.load(f)
    return data["facts"]


class TestSaveLoadRoundtrip:
    """Save then load into a fresh GraphStore — everything must match."""

    def test_save_load_roundtrip(
        self, hash_stub, tmp_graph_path, monkeypatch,
    ) -> None:
        import src.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        graph = GraphStore(tmp_graph_path)
        raw_facts = _load_expected("conv_01.expected.json")
        store_facts([{"fact": f, "topic": ""} for f in raw_facts], "src1", None, graph)

        # Capture in-memory state
        nodes_before = graph.all_nodes()

        # Load into a fresh store
        graph2 = GraphStore(tmp_graph_path)
        graph2.load()

        nodes_after = graph2.all_nodes()
        assert len(nodes_after) == len(nodes_before)

        for n_after in nodes_after:
            n_before = next(n for n in nodes_before if n.id == n_after.id)
            assert n_after.content == n_before.content
            assert n_after.confidence == n_before.confidence
            assert n_after.provenance == n_before.provenance
            assert n_after.level == n_before.level

        # Verify DiGraph (edges — there should be none for these distinct facts)
        assert len(graph2._graph.edges()) == 0


class TestMemoryJsonPrettyPrinted:
    """memory.json must be pretty-printed with newlines and indentation."""

    def test_memory_json_pretty_printed(
        self, hash_stub, tmp_graph_path, monkeypatch,
    ) -> None:
        import src.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        graph = GraphStore(tmp_graph_path)
        raw_facts = _load_expected("conv_01.expected.json")
        store_facts([{"fact": f, "topic": ""} for f in raw_facts], "src1", None, graph)

        with open(tmp_graph_path) as f:
            content = f.read()

        assert content.startswith("{")
        assert "\n  " in content
        # Verify it parses as valid JSON
        data = json.loads(content)
        assert "nodes" in data


class TestNodeWithoutEmbedding:
    """Node without a matching embedding raises ValueError on load."""

    def test_node_without_embedding_errors(self, tmp_path) -> None:
        from src.config import EMBEDDINGS_PATH
        mem_path = str(tmp_path / "memory.json")
        emb_path = os.path.join(tmp_path, EMBEDDINGS_PATH)

        # Create a graph with one node
        graph = GraphStore(mem_path)
        graph.add_node("orphan fact", [0.1, 0.2], {"source_id": "s1"})
        graph.save()

        # Wipe the embeddings file
        os.remove(emb_path)

        # Loading should fail
        graph2 = GraphStore(mem_path)
        with pytest.raises(ValueError, match="no embedding"):
            graph2.load()


class TestOrphanEmbedding:
    """Orphan embedding (no matching node) is silently tolerated."""

    def test_orphan_embedding_tolerated(self, tmp_path) -> None:
        from src.config import EMBEDDINGS_PATH
        mem_path = str(tmp_path / "memory.json")
        emb_path = os.path.join(tmp_path, EMBEDDINGS_PATH)

        graph = GraphStore(mem_path)
        node_id = graph.add_node("real fact", [0.1, 0.2], {"source_id": "s1"})
        graph.save()

        # Add an orphan embedding to the sidecar
        with open(emb_path) as f:
            sidecar = json.load(f)
        sidecar["orphan-id"] = [0.9, 0.9]
        with open(emb_path, "w") as f:
            json.dump(sidecar, f)

        # Load should succeed
        graph2 = GraphStore(mem_path)
        graph2.load()

        assert len(graph2.all_nodes()) == 1
        assert graph2.get_node(node_id) is not None

        # Orphan should not appear in similarity results
        results = graph2.find_similar([0.9, 0.9], top_k=10)
        ids = [nid for nid, _ in results]
        assert "orphan-id" not in ids
