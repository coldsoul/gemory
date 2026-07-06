"""Tests for basic fact storage: fresh store, empty facts, embeddings not leaked."""

import json
from pathlib import Path

import pytest

from gemory.extractor import store_facts
from gemory.graph import GraphStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected(path: str) -> list[str]:
    """Load expected facts from a fixture JSON file."""
    with open(FIXTURE_DIR / path) as f:
        data = json.load(f)
    return data["facts"]


class TestFreshStore:
    """Verify a fresh store creates correct nodes from known facts."""

    def test_fresh_store_creates_correct_nodes(
        self, hash_stub, empty_graph, monkeypatch,
    ) -> None:
        """Feed conv_01 expected facts into store_facts with hash_stub."""
        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        facts = _load_expected("conv_01.expected.json")
        source_id = "test-src-fresh"
        label = "test"

        fact_items = [{"fact": f, "topic": ""} for f in facts]
        result = store_facts(fact_items, source_id, label, empty_graph)

        assert result["facts_extracted"] == len(facts)
        assert result["new_nodes"] == len(facts)
        assert result["corroborated"] == 0
        assert result["skipped"] == 0

        nodes = empty_graph.all_nodes()
        assert len(nodes) == len(facts)

        for node in nodes:
            assert len(node.provenance) == 1
            assert node.provenance[0]["source_id"] == source_id
            assert node.confidence == 1.0


class TestMemoryJsonNoEmbeddings:
    """memory.json must not contain embedding vectors."""

    def test_memory_json_has_no_embeddings(
        self, hash_stub, tmp_graph_path, monkeypatch,
    ) -> None:
        """Store facts, save, read memory.json raw, assert no embedding keys."""
        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        graph = GraphStore(tmp_graph_path)
        raw_facts = _load_expected("conv_01.expected.json")
        store_facts([{"fact": f, "topic": ""} for f in raw_facts], "test-src", None, graph)

        # Read memory.json raw
        with open(tmp_graph_path) as f:
            data = json.load(f)

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


class TestEmptyExtraction:
    """store_facts with empty fact list produces zero nodes."""

    def test_empty_extraction_produces_zero_nodes(
        self, hash_stub, empty_graph, monkeypatch,
    ) -> None:
        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        result = store_facts([], "test-src", None, empty_graph)

        assert result == {
            "facts_extracted": 0,
            "new_nodes": 0,
            "corroborated": 0,
            "skipped": 0,
        }
        assert len(empty_graph.all_nodes()) == 0
