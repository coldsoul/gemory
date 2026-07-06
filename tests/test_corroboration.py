"""Tests for cross-source corroboration of matching facts."""

import json
from pathlib import Path

import numpy as np
import pytest

from gemory import config
from gemory.extractor import store_facts
from gemory.graph import GraphStore
from tests.stubs import vectors_with_cosines

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected(path: str) -> list[str]:
    with open(FIXTURE_DIR / path) as f:
        data = json.load(f)
    return data["facts"]


class TestCrossSourceCorroboration:
    """Two different sources share some facts — those should be corroborated."""

    def test_cross_source_corroboration(self, monkeypatch, tmp_path) -> None:
        dedup = config.DEDUP_THRESHOLD
        edge = config.EDGE_THRESHOLD

        # ── Known fact texts ─────────────────────────────────────────────
        # conv_01 expects:
        f1 = "The user is building a memory system called Gemory."
        f2 = "The user works on a VPS."
        # conv_02 expects:
        g1 = "Gemory is the user's long-term memory project."   # matches f1
        g2 = "The user runs their code on a virtual private server."  # matches f2
        g3 = "The user prefers minimal, inspectable implementations."  # new
        g4 = "The user uses uv for package management."                # new
        g5 = "The user flies FPV drones."                              # new

        # ── Build controlled vectors ────────────────────────────────────
        # We need 7 vectors total.  Use 5D space:
        #   f1 and g1 share the first axis (cosine=1.0)
        #   f2 and g2 share the second axis (cosine=1.0)
        #   g3, g4, g5 are orthogonal to the first two axes and each other
        #   All cross pairs not listed are 0.0 (< EDGE_THRESHOLD)
        labels = [f1, f2, g1, g2, g3, g4, g5]
        n = len(labels)
        gram = np.eye(n)

        # f1 <-> g1 (index 0 <-> 2): cosine = 1.0 (same concept)
        gram[0, 2] = gram[2, 0] = 1.0

        # f2 <-> g2 (index 1 <-> 3): cosine = 1.0 (same concept)
        gram[1, 3] = gram[3, 1] = 1.0

        vectors = vectors_with_cosines(gram, dim=n)
        lookup = {label: vectors[i].tolist() for i, label in enumerate(labels)}

        from tests.stubs import LookupStub
        stub = LookupStub(lookup, dim=n)

        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", stub.embed)

        graph = GraphStore(str(tmp_path / "memory.json"))

        # ── First source: conv_01 facts ─────────────────────────────────
        r1 = store_facts(
            [{"fact": f, "topic": ""} for f in [f1, f2]],
            "src1", "conv_01", graph,
        )
        assert r1["new_nodes"] == 2

        # ── Second source: conv_02 facts ────────────────────────────────
        r2 = store_facts(
            [{"fact": f, "topic": ""} for f in [g1, g2, g3, g4, g5]],
            "src2", "conv_02", graph,
        )
        assert r2["new_nodes"] == 3        # g3, g4, g5
        assert r2["corroborated"] == 2     # g1->f1, g2->f2
        assert r2["skipped"] == 0

        # ── Assertions ──────────────────────────────────────────────────
        nodes = graph.all_nodes()
        assert len(nodes) == 5  # 2 (conv_01) + 3 new (conv_02)

        for node in nodes:
            if node.content in (f1, f2):
                # Corroborated nodes: 2 provenance entries, bumped confidence
                assert len(node.provenance) == 2
                source_ids = {p["source_id"] for p in node.provenance}
                assert source_ids == {"src1", "src2"}
                assert node.confidence == 1.0 + config.CONFIDENCE_INCREMENT
            else:
                # New nodes: 1 provenance entry, base confidence
                assert len(node.provenance) == 1
                assert node.provenance[0]["source_id"] == "src2"
                assert node.confidence == 1.0
