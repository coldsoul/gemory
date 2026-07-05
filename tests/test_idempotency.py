"""Tests for idempotency: exact rerun and grown transcript."""

import json
from pathlib import Path

import pytest

from gemory.extractor import compute_source_id, store_facts
from tests.graph_diff import diff, snapshot

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected(path: str) -> list[str]:
    with open(FIXTURE_DIR / path) as f:
        data = json.load(f)
    return data["facts"]


class TestExactRerunIdempotency:
    """Running store_facts twice with identical inputs changes nothing."""

    def test_exact_rerun_idempotency(
        self, hash_stub, empty_graph, monkeypatch,
    ) -> None:
        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        facts = _load_expected("conv_01.expected.json")
        transcript_path = FIXTURE_DIR / "conv_01.txt"
        source_id = compute_source_id(transcript_path.read_text())

        # First run
        store_facts(facts, source_id, None, empty_graph)
        snap_before = snapshot(empty_graph)

        # Second run — exact same input
        store_facts(facts, source_id, None, empty_graph)
        snap_after = snapshot(empty_graph)

        d = diff(snap_before, snap_after)
        assert d.is_empty, f"Expected no changes on exact rerun:\n{d.summary()}"


class TestGrownTranscriptIdempotency:
    """Rerunning with extra facts only adds the new ones; originals are skipped."""

    def test_grown_transcript_idempotency(
        self, hash_stub, empty_graph, monkeypatch, tmp_path,
    ) -> None:
        import gemory.extractor as ext_mod
        monkeypatch.setattr(ext_mod.llm, "embed", hash_stub.embed)

        original_facts = _load_expected("conv_01.expected.json")
        transcript_path = FIXTURE_DIR / "conv_01.txt"
        source_id = compute_source_id(transcript_path.read_text())

        # First run: store original facts
        store_facts(original_facts, source_id, None, empty_graph)
        snap_before = snapshot(empty_graph)

        assert len(snap_before.node_contents) == len(original_facts)

        # Second run: original facts + one new fact, SAME source_id
        new_fact = "The user wants to add a consolidation process."
        grown_facts = original_facts + [new_fact]

        r = store_facts(grown_facts, source_id, None, empty_graph)
        assert r["new_nodes"] == 1
        assert r["skipped"] == len(original_facts)
        assert r["corroborated"] == 0

        snap_after = snapshot(empty_graph)
        assert len(snap_after.node_contents) == len(grown_facts)

        # Original nodes should have unchanged provenance and confidence
        for nid, orig_prov in snap_before.node_provenance.items():
            after_prov = snap_after.node_provenance[nid]
            assert after_prov == orig_prov, (
                f"Provenance changed for {nid[:8]} on rerun"
            )
        for nid, orig_conf in snap_before.node_confidences.items():
            after_conf = snap_after.node_confidences[nid]
            assert after_conf == orig_conf, (
                f"Confidence changed for {nid[:8]} on rerun"
            )
