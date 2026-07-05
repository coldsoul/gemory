"""End-to-end dump: feed fixtures through the full real stack and dump the
resulting graph for eyeballing.

Skipped unless GEMORY_LIVE=1 is set in the environment.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GEMORY_LIVE") != "1",
    reason="Live suite requires GEMORY_LIVE=1",
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _load_facts(name: str) -> list[str]:
    path = FIXTURE_DIR / f"{name}.expected.json"
    with open(path) as f:
        return json.load(f)["facts"]


class TestEndToEndDump:
    """Run the full real stack and dump the graph for manual inspection."""

    def test_end_to_end(self, tmp_path):
        """Feed conv_01 through real extractor and store, dump graph."""
        import gemory.llm as llm
        from gemory.graph import GraphStore
        from gemory.extractor import compute_source_id, store_facts

        transcript = (FIXTURE_DIR / "conv_01.txt").read_text()

        # Extract facts with real model
        facts = llm.extract_facts(transcript)
        print(f"\nExtracted {len(facts)} facts from conv_01")

        # Store with real embeddings
        source_id = compute_source_id(transcript)
        memory_path = str(tmp_path / "memory.json")
        graph = GraphStore(memory_path)
        summary = store_facts(facts, source_id, "e2e_dump", graph)

        print(f"Summary: {json.dumps(summary)}")

        # Dump graph
        nodes = graph.all_nodes()
        print(f"\nGraph ({len(nodes)} nodes):")
        for node in nodes:
            print(f"  [{node.confidence:.1f}] {node.content}")
            for p in node.provenance:
                print(f"    source={p['source_id']}, label={p.get('label', '')}")

        # Dump edges
        for node in nodes:
            neighbors = graph.get_neighbors(node.id)
            if neighbors:
                for n in neighbors:
                    content = graph.get_node(n).content
                    print(f"  EDGE: {node.content[:40]} → {content[:40]}")

        print("\nReview extraction quality, dedup behavior, and edge creation.")
