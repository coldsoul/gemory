"""Live end-to-end: extract facts with topics from a real transcript
and review the topic assignments. Skipped unless GEMORY_LIVE=1."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GEMORY_LIVE") != "1",
    reason="Live suite requires GEMORY_LIVE=1",
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


class TestTopicEndToEnd:
    """Run extraction with topics on a real transcript and review output."""

    def test_topic_extraction_review(self):
        """Print fact+topic pairs for conv_02 for human review."""
        import gemory.llm as llm

        transcript = (FIXTURE_DIR / "conv_02.txt").read_text()
        facts = llm.extract_facts(transcript)

        print(f"\nExtracted {len(facts)} facts with topics:")
        print("-" * 60)
        for i, item in enumerate(facts, 1):
            fact_text = item.get("fact", "")
            topic = item.get("topic", "")
            print(f"  {i}. [{topic or 'no topic'}]")
            print(f"     {fact_text}")

        non_empty = [f for f in facts if f.get("topic", "").strip()]
        empty = len(facts) - len(non_empty)
        print(f"\nSummary: {len(non_empty)} with topics, {empty} without")
        print("Review: Are topics consistent (same subject → same phrase)?")
        print("Are empty topics correctly assigned to untopical facts?")
