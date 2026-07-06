"""Live extraction quality check — run a real transcript through the live
extractor and print the extracted facts for human review.

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


def _load_expected(name: str) -> dict:
    path = FIXTURE_DIR / f"{name}.expected.json"
    with open(path) as f:
        return json.load(f)


class TestExtractionQuality:
    """Non-assertion review: print extracted facts for human inspection."""

    def test_extract_conv_01(self):
        """Print facts extracted from conv_01.txt for atomicity review."""
        import gemory.llm as llm

        transcript = (FIXTURE_DIR / "conv_01.txt").read_text()
        expected = _load_expected("conv_01")
        facts = llm.extract_facts(transcript)

        print(f"\n--- conv_01 extraction ({len(facts)} facts) ---")
        for i, item in enumerate(facts, 1):
            topic = item.get("topic", "")
            topic_suffix = f" [topic: {topic}]" if topic else ""
            print(f"  {i}. {item['fact']}{topic_suffix}")

        print(f"\nExpected ({len(expected['facts'])} facts):")
        for i, fact in enumerate(expected["facts"], 1):
            print(f"  {i}. {fact}")

        # Not assertion-strict: the live model may return different wording.
        print("\nReview atomicity, self-containment, and durable-vs-transient.")

    def test_extract_conv_02(self):
        """Print facts extracted from conv_02.txt."""
        import gemory.llm as llm

        transcript = (FIXTURE_DIR / "conv_02.txt").read_text()
        expected = _load_expected("conv_02")
        facts = llm.extract_facts(transcript)

        print(f"\n--- conv_02 extraction ({len(facts)} facts) ---")
        for i, item in enumerate(facts, 1):
            topic = item.get("topic", "")
            topic_suffix = f" [topic: {topic}]" if topic else ""
            print(f"  {i}. {item['fact']}{topic_suffix}")

        print(f"\nExpected ({len(expected['facts'])} facts):")
        for i, fact in enumerate(expected["facts"], 1):
            print(f"  {i}. {fact}")

        print("\nCheck that corroboration pairs are recognized.")
