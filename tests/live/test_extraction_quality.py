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
        import src.llm as llm

        transcript = (FIXTURE_DIR / "conv_01.txt").read_text()
        expected = _load_expected("conv_01")
        facts = llm.extract_facts(transcript)

        print(f"\n--- conv_01 extraction ({len(facts)} facts) ---")
        for i, item in enumerate(facts, 1):
            topics = item.get("topics", [])
            topic_str = ", ".join(topics) if topics else "no topic"
            print(f"  {i}. [{topic_str}]")
            print(f"     {item['fact']}")

        multi = [f for f in facts if len(f.get("topics", [])) > 1]
        single = len(facts) - len(multi)
        print(f"\nTopics: {single} single-topic, {len(multi)} multi-topic")
        print("Review: Are topics the TRUE subject (not grammatical subject)?")
        print("Is 'user profile' used only for durable personal attributes?")
        print("Are multi-topic assignments substantive (genuine dual-belonging)?")

    def test_extract_conv_02(self):
        """Print facts extracted from conv_02.txt."""
        import src.llm as llm

        transcript = (FIXTURE_DIR / "conv_02.txt").read_text()
        expected = _load_expected("conv_02")
        facts = llm.extract_facts(transcript)

        print(f"\n--- conv_02 extraction ({len(facts)} facts) ---")
        for i, item in enumerate(facts, 1):
            topics = item.get("topics", [])
            topic_str = ", ".join(topics) if topics else "no topic"
            print(f"  {i}. [{topic_str}]")
            print(f"     {item['fact']}")

        multi = [f for f in facts if len(f.get("topics", [])) > 1]
        single = len(facts) - len(multi)
        print(f"\nTopics: {single} single-topic, {len(multi)} multi-topic")
        print("Spot-check: are multi-topic facts rare (the exception)?")
        print("Check that corroboration pairs are recognized.")
