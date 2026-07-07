"""Live sanity: run the dreamer against the real graph in diff mode.

Runs --dry-run (diff default) to compare algorithm vs LLM clustering
on the real accumulated graph. The resulting agreement/algorithm-only/llm-only
buckets are the go/no-go artifact for the consolidation amendment.

Skipped unless GEMORY_LIVE=1.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GEMORY_LIVE") != "1",
    reason="Live suite requires GEMORY_LIVE=1",
)


class TestDreamerLive:
    def test_diff_dry_run_on_real_graph(self):
        """Run dreamer --cluster-method diff on the real graph."""
        result = subprocess.run(
            [sys.executable, "src/dreamer.py", "--cluster-method", "diff"],
            capture_output=True, text=True, timeout=120,
        )
        print("\n" + "=" * 70)
        print("DREAMER DIFF DRY-RUN (live, on real graph)")
        print("=" * 70)
        print(result.stdout)
        if result.stderr:
            print("\n--- Stderr ---")
            print(result.stderr)
        print("=" * 70)

        print("\nReview checklist (§11 of dreamer spec):")
        print("  1. Agreement bucket: do dense project topics appear here?")
        print("  2. LLM-only bucket: does the user-attribute grouping appear?")
        print("     (cosine cannot detect it, LLM should)")
        print("  3. Unrelated projects: are they NOT forced into a spurious theme?")
        print("  4. Reach: do abstractions clear MIN_REACH?")
        print("\nIf all four pass, the consolidation amendment is validated.")

    def test_hybrid_dry_run_on_real_graph(self):
        """Run dreamer --cluster-method hybrid (dry-run) on the real graph."""
        result = subprocess.run(
            [sys.executable, "src/dreamer.py", "--cluster-method", "hybrid"],
            capture_output=True, text=True, timeout=120,
        )
        print("\n" + "=" * 70)
        print("DREAMER HYBRID DRY-RUN (live, on real graph)")
        print("=" * 70)
        print(result.stdout)
        if result.stderr:
            print("\n--- Stderr ---")
            print(result.stderr)
        print("=" * 70)

        print("\nReview: Are the proposed abstractions meaningful?")
        print("Does the hybrid method produce sensible hierarchy?")
