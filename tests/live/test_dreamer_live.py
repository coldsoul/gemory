"""Live sanity: run the dreamer against the real graph and print proposed
abstractions for human review. Skipped unless GEMORY_LIVE=1."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GEMORY_LIVE") != "1",
    reason="Live suite requires GEMORY_LIVE=1",
)


class TestDreamerLive:
    def test_dry_run_on_real_graph(self):
        """Run dreamer --dry-run on the real memory.json and print output."""
        result = subprocess.run(
            [sys.executable, "dreamer.py", "--dry-run"],
            capture_output=True, text=True, timeout=120,
        )
        print("\n--- Dreamer dry-run output ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- Stderr ---")
            print(result.stderr)
        print("\n--- End dreamer output ---")

        # Not assertion-strict: human reviews this.
        print("\nReview: Are the proposed abstractions meaningful?")
