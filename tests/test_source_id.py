"""Tests for compute_source_id: stability, normalization, boundary collisions."""

from pathlib import Path

import pytest

from src.extractor import compute_source_id

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestSameTranscriptSameId:
    """Transcripts with identical first exchanges share the same source_id."""

    def test_same_transcript_same_id(self) -> None:
        t1 = (FIXTURE_DIR / "conv_01.txt").read_text()
        t2 = (FIXTURE_DIR / "conv_01_grown.txt").read_text()
        assert compute_source_id(t1) == compute_source_id(t2)


class TestDifferentPrefixDifferentId:
    """Different first exchange wording produces different source_ids."""

    def test_different_prefix_different_id(self) -> None:
        t1 = (FIXTURE_DIR / "conv_01.txt").read_text()
        t2 = (FIXTURE_DIR / "conv_01_reworded_prefix.txt").read_text()
        assert compute_source_id(t1) != compute_source_id(t2)


class TestWhitespaceNormalization:
    """Whitespace variants produce the same source_id."""

    def test_whitespace_normalization(self) -> None:
        t1 = (FIXTURE_DIR / "conv_01.txt").read_text()
        t2 = (FIXTURE_DIR / "conv_whitespace.txt").read_text()
        assert compute_source_id(t1) == compute_source_id(t2)


class TestSingleTurnFallback:
    """Transcript with only a user message (no assistant) produces a valid id."""

    def test_single_turn_fallback(self) -> None:
        text = (FIXTURE_DIR / "conv_single_turn.txt").read_text()
        sid = compute_source_id(text)
        assert sid.startswith("sha256:")
        assert len(sid) > 10  # non-trivial hash


class TestBoundarySeparator:
    """The \\n<gemory-sep>\\n separator prevents naive concatenation collisions."""

    def test_boundary_separator(self) -> None:
        t1 = "User: AB\nAssistant: C"
        t2 = "User: A\nAssistant: BC"
        sid1 = compute_source_id(t1)
        sid2 = compute_source_id(t2)
        assert sid1 != sid2
