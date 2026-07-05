"""Shared pytest fixtures for the Gemory test harness."""

import pytest

from gemory.graph import GraphStore
from tests.stubs import HashStub, LookupStub


@pytest.fixture
def tmp_graph_path(tmp_path):
    """A temporary path for memory.json in a temp directory."""
    return str(tmp_path / "memory.json")


@pytest.fixture
def empty_graph(tmp_graph_path):
    """A fresh, empty GraphStore backed by a temp file."""
    store = GraphStore(tmp_graph_path)
    return store


@pytest.fixture
def hash_stub():
    """A deterministic hash-based stub embedder (1536-dim)."""
    return HashStub(dim=1536, seed=42)


def make_lookup_stub(
    lookup: dict[str, list[float]] | None = None,
    dim: int = 1536,
) -> LookupStub:
    """Factory for creating a LookupStub with controlled vectors."""
    return LookupStub(lookup or {}, dim=dim)
