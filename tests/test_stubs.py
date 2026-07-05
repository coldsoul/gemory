"""Test the test infrastructure -- vectors_with_cosines must be correct."""

import numpy as np
import pytest

from tests.stubs import LookupStub, HashStub, vectors_with_cosines


class TestVectorsWithCosines:
    def test_produces_target_cosines(self):
        """The linchpin test: constructed vectors must match target cosines."""
        gram = np.array([
            [1.0, 0.85, 0.70],
            [0.85, 1.0, 0.70],
            [0.70, 0.70, 1.0],
        ])
        v = vectors_with_cosines(gram)
        for i in range(3):
            for j in range(3):
                got = float(np.dot(v[i], v[j]))
                assert abs(got - gram[i, j]) < 1e-6, (i, j, got, gram[i, j])

    def test_unit_norm(self):
        gram = np.eye(5)
        v = vectors_with_cosines(gram)
        for i in range(5):
            assert abs(np.linalg.norm(v[i]) - 1.0) < 1e-6

    def test_higher_dim_padding(self):
        gram = np.array([[1.0, 0.5], [0.5, 1.0]])
        v = vectors_with_cosines(gram, dim=10)
        assert v.shape == (2, 10)
        # Padded dimensions should be zero
        assert np.allclose(v[:, 2:], 0.0)

    def test_rejects_non_square(self):
        with pytest.raises(AssertionError):
            vectors_with_cosines(np.array([[1.0, 0.5]]))

    def test_rejects_non_symmetric(self):
        with pytest.raises(AssertionError):
            vectors_with_cosines(np.array([[1.0, 0.5], [0.3, 1.0]]))


class TestLookupStub:
    def test_returns_known_vector(self):
        stub = LookupStub({"fact_a": [1.0, 0.0]}, dim=2)
        assert stub.embed("fact_a") == [1.0, 0.0]

    def test_returns_far_for_unknown(self):
        stub = LookupStub({}, dim=3)
        v = stub.embed("unknown")
        assert v == [1.0, 0.0, 0.0]

    def test_embed_batch(self):
        stub = LookupStub({"a": [1.0, 0.0]}, dim=2)
        result = stub.embed_batch(["a", "b"])
        assert result[0] == [1.0, 0.0]
        assert result[1] == [1.0, 0.0]  # far vector for "b"


class TestHashStub:
    def test_same_text_same_vector(self):
        stub = HashStub(dim=10)
        assert stub.embed("hello") == stub.embed("hello")

    def test_different_texts_different_vectors(self):
        stub = HashStub(dim=128)
        v1 = stub.embed("hello")
        v2 = stub.embed("world")
        assert v1 != v2

    def test_unit_norm(self):
        stub = HashStub(dim=64)
        v = stub.embed("test")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_embed_batch(self):
        stub = HashStub(dim=10)
        result = stub.embed_batch(["a", "b"])
        assert len(result) == 2
        assert result[0] == stub.embed("a")
        assert result[1] == stub.embed("b")
