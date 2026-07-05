"""Deterministic stub embedders for testing. Never touches the network."""

import hashlib

import numpy as np


def vectors_with_cosines(target_gram: np.ndarray, dim: int = 1536) -> np.ndarray:
    """
    Return an (n, dim) array of unit vectors whose pairwise cosine similarities
    equal target_gram[i, j] exactly (within float precision).

    Parameters
    ----------
    target_gram
        (n, n) symmetric matrix, diagonal all 1.0, off-diagonals the desired
        cosine similarities. Must be positive semi-definite.
    dim
        Embedding dimension (must be >= n).
    """
    g = np.asarray(target_gram, dtype=float)
    n = g.shape[0]
    assert g.shape == (n, n), "gram must be square"
    assert np.allclose(g, g.T), "gram must be symmetric"
    assert np.allclose(np.diag(g), 1.0), "diagonal must be 1.0 (unit vectors)"
    # Cholesky: g = L @ L.T ; rows of L are unit vectors with the target dots.
    try:
        L = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(g + 1e-9 * np.eye(n))
    assert dim >= n, f"embedding dim ({dim}) must be >= number of vectors ({n})"
    out = np.zeros((n, dim), dtype=float)
    out[:, :n] = L
    # rows are already unit-norm by construction; normalize defensively
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


class LookupStub:
    """
    Dictionary mapping exact fact strings to hand-chosen vectors.

    Any string not in the lookup returns a fixed orthogonal "far" vector
    (the first basis element ``[1, 0, 0, ...]``).
    """

    def __init__(self, lookup: dict[str, list[float]], dim: int = 1536):
        self._lookup = {k: np.array(v, dtype=float) for k, v in lookup.items()}
        self._dim = dim
        self._far = np.zeros(dim, dtype=float)
        self._far[0] = 1.0

    def embed(self, text: str) -> list[float]:
        v = self._lookup.get(text)
        if v is not None:
            return v.tolist()
        return self._far.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashStub:
    """
    Deterministically derives a vector from SHA-256 of the text.

    Same text -> same vector, different text -> effectively orthogonal.
    Use where tests only need "identical strings match, different don't."
    """

    def __init__(self, dim: int = 1536, seed: int = 42):
        self._dim = dim
        self._rng = np.random.RandomState(seed)

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Mask to 32 bits so numpy's RandomState doesn't complain.
        seed = int.from_bytes(digest[:8], "big") & 0xFFFF_FFFF
        local_rng = np.random.RandomState(seed)
        v = local_rng.randn(self._dim).astype(float)
        v /= np.linalg.norm(v)
        return v.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
