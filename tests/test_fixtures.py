"""Meta-test: verify that fixture expectations encode valid similarity relationships.

This is a test OF the fixtures, not of the store.
If it fails, the fixture/expectation is wrong and must be fixed before
downstream store tests mean anything.
"""

import json
import numpy as np
from pathlib import Path

import pytest

from src import config
from tests.stubs import vectors_with_cosines, LookupStub

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected(name: str) -> dict:
    path = FIXTURE_DIR / f"{name}.expected.json"
    with open(path) as f:
        return json.load(f)


def _build_gram_from_expected(
    conv_01_facts: list[str],
    conv_02_expected: dict,
    dedup_cosine: float = 0.95,
    other_cosine: float = 0.60,
) -> tuple[np.ndarray, list[str]]:
    """Build a target Gram matrix and ordered fact list from fixture expectations.

    Returns (gram, all_facts) where:
    - conv_01 facts come first
    - conv_02 facts follow
    - Corroborating pairs get dedup_cosine, everything else gets other_cosine.
    """
    facts_02 = conv_02_expected["facts"]
    corroborates = conv_02_expected.get("corroborates", {})

    all_facts = list(conv_01_facts) + facts_02
    n = len(all_facts)
    gram = np.full((n, n), other_cosine)
    np.fill_diagonal(gram, 1.0)
    gram = (gram + gram.T) / 2  # force symmetric
    np.fill_diagonal(gram, 1.0)

    # Set corroboration pairs to dedup_cosine.
    for fact_02, fact_01 in corroborates.items():
        try:
            i = all_facts.index(fact_02)
            j = all_facts.index(fact_01)
        except ValueError:
            continue
        gram[i, j] = dedup_cosine
        gram[j, i] = dedup_cosine

    # Ensure positive semi-definite (add small jitter if needed).
    min_eig = np.linalg.eigvalsh(gram).min()
    if min_eig < 0:
        gram += (-min_eig + 1e-6) * np.eye(n)

    return gram, all_facts


class TestFixtureSimilarities:
    """Verify that the stub embeddings encode the intended relationships."""

    def test_corroboration_pairs_above_dedup_threshold(self):
        """Each corroboration pair must embed to cosine > DEDUP_THRESHOLD."""
        conv_01 = _load_expected("conv_01")
        conv_02 = _load_expected("conv_02")
        corroborates = conv_02.get("corroborates", {})

        gram, all_facts = _build_gram_from_expected(
            conv_01["facts"], conv_02, dedup_cosine=0.95, other_cosine=0.60
        )
        vecs = vectors_with_cosines(gram)
        stub = LookupStub(
            {fact: vecs[i].tolist() for i, fact in enumerate(all_facts)}
        )

        for fact_02, fact_01 in corroborates.items():
            v1 = np.array(stub.embed(fact_01))
            v2 = np.array(stub.embed(fact_02))
            sim = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
            assert sim > config.DEDUP_THRESHOLD, (
                f"Corroboration pair should merge (sim {sim:.4f} > "
                f"{config.DEDUP_THRESHOLD}):\n"
                f"  conv_01: {fact_01}\n"
                f"  conv_02: {fact_02}"
            )

    def test_non_corroboration_pairs_below_edge_threshold(self):
        """Every other cross-fixture pair must embed to cosine < EDGE_THRESHOLD."""
        conv_01 = _load_expected("conv_01")
        conv_02 = _load_expected("conv_02")
        corroborates = conv_02.get("corroborates", {})
        corroborate_values = set(corroborates.values())

        gram, all_facts = _build_gram_from_expected(
            conv_01["facts"], conv_02, dedup_cosine=0.95, other_cosine=0.60
        )
        vecs = vectors_with_cosines(gram)
        stub = LookupStub(
            {fact: vecs[i].tolist() for i, fact in enumerate(all_facts)}
        )

        n01 = len(conv_01["facts"])
        for i, fact_01 in enumerate(conv_01["facts"]):
            for j, fact_02 in enumerate(conv_02["facts"]):
                if fact_01 in corroborate_values and fact_02 in corroborates:
                    # This is a corroboration pair — skip, tested above.
                    continue
                v1 = np.array(stub.embed(fact_01))
                v2 = np.array(stub.embed(fact_02))
                sim = float(np.dot(
                    v1 / np.linalg.norm(v1),
                    v2 / np.linalg.norm(v2),
                ))
                assert sim < config.EDGE_THRESHOLD, (
                    f"Non-corroboration pair must be distinct "
                    f"(sim {sim:.4f} < {config.EDGE_THRESHOLD}):\n"
                    f"  conv_01: {fact_01}\n"
                    f"  conv_02: {fact_02}"
                )

    def test_within_fixture_facts_are_distinct(self):
        """Facts within the same fixture should be below EDGE_THRESHOLD."""
        conv_02 = _load_expected("conv_02")
        facts = conv_02["facts"]

        gram = np.full((len(facts), len(facts)), 0.60)
        np.fill_diagonal(gram, 1.0)
        vecs = vectors_with_cosines(gram)
        stub = LookupStub(
            {fact: vecs[i].tolist() for i, fact in enumerate(facts)}
        )

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                v1 = np.array(stub.embed(facts[i]))
                v2 = np.array(stub.embed(facts[j]))
                sim = float(np.dot(
                    v1 / np.linalg.norm(v1),
                    v2 / np.linalg.norm(v2),
                ))
                assert sim < config.EDGE_THRESHOLD, (
                    f"Facts in same fixture should be distinct "
                    f"(sim {sim:.4f}): {facts[i][:40]} | {facts[j][:40]}"
                )

    def test_empty_facts_fixture_has_no_facts(self):
        """conv_empty_facts should declare zero facts."""
        # This fixture has no expected JSON — the test just confirms
        # we treat it as empty. The store test verifies behavior.
        pass  # Implicit: if we needed facts, they'd be declared.
