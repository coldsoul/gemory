# Gemory — Testing Harness

The Gemory memory store has correctness properties that depend on exactly which facts arrive and how similar their embeddings are.
Live transcripts and real embedding APIs are non-deterministic — they produce different vectors across runs and model versions, making deterministic assertions impossible.

This harness replaces live calls with **deterministic stubs** and **frozen fixtures** so every threshold decision and dedup behavior can be asserted exactly.

## Test layers

| Layer | Embeddings | Deterministic | CI | Purpose |
|---|---|---|---|---|
| **Unit** | Injected stub vectors | Yes | Yes | Store logic: dedup, edge, idempotency, corroboration |
| **Integration (offline)** | Injected stub vectors | Yes | Yes | Full `remember`/`recall` flow end-to-end |
| **Live sanity** | Real embedder | No | No (manual) | Catch real-embedding surprises |

The unit + integration layers run in CI with zero network access.
The live suite is skipped unless `GEMORY_LIVE=1` is set.

## Run

```bash
# Deterministic suite (CI — no network, no API keys)
uv run pytest tests/ -v

# Live sanity (manual — requires API keys, GEMORY_LIVE=1)
GEMORY_LIVE=1 uv run pytest tests/live/ -v -s

# Calibration script (manual threshold tuning)
GEMORY_LIVE=1 uv run python tests/live/calibrate.py memory.json embeddings.json
```

## Stub embedders

Two stub strategies live in `tests/stubs.py`:

### Lookup stub (preferred for asserted tests)
A dictionary mapping exact fixture fact strings → hand-chosen vectors.
Use `vectors_with_cosines()` to build vectors with exact pairwise cosine similarities via Cholesky decomposition.

```python
from tests.stubs import vectors_with_cosines, LookupStub
import numpy as np

# Build three vectors where A·B=0.95, A·C=0.78, B·C=0.50
gram = np.array([[1.0, 0.95, 0.78],
                 [0.95, 1.0, 0.50],
                 [0.78, 0.50, 1.0]])
vecs = vectors_with_cosines(gram)

lookup = {
    "fact_a": vecs[0].tolist(),
    "fact_b": vecs[1].tolist(),
    "fact_c": vecs[2].tolist(),
}
stub = LookupStub(lookup)
```

### Hash stub (for tests where only identity matters)
Same text → same vector, different text → effectively orthogonal.

```python
from tests.stubs import HashStub
stub = HashStub(dim=1536, seed=42)
```

## Fixtures

Frozen, version-controlled transcript files under `tests/fixtures/`:

| File | Purpose |
|---|---|
| `conv_01.txt` | Base conversation (2 facts) |
| `conv_02.txt` | Corroboration + new facts (2 merge, 3 new) |
| `conv_01_grown.txt` | Extended conversation, same first exchange |
| `conv_01_reworded_prefix.txt` | Different first exchange |
| `conv_whitespace.txt` | Whitespace-only differences |
| `conv_single_turn.txt` | User-only message |
| `conv_empty_facts.txt` | Small talk (zero facts) |

Each `conv_N.txt` may have a sibling `conv_N.expected.json` declaring the expected fact strings and any intended corroboration mappings.

### Adding a fixture
1. Write the transcript as `tests/fixtures/your_test.txt`
2. Write `tests/fixtures/your_test.expected.json` with expected facts and optional `corroborates` map
3. Run `test_fixtures.py` to verify similarity assignments under the stub
4. Write store tests that reference the fixture

## Graph diff helper

`tests/graph_diff.py` provides `snapshot()` and `diff()` for declarative before/after assertions:

```python
from tests.graph_diff import snapshot, diff

before = snapshot(graph)
store_facts(facts, source_id, label, graph)
after = snapshot(graph)
d = diff(before, after)
assert d.is_empty  # nothing changed
```

## Determinism rules

- No network calls in Unit or Integration layers
- Stub vectors are fixed and constructed, never random
- Fixtures are frozen — change them deliberately with updated expectations in the same commit
- Each test starts from a known empty graph in a temp directory
- `source_id` does not depend on time

## File layout

```
tests/
├── conftest.py              # Stub injection, graph fixtures, temp paths
├── stubs.py                 # LookupStub, HashStub, vectors_with_cosines
├── graph_diff.py            # Snapshot + diff helper
├── fixtures/
│   ├── conv_01.txt          conv_01.expected.json
│   ├── conv_02.txt          conv_02.expected.json
│   ├── conv_01_grown.txt
│   ├── conv_01_reworded_prefix.txt
│   ├── conv_whitespace.txt
│   ├── conv_single_turn.txt
│   └── conv_empty_facts.txt
├── test_stubs.py            # Self-verification of vector helper
├── test_fixtures.py         # Fixture similarity meta-test
├── test_graph_diff.py       # Diff helper tests
├── test_store_basic.py      # Fresh store, embedding separation, empty facts
├── test_persistence.py      # Round-trip, pretty-print, orphan/missing
├── test_dedup_edge.py       # Merge/edge/no-edge at controlled cosines
├── test_idempotency.py      # Exact rerun + grown transcript
├── test_corroboration.py   # Cross-source with confidence tracking
├── test_source_id.py        # Stability, normalization, fallback
├── test_recall.py           # Ranking, top_k, empty graph
└── live/
    ├── test_extraction_quality.py  # Non-assertion human review
    ├── calibrate.py                # Threshold tuning from real data
    └── test_e2e_dump.py            # Full stack dump for eyeballing
```
