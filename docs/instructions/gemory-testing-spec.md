# Gemory — Testing Harness Spec (`TESTING.md`)

*Companion to the MCP implementation spec. This document defines a **deterministic
testing harness** for the Gemory memory store. It is written to be handed to a
coding agent alongside the implementation spec. Build the harness described here;
do not rely on live, interactively-generated transcripts to validate store
behaviour.*

---

## 0. Why this exists (the core principle)

The store's correctness properties — deduplication, edge creation, idempotency,
corroboration — depend on **exactly which facts arrive and how similar their
embeddings are**. Two things make live/manual testing unreliable:

1. **Live transcripts are non-deterministic.** Asking an assistant to "remember
   this conversation" regenerates a paraphrased transcript each time, so the same
   underlying fact arrives worded differently on each run. That produces
   near-duplicates that land unpredictably around the threshold — the opposite of
   a controlled test.
2. **Live embeddings drift and cost money.** Real embedding APIs return slightly
   different vectors over time and across model versions, and charge per call.
   Logic that must pass deterministically cannot depend on them.

**Therefore: the assistant / live model is NOT part of the test harness.** Tests
run against **frozen transcript fixtures** with **injected (stub) embeddings** and
**asserted outcomes**. A separate, clearly-marked "live" suite exercises the real
embedder for occasional sanity checks only — never in CI, never assertion-strict.

This harness is a prerequisite for the dreamer work: consolidation changes graph
structure in bulk, and without deterministic before/after assertions there is no
safe way to know a dreamer run did what it should.

---

## 1. Test layers

Three layers, in order of authority:

| Layer | Embeddings | Deterministic | Runs in CI | Purpose |
|-------|-----------|---------------|-----------|---------|
| **Unit** | Injected stub vectors | Yes | Yes | Store logic: dedup / edge / idempotency / corroboration branching |
| **Integration (offline)** | Injected stub vectors | Yes | Yes | Full `remember` / `recall` flow end-to-end, transcript → graph, using fixtures |
| **Live sanity** | Real embedder | No | **No** (manual) | Catch real-embedding surprises (calibration, extraction quality) |

The first two layers are the harness. The third is a manual tool.

---

## 2. The stub embedder (foundation of determinism)

Every deterministic test injects a stub in place of `llm.embed` / `embed_batch`.
The stub must be swappable via the same seam the implementation spec already
mandates (all embedding calls behind `llm.py`), e.g. dependency injection or a
config flag the test sets.

Two stub strategies, both required:

### 2a. Lookup stub (preferred for asserted tests)
A dictionary mapping **exact fixture fact strings → hand-chosen vectors**. Vectors
are constructed so chosen pairs have chosen cosine similarities. This gives total
control: you decide that fact A and fact B sit at 0.85 (should merge at a 0.80
threshold) while A and C sit at 0.70 (should stay distinct with an edge).

- Any string not in the lookup returns a fixed orthogonal "far" vector, so
  incidental facts never accidentally match.

**The vector-construction helper is the linchpin of every threshold test — use
the reference implementation below, do not improvise it.** The subtle part is
placing *three or more* vectors at simultaneously-controlled pairwise cosines
(A·B and A·C and B·C all specified). The clean way is to build an explicit Gram
matrix of the target cosines and Cholesky-decompose it: the rows of the Cholesky
factor are unit vectors whose pairwise dot products equal the targets exactly.
Embed those into the full embedding dimension by zero-padding.

```python
import numpy as np

def vectors_with_cosines(target_gram: np.ndarray, dim: int = 1536,
                         seed: int = 0) -> np.ndarray:
    """
    Return an (n, dim) array of unit vectors whose pairwise cosine similarities
    equal target_gram[i, j] exactly (within float precision).

    target_gram: (n, n) symmetric matrix, diagonal all 1.0, off-diagonals the
                 desired cosine similarities. Must be positive semi-definite
                 (any valid set of cosines is).
    """
    g = np.asarray(target_gram, dtype=float)
    n = g.shape[0]
    assert g.shape == (n, n), "gram must be square"
    assert np.allclose(g, g.T), "gram must be symmetric"
    assert np.allclose(np.diag(g), 1.0), "diagonal must be 1.0 (unit vectors)"
    # Cholesky: g = L @ L.T ; rows of L are unit vectors with the target dots.
    # Add a tiny jitter if a target set is exactly singular.
    try:
        L = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(g + 1e-9 * np.eye(n))
    assert dim >= n, "embedding dim must be >= number of vectors"
    out = np.zeros((n, dim), dtype=float)
    out[:, :n] = L
    # rows are already unit-norm by construction; normalize defensively
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out
```

Usage example — three facts with A·B=0.85 (merge), A·C=0.70, B·C=0.70 (both
distinct):

```python
gram = np.array([
    [1.00, 0.85, 0.70],
    [0.85, 1.00, 0.70],
    [0.70, 0.70, 1.00],
])
vecs = vectors_with_cosines(gram)   # vecs[0], vecs[1], vecs[2]
lookup = {FACT_A: vecs[0], FACT_B: vecs[1], FACT_C: vecs[2]}
```

**Mandatory self-verification of the helper (test the test infrastructure).**
Include a test that constructs vectors from a known Gram matrix and asserts the
*actual* pairwise cosines match the targets within a tight tolerance (e.g. 1e-6).
This catches a broken helper immediately, rather than letting it silently corrupt
every downstream threshold test:

```python
def test_helper_produces_target_cosines():
    gram = np.array([[1.0, 0.85, 0.70],
                     [0.85, 1.0, 0.70],
                     [0.70, 0.70, 1.0]])
    v = vectors_with_cosines(gram)
    for i in range(3):
        for j in range(3):
            got = float(np.dot(v[i], v[j]))
            assert abs(got - gram[i, j]) < 1e-6, (i, j, got, gram[i, j])
```

### 2b. Hash stub (for smoke/round-trip where similarity values don't matter)
Deterministically derives a vector from the SHA-256 of the text. Same text →
same vector, different text → effectively orthogonal. Use where a test only needs
"identical strings match, different strings don't," not specific similarity values.

Neither stub hits the network. Both are pure functions of their input.

---

## 3. Fixtures

Frozen, version-controlled transcript files plus their expected outcomes. Store
under `tests/fixtures/`.

### 3a. Transcript fixtures
Hand-written `*.txt` transcripts, realistic but fixed. At minimum:

- **`conv_01.txt`** — a base conversation yielding a known set of atomic facts
  (call them F1..Fn).
- **`conv_02.txt`** — a *different* conversation that **deliberately restates two
  of conv_01's facts in different wording** and introduces three genuinely new
  facts. This is the corroboration fixture — you know exactly which should merge.
- **`conv_01_grown.txt`** — `conv_01.txt` with additional turns **appended** and
  the original turns byte-for-byte unchanged. This is the growing-conversation
  fixture: same first exchange ⇒ same `source_id`.
- **`conv_01_reworded_prefix.txt`** — `conv_01.txt` with the **first exchange
  reworded**. Used to prove `source_id` is sensitive to the first exchange (this
  should be treated as a *different* source).
- **`conv_whitespace.txt`** — `conv_01.txt` with only whitespace / line-ending
  differences in the first exchange. Used to prove normalization makes
  `source_id` identical to `conv_01`'s.
- **`conv_single_turn.txt`** — a user message with no assistant reply. Exercises
  the single-turn `source_id` fallback.
- **`conv_empty_facts.txt`** — pure small talk yielding **zero** extractable
  facts. Exercises the `[]` path.

### 3b. Expectation files
For each transcript that feeds asserted tests, a sibling expectation
(`conv_01.expected.json` or equivalent) declaring:
- The expected set of atomic fact strings (so extraction can be checked when a
  stubbed/fixture extractor is used — see §4).
- The expected `source_id` (or an assertion that two fixtures share / differ in
  it).

Because live fact extraction is itself non-deterministic, asserted store tests
**bypass the live extractor**: feed the known fact list directly into the store
path, or stub `llm.extract_facts` to return the fixture's expected facts. Live
extraction quality is checked separately in the live suite (§6).

### 3c. Worked reference example (pattern to copy for every fixture pair)
Author fixtures by pattern-matching this concrete example, so intent is explicit
and machine-checkable rather than left to interpretation.

`conv_01.expected.json`:
```json
{
  "source_id_note": "hash of first exchange; asserted equal to conv_01_grown, conv_whitespace",
  "facts": [
    "The user is building a memory system called Gemory.",
    "The user works on a VPS."
  ]
}
```

`conv_02.expected.json` (restates the two conv_01 facts in different words, adds
three new):
```json
{
  "facts": [
    "Gemory is the user's long-term memory project.",
    "The user runs their code on a virtual private server.",
    "The user prefers minimal, inspectable implementations.",
    "The user uses uv for package management.",
    "The user flies FPV drones."
  ],
  "corroborates": {
    "Gemory is the user's long-term memory project.": "The user is building a memory system called Gemory.",
    "The user runs their code on a virtual private server.": "The user works on a VPS."
  }
}
```

The `corroborates` map states, for the record, **which conv_02 fact is intended
to merge into which conv_01 fact**. The stub-embedding lookup must then assign
each such pair a cosine **above `DEDUP_THRESHOLD`**, and every other cross-pair a
cosine **below `DEDUP_THRESHOLD`** (and, where an edge is intended, between the
thresholds). This makes the fixture's intent explicit and directly drives the
stub vector construction (§2a).

### 3d. Mandatory fixture-similarity self-verification (don't grade your own homework)
Because the agent may author both the fixtures and their expectations, add a
**meta-test** that validates the fixtures actually encode the intended
similarities under the stub — otherwise a store test can pass while testing
something other than intended. For every fixture pair, assert:
- each pair listed in a `corroborates` map stub-embeds to a cosine **strictly
  above `DEDUP_THRESHOLD`**;
- every pair intended to be an *edge* stub-embeds **between the thresholds**;
- every other cross-fixture pair stub-embeds **below `EDGE_THRESHOLD`**.

This meta-test is a test *of the fixtures*, not of the store. If it fails, the
fixture/expectation is wrong and must be fixed before the store tests mean
anything. **A human should still eyeball the fixture transcripts** — they encode
your definition of what counts as a "duplicate," which is a judgment call the
harness cannot make for you.

---

## 4. What to assert (the core correctness properties)

Each maps to a decision made in the implementation spec. All run with stub
embeddings and fixture facts.

### 4a. Fresh store
Reset graph → store `conv_01`'s facts → assert:
- node count == number of distinct facts,
- every result is **new**, corroborated == 0, skipped == 0,
- each node has exactly one provenance entry,
- `memory.json` contains no embedding vectors (sidecar holds them).

### 4b. Dedup vs edge (the threshold boundary)
Using lookup-stub vectors with known similarities around `DEDUP_THRESHOLD` and
`EDGE_THRESHOLD`:
- A pair at similarity **above `DEDUP_THRESHOLD`** → **one** node, not two
  (merge). The surviving node gains a provenance entry; no edge between them
  (they are the same node).
- A pair **between the thresholds** → **two** nodes **plus a directed edge**.
- A pair **below `EDGE_THRESHOLD`** → two nodes, **no edge**.
- Assert exact node count, edge count, and edge endpoints.

### 4c. Exact idempotency
Store `conv_01`, snapshot the graph, store `conv_01` **again**, assert:
- node count unchanged,
- **no** provenance entries added (same `source_id` already present on every
  node — the second run is a pure no-op),
- confidence values unchanged,
- edge set unchanged.

### 4d. Grown-transcript idempotency (the case that matters for real use)
Store `conv_01`, then store `conv_01_grown`, assert:
- facts from the original portion are **skipped** (same `source_id`, already
  recorded) — not re-corroborated,
- only the appended-portion facts are added as new nodes,
- original nodes' provenance and confidence are unchanged.

### 4e. Corroboration across different sources
Store `conv_01`, then store `conv_02`, assert:
- the two restated facts resolve to the **existing** conv_01 nodes
  (corroborated), not new nodes,
- each corroborated node now has **two** provenance entries with **different**
  `source_id`s,
- each corroborated node's confidence increased by exactly
  `CONFIDENCE_INCREMENT`,
- the three genuinely-new facts are added as new nodes,
- report counts: corroborated == 2, new == 3, skipped == 0.

### 4f. `source_id` derivation
- `conv_01` and `conv_whitespace` produce the **same** `source_id` (per-turn
  normalization).
- `conv_01` and `conv_01_reworded_prefix` produce **different** `source_id`s.
- `conv_01` and `conv_01_grown` produce the **same** `source_id` (first exchange
  unchanged).
- `conv_single_turn` falls back to hashing the first user message alone and does
  not error.
- Boundary/separator: constructed inputs where naive concatenation would collide
  (`"AB"+"C"` vs `"A"+"BC"`) yield **different** `source_id`s.

### 4g. Empty extraction
Store `conv_empty_facts` → zero nodes added, no error, result reports 0/0/0.

### 4h. Persistence round-trip
Store some facts → `save()` → new `GraphStore` → `load()` → assert:
- graph is identical (nodes, edges, all fields),
- embeddings identical (sidecar round-trips),
- `memory.json` is pretty-printed and embedding-free,
- graph is a directed `DiGraph`.
- Also: a node present without its sidecar embedding is a clear error on load; a
  sidecar embedding with no matching node is tolerated (ignored).

### 4i. Recall ordering
With a known small graph, a query stub-embedded near a specific node returns that
node ranked above unrelated ones. (Ordering/relevance sanity, not exact scores.)

---

## 5. Graph-diff helper (observability)

A small utility — used by tests and available manually — that takes two graph
snapshots and reports the delta in human-readable form:
- nodes added / removed (by content),
- provenance entries appended (which node, which `source_id`),
- confidence changes,
- edges added / removed (with endpoints and relation).

This turns "does the graph look right?" into "here is exactly what the store
did." Asserted tests can diff against an expected delta; humans use the same tool
to inspect live runs. It becomes essential for the dreamer, where a consolidation
run must be shown to have made precisely the intended structural changes.

Provide a companion **snapshot** helper (serialize current graph+embeddings to a
comparable in-memory structure) so a test can snapshot → act → diff.

---

## 6. Live sanity suite (manual, not CI)

Clearly separated (e.g. `tests/live/`, skipped unless an env flag is set). Hits
the real embedder and, optionally, the real extractor. Not assertion-strict —
its job is to surface real-world behaviour, not to gate builds.

Includes:
- **Extraction quality check:** run a real transcript through the live extractor;
  print the extracted facts for human review of atomicity / self-containment /
  durable-vs-transient calls.
- **Calibration script (reusable):** given a `memory.json` + `embeddings.json`,
  compute the full pairwise cosine-similarity distribution and print the ranked
  top pairs with their contents. Used to locate the "valley" between
  true-duplicate and distinct-but-related pairs and to choose `DEDUP_THRESHOLD` /
  `EDGE_THRESHOLD` from real data. (This is the manual calibration step, made
  repeatable — and a preview of logic the dreamer will eventually automate.)
- **End-to-end dump:** feed fixtures through the full real stack and dump the
  resulting graph for eyeballing.

---

## 7. Suggested layout

```
tests/
├── conftest.py                  # stub-embedder injection, graph fixtures, tmp paths
├── stubs.py                     # lookup stub + hash stub + vector-at-cosine helper
├── fixtures/
│   ├── conv_01.txt
│   ├── conv_01.expected.json
│   ├── conv_02.txt
│   ├── conv_02.expected.json
│   ├── conv_01_grown.txt
│   ├── conv_01_reworded_prefix.txt
│   ├── conv_whitespace.txt
│   ├── conv_single_turn.txt
│   └── conv_empty_facts.txt
├── test_source_id.py            # §4f
├── test_dedup_edge.py           # §4b
├── test_idempotency.py          # §4c, §4d
├── test_corroboration.py        # §4e
├── test_store_basic.py          # §4a, §4g
├── test_persistence.py          # §4h
├── test_recall.py               # §4i
├── test_graph_diff.py           # §5 helper's own tests
├── graph_diff.py                # §5 utility (also importable by app / dreamer)
└── live/                        # §6, skipped unless GEMORY_LIVE=1
    ├── test_extraction_quality.py
    ├── calibrate.py
    └── test_e2e_dump.py
```

---

## 8. Determinism rules (hard requirements)

- No network calls in the Unit or Integration layers. Enforce via the stub seam;
  if a test triggers a real embedder/extractor call, that is a harness bug.
- No wall-clock nondeterminism in assertions: inject a fixed clock (or tolerate
  timestamps by asserting *presence/ordering*, not exact values). `source_id`
  must not depend on time.
- Stub vectors are fixed and constructed, never random (or seeded if generated).
- Fixtures are frozen: changing a fixture is a deliberate act that may change
  expected outcomes, and both change together in the same commit.
- Reset between tests: each test starts from a known empty graph + empty sidecar
  in a temp dir; no shared mutable state across tests.

---

## 9. Acceptance for the harness itself

- The five properties from the implementation spec's acceptance criteria
  (round-trip, exact idempotency, grown-transcript idempotency, dedup-vs-edge,
  persistence/isolation) each have at least one asserted test here.
- **The vector helper is self-verified** (§2a): a test constructs vectors from a
  known Gram matrix and asserts actual pairwise cosines match targets within
  1e-6. This must exist and pass before any threshold test is trusted.
- **The fixtures are self-verified** (§3d): a meta-test asserts every intended
  merge/edge/distinct relationship actually holds under the stub embeddings, at
  the configured thresholds.
- Corroboration (`§4e`) is covered — the property no live run in development has
  yet demonstrated.
- CI runs Unit + Integration with zero network access and passes deterministically
  on repeated runs (run the suite 3× in CI; identical results).
- The live suite is skipped by default and runs only under an explicit flag.
- The calibration script reproduces, on the attached-style data, a ranked pair
  list matching manual inspection (the 0.80 valley finding is re-derivable).

---

## 10. Note on the dreamer (forward pointer, not in scope here)

When the dreamer is built, this harness extends naturally: fixtures become
*graph states* (not just transcripts), and assertions become *before/after
structural deltas* via the graph-diff helper — e.g. "these N near-duplicate nodes
existing in the input graph are merged into one, preserving all provenance," or
"a cluster of related nodes gains a parent abstraction node at level 1." Building
the diff helper and snapshot machinery now is what makes dreamer testing
tractable later. Do not build dreamer tests yet; just don't foreclose them.
