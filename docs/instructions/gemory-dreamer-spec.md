# Gemory Dreamer — Implementation Spec

*Working title. Self-contained brief for a coding agent to implement the
**dreamer** (offline consolidation) of the Gemory graph-memory PoC. Companion to
the MCP implementation spec and the testing spec, and consistent with their field
names and decisions (`DiGraph`, `level`, `provenance`/`source_id`, `confidence`,
`GraphStore`, sidecar embeddings, `config.py` constants). Build exactly what is
specified; respect the "Out of scope" list.*

---

## 0. What the dreamer is, in one paragraph

The MCP server (already built) grows a **flat** graph: atomic fact nodes, with
`related` edges between near-but-not-duplicate facts. The dreamer is a
**separately-run, offline batch process** that periodically reorganizes that flat
graph into **emergent hierarchy** — it finds clusters of related nodes and
creates higher-level **abstraction nodes** above them, so that recall can later
enter at a theme and drill down. It is the novel core of the project and the
thing the PoC exists to validate. In brain terms it is the "sleep" pass that
consolidates recent experience into more abstract, stable structure.

**It is run manually in the PoC** (a CLI script), and its output is meant to be
**reviewed by a human** before being trusted. Human judgement of whether the
abstractions are meaningful is the primary success signal.

---

## 1. Scope

### In scope (this deliverable)
- A CLI script (`dreamer.py`) that loads the graph via `GraphStore`, runs a
  consolidation pass, and writes the result back (or to a copy — see §7).
- **Clustering** of related fact nodes.
- **Abstraction-node creation**: an LLM writes a summary node above each cluster,
  linked by directed hierarchy edges.
- **Recursive consolidation**: the same logic applies above existing abstraction
  nodes, so depth emerges from the data (no fixed number of levels).
- A **dry-run / review mode** that shows proposed changes without applying them.
- Integration with the **graph-diff helper** (from the testing spec) so a run's
  structural changes are legible.
- Determinism controls so runs are reproducible enough to test.

### Explicitly deferred (do NOT build now)
- Contradiction handling / contested states / belief revision.
- Confidence *propagation* up the hierarchy on contradiction.
- Adaptive or density-based *automatic* triggering (PoC trigger is "a human runs
  it"; you may compute density signals for reporting, but do not auto-run).
- Merging of already-existing duplicate nodes as a primary feature (see §6.4 for
  the limited, optional version).
- Cross-edges between abstraction nodes (uber-node-to-uber-node links).
- Real-time / online consolidation.
- Any change to the MCP server, the store schema semantics, or recall. The
  dreamer only *adds* structure the existing schema already allows.

---

## 2. Preconditions and consistency with existing code

The dreamer must reuse, not reinvent:
- **`GraphStore`** (from the MCP spec) is the only way it touches the graph and
  embeddings. It does not import `networkx` directly, and does not read/write
  `memory.json` / the embedding sidecar by hand. If `GraphStore` lacks a method
  the dreamer needs (e.g. `add_parent_edge`, `set_level`, `get_children`), add it
  *to `GraphStore`* with tests, keeping all graph-library access inside that
  module.
- **`llm.py`** is the only place LLM calls happen. The dreamer's summarization
  prompt lives behind a function there (e.g. `summarize_cluster(facts) -> {label,
  summary}`), mirroring how extraction lives behind `extract_facts`.
- **`config.py`** holds every tunable constant the dreamer introduces (see §8).
- The graph is a **directed DAG** (`DiGraph`). Hierarchy edges are directed.

### Node types
The existing schema has only fact nodes at `level` 0. The dreamer introduces
**abstraction nodes**. Distinguish them explicitly with a `kind` field (add to
the node schema, defaulting existing nodes to `"fact"`; abstraction nodes are
`"abstraction"`). Additive change: tolerate its absence on load (treat missing as
`"fact"`).

| Node field additions | Meaning |
|----------------------|---------|
| `kind` | `"fact"` (leaf, extracted) or `"abstraction"` (dreamer-created summary) |
| `level` | already exists; now **maintained** by the dreamer: leaf facts stay 0, an abstraction is `1 + max(child levels)` |

### Edge relations (extends the controlled vocabulary)
The MCP PoC emitted only `"related"`. The dreamer introduces **directed hierarchy
edges** with relation `"parent_of"` (from abstraction → child). Keep `"related"`
as-is. Do not invent others. `child_of` is not stored separately — it is just the
reverse traversal of `parent_of` (the graph is directed; query both directions).

---

## 3. The consolidation pass — high-level algorithm

One dreamer run does the following, in order:

1. **Load** the graph via `GraphStore`.
2. **Select the working set** — which nodes are eligible for consolidation this
   run (§4).
3. **Cluster** the working set into groups of related nodes (§5).
4. For each cluster that qualifies, **create one abstraction node** summarizing it
   and link it to its members with `parent_of` edges (§6).
5. **Recurse** one level up: treat the newly-created abstraction nodes (plus
   existing same-level abstractions) as a new working set and repeat clustering +
   abstraction, until a level produces no qualifying clusters (§6.3).
6. **Recompute `level`** for affected nodes.
7. **Report** the proposed changes via the graph-diff helper; in apply mode,
   `save()`; in dry-run mode, do not.

Everything below details the pieces.

---

## 4. Selecting the working set

Not every node should be reconsidered on every run. Two modes, selectable by CLI
flag:

- **`--full`**: the working set is *all* fact-level nodes (and, in the recursion,
  all abstraction nodes). Simple, more expensive, deterministic. This is the
  default for the PoC because graphs are small and full passes are easiest to
  reason about and test.
- **`--recent N`**: the working set is nodes whose most recent `provenance`
  timestamp is within the last N days, plus their immediate graph neighbours.
  Cheaper, and closer to the eventual "consolidate the hot area" design. Provided
  now but not the default.

Report the working-set size at the start of a run.

**Idempotency of consolidation (important):** running the dreamer twice on an
unchanged graph must not create duplicate abstraction nodes or re-abstract
already-abstracted clusters. Achieve this by: (a) a fact node that already has a
`parent_of` parent is not re-clustered into a *new* sibling abstraction unless new
un-parented members would join it; (b) abstraction creation checks whether an
equivalent abstraction already exists over substantially the same member set and,
if so, updates it rather than duplicating (see §6.2). This mirrors the store's
idempotency property and is a required test (see §9).

---

## 5. Clustering

Group working-set nodes by semantic relatedness using their **embeddings**
(fetched via `GraphStore`; never re-embedded here — reuse the sidecar vectors).

### Method
- Build similarity structure from embeddings. A clean, dependency-light approach:
  construct a graph where an edge connects two nodes with cosine similarity ≥
  `CLUSTER_SIM_THRESHOLD`, then take **connected components** or run a community
  detection algorithm (`networkx.community`, e.g. greedy modularity or Louvain via
  a documented dependency) over it. Community detection is preferred because it
  handles the "one big blob" failure mode better than raw components.
- **Reuse the existing `related` edges** as a prior: they already encode
  store-time similarity. The cluster graph can be the union of existing `related`
  edges and freshly-computed high-similarity edges over the working set.
- Respect a **minimum cluster size** (`MIN_CLUSTER_SIZE`, e.g. 3): clusters
  smaller than this are left alone — abstracting over one or two facts produces
  noise, not insight.
- Respect a **maximum cluster size** (`MAX_CLUSTER_SIZE`): an over-large cluster
  should be split (e.g. sub-cluster it at a higher similarity threshold) rather
  than summarized into a uselessly-broad abstraction. Report when this triggers.

### Output
A list of clusters, each a set of node ids, each of size within
`[MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE]`.

Clustering must be **deterministic** given the same graph and thresholds (fix any
RNG seed the community algorithm uses; sort inputs). This is required for testing.

---

## 6. Abstraction-node creation

### 6.1 Writing the abstraction
For each qualifying cluster, call `llm.summarize_cluster(member_facts)` which
returns a small structured object:

```json
{
  "label": "3-6 word theme label",
  "summary": "1-2 sentence description of what these facts have in common"
}
```

Prompt requirements (mirrors the extraction prompt's discipline):
- Output **strict JSON**, no prose/fences, parse defensively.
- The `label` is short and thematic (it is what recall scans).
- The `summary` states the *common theme*, written to be read out of context
  months later — self-contained, no pronouns dangling.
- **Do not invent facts** not supported by the members. The abstraction describes
  the cluster; it does not add new claims. (Same anti-hallucination reasoning as
  extraction: a fabricated abstraction becomes load-bearing.)
- The summary is written **at consolidation time** and stored, so recall stays
  cheap (never summarize on the fly at query time).

Create the abstraction node with:
- `kind = "abstraction"`,
- `content` = the summary (and store `label` too; keep both — label for scanning,
  summary for context),
- its own `embedding` (embed the label + summary, via `llm.embed`, and store in
  the sidecar like any node),
- `level = 1 + max(level of members)`,
- `provenance`: record that it was dreamer-created — e.g. a provenance entry with
  a synthetic `source_id` like `"dreamer:<run_id>"` and the timestamp, plus the
  list of member ids it was built from (store member ids so a later run can detect
  "substantially the same member set"). Confidence starts at `CONFIDENCE_BASE`.

Link it: add a directed `parent_of` edge from the abstraction node to **each**
member.

### 6.2 Don't duplicate existing abstractions (idempotency)
Before creating, check whether an abstraction already covers substantially the
same member set (e.g. Jaccard overlap of member ids ≥ `ABSTRACTION_OVERLAP` such
as 0.8). If so, **update** the existing one: attach any new members with
`parent_of` edges, and (optionally) refresh its summary. Do not create a parallel
abstraction. This is what makes re-running the dreamer safe.

### 6.3 Recursion (emergent depth)
After a level is abstracted, collect all abstraction nodes at that new level and
run §5–§6 again treating them as the working set — clustering by their *own*
embeddings, summarizing clusters of themes into higher themes. Stop when a level
yields no cluster of size ≥ `MIN_CLUSTER_SIZE`. This is how the "few big roots
over many months, shallow where engagement is thin" shape emerges without fixed
tiers. Cap recursion at `MAX_LEVELS` (e.g. 6) as a safety stop; report if hit.

### 6.4 (Optional) merge of true duplicates — narrow scope only
The dreamer *may* merge nodes that are true duplicates (cosine ≥
`DEDUP_THRESHOLD`, the same constant the store uses) that slipped in before
threshold tuning — merging their provenance (idempotently, by `source_id`) and
re-pointing edges. Keep this conservative and clearly separated from abstraction.
If it adds risk or complexity, **omit it for the first version** — it is not the
point of the dreamer, and over-merging is the invisible-failure direction. Prefer
to leave duplicate cleanup to a future pass.

---

## 7. Safety, dry-run, and output

Because the dreamer mutates accumulated memory, it must be **cautious by default**:

- **`--dry-run` is the default.** It computes the full proposed consolidation and
  prints a report (via the graph-diff helper) — clusters found, abstractions that
  would be created (label + summary + member contents), levels affected — but
  does **not** write. The human reviews this.
- **`--apply`** performs the write, and only after taking a **backup** of
  `memory.json` + the embedding sidecar (timestamped copies) so a bad run is
  recoverable. Never destructive-in-place without a backup.
- A **`--run-id`** (default: timestamp) tags every abstraction created in the run,
  so a run's output can be identified and, if needed, rolled back or filtered.
- Structured logging to stderr: working-set size, clusters found, abstractions
  created/updated, recursion depth reached, anything skipped.

The graph-diff report is the primary human-facing artifact. It must clearly show,
in readable form: new abstraction nodes (with label, summary, and the facts under
them), new `parent_of` edges, and any `level` changes.

---

## 8. Constants (all in `config.py`)

Every number the dreamer uses is a named, documented constant — no inline magic
values, same rule as the store:

- `CLUSTER_SIM_THRESHOLD` — min cosine to link two nodes in the cluster graph
  (start around the `EDGE_THRESHOLD` neighbourhood, ~0.75; tune on real data).
- `MIN_CLUSTER_SIZE` — smallest cluster worth abstracting (e.g. 3).
- `MAX_CLUSTER_SIZE` — largest before forcing a split (e.g. 12; tune).
- `ABSTRACTION_OVERLAP` — Jaccard threshold for "same abstraction already exists"
  (e.g. 0.8).
- `MAX_LEVELS` — recursion safety cap (e.g. 6).
- Reuses existing `DEDUP_THRESHOLD`, `CONFIDENCE_BASE` from the store config.

These will need tuning against a real accumulated graph; keep them trivially
editable and document each. Expect `CLUSTER_SIM_THRESHOLD` and `MIN_CLUSTER_SIZE`
to be the two that most change the character of the output.

---

## 9. Testing (extends the existing harness)

Reuse the deterministic testing approach from the testing spec: **stub
embeddings, frozen fixtures, asserted structural outcomes, graph-diff for
observability.** The LLM summarizer is stubbed for asserted tests (return a fixed
`{label, summary}` for a known member set); summary *quality* is checked only in
the manual live suite.

Required asserted tests (stub embeddings, fixed thresholds):

- **Clustering determinism:** same graph + thresholds → identical clusters across
  repeated runs.
- **Basic abstraction:** a fixture graph with one obvious cluster of ≥
  `MIN_CLUSTER_SIZE` facts produces exactly one abstraction node at level 1, with
  `parent_of` edges to every member, `kind="abstraction"`, and a stored embedding.
- **Min-size respected:** a cluster below `MIN_CLUSTER_SIZE` produces no
  abstraction.
- **Max-size split:** an over-large cluster is split, not summarized whole.
- **Recursion / emergent depth:** a fixture designed so themes-of-themes exist
  yields a level-2 abstraction above level-1 abstractions; a shallow region stays
  shallow. Assert the resulting `level` values.
- **Consolidation idempotency (critical):** running the dreamer twice on an
  unchanged graph creates no new abstractions on the second run and no duplicate
  parents (via §6.2). Assert node/edge counts stable.
- **Incremental:** add a few new fact nodes under an existing theme, re-run →
  they are attached to the existing abstraction (updated), not given a new
  parallel one.
- **Dry-run writes nothing:** `--dry-run` leaves `memory.json` and the sidecar
  byte-identical; `--apply` changes them and leaves a backup.
- **Provenance/level integrity:** abstraction provenance records the run id and
  member ids; `level` of each abstraction equals `1 + max(child levels)`.

Determinism rules from the testing spec apply (no network in asserted tests,
fixed seeds, reset state per test).

### Live sanity (manual, not CI)
- Run the dreamer (`--dry-run`) against the **real accumulated graph** and have a
  human read the proposed abstractions: are the labels/summaries meaningful, are
  the clusters sensible? This is the actual PoC validation — the asserted tests
  prove the machinery is correct; only human review proves the *idea* works.
- The calibration script from the testing spec is useful here to pick
  `CLUSTER_SIM_THRESHOLD` from the real similarity distribution.

---

## 10. Deliverables & acceptance

### Deliverables
1. `dreamer.py` — the CLI (`--dry-run` default, `--apply`, `--full`/`--recent N`,
   `--run-id`), runnable via `uv`.
2. Any new `GraphStore` methods the dreamer needs (with unit tests), keeping all
   graph-library access inside `graph.py`.
3. `summarize_cluster` in `llm.py` with its prompt.
4. New constants in `config.py`, documented.
5. Asserted tests (§9) added to the existing suite; live sanity script.
6. Short `DREAMER.md` in the repo: what it does, how to run a review pass, how to
   apply, how to roll back a run via `--run-id` + backup.

### Acceptance criteria
- A `--dry-run` on a fixture graph prints a readable report of proposed
  abstractions via the graph-diff helper and writes nothing.
- `--apply` creates abstraction nodes with `parent_of` edges, correct `level`
  values, stored embeddings, and a backup of the prior state.
- Re-running is idempotent (no duplicate abstractions).
- All §9 asserted tests pass deterministically (run 3× in CI, identical results).
- `grep` confirms isolation: `networkx` only in `graph.py`, LLM calls only in
  `llm.py`.
- Running against the real accumulated graph produces abstractions a human can
  review — this is the go/no-go artifact for the project's decision gate.

---

## 11. What "good" looks like (for the human reviewer)

When you run the first real `--dry-run`, you are looking for three things
(qualitative, from the project plan):

1. **Meaningful abstractions** — do the theme labels/summaries read as *your own*
   themes, or as noise? Recognizing them is the strongest positive signal.
2. **Sensible hierarchy shape** — deeper where you've engaged more, shallow where
   you haven't; roots that are genuinely high-level.
3. **Sensible clustering** — related facts grouped, unrelated facts not forced
   together.

If those hold, the core hypothesis of the project is supported and the deferred
machinery (contradiction handling, adaptive triggers, cross-edges, richer recall)
becomes worth building. If they don't, that is the cheapest possible place to
learn it — which is the entire point of having built to this gate.
