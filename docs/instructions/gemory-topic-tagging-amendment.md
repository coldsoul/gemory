# Gemory — Amendment: Topic Tagging & Topic-Based Consolidation

*Change spec for the implementing agent. Amends three existing components — the
**extractor**, the **store/`GraphStore`**, and the **dreamer** — and adds one new
component, the **topic registry**. Consistent with all prior specs (`DiGraph`,
`kind`, `level`, `parent_of`, `provenance`/`source_id`, sidecar embeddings,
`GraphStore` as sole graph seam, `config.py` constants, stub-embedding tests).*

---

## 0. Why this change (the finding that motivates it)

The dreamer's original clustering step assumed atomic fact **embeddings** would
form tight enough neighbourhoods that a cosine threshold could discover themes.
Measured against a real 67-node graph, that assumption is false:

- Across all pairs, the **maximum** cosine similarity is ~0.75; the mean is ~0.26.
- Eight facts that are unambiguously the **same project** (MCP server, testing
  harness, dreamer, memory.json, dedup threshold, …) have an intra-group
  fact-sentence cosine of **mean 0.48, min 0.33**.
- No `CLUSTER_SIM_THRESHOLD` can group those eight while excluding unrelated
  facts, because as *sentences* they are not close.

Root cause: **atomicity makes facts semantically narrow.** "The user implemented
the MCP server" and "The user implemented the testing harness" share a subject
but differ in content, so sentence embeddings place them only mildly close. The
property that makes facts good graph nodes makes them poor clustering targets by
raw cosine.

Key observation that fixes it: the facts that belong together **share a topic**
("the memory system"), and **topic strings are short and directly about the
subject, so they embed tightly** even when the full fact sentences do not. So:
assign each fact a topic, canonicalize topics so they don't fragment, and cluster
on topic — not on fact-sentence cosine.

---

## 1. Summary of the change

1. **Extractor** now returns, per fact, a proposed **topic** (a short subject
   phrase) alongside the fact text — using the full-conversation context it
   already has (the best possible moment to know the topic).
2. A new **topic registry** canonicalizes proposed topics via **match-or-create**
   (embed the topic string; reuse an existing topic if close, else mint a new
   one). This mirrors the existing node-dedup pattern and prevents tag
   fragmentation ("Gemory" vs "the memory system" → one canonical topic).
3. **Topics are stored as abstraction nodes** (`kind="abstraction"`, `level=1`),
   and each fact gets a directed `parent_of` edge **from its topic to the fact**.
   Level-1 structure is therefore built cheaply **at write time**.
4. The **dreamer's level-1 clustering changes** from cosine-over-facts to simply
   reading topic membership (near-free). The dreamer's real work moves **up**: it
   groups *topics* into broader level-2+ themes, where the input count is small
   enough that even LLM-driven grouping is cheap.

This makes online writes do the frequent, cheap structure-forming and the offline
dreamer do the sparse, higher-level abstraction — which is closer to the original
design intent than cosine-clustering ever was.

---

## 2. Extractor changes (`extractor.py` / `llm.py`)

### 2.1 Output shape
`extract_facts` now returns, instead of a list of strings, a list of objects:

```json
[
  {"fact": "The user has implemented the MCP server for their memory system.",
   "topic": "Gemory memory system"},
  {"fact": "The user flies FPV drones.",
   "topic": "FPV drones"}
]
```

Parse defensively (strip fences/preamble; tolerate a bare string by treating
`topic` as empty → falls to the "no confident topic" path in §3.4).

### 2.2 Prompt additions
Extend the existing extraction prompt (keep all current atomicity /
self-containment / durable-vs-transient / anti-hallucination rules) with:

- For each fact, also give a **topic**: a short (2–5 word) noun phrase naming the
  **subject** the fact is about — the project, entity, or area — not a restatement
  of the fact.
  - Good: fact "The user implemented the testing harness for their memory system"
    → topic "Gemory memory system".
  - The topic should be the **same phrase** for all facts about the same subject
    within this conversation (be consistent), so downstream matching is easy.
  - Prefer a concrete project/entity name if one is present in the conversation
    (e.g. "Gemory", "Sofia transit tracker", "MS information website").
- If a fact does not clearly belong to any subject, output an **empty** topic
  (`""`); do not force one. (These facts stay un-topiced; see §3.4.)
- The topic must be supported by the conversation; **do not invent** a subject
  not present.

### 2.3 Note
Live extraction is non-deterministic, so asserted tests stub the extractor to
return fixed `{fact, topic}` objects (see §6). Extraction/topic *quality* is
reviewed only in the live suite.

---

## 3. Topic registry (new component)

A small module (e.g. `topics.py`), owned like everything else through
`GraphStore` for persistence, that turns free-text proposed topics into
**canonical topic nodes** so tags don't fragment.

### 3.1 What a topic is
A topic is an **abstraction node** in the same graph:
- `kind = "abstraction"`,
- `level = 1`,
- `content` = canonical topic label (the first proposed spelling that created it,
  or a cleaned form),
- `embedding` = embedding of the **topic string** (short → embeds tightly),
  stored in the sidecar like any node,
- `provenance` = origin marker (e.g. `source_id = "topic-registry:<...>"`).
- It is distinguishable from dreamer-created higher abstractions by `level==1`
  and by an optional `abstraction_kind = "topic"` field (additive; higher
  dreamer abstractions use `abstraction_kind = "theme"`). Tolerate absence.

### 3.2 Match-or-create (mirrors node dedup)
Given a proposed topic string `t`:
1. `e = llm.embed(t)`.
2. Compare `e` against embeddings of existing **topic** nodes (level-1
   abstractions with `abstraction_kind="topic"`), cosine similarity.
3. If the best match ≥ `TOPIC_MATCH_THRESHOLD` → **reuse** that topic's id.
4. Else → **create** a new topic node with label `t` and embedding `e`.

This is exactly the store's dedup pattern applied to topic strings. Because topic
strings are short and subject-focused, `TOPIC_MATCH_THRESHOLD` can be set
meaningfully high (see §5) — this is the regime where cosine works well, unlike
fact sentences.

### 3.3 Linking facts to topics
When a fact node is stored (or corroborated), after its topic is resolved to a
canonical topic id, add a directed edge **`parent_of` from the topic node to the
fact node** (topic is the parent; fact is the child). Idempotent: do not add a
duplicate `parent_of` edge if it already exists. A fact has at most one topic
parent in this PoC (single-topic assumption; multi-topic is a future extension —
do not build it now, but do not structurally forbid it either).

### 3.4 Un-topiced facts
If the proposed topic is empty, store the fact node normally with **no** topic
parent. These facts are simply not part of level-1 topic structure yet; a later
dreamer pass may still pick them up (or a future version may re-topic them). Do
not block storage on having a topic.

### 3.5 Interaction with existing node dedup
Topic resolution happens **per fact, at store time**, and is **independent of**
the existing fact-node dedup (which still runs on fact-sentence embeddings with
`DEDUP_THRESHOLD`). Order: resolve/store the fact node first (dedup as today),
then resolve its topic and link. If the fact was a corroboration of an existing
node, still ensure the topic edge exists (idempotently) — a repeat mention
shouldn't lose the topic link.

---

## 4. Dreamer changes (`dreamer.py`)

### 4.1 Level-1 clustering: replace cosine with topic membership
The dreamer no longer clusters facts by fact-sentence cosine at level 1. Instead:

- **Level-1 clusters are given**: each topic node and its `parent_of` fact
  children *is* a cluster, already built at write time.
- The dreamer's level-1 job shrinks to **maintenance and enrichment**:
  - For each topic with ≥ `MIN_CLUSTER_SIZE` children, ensure it has a written
    **summary** (call `summarize_cluster` on the child facts to produce a richer
    `summary` beyond the bare topic label — the label came from a single fact's
    context; the summary sees all members).
  - Topics with fewer than `MIN_CLUSTER_SIZE` children are left as-is (a thin
    topic is fine; it just isn't enriched yet).
- The old `CLUSTER_SIM_THRESHOLD` fact-cosine path is **removed** as the level-1
  mechanism. (It may remain available as a fallback for **un-topiced** facts
  only — optional; if included, keep it clearly separated and off by default.)

### 4.2 Level-2+ : group topics into themes (this is the dreamer's real work)
Now the recursion starts from **topics**, not facts:

- Working set for level 2 = all topic nodes (level-1 abstractions).
- Cluster **topics** into broader themes. Because there are few topics (tens, not
  thousands) and topic strings/summaries embed tightly, **either** method works
  here:
  - cosine over topic embeddings with a threshold (now viable — topics cluster
    well), **or**
  - LLM grouping: hand the LLM the list of topic labels+summaries and ask for
    thematic groups. Cheap at this scale, and this is where a C-style LLM
    grouping is affordable precisely because the input is small.
- Create level-2 `theme` abstractions (`abstraction_kind="theme"`) over the topic
  clusters, with `parent_of` edges topic→... reversed appropriately (theme is
  parent of its topics). Summaries written at consolidation time.
- Recurse upward as before (themes of themes) until a level yields no cluster ≥
  `MIN_CLUSTER_SIZE`, capped at `MAX_LEVELS`.

Everything else about the dreamer is unchanged: dry-run default, backup on apply,
`--run-id` tagging, idempotency (a topic already summarized / a theme already
covering the same topic set is updated, not duplicated — reuse
`ABSTRACTION_OVERLAP`), graph-diff reporting, structured logging.

### 4.3 `level` recomputation
Facts stay level 0. Topics are level 1. Themes are `1 + max(child level)`. Keep
the level field maintained as structure grows.

---

## 5. Config additions (`config.py`)

- `TOPIC_MATCH_THRESHOLD` — cosine for match-or-create of topic strings. Set
  **high** (e.g. 0.80–0.85) — topic strings are short and subject-focused, so
  genuine same-topic phrasings score high and this threshold can be strict
  without fragmenting. Tune against the topic-string distribution (see §7).
- Reuse existing `MIN_CLUSTER_SIZE`, `MAX_CLUSTER_SIZE`, `ABSTRACTION_OVERLAP`,
  `MAX_LEVELS`.
- `CLUSTER_SIM_THRESHOLD` is now used (if at all) only for the optional
  un-topiced-fact fallback and for level-2 topic clustering if the cosine method
  is chosen there. Document its narrowed role.

---

## 6. Testing (extends the harness; stub embeddings, frozen fixtures)

Add asserted tests. Stub the extractor to emit fixed `{fact, topic}` objects and
stub embeddings so topic-string similarities are controlled (reuse the
`vectors_with_cosines` helper).

- **Topic match-or-create:** two facts whose proposed topics are different
  spellings of the same subject (topic-string vectors set above
  `TOPIC_MATCH_THRESHOLD`) resolve to **one** topic node; a genuinely different
  topic (below threshold) creates a **second**. Assert topic-node count.
- **Fragmentation prevented:** the classic case — facts tagged "Gemory" and "the
  memory system" (topic vectors set high) share one topic node.
- **Fact→topic linking:** every stored fact with a non-empty topic has exactly one
  incoming `parent_of` edge from its topic; idempotent on re-store.
- **Un-topiced facts:** empty topic → fact stored, no topic parent, no error.
- **Corroboration keeps topic link:** re-storing a fact that corroborates an
  existing node still ensures the topic edge exists (no lost link, no duplicate).
- **Dreamer level-1 is topic-driven:** given topics with children, the dreamer
  enriches topics with ≥ `MIN_CLUSTER_SIZE` children (summary written) and skips
  thinner ones — with **no** dependence on fact-sentence cosine. Assert that a
  set of facts that are far apart in fact-embedding space but share a topic are
  correctly grouped under that topic (this is the regression test for the whole
  motivating finding).
- **Dreamer level-2 grouping:** several topics that belong to one theme (topic
  vectors/summaries set close) produce one level-2 theme abstraction; assert
  `level` values and `parent_of` structure.
- **Idempotency:** re-running extraction and the dreamer creates no duplicate
  topics, edges, or abstractions.

Live sanity (manual): run the real extractor on a real conversation and eyeball
the proposed topics for consistency; run the dreamer `--dry-run` on the real graph
and review whether topic groupings and level-2 themes are meaningful.

---

## 7. Calibration note

Before trusting `TOPIC_MATCH_THRESHOLD`, verify the core assumption on real data
(the one-minute check): once some facts are tagged, embed the **topic strings**
only and confirm same-subject topics score high (≥ ~0.8) while different subjects
score low. The existing calibration script can be pointed at the topic-node
embeddings to produce the ranked pair list. Expectation, based on the motivating
finding: topic strings will separate far more cleanly than fact sentences did
(fact sentences maxed at 0.75 and averaged 0.26; short subject phrases should show
a much clearer high/low split). If for some reason they do not, fall back to an
**enumerated canonical topic list** the extractor must choose from, rather than
free-texting topics — collision-proof at the cost of flexibility.

---

## 8. Order of implementation

1. Extractor output shape + prompt (§2), with stubbed tests.
2. Topic registry match-or-create (§3), mirroring node dedup; tests.
3. Fact→topic linking at store time (§3.3–3.5); tests.
4. Dreamer level-1 switch to topic membership (§4.1); regression test for the
   far-apart-but-same-topic case.
5. Dreamer level-2 topic grouping + recursion (§4.2); tests.
6. Calibration check (§7) on the real graph, then a real `--dry-run` review.

Step 4's regression test is the one that proves this whole amendment did its job:
facts that cosine-clustering could not group (mean 0.48, min 0.33) must now be
grouped correctly because they share a topic.
