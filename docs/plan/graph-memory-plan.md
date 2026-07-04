# Gemory — Proof of Concept Plan

*Working title (graph + memory) — everything here, name included, is provisional
and may change depending on the outcome of the PoC.*

A long-term agentic memory system structured as an evolving DAG of facts, with
periodic offline consolidation ("the dreamer") that grows emergent, multi-level
abstraction. This document scopes the **minimal PoC** whose only job is to answer
one question:

> Does hierarchical graph memory with periodic consolidation produce more useful
> long-term recall than a flat fact store — and can a human recognise the
> abstractions the dreamer produces as meaningful?

Everything not serving that question is deferred.

---

## Guiding principles

- **Prove the concept before polishing it.** Flat graph first; hierarchy only
  once the plumbing works and real data exists.
- **Manual over automated.** The dreamer is a script you run by hand and inspect
  with your own eyes. Human judgement is the primary evaluator at this stage.
- **Dogfood on real conversations.** The fastest signal is recognising your own
  facts and abstractions, not synthetic benchmarks.
- **Keep it greppable.** The whole system should be understandable without an
  architecture diagram. Five files, one readable JSON graph.
- **Wrap the graph.** All graph-library calls live behind one module so the
  storage backend can be swapped later without touching anything else.

---

## Scope

### In scope (PoC)
- Local MCP server exposing `remember` and `recall` to the agent.
- Flat graph (nodes + edges), JSON persistence, human-inspectable.
- Post-conversation fact extraction (separate LLM call, atomic facts).
- Idempotent store with similarity-based deduplication.
- Manually-run dreamer that proposes consolidations for human review.
- Interactive visualiser over the graph.

### Explicitly deferred (only if the concept holds)
- Contradiction handling & contested states
- Confidence propagation up the hierarchy
- Adaptive / density-based dreamer triggers
- Multi-parent DAG semantics (start tree-ish; DAG is the eventual target)
- Automated real-time extraction during conversation
- Graph database backend (Neo4j etc.)
- Remote / VPS deployment, concurrency, locking

---

## Components

### 1. MCP Server (`server.py`)
The agent's only interface to memory. Two tools, nothing more.

- `recall(query)` — returns relevant stored context to seed a conversation.
- `remember(transcript)` — accepts a conversation transcript for extraction.
  *In the PoC this is triggered manually / on demand rather than automatically.*

The server knows nothing about graph internals — it calls into `graph.py` and
the extractor. Kept deliberately thin.

### 2. Graph wrapper (`graph.py`)
The single chokepoint for all graph-library access. Small, deliberate interface:

- `add_node`, `add_edge`
- `find_similar` (embedding cosine similarity — dedup + connection detection)
- `get_neighbors`, `get_roots`
- `save`, `load`

Every backend-specific call lives here and nowhere else. This is the seam along
which a future Neo4j migration happens.

### 3. Fact extractor (part of `remember` flow)
A dedicated LLM call, separate from the conversational agent, prompted
specifically for extraction.

- Input: conversation transcript.
- Output: **atomic** facts — one discrete claim each — plus proposed
  connections to existing nodes.
- Each stored fact carries **provenance** (conversation ID + timestamp) even
  though later features consume it, not the PoC.

### 4. Dreamer (`dreamer.py`)
A manually-run CLI script. This is the novel part and where most design effort
goes.

- Loads the graph, runs community detection over recently-active nodes.
- Asks an LLM to propose abstractions (higher-level nodes) for each cluster.
- Writes the abstraction summary **at consolidation time** (it has full context
  then; recall should stay cheap).
- Output is surfaced for **human review** before/after applying — the core
  qualitative validation step.

### 5. Recall / retrieval (part of `recall` flow)
Start simple, sophisticate only if data demands it.

- PoC: cosine similarity over embeddings of leaf nodes.
- Later: scan roots → select entry points → beam search with per-node exit
  summaries → merge results.

### 6. Visualiser (`visualize.py`)
Turns the JSON graph into something a human can actually explore.

- `pyvis` → standalone interactive HTML (drag/zoom/hover).
- Node **colour by hierarchy level**, **size by confidence/reach**.
- **Filtering** built in from the start: threshold by confidence, or show only
  a subgraph N hops from a chosen node, so large graphs stay legible.
- Keep a one-line GraphML export (`networkx.write_gexf`) so Gephi's heavier
  analysis is available later without rework.

---

## Proposed file layout

```
gemory/
├── server.py        # MCP server: remember + recall
├── graph.py         # graph-library wrapper: load/save/query/similarity
├── dreamer.py       # CLI, run manually post-conversation
├── visualize.py     # pyvis rendering + filtering
├── llm.py           # thin LLM-client wrapper (swap models per task)
├── memory.json      # the graph — human-readable
└── pyproject.toml   # uv-managed
```

---

## Tech stack

| Concern            | Choice                        | Rationale                                             |
|--------------------|-------------------------------|-------------------------------------------------------|
| Language / pkg mgr | Python + `uv`                 | Existing comfort; zero-friction envs                  |
| MCP protocol       | official `mcp` Python SDK     | Handles the protocol plumbing                         |
| Graph              | `networkx` + JSON persistence | Zero infra, fully inspectable, built-in algorithms    |
| LLM (extraction/dreamer) | DeepSeek via `openai` SDK | Cheap for high-token extraction/consolidation work    |
| LLM client         | thin wrapper (`llm.py`)       | Swap model per task (extract vs consolidate) later    |
| Embeddings         | API-based, `numpy` for math   | Removes a local dependency at PoC scale               |
| CLI                | `click`                       | Clean manual-run interface for the dreamer            |
| Visualisation      | `pyvis` (→ Gephi via GraphML) | Interactive, local, same Python process               |

**Deliberately NOT used yet:** Neo4j / graph DB, local embedding models,
async frameworks, LangChain / LlamaIndex (keep the graph visible, not abstracted
away).

---

## Key design decisions (settled)

- **Store neither raw transcript nor agent-inline flags.** A dedicated
  post-conversation extractor decides *what* to store; only distilled atomic
  facts are persisted. Transcripts are input, not graph content.
- **Idempotent store.** Re-running extraction on the same conversation must not
  create duplicates — worst case is a few confidence bumps / new edges.
- **Dedup = merge sensitivity.** One similarity threshold governs both. Start
  **conservative (high threshold)**: false duplicates are easy to spot and fix;
  wrongly-merged distinct facts are hard to notice. Loosen as data reveals.
- **Re-run story.** Transcripts aren't kept in the graph; if re-extraction is
  needed, feed a saved conversation back through `remember`. Idempotency makes
  this safe.
- **Level is structural, not labelled.** A node's "level" is its distance from
  leaves; roots (no parents) are the retrieval entry points. Depth emerges from
  engagement, not from fixed tiers.

---

## High-level deliverables & build order

Ordered so that a **baseline measurement exists before the dreamer**, and so
each step produces something testable.

1. **MCP server skeleton** — responds to tool calls, persists a JSON file.
   *Deliverable:* agent can call a no-op `remember`/`recall` round-trip.
2. **`remember` + extractor** — atomic-fact extraction, nodes added with
   provenance, idempotent dedup.
   *Deliverable:* a real conversation produces sensible atomic nodes; re-running
   it adds no duplicates.
3. **`recall`** — similarity retrieval over leaf nodes.
   *Deliverable:* a query returns relevant stored facts.
4. **Visualiser** — pyvis view with colour/size/filter.
   *Deliverable:* the graph is explorable in a browser.
5. **Use it for ~1 week** of real conversations. Accumulate real data.
   *Deliverable:* a non-trivial graph grown from genuine use.
6. **Dreamer** — community detection + LLM abstraction proposals, human-reviewed.
   *Deliverable:* first consolidation run; abstractions inspected by eye.
7. **Decision gate** — look at what the dreamer produced and decide whether the
   concept holds and is worth building out.

---

## Testing & verification

### Functional (does the plumbing work?)
- **Round-trip:** agent calls `remember` then `recall`; stored facts come back.
- **Atomicity:** extractor output is one-claim-per-node, not compound blobs.
  Spot-check against transcripts.
- **Idempotency:** run extraction twice on the same transcript → node count
  stable, only confidence/edge changes. This is the key correctness property.
- **Dedup threshold:** deliberately feed near-duplicate and genuinely-distinct
  facts; confirm the former merge and the latter don't. Tune the knob here.
- **Persistence:** save → load → graph is identical.

### Qualitative (does the *concept* work?) — the important part
- **Abstraction sniff test:** after a dreamer run, a human judges whether the
  proposed uber-nodes are genuinely useful summaries or noise. This is the
  primary PoC signal.
- **Recognition:** do the abstractions read as *your own* themes? Dogfooding
  makes this the fastest, most honest evaluator.
- **Community sanity check:** compare the dreamer's clusters against Gephi's
  independent community detection on the same graph — do they roughly agree?
- **Legibility:** can you understand the graph's state from the pyvis view
  without reading `memory.json`?

### Comparative (later, if the concept holds)
Introduce synthetic ground truth and baselines to move from "looks good" to
"measurably better":
- Fictional persona with a **known** fact set across several life domains;
  50–100 LLM-generated conversations revealing facts over "time" (some
  reinforced, some contradicting). Ground truth enables precise recall scoring.
- **Baselines:** flat vector-search fact store; recency-weighted flat store.
- **Success = graceful degradation:** as conversation count grows, flat stores
  bury early facts under noise; the graph system should stay navigable via
  consolidation. Plus meaningful abstractions and better cross-domain retrieval.

---

## Rough effort estimate

- MCP server + flat graph + extractor + recall + visualiser: ~a weekend.
- Dreamer: ~another weekend.
- → Something genuinely testable within ~2 weeks of part-time work, hooked into
  a tool used daily.

---

## Prior art to read first

- **Generative Agents** (Park et al., 2023) — the reflection mechanism ≈ dreamer.
  arxiv.org/abs/2304.03442
- **GraphRAG / From Local to Global** (Edge et al., 2024) — hierarchical
  community summaries. arxiv.org/abs/2404.16130
- **MemGPT** (Packer et al., 2023) — tiered memory. arxiv.org/abs/2310.08560
- **Zep / Graphiti** (2025) — conversational temporal knowledge graphs; closest
  to the online-phase design. arxiv.org/abs/2501.13956
- **Cognitive Architectures for Language Agents** (Sumers et al., 2023) — survey
  context. arxiv.org/abs/2309.02427

**Novelty sits in the combination:** an evolving conversational graph where
consolidation is graph-aware, producing emergent multi-level abstraction with
confidence semantics — not cleanly solved by any single system above.
