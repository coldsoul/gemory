# Gemory MCP Server — Implementation Spec

*Working title. This document is a self-contained brief for a coding agent to
implement the **MCP server layer** of the Gemory graph-memory PoC. It assumes no
prior context. Build exactly what is specified; do not add features from the
"Out of scope" list.*

---

## 1. What you are building

A **local MCP (Model Context Protocol) server** in Python that gives an LLM agent
a long-term memory backed by a graph. The server exposes a small set of tools.
Behind those tools sits a graph of atomic facts persisted to a single JSON file.

The server is one layer of a larger PoC. **You are implementing the MCP server,
the graph wrapper, the LLM client wrapper, the extraction flow, and the recall
flow.** The offline "dreamer" consolidation process and the visualiser are
separate deliverables and are **out of scope** here — but the code you write must
not preclude them (see §9).

The guiding ethos: **small, greppable, inspectable.** The entire graph must be
readable as JSON. All graph-library access must sit behind one module so the
backend can be swapped later.

---

## 2. Tech stack (required)

| Concern            | Choice                                  |
|--------------------|-----------------------------------------|
| Language           | Python 3.11+                            |
| Package manager    | `uv`                                    |
| MCP protocol       | Official `mcp` Python SDK               |
| Graph              | `networkx`, persisted to JSON           |
| LLM (extraction)   | DeepSeek via the `openai` SDK (OpenAI-compatible endpoint) |
| Embeddings         | API-based; `numpy` for vector math      |
| Config             | Environment variables (`.env` via `python-dotenv`) |

Do **not** introduce: Neo4j or any graph database, local embedding models,
LangChain / LlamaIndex, async frameworks beyond what the MCP SDK requires, or any
web framework.

---

## 3. File layout (produce exactly this)

```
gemory/
├── server.py        # MCP server: defines and wires up the tools
├── graph.py         # Graph wrapper — the ONLY module that imports networkx
├── extractor.py     # Transcript -> atomic facts (LLM call)
├── recall.py        # Query -> relevant nodes (similarity search)
├── llm.py           # Thin LLM + embeddings client wrapper
├── models.py        # Dataclasses / types for Node, Edge, Fact
├── config.py        # Env-var loading, constants (thresholds, confidence base/increment, paths, model names)
├── memory.json      # The graph, human-readable, embedding-free (runtime; gitignored)
├── embeddings.json  # Embedding sidecar, keyed by node id (runtime; gitignored)
├── pyproject.toml   # uv-managed
├── .env.example     # Documented env vars, no secrets
└── README.md        # How to run + how to register with an MCP client
```

Keep `networkx` imports **only** in `graph.py`. Keep LLM/embedding API calls
**only** in `llm.py`. `server.py`, `extractor.py`, and `recall.py` must depend on
those wrappers, never on the libraries directly. This separation is a hard
requirement, not a style preference — it is the seam for future backend swaps.

---

## 4. Data model (`models.py`)

Define these as dataclasses. Field names are part of the spec.

### Global conventions (apply everywhere)
- **All timestamps are ISO-8601 UTC strings.** This applies uniformly to
  `created_at`, `updated_at`, and the `timestamp` inside each provenance entry.
  One format, one timezone, no exceptions.
- **The graph is DIRECTED** — a `networkx.DiGraph`, treated as a DAG. Even though
  PoC association edges are symmetric in *meaning*, the graph is directed so that
  hierarchy (`parent_of` / `child_of`), which is inherently directional, can be
  added later without a structural migration. Do not use an undirected `Graph`.
- **Behavioural constants live in `config.py`**, never hard-coded inline: the
  confidence base value and the confidence increment (alongside the thresholds
  already specified in §7).

### Node
Represents one atomic fact (PoC has only fact-level nodes; abstraction nodes come
later via the dreamer, so leave room but don't build them).

| Field         | Type              | Notes                                                   |
|---------------|-------------------|---------------------------------------------------------|
| `id`          | `str`             | UUID4 string, generated on creation                     |
| `content`     | `str`             | The atomic fact, one claim                              |
| `confidence`  | `float`           | Initialised to `CONFIDENCE_BASE` (config); increased by `CONFIDENCE_INCREMENT` (config) on each *new-source* corroboration |
| `provenance`  | `list[dict]`      | Append-only, **idempotent** list of `{source_id, label, timestamp}` (see §7.1) |
| `created_at`  | `str` (ISO 8601)  | UTC                                                     |
| `updated_at`  | `str` (ISO 8601)  | UTC; updated on any mutation                            |
| `level`       | `int`             | **Computed/derived, not hand-set.** Structural: distance-from-leaf. PoC leaf facts are all `0`. Reserved for hierarchy; do not assign arbitrary values. |

**Embeddings are NOT stored on the node.** See "Embedding storage" below.

### Edge
A directed relationship between two nodes.

| Field        | Type    | Notes                                                        |
|--------------|---------|--------------------------------------------------------------|
| `source`     | `str`   | Node id (edge origin)                                        |
| `target`     | `str`   | Node id (edge destination)                                   |
| `weight`     | `float` | Similarity / association strength                            |
| `relation`   | `str`   | Controlled vocabulary. **PoC uses exactly `"related"`.**     |

**`relation` vocabulary is controlled, not free-text.** The PoC emits only
`"related"` (a symmetric association; stored in a single canonical direction —
e.g. `source` = the newer node, `target` = the matched existing node). Values
such as `parent_of` / `child_of` / `contradicts` are **reserved for later
phases** — do NOT invent or emit other relation values in the PoC.

Do not model multi-parent hierarchy or contested/temporal edges yet — just leave
the schema open enough that adding fields later is non-breaking.

### Embedding storage (sidecar, not inline)
Embeddings are stored in a **separate sidecar file**, not on the node and not in
`memory.json`. Rationale: `memory.json` is reviewed by a human throughout the PoC,
and inline 1000+-dimension float arrays would drown the actual fact content and
make the file unreadable — which defeats the inspectability principle the whole
review-driven validation depends on.

- Sidecar file (e.g. `embeddings.json` or a small on-disk key-value store),
  keyed by node `id` → `list[float]`.
- Written/updated atomically alongside `memory.json` (§5 persistence rules apply
  to both; keep them consistent — a node in the graph must have a corresponding
  embedding, and orphaned embeddings should be tolerated on load, not fatal).
- `GraphStore` owns both files and hides the split behind its interface; callers
  never touch the sidecar directly.
- `memory.json` therefore contains only human-legible fields: content,
  confidence, provenance, timestamps, level, and edges.

---

## 5. Graph wrapper (`graph.py`) — the core module

This wraps a `networkx.DiGraph` (directed, treated as a DAG — see §4) and is the
single source of truth for graph operations. It owns **both** `memory.json` and
the embedding sidecar, hiding the split behind its interface. Required interface:

```python
class GraphStore:
    def __init__(self, path: str): ...     # owns memory.json + embedding sidecar
    def load(self) -> None: ...            # read both files; empty graph if absent
    def save(self) -> None: ...            # atomic write of BOTH files (see below)

    def add_node(self, content: str, embedding: list[float],
                 provenance: dict) -> str: ...
        # stores content/provenance/etc in the graph, embedding in the sidecar
        # keyed by the new node id; returns node id
    def add_edge(self, source: str, target: str,
                 weight: float, relation: str = "related") -> None: ...
        # directed edge; PoC relation is always "related" (§4)

    def find_similar(self, embedding: list[float],
                     threshold: float | None = None,
                     top_k: int = 10) -> list[tuple[str, float]]: ...
        # cosine similarity against sidecar embeddings
        # returns [(node_id, cosine_similarity), ...] sorted desc

    def get_node(self, node_id: str) -> Node: ...
    def get_neighbors(self, node_id: str) -> list[str]: ...
    def get_roots(self) -> list[str]: ...   # nodes with no parents; PoC: may be all
    def bump_confidence(self, node_id: str, provenance: dict) -> bool: ...
        # idempotent: if provenance['source_id'] already present on node, do
        # nothing and return False. Otherwise append provenance, increment
        # confidence by CONFIDENCE_INCREMENT, update updated_at, return True.
        # (See §7.1)
    def all_nodes(self) -> list[Node]: ...
```

### Persistence requirements
- `memory.json` must be **human-readable**: pretty-printed, stable key ordering,
  and **embedding-free** (embeddings live in the sidecar per §4).
- Use `networkx`'s node-link JSON form (`networkx.node_link_data` /
  `node_link_graph`) or an equivalent you document. Round-trip must be lossless:
  `save()` then `load()` yields an identical graph **and** identical embeddings.
- **Atomic writes:** write each file (graph and embedding sidecar) to a temp file
  and `os.replace()` it into place, so a crash mid-write can't corrupt either
  store. On load, tolerate an embedding present without a matching node (orphan —
  ignore) but treat a node without a matching embedding as a clear error.

### Similarity
- Cosine similarity via `numpy`. Precompute/normalise as convenient.
- At PoC scale (thousands of nodes) a linear scan is fine; do not add an index.

---

## 6. LLM wrapper (`llm.py`)

Isolate every external model call here so models can be swapped per task.

```python
def extract_facts(transcript: str) -> list[str]: ...   # returns atomic facts
def embed(text: str) -> list[float]: ...               # single embedding
def embed_batch(texts: list[str]) -> list[list[float]]: ...
```

- Point the `openai` client at DeepSeek's base URL; read key + base URL + model
  names from config/env.
- `extract_facts` builds the extraction prompt (see §7), calls the model, parses
  the response, returns a clean `list[str]`. Handle and strip any code-fence or
  preamble the model adds; parse defensively.
- Fail loudly with clear errors on auth/parse failure — never silently return
  empty results in a way that looks like "nothing to store".

---

## 7. Extraction flow (`extractor.py`)

Turns a transcript into stored nodes. This is where store-quality is won or lost.

### The extraction prompt (design carefully)
The single most important instruction is **atomicity**: the model must output
**discrete, single-claim facts**, never compound sentences.

- Bad: `"Radi uses uv and works on a VPS and likes bonsai"`
- Good: three facts — `"Radi uses uv"`, `"Radi works on a VPS"`,
  `"Radi likes bonsai"`.

Prompt requirements:
- Output a **strict JSON array of strings** and nothing else (no prose, no
  fences). Parse defensively regardless.
- One atomic claim per element.
- Extract durable facts about the user / world, not conversational filler or
  one-off pleasantries.
- Prefer stable, self-contained statements that will make sense out of context
  (resolve pronouns where possible).

### Starting prompt (use as the initial implementation)
Implement this as the initial extraction prompt. It is a **starting point meant
to be iterated** once real output is observed — keep it easy to edit in one place.

```
You are a fact extractor for a long-term memory system. You are given a
transcript of a conversation between a user and an assistant. Your job is to
extract durable, atomic facts worth remembering about the user and their world.

Output ONLY a JSON array of strings. No prose, no explanation, no markdown code
fences. If there are no facts worth storing, output [].

Rules for what to extract:
- Extract DURABLE facts: things likely to remain true and be useful in future
  conversations (preferences, background, ongoing projects, relationships,
  goals, constraints, decisions, stable opinions).
- Do NOT extract transient or conversational content: greetings, the assistant's
  suggestions, questions, one-off task details, or anything that only matters
  within this single conversation.
- Extract facts about the USER and their world, not about the assistant.

Rules for HOW to write each fact (this is the most important part):
- Each fact must be ATOMIC: exactly one claim. Never combine claims with "and".
  Bad:  "The user uses uv and works on a VPS and likes bonsai"
  Good: "The user uses uv"
        "The user works on a VPS"
        "The user likes bonsai"
- Each fact must be SELF-CONTAINED: it must make sense on its own, months later,
  with no access to this conversation. Resolve pronouns and vague references to
  concrete nouns.
  Bad:  "He wants to build it in Python"
  Good: "The user wants to build the memory system in Python"
- Write each fact as a complete, present-tense statement.
- Use a consistent way of referring to the user across facts (e.g. always "The
  user ..."). Do not use their name unless it is itself the fact being stored.
- State only what the transcript supports. Do not infer, speculate, or embellish.
  If something is uncertain or hypothetical, either omit it or state the
  uncertainty explicitly as part of the fact.

Output format: ["fact one", "fact two", ...]
```

### Why the prompt is shaped this way (context for tuning it)
- **Atomicity uses a worked example, not just a rule.** Models follow
  demonstrated patterns more reliably than abstract instructions; the "and"
  conjunction is a concrete surface cue for the most common compound-fact case.
  This instruction most affects graph quality.
- **Self-containment has its own example** because unresolved pronouns are
  invisible in the moment but useless when a node is retrieved out of context.
  Self-contained facts also embed and match better, which serves recall.
- **The consistent-referent rule ("always 'The user'") quietly aids dedup.**
  Dedup/merge runs on embedding similarity; equal facts phrased with different
  subjects ("Radi uses uv" vs "The user works with uv") embed further apart and
  can slip past `DEDUP_THRESHOLD`. Consistent phrasing tightens the space so
  equal facts land close together.
- **"Do not infer or speculate"** matters more than in ordinary extraction: a
  false fact here doesn't sit inertly — it gets corroborated, connected, and
  eventually consolidated into an abstraction. Under-extract rather than pollute.
- **Explicit `[]`** stops the model manufacturing facts from small talk to seem
  useful.

### Two knobs to expect to tune (leave the prompt trivially editable)
- **Durable vs transient** is the fuzziest judgment and the likeliest thing to
  adjust. If the extractor over-captures in-conversation brainstorming (e.g. an
  idea that was considered then discarded a message later), add: "prefer facts
  the user states about themselves over decisions reached during the
  conversation." If it under-captures, loosen it.
- **Atomic granularity has an empirical floor.** "The user is developing a bald
  cypress bonsai" could be one fact or two ("the user does bonsai" + "the user
  has a bald cypress"). Over-splitting yields trivially-thin nodes that are hard
  to connect; under-splitting yields compound-ish facts. The starting prompt errs
  slightly toward splitting; find the right grain by inspecting a week of real
  output. There is no principled answer — the graph tells you.

Iterate against a **fixed sample transcript** (the same fixture used in tests):
re-run after each prompt edit and watch how the extracted set changes. This is
the fastest way to converge on a version to trust.


### 7.1 Source identity & idempotent provenance — read this before the store algorithm

**Design tension being solved:** the point of long-term memory is that
conversations *grow*. A conversation is an append-only stream, not a fixed
artifact. So identifying a source by hashing the *whole* transcript is wrong: a
grown, re-submitted conversation would hash differently and be counted as a brand
new source, inflating corroboration counts — and corroboration (source diversity)
is a load-bearing confidence signal. **Do NOT hash the whole transcript.**

Two mechanisms together resolve this. Implement both.

**(1) Correctness guarantee — idempotent provenance.**
A provenance entry is `{source_id, label, timestamp}`. Provenance is a *set* in
behaviour, not a bag: **a given `source_id` may appear on a given node at most
once.** Before appending a provenance entry to a node (whether creating it or
corroborating it), check whether an entry with the same `source_id` already
exists on that node; if so, do not append and do not increment confidence for it
again. This is what makes re-processing a grown transcript safe: facts from its
earlier portion hit existing nodes via similarity dedup, and their provenance is
already recorded, so nothing double-counts. **Conversation identity is therefore
not load-bearing** — this guarantee holds even if the `source_id` is imperfect.

**(2) Stable-enough source id — first-exchange hash, computed server-side.**
Because a conversation only grows by appending, an earlier submission is a
*prefix* of a later one. Derive `source_id` from the **first exchange** (first
user message + first assistant response), not the whole transcript. Using the
first *exchange* rather than just the first user message matters: opening user
turns are low-entropy ("hi", "help me debug this") and can collide across
conversations, but the assistant's first response is high-entropy and diverges
immediately, so the pair is effectively unique.

Definition:

```
source_id = "sha256:" + hexdigest(
    normalize(first_user_message) + "\n<gemory-sep>\n" + normalize(first_assistant_message)
)
```

- `normalize` applied **per turn, before hashing** (this is load-bearing, not
  cosmetic — the id must be identical when a grown transcript is re-submitted):
  strip leading/trailing whitespace, collapse internal whitespace runs, normalize
  line endings.
- Turns are joined with a **fixed separator** (`\n<gemory-sep>\n` above, or any
  documented constant) so that concatenation is unambiguous — otherwise
  `"AB" + "C"` and `"A" + "BC"` could hash to the same value.
- **Idempotency requirement:** the first user turn and first assistant turn are
  fixed history by the time `remember` is called (extraction is post-conversation;
  earlier turns don't mutate), so the hash is stable across re-submission of a
  grown transcript. Do **not** hash anything that a client might re-render
  differently on a later pass — normalization defends against incidental
  whitespace/line-ending drift.
- **Single-turn fallback:** if the transcript has no first assistant message
  (e.g. a truncated transcript — shouldn't occur for a post-conversation
  `remember`, but define it rather than improvise), fall back to
  `sha256(normalize(first_user_message))` alone.
- Computed **inside the server**, never supplied by the agent.

Collision behaviour is deliberately on the safe side: in the rare event two
conversations share an identical first exchange, they'd share a `source_id` and
be treated as one source — this *under*-counts corroboration (conservative),
whereas whole-transcript hashing *over*-counts (corrupting). Combined with (1),
even when the id is imperfect the fact-level graph stays correct.

`label` is the optional human-readable `conversation_name` (see §8), stored for
legibility only — **never** used in dedup, identity, or idempotency logic. A
wrong/missing/duplicate label can never corrupt the graph.

**Future upgrade (not now):** if a client reliably supplies a stable conversation
ID, it can replace the prefix hash as `source_id`. Nothing here blocks that; the
idempotent-provenance guarantee (1) makes the source of the id swappable.

### Store algorithm (idempotent + dedup) — implement exactly
Given the batch of extracted facts and a single `source_id` + `label` computed
once for the whole transcript (§7.1):

For each extracted fact string:
1. `embedding = llm.embed(fact)`.
2. `matches = graph.find_similar(embedding, threshold=DEDUP_THRESHOLD, top_k=1)`.
3. If a match exists above `DEDUP_THRESHOLD` (same fact, existing node):
   - If this `source_id` is **not** already in the node's provenance: append
     `{source_id, label, timestamp}` and bump confidence.
   - If this `source_id` **is** already present: do nothing (idempotent — this is
     what makes re-running on the same or grown transcript safe).
   - Either way, do **not** create a new node.
4. Else (new fact):
   - `new_id = graph.add_node(content=fact, embedding=embedding,
     provenance=[{source_id, label, timestamp}])`.
   - **Connection detection:** query
     `find_similar(embedding, threshold=EDGE_THRESHOLD, top_k=K)` and add an edge
     from `new_id` to each returned node that is below the dedup threshold but
     above `EDGE_THRESHOLD`. (Close-but-not-identical ⇒ edge, not merge.)
5. `graph.save()` once at the end of the batch.

`timestamp` is UTC now. `source_id` and `label` are computed once per `remember`
call per §7.1, not per fact.

### Tunable constants (all in `config.py`)
- `DEDUP_THRESHOLD` — **start conservative / high** (e.g. 0.92). False duplicates
  are easy to spot and fix; wrongly-merged distinct facts are hard to notice.
- `EDGE_THRESHOLD` — lower (e.g. 0.75). Tunable.
- `CONFIDENCE_BASE` — initial confidence for a newly created node (e.g. 1.0).
- `CONFIDENCE_INCREMENT` — added on each *new-source* corroboration (e.g. 1.0).
- All must be single named constants, easy to change, each with a documented
  meaning. No behavioural number is hard-coded outside `config.py`.

---

## 8. MCP server + tools (`server.py`)

Use the official `mcp` Python SDK to expose tools over stdio (the transport
Claude Desktop and similar clients use for local servers).

### Tools to expose

**`recall`**
- Input: `query: str`, optional `top_k: int`.
- Behaviour: embed the query, run similarity search over nodes, return the
  top matches as readable text (fact content + light metadata like confidence).
- Output: a compact, human/agent-readable string or structured list of facts.
  This is what an agent calls at the **start** of a conversation to load context.

**`remember`**
- Input: `transcript: str` (required); `conversation_name: str | None`
  (optional — stored as a human-readable **label** only; see §7.1).
- Behaviour: compute `source_id` from a stable transcript prefix and `label` from
  `conversation_name` (§7.1), then run the extraction flow (§7) over the
  transcript. **Do not** accept a client-supplied identity for dedup/idempotency —
  the `source_id` is always derived server-side.
- Output: a short summary — facts extracted, how many were new vs corroborated,
  and how many were skipped as already-recorded for this `source_id` (idempotent
  no-ops). This confirms the manual post-conversation step worked.

### Server requirements
- Load `GraphStore` once at startup; `load()` the graph.
- Each tool call operates on the in-memory graph and `save()`s appropriately.
- Clear structured logging to stderr (never stdout — stdout is the MCP channel):
  extraction counts, merges, errors.
- Graceful handling if `memory.json` is missing (start empty) or malformed (fail
  with a clear message rather than silently resetting).

### Registration
The `README.md` must show the exact JSON snippet to register this server with an
MCP client (Claude Desktop config), including how the `uv`/python entrypoint is
invoked and where `memory.json` lives.

---

## 9. Forward-compatibility constraints (do not violate)

These keep deferred features buildable without a rewrite:
- **`level` field on nodes** exists now (0 for all), so hierarchy can be added.
- **`provenance` is a list**, so multiple corroborations accumulate — this feeds
  future confidence/source-diversity logic.
- **Graph access is behind `GraphStore`** so a Neo4j backend can replace it by
  reimplementing one file.
- **LLM/embeddings behind `llm.py`** so models can be swapped per task.
- Edge/Node schemas are **additive-friendly**: adding fields later must not break
  existing `memory.json` files (tolerate missing keys on load).

---

## 10. Out of scope (do NOT build)

- The dreamer / consolidation / community detection.
- Any hierarchy creation, abstraction nodes, or multi-parent DAG logic.
- Contradiction handling, contested states, confidence propagation.
- The visualiser (pyvis / Gephi).
- Automated real-time extraction during conversation (extraction is a manual,
  explicit `remember` call in the PoC).
- Remote/VPS deployment, multi-user, concurrency/locking beyond atomic file
  writes.
- Beam-search retrieval / exit summaries (recall is flat similarity for now).

---

## 11. Deliverables & acceptance

### Deliverables
1. All files in §3, runnable via `uv`.
2. `README.md`: setup, env vars, run command, MCP client registration snippet.
3. `.env.example` documenting every required variable.
4. A minimal test suite (see below).

### Acceptance criteria
- **Round-trip:** an MCP client can call `remember` with a sample transcript,
  then `recall` with a related query and get the stored facts back.
- **Atomicity:** given a compound-fact transcript, extraction yields separate
  one-claim nodes (verify against a fixture).
- **Idempotency — exact re-run (critical):** calling `remember` twice with the
  same transcript leaves node count unchanged on the second run, and — because
  provenance is idempotent per `source_id` (§7.1) — confidence and provenance are
  *also* unchanged (the second run is a pure no-op at the provenance level).
  Cover with an automated test.
- **Idempotency — grown transcript (critical):** submit a transcript, then submit
  a longer version that appends new content to the same conversation. Facts from
  the original portion must **not** re-corroborate (same `source_id` via prefix
  hash, already recorded); only genuinely new facts from the appended portion are
  added. This is the case that matters for real long-term use. Cover with an
  automated test.
- **Dedup vs edge:** near-duplicate facts merge (no new node); close-but-distinct
  facts create a new node **and** an edge. Cover with a test using controlled
  embeddings.
- **Persistence:** `save()` → `load()` yields an identical graph **and identical
  embeddings**; `memory.json` is pretty-printed, diff-friendly, and contains **no
  embedding vectors** (they live in the sidecar). A directed `DiGraph` is used.
- **Isolation:** `grep -r networkx` finds hits only in `graph.py`; LLM/embedding
  calls appear only in `llm.py`.

### Testing notes
- Unit-test `graph.py` (add/dedup/edge/save/load/similarity) with **stubbed
  embeddings** (inject vectors directly) so tests are deterministic and don't hit
  the network.
- Unit-test the store algorithm's idempotency and dedup/edge branching with
  hand-crafted vectors around the thresholds.
- Unit-test `source_id` derivation (§7.1): the same first exchange (first user
  message + first assistant response) yields the same id across an original and a
  grown transcript; whitespace/line-ending-only differences do not change it
  (per-turn normalization); the fixed separator prevents boundary collisions
  (`"AB"+"C"` ≠ `"A"+"BC"`); the single-turn fallback (no assistant message)
  hashes the first user message alone; and idempotent-provenance rejects a
  duplicate `source_id` on a node.
- Extraction prompt output can be tested against a recorded/fixture LLM response;
  don't require a live model for the test suite to pass.
- Keep a separate, clearly-marked manual/integration check that does hit DeepSeek
  + embeddings for end-to-end sanity, runnable on demand but not in CI.

---

## 12. Suggested build order

1. `config.py`, `models.py`, `pyproject.toml`, `.env.example` — scaffolding.
2. `graph.py` + its unit tests (stubbed embeddings). Get persistence and
   similarity solid first — everything rests on this.
3. `llm.py` — extraction + embeddings wrappers against DeepSeek.
4. `extractor.py` — the store algorithm; test idempotency and dedup/edge here.
5. `recall.py` — similarity retrieval.
6. `server.py` — wire the two tools over stdio; structured stderr logging.
7. `README.md` + registration snippet + end-to-end manual check.

Start with the graph and its tests. If the store algorithm's idempotency and
dedup behaviour are correct and well-tested, the rest is straightforward wiring.
