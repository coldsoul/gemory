# Dreamer -- offline graph consolidation

The Dreamer is an offline consolidation process for the Gemory memory
system. It finds clusters of related facts in the graph, creates
abstraction nodes that summarize them, and builds an emergent hierarchy
through recursive consolidation. The result is a knowledge graph that
organizes itself from flat facts into meaningful themes and sub-themes.

## How to run

### Review pass (dry-run, default)

This is the safe way to preview what the Dreamer would do. It performs
the full clustering and abstraction in memory but never touches your
graph files on disk:

```
uv run python gemory/dreamer.py
```

The output shows each proposed abstraction: its level, label, and the
member facts it would summarize. Review this output carefully.

### Apply changes

Once you are satisfied with the dry-run output, apply the changes:

```
uv run python gemory/dreamer.py --apply
```

This creates timestamped backups of both ``memory.json`` and
``embeddings.json`` before writing any changes. The backup files
have ``.bak`` suffixes with a UTC timestamp, e.g.
``memory.json.20260706T180000Z.bak``.

### Incremental runs (--recent N)

To consolidate only recently active nodes (useful when you have a large
graph and run the Dreamer periodically):

```
uv run python gemory/dreamer.py --recent 7
```

This selects nodes whose provenance timestamps fall within the last
7 days, plus their immediate neighbours (successors, parents, children).
The rest of the graph is left untouched in that run.

### Specifying a custom memory path

```
uv run python gemory/dreamer.py --memory-path /path/to/memory.json
```

### Setting the run id

```
uv run python gemory/dreamer.py --run-id "my-run-20260706"
```

The run id appears in abstraction provenance for traceability.

## How to roll back

If an applied consolidation produces undesirable results:

1. Find the backup file created just before the run:
   ``memory.json.<timestamp>.bak``

2. Stop any process using the graph (e.g. the MCP server).

3. Copy the backup back into place:

```
cp memory.json.<timestamp>.bak memory.json
cp embeddings.json.<timestamp>.bak embeddings.json
```

4. Restart the MCP server.

## What to look for in the output

When reviewing a dry-run, ask these questions:

* **Are the abstractions meaningful?** Each abstraction should capture
  a genuine common theme among its member facts. If members seem
  unrelated, the ``CLUSTER_SIM_THRESHOLD`` may be too low.

* **Is the hierarchy sensible?** Level-1 abstractions group raw facts.
  Level-2 abstractions group level-1 abstractions. The hierarchy should
  go from specific (level 0) to general (higher levels).

* **Are clusters the right size?** Clusters smaller than
  ``MIN_CLUSTER_SIZE`` (default: 3) are discarded. Clusters larger than
  ``MAX_CLUSTER_SIZE`` (default: 12) are automatically split.

* **Are there duplicate abstractions?** The Dreamer checks Jaccard
  overlap against existing abstractions before creating new ones. A high
  overlap (default >= 80%) reuses the existing abstraction.

## Architecture

The Dreamer builds on four existing modules:

| Module | Role |
|--------|------|
| ``gemory.graph.GraphStore`` | Stores nodes, edges, embeddings. Provides ``_embeddings`` (sidecar dict), ``add_node``, ``add_edge``, ``add_parent_edge``, ``get_parents``, ``get_children``, ``get_all_edges``. |
| ``gemory.cluster.cluster_nodes`` | Louvain community detection over a similarity graph built from embedding cosines and existing ``related`` edges. Enforces size constraints. |
| ``gemory.llm.summarize_cluster`` | LLM call to produce a ``{label, summary}`` dict from a list of member fact strings. |
| ``gemory.llm.embed`` | LLM call to produce a vector for the abstraction text. |

### Consolidation flow

```
1. Load the graph from disk
2. Select working set (--full or --recent)
3. Level 1:
   a. cluster_nodes() -> list of clusters
   b. For each cluster:
      i.   summarize_cluster(member_facts) -> label, summary
      ii.  embed(label + ". " + summary) -> vector
      iii. Create abstraction node (kind="abstraction", level=1)
      iv.  add_parent_edge(abstraction, each member)
4. Level 2:
   a. Working set = the level-1 abstraction node ids
   b. Repeat clustering + abstraction
5. Continue until no more clusters or MAX_LEVELS reached
```

### Size constraints

* ``MIN_CLUSTER_SIZE`` (default: 3): clusters smaller than this are
  discarded (too noisy).
* ``MAX_CLUSTER_SIZE`` (default: 12): clusters larger than this are
  split by re-running community detection on the subgraph.
* ``MAX_LEVELS`` (default: 6): safety cap on recursion depth.

### Idempotency

The ``_create_abstraction`` function checks Jaccard overlap between a
proposed cluster and all existing abstractions' member sets. If the
overlap >= ``ABSTRACTION_OVERLAP`` (default: 80%), the existing
abstraction is reused (optionally extended with any new members).

This means re-running the Dreamer on the same graph produces the same
abstractions, not duplicates.

### Backups

Before any ``--apply`` run, the Dreamer copies:

* ``memory.json`` -> ``memory.json.<timestamp>.bak``
* ``embeddings.json`` -> ``embeddings.json.<timestamp>.bak``

These are plain file copies (``shutil.copy2``), preserving metadata.
They can be restored by reversing the copy.
