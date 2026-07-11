"""Recall: retrieve and format relevant memories from the graph.

The ``recall`` function embeds a natural-language query, finds the most
similar stored facts, and returns a human-readable summary.

The ``traverse_recall`` function performs hierarchical traversal: it starts
from root nodes, uses the LLM to prune irrelevant branches, descends into
promising ones, and collects leaf facts.
"""

import logging

import numpy as np

from src import config as cfg, llm
from src.graph import GraphStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat (embedding-based) recall
# ---------------------------------------------------------------------------

def recall(query: str, graph: GraphStore, top_k: int = 5) -> str:
    """Search the memory graph for facts relevant to *query*.

    Steps
    -----
    1. Embed the query via :func:`src.llm.embed`.
    2. Run similarity search over all stored embeddings (no threshold).
    3. Format the top-``top_k`` results as a readable string.

    Parameters
    ----------
    query
        Free-text question or description to search for.
    graph
        An initialised :class:`GraphStore` instance.
    top_k
        Maximum number of results to return.

    Returns
    -------
    str
        Formatted memory results, or an empty-state message.
    """
    logger.info("Recalling memories for query (%d chars), top_k=%d", len(query), top_k)

    if not graph._embeddings:
        logger.info("Memory is empty")
        return "Memory is empty. No facts stored yet."

    embedding = llm.embed(query)

    results = graph.find_similar(embedding, top_k=top_k)

    if not results:
        logger.info("No matching memories found")
        return "No matching memories found."

    logger.info("Found %d matching memories", len(results))

    lines: list[str] = []
    for rank, (node_id, similarity) in enumerate(results, start=1):
        node = graph.get_node(node_id)
        lines.append(
            f"[#{rank} | similarity: {similarity:.3f} | confidence: {node.confidence:.1f}]"
        )
        lines.append(node.content)
        lines.append("")  # blank separator

    return "\n".join(lines).rstrip("\n")


# ---------------------------------------------------------------------------
# Traversal-based recall
# ---------------------------------------------------------------------------

def traverse_recall(
    query: str,
    graph: GraphStore,
    top_k: int = 10,
) -> tuple[str, dict]:
    """Traversal-based recall: roots -> prune -> descend -> collect facts.

    Returns ``(formatted_text, metrics_dict)`` where *metrics_dict* contains
    ``layers_visited``, ``branches_pruned``, ``branches_kept``,
    ``facts_collected``, and ``prune_decisions``.
    """
    from src.llm import prune_branches

    metrics = {
        "layers_visited": 0,
        "branches_pruned": 0,
        "branches_kept": 0,
        "facts_collected": 0,
        "prune_decisions": [],
    }

    # Get root nodes (nodes with no parent_of parent).
    all_nodes = {n.id: n for n in graph.all_nodes()}
    roots = []
    for n in all_nodes.values():
        if not graph.get_parents(n.id):
            roots.append(n.id)

    if not roots:
        return "No root nodes found in graph.", metrics

    logger.info(
        "Traversal recall: %d roots, query=%r", len(roots), query[:80],
    )

    frontier = roots
    collected_facts: list = []
    depth = 0

    while frontier and depth < cfg.MAX_TRAVERSAL_DEPTH:
        abstractions = []
        leaf_facts = []
        for nid in frontier:
            node = all_nodes[nid]
            if node.kind == "abstraction" or node.level > 0:
                abstractions.append(node)
            else:
                leaf_facts.append(node)

        collected_facts.extend(leaf_facts)
        metrics["facts_collected"] += len(leaf_facts)

        if not abstractions:
            break

        # Build candidate list for pruning.
        candidates = []
        for node in abstractions:
            candidates.append({
                "id": node.id,
                "label": node.label or node.content[:40],
                "summary": node.summary or node.content,
                "reach": node.reach,
            })

        try:
            kept_ids = prune_branches(query, candidates)
        except Exception:
            logger.exception("Prune failed at depth %d, keeping all", depth)
            kept_ids = [c["id"] for c in candidates]

        if len(kept_ids) > cfg.MAX_BRANCHES_PER_LEVEL:
            kept_ids = kept_ids[:cfg.MAX_BRANCHES_PER_LEVEL]

        discarded = [c["id"] for c in candidates if c["id"] not in kept_ids]
        metrics["branches_kept"] += len(kept_ids)
        metrics["branches_pruned"] += len(discarded)
        metrics["prune_decisions"].append({
            "layer": depth,
            "kept": kept_ids,
            "discarded": discarded,
        })

        if not kept_ids:
            logger.warning(
                "TOTAL PRUNE at depth %d -- all %d branches discarded. "
                "Query: %r",
                depth, len(candidates), query[:80],
            )
            break

        # Expand: next frontier = children of kept abstractions.
        next_frontier: set[str] = set()
        for kid in kept_ids:
            children = graph.get_children(kid)
            next_frontier.update(children)

        frontier = list(next_frontier)
        depth += 1
        metrics["layers_visited"] = depth

    # Fallback if nothing collected from traversal.
    if not collected_facts and depth > 0:
        return (
            "Traversal found no matching facts. "
            f"(Visited {depth} layers, pruned {metrics['branches_pruned']} branches.)",
            metrics,
        )

    # Rank collected facts by cosine similarity to query.
    if collected_facts and top_k > 0:
        query_emb = llm.embed(query)
        scored: list[tuple[float, str]] = []
        for node in collected_facts:
            emb = graph._embeddings.get(node.id)
            if emb:
                qn = np.linalg.norm(query_emb)
                en = np.linalg.norm(emb)
                if qn > 0 and en > 0:
                    sim = float(np.dot(
                        np.array(query_emb) / qn,
                        np.array(emb) / en,
                    ))
                    scored.append((sim, node.id))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = {nid for _, nid in scored[:min(top_k, cfg.MAX_FACTS_RETURNED)]}
        collected_facts = [graph.get_node(nid) for nid in top_ids]

    # Format output.
    lines: list[str] = []
    for i, node in enumerate(collected_facts[:top_k]):
        lines.append(f"[#{i + 1} | confidence: {node.confidence:.1f}]")
        lines.append(node.content)
        lines.append("")

    if not lines:
        return "No matching facts found in traversal.", metrics

    return "\n".join(lines), metrics
