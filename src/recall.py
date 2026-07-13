"""Recall: retrieve and format relevant memories from the graph.

The ``recall`` function embeds a natural-language query, finds the most
similar stored facts, and returns a human-readable summary.

The ``traverse_recall`` function performs hierarchical traversal: it starts
from root nodes, uses the LLM to prune irrelevant branches, descends into
promising ones, and collects leaf facts.
"""

import logging

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
    top_k: int | None = None,
) -> tuple[str, dict]:
    """Traversal-based recall: descend by parent_of, prune, return the
    surviving region grouped by branch with summaries attached.

    Returns ``(formatted_text, metrics_dict)`` where *metrics_dict* contains
    ``layers_visited``, ``branches_pruned``, ``branches_kept``,
    ``facts_collected``, ``prune_decisions``, and ``budget_exceeded``.

    No ranking is applied — the caller (an LLM with conversational context)
    is better positioned to rank.  *top_k* is a budget, not a relevance cut:
    if exceeded, pruning continues deeper; if the graph bottom is reached and
    the set is still too large, summaries + partial facts are returned and
    the over-large-node condition is logged loudly.
    """
    from src.llm import prune_branches

    budget = top_k if top_k is not None else cfg.MAX_RETURNED_FACTS

    metrics: dict = {
        "layers_visited": 0,
        "branches_pruned": 0,
        "branches_kept": 0,
        "facts_collected": 0,
        "prune_decisions": [],
        "budget_exceeded": False,
    }

    all_nodes = {n.id: n for n in graph.all_nodes()}
    roots = [n.id for n in all_nodes.values() if not graph.get_parents(n.id)]
    if not roots:
        return "No root nodes found in graph.", metrics

    logger.info("Traversal recall: %d roots, budget=%d, query=%r",
                 len(roots), budget, query[:80])

    frontier = roots
    depth = 0

    # Track the tree of kept nodes: {kept_parent_id: [(child_node, depth), ...]}
    # This is what we'll render at the end.
    kept_tree: dict[str, list] = {}

    while frontier and depth < cfg.MAX_TRAVERSAL_DEPTH:
        abstractions: list = []
        leaf_facts: list = []
        for nid in frontier:
            node = all_nodes[nid]
            if node.kind == "abstraction" or node.level > 0:
                abstractions.append(node)
            else:
                leaf_facts.append(node)

        # Collect leaf facts under their direct parents.
        for fact in leaf_facts:
            for pid in graph.get_parents(fact.id):
                if pid not in kept_tree:
                    kept_tree[pid] = []
                kept_tree[pid].append(fact)
                metrics["facts_collected"] += 1

        if not abstractions:
            break

        # Prune.
        candidates = [
            {
                "id": n.id,
                "label": n.label or n.content[:40],
                "summary": n.summary or n.content,
                "reach": n.reach,
            }
            for n in abstractions
        ]

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
                "TOTAL PRUNE at depth %d -- all %d branches discarded. Query: %r",
                depth, len(candidates), query[:80],
            )
            break

        # Expand children of kept abstractions.
        next_frontier: set[str] = set()
        for kid in kept_ids:
            children = graph.get_children(kid)
            next_frontier.update(children)
            # Record that this kept parent has children to explore.
            if kid not in kept_tree:
                kept_tree[kid] = []

        frontier = list(next_frontier)
        depth += 1
        metrics["layers_visited"] = depth

    # ── budget guard: if the kept region is too large, descend further ──
    total_facts = sum(
        sum(1 for n in nodes if n.level == 0)
        for nodes in kept_tree.values()
    )
    while total_facts > budget and frontier and depth < cfg.MAX_TRAVERSAL_DEPTH:
        # Budget exceeded — prune deeper.
        abstractions = [all_nodes[nid] for nid in frontier
                        if all_nodes[nid].kind == "abstraction" or all_nodes[nid].level > 0]
        leaf_facts = [all_nodes[nid] for nid in frontier
                      if all_nodes[nid].kind != "abstraction" and all_nodes[nid].level == 0]

        for fact in leaf_facts:
            for pid in graph.get_parents(fact.id):
                if pid not in kept_tree:
                    kept_tree[pid] = []
                kept_tree[pid].append(fact)
                metrics["facts_collected"] += 1

        if not abstractions:
            break

        candidates = [
            {"id": n.id, "label": n.label or n.content[:40],
             "summary": n.summary or n.content, "reach": n.reach}
            for n in abstractions
        ]
        try:
            kept_ids = prune_branches(query, candidates)
        except Exception:
            kept_ids = [c["id"] for c in candidates]

        if len(kept_ids) > cfg.MAX_BRANCHES_PER_LEVEL:
            kept_ids = kept_ids[:cfg.MAX_BRANCHES_PER_LEVEL]

        next_frontier = set()
        for kid in kept_ids:
            children = graph.get_children(kid)
            next_frontier.update(children)
            if kid not in kept_tree:
                kept_tree[kid] = []

        frontier = list(next_frontier)
        depth += 1
        metrics["layers_visited"] = depth
        total_facts = sum(
            sum(1 for n in nodes if n.level == 0)
            for nodes in kept_tree.values()
        )

    if total_facts > budget:
        metrics["budget_exceeded"] = True
        logger.warning(
            "BUDGET EXCEEDED: %d facts after %d layers (budget=%d). "
            "Returning summaries + partial facts. An intermediate abstraction "
            "node may be needed (dreamer: split over-large node).",
            total_facts, depth, budget,
        )

    # ── render: grouped by branch, with summaries ──
    lines: list[str] = []
    rendered_facts = 0
    for bid, children in kept_tree.items():
        if bid not in all_nodes:
            continue
        branch = all_nodes[bid]
        label = branch.label or branch.content[:40]
        summary = branch.summary or branch.content
        lines.append(f"## {label}")
        if summary:
            lines.append(f"({summary})")
        lines.append("")

        # Separate child abstractions from leaf facts
        child_abs = [(c.id, c) for c in children if c.kind == "abstraction" or c.level > 0]
        child_facts = [c for c in children if c.kind != "abstraction" and c.level == 0]

        # Render child abstractions as sub-sections
        for cid, cnode in child_abs:
            if budget > 0 and rendered_facts >= budget:
                break
            clabel = cnode.label or cnode.content[:40]
            csummary = cnode.summary or cnode.content
            lines.append(f"### {clabel}")
            if csummary:
                lines.append(f"({csummary})")
            lines.append("")
            # Facts under this child
            sub_facts = [
                n for n in kept_tree.get(cid, [])
                if n.kind != "abstraction" and n.level == 0
            ]
            for f in sub_facts:
                if budget > 0 and rendered_facts >= budget:
                    break
                lines.append(f"- {f.content}")
                rendered_facts += 1
            lines.append("")

        # Render direct leaf facts
        for f in child_facts:
            if budget > 0 and rendered_facts >= budget:
                break
            lines.append(f"- {f.content}")
            rendered_facts += 1
        lines.append("")

    if not lines:
        return "No matching facts found in traversal.", metrics

    if metrics["budget_exceeded"]:
        lines.append(
            f"(Budget of {budget} facts exceeded. "
            f"{total_facts} facts in region; showing {rendered_facts}. "
            "Consider adding intermediate abstraction nodes.)"
        )

    return "\n".join(lines).rstrip("\n"), metrics
