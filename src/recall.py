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

# ---------------------------------------------------------------------------
# Relation expansion
# ---------------------------------------------------------------------------

def _expand_relations(
    graph: GraphStore,
    kept_abstraction_ids: set[str],
    all_nodes: dict[str, object],
) -> list[dict]:
    """Follow relates_to edges one hop from *kept_abstraction_ids*.

    Only follows edges with ``provenance="stated"`` (facts asserted).
    Excludes nodes already in the traversed region.
    Returns a list of dicts: ``{node, edge: (from_id, to_id, data)}``.
    """
    related: dict[str, dict] = {}
    for nid in kept_abstraction_ids:
        for source, target, data in graph.get_edges_by_relation("relates_to"):
            if data.get("provenance") != "stated":
                continue
            other: str | None = None
            if source == nid:
                other = target
            elif target == nid:
                other = source  # treat as bidirectional for one hop
            if other is not None and other not in kept_abstraction_ids and other not in related:
                if other in all_nodes:
                    related[other] = {
                        "node": all_nodes[other],
                        "edge": (source, target, data),
                    }
    return list(related.values())


# ---------------------------------------------------------------------------
# Traversal-based recall
# ---------------------------------------------------------------------------

def traverse_recall(
    query: str,
    graph: GraphStore,
    relation_expansion: bool = True,
) -> tuple[str, dict]:
    """Traversal-based recall: descend by parent_of, prune, return the
    surviving region grouped by branch with summaries attached.

    When *relation_expansion* is true (default), stated ``relates_to``
    edges from kept branches are followed one hop and their targets
    rendered under a ``## Related context`` section.

    Returns ``(formatted_text, metrics_dict)`` where *metrics_dict* contains
    ``layers_visited``, ``branches_pruned``, ``branches_kept``,
    ``facts_collected``, ``prune_decisions``, and ``budget_exceeded``.

    No ranking is applied — the caller (an LLM with conversational context)
    is better positioned to rank.  The budget is controlled by
    :data:`config.MAX_RETURNED_FACTS`: if the kept region exceeds it,
    pruning continues deeper; if the graph bottom is reached and the set
    is still too large, summaries + partial facts are returned and the
    over-large-node condition is logged loudly.
    """
    from src.llm import prune_branches

    budget = cfg.MAX_RETURNED_FACTS

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

    # Track the tree of kept parent→children for rendering.
    # {parent_id: {"node": Node, "children": [child_ids], "kept": bool}}
    kept_tree: dict[str, dict] = {}
    # Track which abstraction nodes were kept at each level (for rendering).
    kept_abstraction_ids: set[str] = set()

    while frontier and depth < cfg.MAX_TRAVERSAL_DEPTH:
        # Separate frontier into abstractions (need pruning) and
        # facts / empty nodes (pass through untouched).
        abstractions: list = []
        pass_through: list[str] = []
        for nid in frontier:
            node = all_nodes[nid]
            if node.kind == "abstraction" or node.level > 0:
                abstractions.append(node)
            else:
                pass_through.append(nid)

        # No abstractions left — all remaining frontier nodes are facts.
        if not abstractions:
            break

        # Prune abstraction candidates.
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
            # A total prune with direct facts in the frontier is a valid
            # "stop descending" signal — the answer may be among those direct
            # facts. Only a total prune at the root layer with nothing to fall
            # back on is a genuine failure.
            if pass_through:
                logger.info(
                    "Pruned all %d abstraction branches at depth %d — "
                    "%d direct facts remain in frontier. Query: %r",
                    len(candidates), depth, len(pass_through), query[:80],
                )
            else:
                logger.warning(
                    "TOTAL PRUNE at depth %d -- all %d branches discarded "
                    "with no direct facts. Query: %r",
                    depth, len(candidates), query[:80],
                )
            break

        # Mark kept abstractions for rendering.
        for kid in kept_ids:
            kept_abstraction_ids.add(kid)
            if kid not in kept_tree:
                kept_tree[kid] = {"node": all_nodes[kid], "children": []}

        # Expand: next frontier = pass-through facts + children of kept abstractions.
        next_frontier: set[str] = set(pass_through)
        for kid in kept_ids:
            children = graph.get_children(kid)
            next_frontier.update(children)
            kept_tree[kid]["children"].extend(children)

        frontier = list(next_frontier)
        depth += 1
        metrics["layers_visited"] = depth

        # Budget-aware: if too many facts would be returned, prune deeper.
        fact_count = sum(
            sum(1 for cid in info.get("children", [])
                if cid in all_nodes and all_nodes[cid].level == 0)
            for info in kept_tree.values()
        )
        all_children = sum(len(info.get("children", [])) for info in kept_tree.values())
        if fact_count > budget and all_children > 0:
            continue  # The next iteration will prune deeper
        elif fact_count == 0:
            # No facts yet — continue descending
            continue
    # The frontier now contains only facts (no abstractions left to prune)
    # or we broke out early.  Collect metrics.
    metrics["facts_collected"] = sum(
        sum(1 for cid in info.get("children", [])
            if cid in all_nodes and all_nodes[cid].level == 0)
        for info in kept_tree.values()
    )

    # ── relation expansion: one-hop from kept branches ──
    related_nodes: list[dict] = []
    if relation_expansion and kept_abstraction_ids:
        related_nodes = _expand_relations(graph, kept_abstraction_ids, all_nodes)
        if related_nodes:
            logger.info(
                "Relation expansion: %d related nodes from %d kept branches",
                len(related_nodes), len(kept_abstraction_ids),
            )

    # ── render: grouped by branch, with summaries ──
    lines: list[str] = []
    rendered_facts = 0
    for bid, info in kept_tree.items():
        if bid not in all_nodes:
            continue
        branch = info["node"]
        label = branch.label or branch.content[:40]
        summary_text = branch.summary or branch.content
        lines.append(f"## {label}")
        if summary_text:
            lines.append(f"({summary_text})")
        lines.append("")

        child_ids = info.get("children", [])
        # Separate child abstractions from leaf facts
        child_abs = [(cid, all_nodes[cid]) for cid in child_ids
                     if cid in all_nodes and (all_nodes[cid].kind == "abstraction" or all_nodes[cid].level > 0)]
        child_facts = [all_nodes[cid] for cid in child_ids
                       if cid in all_nodes and all_nodes[cid].kind != "abstraction" and all_nodes[cid].level == 0]

        # Render child abstractions as sub-sections (recursive)
        for cid, cnode in child_abs:
            clabel = cnode.label or cnode.content[:40]
            csummary = cnode.summary or cnode.content
            lines.append(f"### {clabel}")
            if csummary:
                lines.append(f"({csummary})")
            lines.append("")
            # Facts under this child abstraction
            sub_info = kept_tree.get(cid)
            if sub_info:
                sub_child_ids = sub_info.get("children", [])
                for scid in sub_child_ids:
                    if scid in all_nodes and all_nodes[scid].level == 0:
                        lines.append(f"- {all_nodes[scid].content}")
                        rendered_facts += 1
            lines.append("")

        # Render direct leaf facts
        for f in child_facts:
            lines.append(f"- {f.content}")
            rendered_facts += 1
        lines.append("")

    # ── related context section ──
    if related_nodes:
        lines.append("## Related context (via relations, 1 hop)")
        lines.append("")
        for item in related_nodes:
            node = item["node"]
            source, target, data = item["edge"]
            label = node.label or node.content[:40]
            summary_text = node.summary or node.content
            edge_desc = (
                f"{source[:8]} -> {target[:8]}, {data.get('provenance', '')}"
            )
            lines.append(f"### {label}  [reached via: {edge_desc}]")
            if summary_text:
                lines.append(f"({summary_text})")
            lines.append("")
            children = graph.get_children(node.id)
            fact_children = [
                c for c in children
                if c in all_nodes and all_nodes[c].level == 0
            ]
            for fc in fact_children[:cfg.MAX_RELATED_FACTS]:
                lines.append(f"- {all_nodes[fc].content}")
                rendered_facts += 1
            if len(fact_children) > cfg.MAX_RELATED_FACTS:
                lines.append(
                    f"(+{len(fact_children) - cfg.MAX_RELATED_FACTS} more "
                    f"facts -- see summary above)"
                )
            lines.append("")

    if not lines:
        return "No matching facts found in traversal.", metrics

    total_facts = sum(
        sum(1 for cid in info.get("children", [])
            if cid in all_nodes and all_nodes[cid].level == 0)
        for info in kept_tree.values()
    )
    if total_facts > budget:
        metrics["budget_exceeded"] = True
        logger.warning(
            "BUDGET EXCEEDED: %d facts after %d layers (budget=%d). "
            "Returning summaries + partial facts. An intermediate abstraction "
            "node may be needed (dreamer: split over-large node).",
            total_facts, metrics["layers_visited"], budget,
        )
        lines.append(
            f"(Budget of {budget} facts exceeded. "
            f"{total_facts} facts in region; showing {rendered_facts}. "
            "Consider adding intermediate abstraction nodes.)"
        )

    return "\n".join(lines).rstrip("\n"), metrics
