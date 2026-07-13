"""Consolidation helpers: split cluster() from summarize() with input contracts.

Clustering receives labels+summaries (the gestalt).
Summarizing receives children content (compound-upward).
"""

import logging
from dataclasses import dataclass

from src.graph import GraphStore
from src.llm import summarize_cluster
from src.reach import compute_reach

logger = logging.getLogger(__name__)


@dataclass
class NodeGestalt:
    """The gestalt of a node -- what clustering should see."""

    node_id: str
    label: str       # Short label (or fact text for facts)
    summary: str     # Content/summary (fact text or abstraction summary)


def cluster_layer(
    graph: GraphStore,
    node_ids: list[str],
    method: str = "algorithm",
    seed: int = 42,
    context: str | None = None,
) -> list[set[str]]:
    """Cluster *node_ids* into groups.

    Parameters
    ----------
    graph
        The graph store.
    node_ids
        Node IDs to cluster.
    method
        ``"algorithm"`` (cosine via Louvain), ``"llm"`` (LLM-based grouping on
        labels+summaries), or ``"hybrid"`` (algorithm first, then LLM on
        leftovers).
    seed
        Random seed for deterministic algorithm clustering.
    context
        Optional context string describing the subject being partitioned (used
        by LLM clustering to tune toward aspect finding rather than broad theme
        identification).

    Returns
    -------
    list[set[str]]
        List of clusters, each a set of node IDs.
    """
    if len(node_ids) < 2:
        return []

    # Build gestalt for all nodes (needed by LLM and hybrid methods).
    id_list = list(node_ids)
    gestalts: list[dict] = []
    for nid in id_list:
        node = graph.get_node(nid)
        label = node.label or node.content[:50]
        summary = node.content
        gestalts.append({
            "index": len(gestalts),
            "id": nid,
            "label": label,
            "summary": summary,
        })

    if method == "algorithm":
        return _cluster_algorithm(graph, id_list, seed)

    elif method == "llm":
        return _cluster_llm(gestalts, id_list, context=context)

    elif method == "hybrid":
        # Run algorithm first.
        algo_clusters = _cluster_algorithm(graph, id_list, seed)
        algo_ids: set[str] = set()
        for c in algo_clusters:
            algo_ids.update(c)

        leftover_ids = [nid for nid in id_list if nid not in algo_ids]
        if not leftover_ids or len(leftover_ids) < 2:
            return algo_clusters

        # Filter gestalts to leftovers only and re-index.
        leftover_gestalts = [g for g in gestalts if g["id"] in leftover_ids]
        for i, g in enumerate(leftover_gestalts):
            g["index"] = i

        llm_clusters = _cluster_llm(
            leftover_gestalts, leftover_ids, context=context,
        )
        return algo_clusters + llm_clusters

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _cluster_algorithm(
    graph: GraphStore,
    node_ids: list[str],
    seed: int,
) -> list[set[str]]:
    """Cosine-based clustering over node embeddings (existing Louvain logic)."""
    from src.cluster import cluster_nodes
    return cluster_nodes(graph, node_ids, seed=seed)


def _cluster_llm(
    gestalts: list[dict],
    id_list: list[str],
    context: str | None = None,
) -> list[set[str]]:
    """LLM-based clustering over labels+summaries."""
    from src.llm import cluster_by_llm

    if not gestalts:
        return []

    # Map indices back to node IDs.
    idx_to_id: dict[int, str] = {g["index"]: g["id"] for g in gestalts}

    llm_input = [
        {"index": g["index"], "label": g["label"], "summary": g["summary"]}
        for g in gestalts
    ]

    try:
        raw_clusters = cluster_by_llm(llm_input, context=context)
    except Exception:
        logger.exception("LLM clustering failed, returning no clusters")
        return []

    # Map indices back to node IDs.
    result: list[set[str]] = []
    for cluster in raw_clusters:
        mapped = {idx_to_id[idx] for idx in cluster if idx in idx_to_id}
        if len(mapped) >= 2:
            result.append(mapped)

    return result


def summarize_layer(
    graph: GraphStore,
    cluster: set[str],
) -> dict[str, str]:
    """Write a summary for a cluster using children's content.

    Input contract: summarize from children's content.
    - At level 1 (children are facts): use raw fact texts.
    - Above level 1 (children are abstractions): use their summaries.

    Returns a dict with ``"label"`` and ``"summary"`` keys.
    """
    member_ids = list(cluster)
    child_contents: list[str] = []

    for mid in member_ids:
        node = graph.get_node(mid)
        if node.level == 0:
            # Facts: use raw fact text
            child_contents.append(node.content)
            has_facts = True
        else:
            # Abstractions: compound upward — use label + summary,
            # never the bare label alone. Per §4: "children are
            # abstractions → use their summaries, not raw leaves."
            label = node.label or ""
            summary = node.summary or ""
            if summary:
                child_contents.append(f"{label}: {summary}" if label else summary)
            else:
                # Legacy node without summary field — fall back to content
                child_contents.append(node.content)
            has_abstractions = True

    # Distinguish facts from abstractions in the log
    fact_count = sum(1 for mid in member_ids if graph.get_node(mid).level == 0)
    abs_count = len(member_ids) - fact_count
    if fact_count > 0 and abs_count == 0:
        logger.info("Summarizing cluster of %d facts", len(child_contents))
    elif abs_count > 0 and fact_count == 0:
        logger.info("Summarizing cluster of %d abstractions", len(child_contents))
    else:
        logger.info("Summarizing cluster of %d members (%d facts, %d abstractions)",
                     len(child_contents), fact_count, abs_count)

    return summarize_cluster(child_contents, is_abstraction=(fact_count == 0))
