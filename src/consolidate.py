"""Consolidation helpers: split cluster() from summarize() with input contracts.

Clustering receives labels+summaries (the gestalt).
Summarizing receives children content (compound-upward).
"""

import logging
from dataclasses import dataclass

from src.cluster import cluster_nodes
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
    seed: int = 42,
) -> list[set[str]]:
    """Cluster *node_ids* using their gestalt (label + summary).

    This is the clustering input contract: clustering sees
    labels+summaries, NOT full transitive facts.

    Returns clusters as sets of node IDs.
    """
    # Build gestalt for each node (for logging/audit, not for embedding —
    # the embedding was already stored with label+summary at creation time).
    for nid in node_ids:
        node = graph.get_node(nid)
        label = node.label or node.content[:50]
        _gestalt = NodeGestalt(node_id=nid, label=label, summary=node.content)

    # Delegate to the existing Louvain-based clusterer.
    return cluster_nodes(graph, node_ids, seed=seed)


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
        # For facts (level 0): use raw fact text.
        # For abstractions: use their summary (content field, which
        # should be the summary written at their creation time).
        child_contents.append(node.content)

    return summarize_cluster(child_contents)
