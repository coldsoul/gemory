"""Clustering: group related nodes using Louvain community detection."""

import logging

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities

from src import config
from src.graph import GraphStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_nodes(
    graph: GraphStore,
    node_ids: list[str],
    seed: int = 42,
) -> list[set[str]]:
    """Group *node_ids* into clusters of semantically related nodes.

    Uses Louvain community detection on a similarity graph built from
    embedding cosines and existing ``related`` edges.

    Returns a list of clusters, each a set of node IDs, where each cluster
    is within ``[MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE]`` inclusive.
    """
    if len(node_ids) < config.MIN_CLUSTER_SIZE:
        logger.info(
            "Working set too small (%d nodes < %d min), skipping",
            len(node_ids), config.MIN_CLUSTER_SIZE,
        )
        return []

    # Build a similarity graph over the working set.
    cluster_graph = _build_similarity_graph(graph, node_ids)

    # Run Louvain community detection.
    raw_clusters = _detect_communities(cluster_graph, seed)

    # Enforce size constraints.
    clusters = _enforce_size_constraints(raw_clusters, cluster_graph)

    logger.info(
        "Clustering: %d nodes -> %d clusters (sizes: %s)",
        len(node_ids), len(clusters),
        [len(c) for c in clusters],
    )
    return clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_similarity_graph(graph: GraphStore, node_ids: list[str]) -> nx.Graph:
    """Build an undirected similarity graph over *node_ids*.

    Two nodes are connected when either:

    1. Their embedding cosine similarity >= ``CLUSTER_SIM_THRESHOLD``, or
    2. They share an existing ``related`` edge in the store.
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)

    # Build embedding matrix from the sidecar.
    vectors: list[list[float]] = []
    valid_ids: list[str] = []
    for nid in node_ids:
        emb = graph._embeddings.get(nid)
        if emb is not None:
            vectors.append(emb)
            valid_ids.append(nid)

    if len(valid_ids) < 2:
        return g

    vec_matrix = np.array(vectors, dtype=float)
    norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec_matrix /= norms
    sim_matrix = vec_matrix @ vec_matrix.T

    id_to_idx = {nid: i for i, nid in enumerate(valid_ids)}

    # Connect nodes whose cosine >= CLUSTER_SIM_THRESHOLD.
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            sim = float(sim_matrix[i, j])
            if sim >= config.CLUSTER_SIM_THRESHOLD:
                g.add_edge(valid_ids[i], valid_ids[j], weight=sim)

    # Also add edges for explicit ``related`` edges from the store.
    # (NOT ``relates_to`` — relations are directional links between
    # entities, not cohesion signals and must never be used for clustering.)
    for u, v, data in graph.get_edges_by_relation("related"):
        if u in id_to_idx and v in id_to_idx:
            if not g.has_edge(u, v):
                g.add_edge(u, v, weight=data.get("weight", 0.5))

    return g


def _detect_communities(cluster_graph: nx.Graph, seed: int) -> list[set[str]]:
    """Run Louvain community detection on *cluster_graph*."""
    if cluster_graph.number_of_edges() == 0:
        # Every node is isolated — each is its own "community".
        return [{n} for n in cluster_graph.nodes()]

    communities = louvain_communities(
        cluster_graph,
        weight="weight",
        seed=seed,
    )
    return [set(c) for c in communities]


def _enforce_size_constraints(
    raw_clusters: list[set[str]],
    cluster_graph: nx.Graph,
) -> list[set[str]]:
    """Discard clusters below ``MIN_CLUSTER_SIZE`` and split oversize ones."""
    result: list[set[str]] = []
    for cluster in raw_clusters:
        size = len(cluster)
        if size < config.MIN_CLUSTER_SIZE:
            continue  # too small, discard
        if size <= config.MAX_CLUSTER_SIZE:
            result.append(cluster)
        else:
            # Too large — split by re-running detection on the subgraph.
            logger.info(
                "Cluster of size %d exceeds max %d, splitting...",
                size, config.MAX_CLUSTER_SIZE,
            )
            subgraph = cluster_graph.subgraph(cluster).copy()
            sub_clusters = _detect_communities(subgraph, seed=42)
            for sc in sub_clusters:
                sc_size = len(sc)
                if config.MIN_CLUSTER_SIZE <= sc_size <= config.MAX_CLUSTER_SIZE:
                    result.append(sc)
                elif sc_size > config.MAX_CLUSTER_SIZE:
                    # Still too large — accept it with a warning.
                    logger.warning(
                        "Sub-cluster of size %d still exceeds max, truncating",
                        sc_size,
                    )
                    result.append(sc)
    return result
