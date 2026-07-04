"""Recall: retrieve and format relevant memories from the graph.

The ``recall`` function embeds a natural-language query, finds the most
similar stored facts, and returns a human-readable summary.
"""

import logging

from gemory import llm
from gemory.graph import GraphStore

logger = logging.getLogger(__name__)


def recall(query: str, graph: GraphStore, top_k: int = 5) -> str:
    """Search the memory graph for facts relevant to *query*.

    Steps
    -----
    1. Embed the query via :func:`gemory.llm.embed`.
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

    # Check for empty graph before making any API call.
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
