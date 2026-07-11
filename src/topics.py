"""Topic registry: canonical topic nodes with match-or-create logic.

Mirrors the store's node-dedup pattern applied to topic strings.
Because topic strings are short and subject-focused, cosine similarity
works reliably here (unlike fact sentences).
"""

import logging
from datetime import datetime, timezone

import numpy as np

from src import config
from src.graph import GraphStore
from src.llm import embed

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def resolve_topic(graph: GraphStore, proposed_topic: str) -> str | None:
    """Resolve a proposed topic string to a canonical topic node ID.

    Returns the topic node ID, or ``None`` if *proposed_topic* is empty or
    whitespace-only.  Creates a new topic node if no existing topic matches
    above :data:`config.TOPIC_MATCH_THRESHOLD`.
    """
    if not proposed_topic or not proposed_topic.strip():
        return None

    proposed_topic = proposed_topic.strip()

    # Embed the proposed topic string.
    topic_embedding = embed(proposed_topic)

    # Check existing topic nodes for a match above threshold.
    existing_topics = graph.get_topic_nodes()
    if existing_topics:
        query = np.array(topic_embedding, dtype=float)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        query_normalised = query / query_norm

        best_id: str | None = None
        best_sim = -1.0
        for topic_node in existing_topics:
            emb = graph.get_node_embedding(topic_node.id)
            if not emb:
                continue
            emb_arr = np.array(emb, dtype=float)
            norm = np.linalg.norm(emb_arr)
            if norm == 0:
                continue
            sim = float(np.dot(emb_arr / norm, query_normalised))
            if sim > best_sim:
                best_sim = sim
                best_id = topic_node.id

        if best_id is not None and best_sim >= config.TOPIC_MATCH_THRESHOLD:
            logger.info(
                "Matched topic %r -> existing %s (sim=%.3f)",
                proposed_topic, best_id[:8], best_sim,
            )
            return best_id

    # No match -- create a new topic node.
    now = _now_iso()
    topic_id = graph.add_node(
        content=proposed_topic,
        embedding=topic_embedding,
        provenance={
            "source_id": "topic-registry",
            "label": proposed_topic,
            "timestamp": now,
        },
        kind="abstraction",
        label=proposed_topic,
        abstraction_kind="topic",
        summary="",
    )
    # Set level to 1.
    graph.set_node_attr(topic_id, "level", 1)

    logger.info("Created new topic %s: %r", topic_id[:8], proposed_topic)
    return topic_id
