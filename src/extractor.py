"""Extractor: compute source ids, run the idempotent store algorithm.

The ``compute_source_id`` function derives a stable hash from the first
exchange in a conversation transcript.  The ``store_facts`` function embeds
each fact, deduplicates against the graph, creates edges for close-but-distinct
facts, and persists everything.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone

from src import config
from src import llm
from src.graph import GraphStore
from src.topics import resolve_topic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role-prefix detection
# ---------------------------------------------------------------------------

_USER_PREFIXES = ("user:", "human:")
_ASSISTANT_PREFIXES = ("assistant:", "ai:")
_ALL_PREFIXES = _USER_PREFIXES + _ASSISTANT_PREFIXES


# ---------------------------------------------------------------------------
# Transcript parsing helpers
# ---------------------------------------------------------------------------

def _detect_role(line: str) -> str | None:
    """Return ``"user"``, ``"assistant"``, or ``None`` for a role-prefixed line."""
    lower = line.strip().lower()
    for prefix in _USER_PREFIXES:
        if lower.startswith(prefix):
            return "user"
    for prefix in _ASSISTANT_PREFIXES:
        if lower.startswith(prefix):
            return "assistant"
    return None


def _extract_content(line: str) -> str:
    """Remove the recognised role prefix from *line* and return the remainder,
    stripped."""
    stripped = line.strip()
    lower = stripped.lower()
    for prefix in _ALL_PREFIXES:
        if lower.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _parse_first_turns(transcript: str) -> tuple[str, str]:
    """Extract the **first** user message and **first** assistant message.

    Returns
    -------
    (first_user_message, first_assistant_message)
        Either string may be empty if that role was not found.
    """
    first_user = ""
    first_assistant = ""
    current_role: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal first_user, first_assistant
        if current_role == "user" and not first_user:
            first_user = "\n".join(current_lines)
        elif current_role == "assistant" and not first_assistant:
            first_assistant = "\n".join(current_lines)

    for raw_line in transcript.split("\n"):
        role = _detect_role(raw_line)

        if role is not None:
            _flush()
            current_role = role
            content = _extract_content(raw_line)
            current_lines = [content] if content else []
        else:
            if current_role is not None:
                current_lines.append(raw_line.rstrip())

        if first_user and first_assistant:
            break

    _flush()
    return first_user, first_assistant


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_turn(text: str) -> str:
    """Normalize a turn string for stable hashing.

    * Replaces ``\\r\\n`` and ``\\r`` with ``\\n``.
    * Strips leading/trailing whitespace.
    * Collapses runs of spaces and tabs into a single space.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_source_id(transcript: str) -> str:
    """Derive a stable ``"sha256:..."`` source identifier from *transcript*.

    The hash covers the normalised first user exchange and (when present) the
    first assistant exchange, separated by ``"\\n<gemory-sep>\\n"``.

    Raises :class:`ValueError` if no user message can be identified.
    """
    logger.info("Computing source_id from transcript (%d chars)", len(transcript))

    user_msg, assistant_msg = _parse_first_turns(transcript)

    if not user_msg:
        raise ValueError(
            "Cannot compute source_id: no user message found in transcript"
        )

    norm_user = _normalize_turn(user_msg)

    if assistant_msg:
        norm_assistant = _normalize_turn(assistant_msg)
        combined = norm_user + "\n<gemory-sep>\n" + norm_assistant
    else:
        combined = norm_user

    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def store_facts(
    facts: list[dict],
    source_id: str,
    label: str | None,
    graph: GraphStore,
) -> dict[str, int]:
    """Embed, deduplicate, and store *facts* into *graph*.

    Each element of *facts* is ``{"fact": "...", "topics": [...]}`` where
    ``"topics"`` is optional (defaults to ``[]``).

    Steps for each fact
    -------------------
    1. Embed the fact string via :func:`src.llm.embed`.
    2. Look up the embedding in the graph at ``DEDUP_THRESHOLD``.
    3. If a match is found → ``bump_confidence`` (corroborate or skip).
    4. Otherwise → ``add_node`` and connect to close-but-distinct neighbours
       (``EDGE_THRESHOLD`` ≤ similarity < ``DEDUP_THRESHOLD``).
    5. Save the graph once after all facts are processed.

    Parameters
    ----------
    facts
        List of dicts with keys ``"fact"`` (required) and ``"topics"`` (optional,
        list of strings), as returned by :func:`src.llm.extract_facts`.
    source_id
        Stable source identifier (e.g. from :func:`compute_source_id`).
    label
        Optional human-readable label for provenance legibility.
    graph
        An initialised :class:`GraphStore` instance.

    Returns
    -------
    dict
        ``{"facts_extracted": int, "new_nodes": int, "corroborated": int,
            "skipped": int, "topics_linked": int}``
    """
    logger.info("Storing %d facts with source_id=%s", len(facts), source_id)

    now = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {
        "facts_extracted": len(facts),
        "new_nodes": 0,
        "corroborated": 0,
        "skipped": 0,
        "topics_linked": 0,
    }

    for fact_item in facts:
        fact_text = fact_item["fact"]
        topic_texts = fact_item.get("topics", [])
        # Backward compat: old single "topic" string.
        if not topic_texts and "topic" in fact_item:
            t = fact_item["topic"]
            topic_texts = [t] if t else []

        if topic_texts:
            logger.info("Fact topics: %r", topic_texts)

        embedding = llm.embed(fact_text)
        provenance = {
            "source_id": source_id,
            "label": label or "",
            "timestamp": now,
        }

        matches = graph.find_similar(
            embedding, threshold=config.DEDUP_THRESHOLD, top_k=1
        )

        node_id: str | None = None

        if matches:
            node_id = matches[0][0]
            if graph.bump_confidence(node_id, provenance):
                counts["corroborated"] += 1
                logger.info("Corroborated: %s", node_id)
            else:
                counts["skipped"] += 1
                logger.info("Skipped: %s", node_id)
        else:
            node_id = graph.add_node(
                content=fact_text, embedding=embedding, provenance=provenance,
                kind="fact", label="",
            )
            counts["new_nodes"] += 1
            logger.info("New node: %s", node_id)

            # Connect to close-but-distinct neighbours.
            candidates = graph.find_similar(
                embedding, threshold=config.EDGE_THRESHOLD, top_k=10
            )
            for cand_id, cand_sim in candidates:
                if cand_sim < config.DEDUP_THRESHOLD:
                    graph.add_edge(
                        node_id, cand_id, weight=cand_sim, relation="related",
                    )

        # Topic linking (for both new and corroborated facts)
        for topic_text in topic_texts:
            if topic_text and topic_text.strip():
                topic_id = resolve_topic(graph, topic_text.strip())
                if topic_id:
                    parents = graph.get_parents(node_id)
                    if topic_id not in parents:
                        graph.add_parent_edge(topic_id, node_id)
                        counts["topics_linked"] += 1
                        logger.info(
                            "Linked fact %s to topic %s",
                            node_id[:8], topic_id[:8],
                        )

    graph.save()
    logger.info(
        "Saved graph (%d new, %d corroborated, %d skipped, %d topics linked)",
        counts["new_nodes"],
        counts["corroborated"],
        counts["skipped"],
        counts["topics_linked"],
    )

    return counts
