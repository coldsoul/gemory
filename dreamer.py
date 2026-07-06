#!/usr/bin/env python3
"""Dreamer: offline graph consolidation for the Gemory memory system.

Finds clusters of related facts, creates abstraction nodes summarizing
them, and builds emergent hierarchy through recursive consolidation.

Run with --dry-run (default) to preview changes without applying.
Run with --apply to write changes (creates a backup first).
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

# Ensure project root on sys.path so ``from gemory.*`` imports resolve
# when this script is run directly (e.g. ``uv run dreamer.py``).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("dreamer")

from gemory.config import (
    ABSTRACTION_OVERLAP,
    CONFIDENCE_BASE,
    MAX_CLUSTER_SIZE,
    MAX_LEVELS,
    MEMORY_PATH,
    MIN_CLUSTER_SIZE,
)
from gemory.cluster import cluster_nodes
from gemory.graph import GraphStore
from gemory.llm import embed, summarize_cluster
from tests.graph_diff import snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _backup(memory_path: str) -> str:
    """Create timestamped backups of ``memory.json`` and the embeddings sidecar.

    Returns the backup timestamp string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dir_name = os.path.dirname(memory_path) or "."
    embeddings_path = os.path.join(dir_name, "embeddings.json")

    backup_memory = f"{memory_path}.{timestamp}.bak"
    backup_embeddings = f"{embeddings_path}.{timestamp}.bak"

    if os.path.exists(memory_path):
        shutil.copy2(memory_path, backup_memory)
        logger.info("Backed up %s -> %s", memory_path, backup_memory)
    if os.path.exists(embeddings_path):
        shutil.copy2(embeddings_path, backup_embeddings)
        logger.info("Backed up %s -> %s", embeddings_path, backup_embeddings)

    return timestamp


def _select_working_set(
    graph: GraphStore,
    mode: str,
    recent_days: int = 7,
) -> list[str]:
    """Select node IDs eligible for consolidation this run."""
    all_nodes = graph.all_nodes()

    if mode == "full":
        # All nodes
        node_ids = [n.id for n in all_nodes]
        logger.info("Working set (--full): %d nodes", len(node_ids))
        return node_ids

    if mode == "recent":
        cutoff = _now_iso()
        threshold = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()

        recent_ids = set()
        for node in all_nodes:
            for prov in node.provenance:
                ts = prov.get("timestamp", "")
                if ts and ts >= threshold:
                    recent_ids.add(node.id)
                    break

        # Include immediate neighbours
        extended = set(recent_ids)
        for nid in recent_ids:
            for neighbor in graph.get_neighbors(nid):
                extended.add(neighbor)
            for parent in graph.get_parents(nid):
                extended.add(parent)
            for child in graph.get_children(nid):
                extended.add(child)

        logger.info(
            "Working set (--recent %d): %d recent + neighbours = %d total",
            recent_days, len(recent_ids), len(extended),
        )
        return list(extended)

    return []


def _create_abstraction(
    graph: GraphStore,
    cluster: set[str],
    run_id: str,
    existing_abstractions: list[dict],
) -> str | None:
    """Create an abstraction node for a cluster, or update an existing one.

    Checks existing abstractions via Jaccard overlap before creating.
    Returns the abstraction node ID, or ``None`` if no creation was needed.
    """
    member_ids = list(cluster)
    member_nodes = [graph.get_node(mid) for mid in member_ids]
    member_facts = [n.content for n in member_nodes]

    # Check for existing overlapping abstraction.
    for existing in existing_abstractions:
        existing_members = set(existing["member_ids"])
        overlap = len(cluster & existing_members) / max(
            len(cluster | existing_members), 1,
        )
        if overlap >= ABSTRACTION_OVERLAP:
            abs_id = existing["id"]
            new_members = cluster - existing_members
            if new_members:
                for mid in new_members:
                    graph.add_parent_edge(abs_id, mid)
                logger.info(
                    "Updated abstraction %s: attached %d new members",
                    abs_id[:8], len(new_members),
                )
            else:
                logger.info(
                    "Abstraction %s already covers this cluster, skipping",
                    abs_id[:8],
                )
            return abs_id

    # Create new abstraction.
    try:
        result = summarize_cluster(member_facts)
    except Exception:
        logger.exception("Summarization failed for cluster of %d facts", len(cluster))
        return None

    label = result.get("label", "Unlabeled cluster")
    summary = result.get("summary", "No summary available")

    # Embed the abstraction text.
    embedding = embed(f"{label}. {summary}")

    # Compute level as one above the highest child level.
    max_child_level = max((n.level for n in member_nodes), default=0)
    abs_level = max_child_level + 1

    # Create the abstraction node.
    abs_id = graph.add_node(
        content=summary,
        embedding=embedding,
        provenance={
            "source_id": f"dreamer:{run_id}",
            "label": label,
            "timestamp": _now_iso(),
            "member_ids": member_ids,
        },
        kind="abstraction",
        label=label,
    )

    # Set level.
    graph.set_node_attr(abs_id, "level", abs_level)

    # Link to members.
    for mid in member_ids:
        graph.add_parent_edge(abs_id, mid)

    logger.info(
        "Created abstraction %s (level %d, label=%r, %d members)",
        abs_id[:8], abs_level, label, len(member_ids),
    )

    return abs_id


def _consolidate_level(
    graph: GraphStore,
    node_ids: list[str],
    run_id: str,
    existing_abstractions: list[dict],
    _level: int,
) -> list[dict]:
    """Run one level of consolidation: cluster -> abstract.

    Returns a list of new abstraction dicts ``{id, member_ids}``.
    """
    logger.info("--- Consolidation level %d (%d nodes) ---", _level, len(node_ids))

    clusters = cluster_nodes(graph, node_ids)
    if not clusters:
        logger.info("No qualifying clusters at level %d, stopping", _level)
        return []

    new_abstractions: list[dict] = []
    for cluster in clusters:
        abs_id = _create_abstraction(
            graph, cluster, run_id, existing_abstractions,
        )
        if abs_id:
            new_abstractions.append({
                "id": abs_id,
                "member_ids": list(cluster),
            })

    return new_abstractions


def _print_dry_run_report(
    all_abstractions: list[dict],
    graph: GraphStore,
) -> None:
    """Print a human-readable report of what *would* be created."""
    print(f"\nWould create {len(all_abstractions)} abstraction nodes:")
    for abs_info in all_abstractions:
        try:
            abs_node = graph.get_node(abs_info["id"])
            print(f"  [{abs_node.level}] {abs_node.label}")
            print(f"       {abs_node.content[:100]}")
            members = abs_info.get("member_ids", [])
            for mid in members[:5]:
                try:
                    m = graph.get_node(mid)
                    print(f"         - {m.content[:60]}")
                except KeyError:
                    pass
            if len(members) > 5:
                print(f"         ... and {len(members) - 5} more")
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemory Dreamer -- offline graph consolidation",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply changes (default: dry-run, preview only)",
    )
    parser.add_argument(
        "--full", action="store_true", default=True,
        help="Consolidate all nodes (default)",
    )
    parser.add_argument(
        "--recent", type=int, metavar="N",
        help="Consolidate only nodes active in the last N days",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Identifier for this run (default: timestamp)",
    )
    parser.add_argument(
        "--memory-path", default=MEMORY_PATH,
        help=f"Path to memory.json (default: {MEMORY_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic clustering (default: 42)",
    )
    args = parser.parse_args()

    run_id = args.run_id or _now_iso()
    apply_mode = args.apply
    mode = "recent" if args.recent else "full"
    recent_days = args.recent or 7

    logger.info("Dreamer run: %s mode=%s apply=%s", run_id, mode, apply_mode)

    # Load graph.
    graph = GraphStore(args.memory_path)
    try:
        graph.load()
    except FileNotFoundError:
        logger.info("No existing graph, starting empty")
    node_count = len(graph.all_nodes())
    logger.info("Loaded graph: %d nodes", node_count)

    if node_count == 0:
        print("Graph is empty -- nothing to consolidate.")
        return

    # Snapshot before (only used to compute diff in non-apply mode, but
    # we capture it here to keep the interface consistent).
    _before = snapshot(graph) if not apply_mode else None

    if not apply_mode:
        print(f"\n--- DRY RUN (run_id={run_id}) ---")

    # Select working set.
    working_set = _select_working_set(graph, mode, recent_days)

    # Consolidate recursively.
    all_abstractions: list[dict] = []
    current_level = 1
    next_working_set = working_set

    while current_level <= MAX_LEVELS:
        new_abs = _consolidate_level(
            graph, next_working_set, run_id, all_abstractions, current_level,
        )
        if not new_abs:
            break

        all_abstractions.extend(new_abs)
        # Next level: cluster the newly created abstractions.
        next_working_set = [a["id"] for a in new_abs]
        current_level += 1

    if current_level > MAX_LEVELS:
        logger.warning("Hit MAX_LEVELS=%d -- hierarchy capped", MAX_LEVELS)

    # Report / apply.
    if apply_mode:
        backup_ts = _backup(args.memory_path)
        logger.info("Backup timestamp: %s", backup_ts)

        graph.save()
        logger.info("Saved graph: %d nodes", len(graph.all_nodes()))
        print(f"\nApplied {len(all_abstractions)} abstractions (backup: {backup_ts})")
    else:
        _print_dry_run_report(all_abstractions, graph)
        print(f"\nWARNING: Dry-run -- no changes written.")
        print(f"Run with --apply to commit these changes.")


if __name__ == "__main__":
    main()
