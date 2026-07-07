#!/usr/bin/env python3
"""Dreamer: offline graph consolidation for the Gemory memory system.

Finds clusters of related facts, creates abstraction nodes summarizing
them, and builds emergent hierarchy through recursive consolidation.

Run with --dry-run (default) to preview changes without applying.
Run with --apply to write changes (creates a backup first).

Clustering methods
------------------
* ``algorithm`` -- cosine-based Louvain community detection.
* ``llm`` -- LLM-based thematic grouping of labels+summaries.
* ``hybrid`` (apply default) -- algorithm first, then LLM on leftovers.
* ``diff`` (dry-run default) -- run both and compare results.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone

# Ensure project root on sys.path so ``from src.*`` imports resolve
# when this script is run directly (e.g. ``uv run src/dreamer.py``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("dreamer")

from src.config import (
    ABSTRACTION_OVERLAP,
    CONFIDENCE_BASE,
    MAX_CLUSTER_SIZE,
    MAX_LEVELS,
    MEMORY_PATH,
    MIN_CLUSTER_SIZE,
    MIN_REACH,
)
from src.consolidate import cluster_layer, summarize_layer
from src.graph import GraphStore
from src.llm import embed
from src.reach import compute_reach, update_reach
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
        node_ids = [n.id for n in all_nodes]
        logger.info("Working set (--full): %d nodes", len(node_ids))
        return node_ids

    if mode == "recent":
        threshold = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()

        recent_ids = set()
        for node in all_nodes:
            for prov in node.provenance:
                ts = prov.get("timestamp", "")
                if ts and ts >= threshold:
                    recent_ids.add(node.id)
                    break

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


# ---------------------------------------------------------------------------
# Core consolidation
# ---------------------------------------------------------------------------

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
        result = summarize_layer(graph, cluster)
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


def _consolidate_layer(
    graph: GraphStore,
    node_ids: list[str],
    run_id: str,
    existing_abstractions: list[dict],
    method: str = "hybrid",
) -> list[dict]:
    """Consolidate one layer: cluster, gate by reach, create abstractions.

    Level-agnostic: identical operation at every height.  No level
    number is referenced.
    """
    logger.info(
        "--- Consolidating layer (%d nodes, method=%s) ---",
        len(node_ids), method,
    )

    clusters = cluster_layer(graph, node_ids, method=method)
    if not clusters:
        logger.info("No clusters formed -- layer complete")
        return []

    new_abstractions: list[dict] = []
    for cluster in clusters:
        # Gate: only create abstraction if reach >= MIN_REACH.
        reach = compute_reach(graph, list(cluster))
        if reach < MIN_REACH:
            logger.info(
                "Skipping cluster of size %d: reach=%d < MIN_REACH=%d",
                len(cluster), reach, MIN_REACH,
            )
            continue

        abs_id = _create_abstraction(
            graph, cluster, run_id, existing_abstractions,
        )
        if abs_id:
            update_reach(graph, abs_id)
            member_ids = list(cluster)
            new_abstractions.append({"id": abs_id, "member_ids": member_ids})

    return new_abstractions


def _run_consolidation_pass(
    graph: GraphStore,
    run_id: str,
    method: str,
) -> list[dict]:
    """Run one full consolidation pass with the given method.

    Returns a list of ``{id, member_ids}`` dicts for each abstraction
    created.  Does **not** save to disk.
    """
    all_node_ids = [n.id for n in graph.all_nodes()]
    layer = all_node_ids
    all_abstractions: list[dict] = []

    while True:
        new_abs = _consolidate_layer(
            graph, layer, run_id, all_abstractions, method=method,
        )
        if not new_abs:
            break
        all_abstractions.extend(new_abs)
        layer = [a["id"] for a in new_abs]

        max_level = max(
            (graph.get_node(a["id"]).level for a in all_abstractions),
            default=0,
        )
        if max_level >= MAX_LEVELS:
            break

    return all_abstractions


# ---------------------------------------------------------------------------
# Diff dry-run
# ---------------------------------------------------------------------------

def _diff_consolidation(graph: GraphStore, run_id: str) -> dict:
    """Run both algorithm and LLM clustering, then diff the results.

    Returns a dict with keys ``"agreement"``, ``"algorithm_only"``,
    ``"llm_only"``.
    """
    tmpdir = tempfile.mkdtemp(prefix="gemory-diff-")
    orig_mem = os.path.join(tmpdir, "memory.json")
    orig_emb = os.path.join(tmpdir, "embeddings.json")

    # Save current state and copy to temp.
    graph.save()
    mem_path = graph._path
    emb_path = graph._embeddings_path
    if os.path.exists(mem_path):
        shutil.copy2(mem_path, orig_mem)
    if os.path.exists(emb_path):
        shutil.copy2(emb_path, orig_emb)

    try:
        # Run algorithm pass.
        algo_results = _run_consolidation_pass(graph, run_id, "algorithm")

        # Restore original state.
        if os.path.exists(orig_mem):
            shutil.copy2(orig_mem, mem_path)
        if os.path.exists(orig_emb):
            shutil.copy2(orig_emb, emb_path)
        graph.load()

        # Run LLM pass.
        llm_results = _run_consolidation_pass(graph, run_id, "llm")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Bucket by member sets.
    algo_sets = {frozenset(r["member_ids"]): r for r in algo_results}
    llm_sets = {frozenset(r["member_ids"]): r for r in llm_results}

    agreement: dict = {}
    algorithm_only: dict = {}
    llm_only: dict = {}

    all_keys = set(algo_sets.keys()) | set(llm_sets.keys())
    for members in all_keys:
        info = algo_sets.get(members) or llm_sets.get(members) or {}
        label = info.get("label", f"cluster of {len(members)} nodes")

        if members in algo_sets and members in llm_sets:
            agreement[label] = list(members)
        elif members in algo_sets:
            algorithm_only[label] = list(members)
        else:
            llm_only[label] = list(members)

    return {
        "agreement": agreement,
        "algorithm_only": algorithm_only,
        "llm_only": llm_only,
    }


def _print_diff_report(diff: dict) -> None:
    """Print a human-readable diff report."""
    print("\n" + "=" * 70)
    print("CLUSTERING METHOD DIFF")
    print("=" * 70)

    agreement = diff.get("agreement", {})
    algo_only = diff.get("algorithm_only", {})
    llm_only = diff.get("llm_only", {})

    print(
        "\nAgreement (both methods): %d groups" % len(agreement),
    )
    for label, members in agreement.items():
        print("  [common] %s (%d nodes)" % (label, len(members)))

    print(
        "\nAlgorithm-only (cosine found, LLM did not): %d groups" % len(algo_only),
    )
    for label, members in algo_only.items():
        print("  [algo] %s" % label)

    print(
        "\nLLM-only (LLM found, cosine missed): %d groups" % len(llm_only),
    )
    for label, members in llm_only.items():
        print("  [llm] %s" % label)

    if llm_only:
        print(
            "\nLLM-only groups are the value the LLM adds -- "
            "categorical groupings cosine cannot detect."
        )
    print("=" * 70)


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
    parser.add_argument(
        "--cluster-method", choices=["algorithm", "llm", "hybrid", "diff"],
        default=None,
        help=(
            "Clustering method. 'diff' runs both and compares (dry-run default). "
            "'hybrid' runs algorithm then LLM on leftovers (apply default). "
            "Leave unset for auto-selection."
        ),
    )
    args = parser.parse_args()

    run_id = args.run_id or _now_iso()
    apply_mode = args.apply

    # Auto-select method if not explicitly set.
    if args.cluster_method:
        method = args.cluster_method
    elif apply_mode:
        method = "hybrid"
    else:
        method = "diff"

    mode = "recent" if args.recent else "full"
    recent_days = args.recent or 7

    logger.info(
        "Dreamer run: %s method=%s apply=%s", run_id, method, apply_mode,
    )

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

    # Diff mode: run both methods and compare, no save.
    if method == "diff":
        diff = _diff_consolidation(graph, run_id)
        _print_diff_report(diff)
        print("\nDiff mode -- no changes written.")
        print("Run with --cluster-method hybrid --apply to commit.")
        return

    # Standard single-method consolidation.
    _before = snapshot(graph) if not apply_mode else None

    if not apply_mode:
        print(f"\n--- DRY RUN (run_id={run_id}) ---")

    initial_layer = _select_working_set(graph, mode, recent_days)
    if not initial_layer:
        logger.info("Working set is empty -- nothing to consolidate")
        return

    all_abstractions = _run_consolidation_pass(graph, run_id, method)

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
