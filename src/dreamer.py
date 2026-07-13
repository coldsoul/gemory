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
    MAX_NODE_CHILDREN,
    MEMORY_PATH,
    MIN_CLUSTER_SIZE,
    MIN_REACH,
    RELATION_LIFT_RATIO,
)
from src.consolidate import cluster_layer, summarize_layer
from src.graph import GraphStore
from src.llm import embed
from src.reach import backfill_reach, compute_reach, update_reach
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


def _find_overlarge_nodes(graph: GraphStore) -> list[str]:
    """Return IDs of nodes whose direct child count exceeds
    ``MAX_NODE_CHILDREN``.

    Only returns nodes that have NOT already been aspect-split (i.e. whose
    children are not already within the threshold due to aspect re-parenting).
    A node whose children are mostly aspects is already structured and does
    not need further splitting.

    Does NOT return nodes at the threshold or below.
    """
    candidates = []
    for node in graph.all_nodes():
        children = graph.get_children(node.id)
        if len(children) > MAX_NODE_CHILDREN:
            candidates.append(node.id)

    if candidates:
        logger.info(
            "Found %d over-large nodes (threshold=%d)",
            len(candidates), MAX_NODE_CHILDREN,
        )
    return candidates


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
    summary_result: dict[str, str] | None = None,
) -> str | None:
    """Create an abstraction node for a cluster, or update an existing one.

    Parameters
    ----------
    graph
        The graph store.
    cluster
        Set of member node IDs.
    run_id
        Identifier for this dreamer run.
    existing_abstractions
        Previously created abstractions (for overlap checking).
    summary_result
        Pre-computed ``{label, summary}`` dict.  When provided the
        ``summarize_layer`` call is skipped (caller already checked).

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
    if summary_result:
        result = summary_result
    else:
        try:
            result = summarize_layer(graph, cluster)
        except Exception:
            logger.exception("Summarization failed for cluster of %d facts", len(cluster))
            return None

    label = result.get("label", "Unlabeled cluster")
    summary_text = result.get("summary", "")

    # Embed the abstraction text (label + summary).
    embedding = embed(f"{label}. {summary_text}")

    # Compute level as one above the highest child level.
    max_child_level = max((n.level for n in member_nodes), default=0)
    abs_level = max_child_level + 1

    # Create the abstraction node.
    # content = short label (for backward compat reading "content")
    # label = authoritative short label
    # summary = authoritative prose description
    abs_id = graph.add_node(
        content=label,
        embedding=embedding,
        provenance={
            "source_id": f"dreamer:{run_id}",
            "label": label,
            "timestamp": _now_iso(),
            "member_ids": member_ids,
        },
        kind="abstraction",
        label=label,
        summary=summary_text,
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

        # Pre-check: summarize the cluster and veto non-theme results.
        try:
            summary_result = summarize_layer(graph, cluster)
        except Exception:
            logger.exception("Summarization failed for cluster, skipping")
            continue

        summary_text = summary_result.get("summary", "").lower()
        label_text = summary_result.get("label", "").lower()
        veto_phrases = [
            "no common theme", "no strong theme", "miscellaneous facts",
            "no clear theme", "unrelated", "without a clear",
            "not a coherent", "no coherent", "miscellaneous",
        ]
        # Check both label and summary for veto phrases
        combined = f"{label_text} {summary_text}"
        if any(phrase in combined for phrase in veto_phrases):
            logger.info(
                "Vetoing cluster: summarizer produced non-theme result (%r)",
                summary_text[:80],
            )
            continue

        abs_id = _create_abstraction(
            graph, cluster, run_id, existing_abstractions,
            summary_result=summary_result,
        )
        if abs_id:
            update_reach(graph, abs_id)
            member_ids = list(cluster)
            new_abstractions.append({"id": abs_id, "member_ids": member_ids})

            # Lift relations from common sources to the new category.
            lifted = _lift_relations(graph, abs_id, member_ids)
            if lifted:
                logger.info(
                    "Lifted %d relations to abstraction %s",
                    lifted, abs_id[:8],
                )

    return new_abstractions


def _run_consolidation_pass(
    graph: GraphStore,
    run_id: str,
    method: str,
) -> list[dict]:
    """Run one full consolidation pass with the given method.

    The first layer is the **topic layer** -- we do NOT re-cluster raw
    facts.  Topics and their fact children are the given level-1 structure.

    Returns a list of ``{id, member_ids}`` dicts for each abstraction
    created.  Does **not** save to disk.
    """
    # Backfill reach on all abstraction nodes before starting.
    backfilled = backfill_reach(graph)
    if backfilled:
        logger.info("Backfilled reach on %d abstraction nodes", backfilled)

    # Start from the topic layer (level-1 abstractions).
    topic_nodes = graph.get_topic_nodes()
    if topic_nodes:
        # Enrich topics first (write summaries from child facts).
        for topic in topic_nodes:
            children = graph.get_children(topic.id)
            if children:
                try:
                    result = summarize_layer(graph, set(children))
                    summary_text = result.get("summary", "")
                    new_label = result.get("label", topic.label)
                    # Set summary on the topic — do NOT overwrite content (label).
                    if summary_text:
                        graph.set_node_attr(topic.id, "summary", summary_text)
                    # Optionally refine label if summarizer has a better one.
                    if new_label and new_label != topic.label and len(new_label) < 50:
                        graph.set_node_attr(topic.id, "label", new_label)
                    r = compute_reach(graph, [topic.id])
                    graph.set_node_attr(topic.id, "reach", r)
                    logger.info(
                        "Enriched topic %s: reach=%d, %d children",
                        topic.label, r, len(children),
                    )
                except Exception:
                    logger.exception("Failed to enrich topic %s", topic.label)

        layer = [t.id for t in topic_nodes]
    else:
        # No topics -- fall back to fact level.
        logger.warning("No topic nodes found -- starting from facts")
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


# ---------------------------------------------------------------------------
# Relation lifting
# ---------------------------------------------------------------------------

def _lift_relations(
    graph: GraphStore,
    category_id: str,
    member_ids: list[str],
) -> int:
    """Lift relates_to edges from common sources to a newly created category.

    If a source node X has relates_to edges to all (or >=
    RELATION_LIFT_RATIO fraction) of the category's members, create a
    derived ``relates_to`` edge X -> category.

    Leaf edges are KEPT -- lifting adds, never replaces.
    Returns the number of lifted edges created.
    """
    # Gather all relates_to edges incoming to any member.
    incoming: dict[str, set[str]] = {}
    for source, target, data in graph.get_edges_by_relation("relates_to"):
        if target in member_ids:
            incoming.setdefault(source, set()).add(target)

    threshold = max(1, int(len(member_ids) * RELATION_LIFT_RATIO))

    lifted = 0
    for source_id, target_members in incoming.items():
        if len(target_members) < threshold:
            continue

        # Check if a lifted edge already exists.
        already = any(
            s == source_id and t == category_id
            for s, t, d in graph.get_edges_by_relation("relates_to")
        )
        if already:
            continue

        # Add derived edge (bypass the idempotency guard because this is
        # a new relation type or the edge does not yet exist).
        graph._graph.add_edge(
            source_id, category_id,
            relation="relates_to",
            provenance="derived",
            origin_fact="",
        )
        lifted += 1
        logger.info(
            "Lifted relation: %s -> %s (from %d/%d members)",
            source_id[:8], category_id[:8],
            len(target_members), len(member_ids),
        )

    return lifted


def _split_node(
    graph: GraphStore,
    parent_id: str,
    run_id: str,
) -> list[str]:
    """Split an over-large node into aspect sub-nodes.

    Reuses the existing consolidation pipeline (cluster -> gate ->
    summarize -> re-parent), pointed downward at *parent_id*'s direct
    children rather than upward at the topic layer.

    Returns the IDs of created aspect nodes.
    """
    children = graph.get_children(parent_id)
    if len(children) <= MAX_NODE_CHILDREN:
        return []

    logger.info(
        "Splitting node %s (%s): %d children exceed threshold %d",
        parent_id[:8],
        (graph.get_node(parent_id).label or graph.get_node(parent_id).content[:30]),
        len(children), MAX_NODE_CHILDREN,
    )

    from src.consolidate import cluster_layer, summarize_layer
    from src.reach import compute_reach, update_reach

    parent_node = graph.get_node(parent_id)
    context = (
        f"[{parent_node.label or parent_node.content[:40]}] "
        f"{parent_node.summary or ''}"
    )

    clusters = cluster_layer(
        graph, children, method="hybrid", context=context,
    )

    # If hybrid was too conservative, retry with LLM-only + context.
    clustered_ids = set()
    for c in clusters:
        clustered_ids.update(c)
    leftover_count = len(children) - len(clustered_ids)

    if leftover_count > MAX_NODE_CHILDREN and context.strip():
        logger.info(
            "Hybrid clusterer grouped %d/%d children; %d leftovers exceed "
            "threshold. Retrying with LLM-only + context...",
            len(clustered_ids), len(children), leftover_count,
        )
        clusters = cluster_layer(
            graph, children, method="llm", context=context,
        )

    if not clusters:
        logger.info(
            "No meaningful clusters found within node %s -- nothing to split",
            parent_id[:8],
        )
        return []

    created_aspects: list[str] = []
    for cluster in clusters:
        # Single-child guard: skip clusters with only one member.
        if len(cluster) <= 1:
            continue
        # Idempotency: check if an existing aspect (already a child of the
        # parent) substantially overlaps this cluster.
        existing_children = graph.get_children(parent_id)
        skip_cluster = False
        for child_id in existing_children:
            child = graph.get_node(child_id)
            if child.kind != "abstraction":
                continue
            existing_members = set(graph.get_children(child_id))
            overlap = len(cluster & existing_members) / max(
                len(cluster | existing_members), 1,
            )
            if overlap >= ABSTRACTION_OVERLAP:
                # Attach any new members to the existing aspect.
                new_members = cluster - existing_members
                for mid in new_members:
                    graph.add_parent_edge(child_id, mid)
                skip_cluster = True
                logger.info(
                    "Aspect already exists for cluster (overlap=%.2f), "
                    "attached %d new members",
                    overlap, len(new_members),
                )
                break
        if skip_cluster:
            continue

        # Gate: only create aspect if union-reach clears MIN_REACH.
        reach = compute_reach(graph, list(cluster))
        if reach < MIN_REACH:
            logger.info(
                "Skipping aspect cluster: reach=%d < MIN_REACH=%d",
                reach, MIN_REACH,
            )
            continue

        # Summarize -- children are facts here, so summarize_layer receives
        # their raw texts (compound-upward contract).
        try:
            summary_result = summarize_layer(graph, cluster)
        except Exception:
            logger.exception(
                "Summarization failed for aspect cluster, skipping",
            )
            continue

        # Non-theme veto (same guards as upward consolidation).
        summary_text = summary_result.get("summary", "").lower()
        label_text = summary_result.get("label", "").lower()
        combined = f"{label_text} {summary_text}"
        veto_phrases = [
            "no common theme", "no strong theme", "miscellaneous facts",
            "no clear theme", "unrelated", "without a clear",
            "not a coherent", "no coherent", "miscellaneous",
        ]
        if any(p in combined for p in veto_phrases):
            logger.info("Vetoing aspect: non-theme result")
            continue

        label = summary_result.get("label", "Aspect")
        summary_text = summary_result.get("summary", "")

        # Embed the aspect.
        try:
            embedding = embed(f"{label}. {summary_text}")
        except Exception:
            embedding = [0.0]  # fallback

        # Determine level: 1 + max(child level).
        child_levels = [graph.get_node(cid).level for cid in cluster]
        aspect_level = max(child_levels) + 1 if child_levels else 1

        # Create the aspect node (ordinary abstraction).
        aspect_id = graph.add_node(
            content=label,
            embedding=embedding,
            provenance={
                "source_id": f"dreamer:{run_id}",
                "label": label,
                "timestamp": _now_iso(),
                "member_ids": list(cluster),
            },
            kind="abstraction",
            label=label,
            summary=summary_text,
        )
        graph.set_node_attr(aspect_id, "level", aspect_level)

        # Re-parent: original node -> aspect -> children.
        graph.add_parent_edge(parent_id, aspect_id)
        for child_id in cluster:
            graph.add_parent_edge(aspect_id, child_id)
            # Remove the direct edge from parent to this child (if it exists),
            # so the parent's direct child count decreases.
            if graph._graph.has_edge(parent_id, child_id):
                graph._graph.remove_edge(parent_id, child_id)

        update_reach(graph, aspect_id)
        created_aspects.append(aspect_id)
        logger.info(
            "Created aspect %s: %r (level=%d, %d children)",
            aspect_id[:8], label, aspect_level, len(cluster),
        )

    if created_aspects:
        # Recompute reach for the original parent (same leaves, one level deeper).
        from src.reach import update_reach as _ur
        _ur(graph, parent_id)

        # Recompute level for the original parent if it is an abstraction.
        parent_node = graph.get_node(parent_id)
        if parent_node.kind == "abstraction":
            child_levels = [
                graph.get_node(cid).level
                for cid in graph.get_children(parent_id)
            ]
            new_level = max(child_levels) + 1 if child_levels else parent_node.level
            if new_level != parent_node.level:
                graph.set_node_attr(parent_id, "level", new_level)
                logger.info(
                    "Updated parent %s level: %d -> %d",
                    parent_id[:8], parent_node.level, new_level,
                )

    return created_aspects


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

    # Downward pass: split any over-large nodes into aspects.
    oversize = _find_overlarge_nodes(graph)
    if oversize:
        logger.info(
            "Downward pass: %d nodes exceed MAX_NODE_CHILDREN=%d",
            len(oversize), MAX_NODE_CHILDREN,
        )
        for node_id in oversize:
            created = _split_node(graph, node_id, run_id)
            if created:
                logger.info(
                    "Node %s split into %d aspects",
                    node_id[:8], len(created),
                )
        # Recompute reach for all abstraction nodes after splits.
        backfilled = backfill_reach(graph)
        if backfilled:
            logger.info("Backfilled reach on %d nodes after downward pass", backfilled)
    else:
        logger.info(
            "No nodes exceed MAX_NODE_CHILDREN=%d -- skipping downward pass",
            MAX_NODE_CHILDREN,
        )

    all_abstractions = _run_consolidation_pass(graph, run_id, method)

    # Final reach/level recomputation for ALL abstraction nodes.
    backfill_reach(graph)
    updated = 0
    for node in graph.all_nodes():
        if node.kind == "abstraction":
            children = graph.get_children(node.id)
            if children:
                child_levels = [graph.get_node(cid).level for cid in children]
                correct_level = max(child_levels) + 1
                if correct_level != node.level:
                    graph.set_node_attr(node.id, "level", correct_level)
                    updated += 1
    if updated:
        logger.info("Recomputed level for %d abstraction nodes", updated)

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
