#!/usr/bin/env python3
"""Evaluate recall: run the query set against flat and traverse methods."""

import json
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import GraphStore
from src.recall import recall, traverse_recall
from src.config import MEMORY_PATH


def load_queries():
    with open("eval/queries.json") as f:
        return json.load(f)


def compute_hit_k(expected_ids: list, returned_text: str, k: int = 10) -> int:
    """Count how many expected fact IDs appear in the returned text."""
    found = 0
    for eid in expected_ids:
        if eid in returned_text:
            found += 1
    return found


def compute_coverage(expected_ids: list, returned_text: str) -> float:
    """Fraction of expected facts present in the returned text."""
    if not expected_ids:
        return 1.0
    found = sum(1 for eid in expected_ids if eid in returned_text)
    return found / len(expected_ids)


def main():
    data = load_queries()
    queries = data["queries"]
    graph = GraphStore(MEMORY_PATH)
    try:
        graph.load()
    except Exception:
        print("No memory graph found. Run the server first.")
        sys.exit(1)

    header = (
        f"{'Query':<6} {'Type':<14} {'Flat Hit@10':>12} "
        f"{'Flat+Sum Hit@10':>17} {'Trav Hit@n':>12} "
        f"{'Trav Cov':>9} {'Trav Layers':>11} {'Trav Kept/Pruned':>17}"
    )
    print(header)
    print("-" * 110)

    type_results: dict = {}

    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]
        expected = q.get("expected_facts", [])

        # --- Flat (facts only) ---
        t0 = time.time()
        flat_text = recall(query_text, graph, top_k=10)
        flat_time = time.time() - t0
        flat_hit = compute_hit_k(expected, flat_text)

        # --- Flat + summaries ---
        t0 = time.time()
        abs_summaries = []
        for node in graph.all_nodes():
            if node.kind == "abstraction" and (node.summary or node.content):
                abs_summaries.append(node.summary or node.content)
        combined_text = flat_text
        if abs_summaries:
            summary_context = "\n\n".join(
                f"[summary] {s}" for s in abs_summaries[:5]
            )
            combined_text = flat_text + "\n\n--- Summaries ---\n" + summary_context
        flat_sum_time = time.time() - t0
        flat_sum_hit = compute_hit_k(expected, combined_text)

        # --- Traversal ---
        t0 = time.time()
        trav_text, trav_metrics = traverse_recall(query_text, graph)
        trav_time = time.time() - t0
        trav_hit = compute_hit_k(
            expected, trav_text,
            k=len(expected) if expected else 10,
        )
        trav_cov = compute_coverage(expected, trav_text)

        # --- Print row ---
        kept = trav_metrics.get("branches_kept", 0)
        pruned = trav_metrics.get("branches_pruned", 0)
        layers = trav_metrics.get("layers_visited", 0)
        print(
            f"{qid:<6} {qtype:<14} "
            f"{flat_hit:>9}/{len(expected):<2} "
            f"{flat_sum_hit:>14}/{len(expected):<2} "
            f"{trav_hit:>9}/{len(expected):<2} "
            f"{trav_cov:>8.2f} "
            f"{layers:>10} "
            f"{kept:>2}/{pruned:<2}"
        )

        # Aggregate per type
        if qtype not in type_results:
            type_results[qtype] = {
                "flat_hit": 0, "flat_sum_hit": 0, "trav_hit": 0,
                "trav_cov": 0.0, "count": 0, "total_expected": 0,
            }
        tr = type_results[qtype]
        tr["flat_hit"] += flat_hit
        tr["flat_sum_hit"] += flat_sum_hit
        tr["trav_hit"] += trav_hit
        tr["trav_cov"] += trav_cov
        tr["count"] += 1
        tr["total_expected"] += len(expected)

    # --- Per-type aggregates ---
    print("\n--- Per-Type Averages ---")
    agg_header = (
        f"{'Type':<14} {'Count':>6} {'Flat Hit%':>10} "
        f"{'Flat+Sum Hit%':>14} {'Trav Hit%':>10} {'Trav Cov%':>10}"
    )
    print(agg_header)
    print("-" * 65)
    for qtype, tr in sorted(type_results.items()):
        n = tr["count"]
        total_exp = tr["total_expected"]
        flat_pct = (tr["flat_hit"] / total_exp * 100) if total_exp else 0
        fs_pct = (tr["flat_sum_hit"] / total_exp * 100) if total_exp else 0
        trav_pct = (tr["trav_hit"] / total_exp * 100) if total_exp else 0
        trav_cov_pct = (tr["trav_cov"] / n * 100) if n else 0
        print(
            f"{qtype:<14} {n:>6} "
            f"{flat_pct:>9.0f}% "
            f"{fs_pct:>13.0f}% "
            f"{trav_pct:>9.0f}% "
            f"{trav_cov_pct:>9.0f}%"
        )


if __name__ == "__main__":
    main()
