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


def compute_hit_k(expected_ids: list, returned_text: str, graph: GraphStore) -> int:
    """Count how many expected facts (by content) appear in the returned text."""
    found = 0
    for eid in expected_ids:
        try:
            content = graph.get_node(eid).content
            if content in returned_text:
                found += 1
        except KeyError:
            print(f"WARNING: expected fact ID {eid!r} not found in graph", file=sys.stderr)
    return found


def compute_coverage(expected_ids: list, returned_text: str, graph: GraphStore) -> float:
    """Fraction of expected facts (by content) present in the returned text."""
    if not expected_ids:
        return 1.0
    found = 0
    for eid in expected_ids:
        try:
            if graph.get_node(eid).content in returned_text:
                found += 1
        except KeyError:
            print(f"WARNING: expected fact ID {eid!r} not found in graph", file=sys.stderr)
    return found / len(expected_ids)


def compute_prune_errors(
    expected_facts: list[str],
    expected_roots: list[str],
    prune_decisions: list[dict],
    graph: GraphStore,
) -> dict[int, dict]:
    """Compute prune-error rate per level using ancestor paths.

    For each expected fact, walks parent_of upward to the root to build
    the set of nodes that must survive at each layer.  A prune error at
    layer *N* = any node on that expected path was discarded at layer *N*,
    not just the root.
    """
    # Build the ancestor path for each expected fact.
    all_expected_nodes: set[str] = set()
    for eid in expected_facts:
        current = eid
        while current:
            try:
                node = graph.get_node(current)
            except KeyError:
                break
            all_expected_nodes.add(current)
            parents = graph.get_parents(current)
            current = parents[0] if parents else None

    per_level = {}
    for decision in prune_decisions:
        layer = decision["layer"]
        kept = set(decision.get("kept", []))
        discarded = set(decision.get("discarded", []))
        # Error: any expected-path node was discarded at this layer.
        error = bool(discarded & all_expected_nodes)
        per_level[layer] = {
            "errors": per_level.get(layer, {}).get("errors", 0) + (1 if error else 0),
            "total": per_level.get(layer, {}).get("total", 0) + 1,
        }
    return per_level


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate recall across query set")
    parser.add_argument("--content", action="store_true",
                        help="Print the actual returned text for each query arm")
    parser.add_argument("--query", type=str, metavar="ID",
                        help="Run only this query (e.g. q07)")
    args = parser.parse_args()

    data = load_queries()
    queries = data["queries"]
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
        if not queries:
            print(f"Query {args.query!r} not found in eval/queries.json")
            sys.exit(1)
    graph = GraphStore(MEMORY_PATH)
    try:
        graph.load()
    except Exception:
        print("No memory graph found. Run the server first.")
        sys.exit(1)

    # Build prefix→full-ID map: eval/queries.json uses 8-char ID prefixes,
    # but the graph stores full UUIDs. Expand on startup so all lookups work.
    id_map: dict[str, str] = {}
    for node in graph.all_nodes():
        nid = node.id
        for length in (8, 12, 16, 20, 24, 28, 32):
            prefix = nid[:length]
            if prefix not in id_map:
                id_map[prefix] = nid
            else:
                # Collision at this length — don't overwrite. 8-char collisions
                # are extremely rare with UUIDs; the first match wins.
                pass

    # Expand all expected IDs and roots in the query set.
    for q in queries:
        if "expected_facts" in q:
            q["expected_facts"] = [
                id_map.get(eid, eid) for eid in q["expected_facts"]
            ]
        if "expected_roots" in q:
            q["expected_roots"] = [
                id_map.get(eid, eid) for eid in q["expected_roots"]
            ]

    header = (
        f"{'Query':<6} {'Type':<14} {'Flat Hit@10':>12} "
        f"{'Flat+Sum Hit@10':>17} {'Trav Hit@n':>12} "
        f"{'Trav+Exp Hit':>14} {'Total Prune':>11} {'Prune= ✓':>8} {'Trav K/P':>9}"
    )
    print(header)
    print("-" * 125)

    type_results: dict = {}
    per_level_errors: dict[int, dict] = {}  # layer → {errors, total}

    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]
        expected = q.get("expected_facts", [])

        # --- Flat (facts only) ---
        t0 = time.time()
        flat_text = recall(query_text, graph, top_k=10)
        flat_time = time.time() - t0
        flat_hit = compute_hit_k(expected, flat_text, graph)

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
        flat_sum_hit = compute_hit_k(expected, combined_text, graph)

        # --- Traversal (expansion OFF) — single prune pass for both arms ---
        t0 = time.time()
        trav_text, trav_metrics = traverse_recall(query_text, graph, relation_expansion=False)
        trav_time = time.time() - t0
        trav_hit = compute_hit_k(expected, trav_text, graph)

        # --- Expansion measured on SAME pruning decisions ---
        # (not a second LLM call — expansion runs post-pruning, so replay it
        #  over the prune_decisions already captured above.)
        from src.recall import _expand_relations
        all_nodes = {n.id: n for n in graph.all_nodes()}
        kept_ids: set[str] = set()
        for d in trav_metrics.get("prune_decisions", []):
            kept_ids.update(d.get("kept", []))
        related = _expand_relations(graph, kept_ids, all_nodes) if kept_ids else []
        if related:
            # Build a synthetic "with expansion" text: original + related section
            trav_exp_text = trav_text + "\n\n## Related context (via relations, 1 hop)\n\n"
            for item in related:
                rn = item["node"]
                rlabel = rn.label or rn.content[:40]
                if rlabel not in trav_text:  # dedup
                    trav_exp_text += f"### {rlabel}\n"
                    if rn.summary or rn.content:
                        trav_exp_text += f"({rn.summary or rn.content})\n\n"
                    for cid in graph.get_children(rn.id):
                        cn = all_nodes.get(cid)
                        if cn and cn.level == 0:
                            trav_exp_text += f"- {cn.content}\n"
            trav_exp_text += "\n"
        else:
            trav_exp_text = trav_text
        trav_exp_time = 0  # no extra LLM call
        trav_exp_hit = compute_hit_k(expected, trav_exp_text, graph)
        trav_exp_metrics = trav_metrics  # same pruning

        # --- Pruning-equality check (always passes — same call) ---
        pruning_identical = True

        # --- Per-level prune-error (using full ancestor paths) ---
        expected_roots = q.get("expected_roots", [])
        if expected or expected_roots:
            pe = compute_prune_errors(expected, expected_roots,
                                      trav_metrics.get("prune_decisions", []), graph)
            for layer, info in pe.items():
                if layer not in per_level_errors:
                    per_level_errors[layer] = {"errors": 0, "total": 0}
                per_level_errors[layer]["errors"] += info["errors"]
                per_level_errors[layer]["total"] += info["total"]

        # --- Detect total prunes ---
        total_prune_layers = [
            d["layer"] for d in trav_metrics.get("prune_decisions", [])
            if not d.get("kept")
        ]
        total_prune_flag = ",".join(str(x) for x in total_prune_layers) if total_prune_layers else "-"

        # --- Print row ---
        kept = trav_metrics.get("branches_kept", 0)
        pruned = trav_metrics.get("branches_pruned", 0)
        prune_eq = "yes" if pruning_identical else "NO"
        print(
            f"{qid:<6} {qtype:<14} "
            f"{flat_hit:>9}/{len(expected):<2} "
            f"{flat_sum_hit:>14}/{len(expected):<2} "
            f"{trav_hit:>9}/{len(expected):<2} "
            f"{trav_exp_hit:>11}/{len(expected):<2} "
            f"{total_prune_flag:>11} "
            f"{prune_eq:>4} "
            f"{kept:>2}/{pruned:<2}"
        )

        # Aggregate per type
        if args.content:
            print(f"\n{'='*60}")
            print(f"[{qid}] {qtype} — {query_text}")
            print(f"{'='*60}")
            print(f"--- FLAT (top-10) ---\n{flat_text[:2000]}")
            print(f"--- TRAVERSAL (no expansion) ---\n{trav_text[:2000]}")
            print(f"--- TRAVERSAL (+expansion) ---\n{trav_exp_text[:2000]}")
            print(f"--- EXPECTED FACTS ---")
            for eid in expected:
                try:
                    print(f"  {graph.get_node(eid).content}")
                except KeyError:
                    print(f"  (unresolvable ID: {eid})")
            print(f"{'='*60}\n")

        if qtype not in type_results:
            type_results[qtype] = {
                "flat_hit": 0, "flat_sum_hit": 0, "trav_hit": 0,
                "trav_exp_hit": 0, "count": 0, "total_expected": 0,
                "prune_identical": 0,
            }
        tr = type_results[qtype]
        tr["flat_hit"] += flat_hit
        tr["flat_sum_hit"] += flat_sum_hit
        tr["trav_hit"] += trav_hit
        tr["trav_exp_hit"] += trav_exp_hit
        if pruning_identical:
            tr["prune_identical"] += 1
        tr["count"] += 1
        tr["total_expected"] += len(expected)

    # --- Per-type aggregates ---
    print("\n--- Per-Type Averages ---")
    agg_header = (
        f"{'Type':<14} {'Count':>6} {'Flat Hit%':>10} "
        f"{'Flat+Sum Hit%':>14} {'Trav Hit%':>10} {'Trav+Exp Hit%':>15} {'Delta':>7}"
    )
    print(agg_header)
    print("-" * 80)
    for qtype, tr in sorted(type_results.items()):
        n = tr["count"]
        total_exp = tr["total_expected"]
        flat_pct = (tr["flat_hit"] / total_exp * 100) if total_exp else 0
        fs_pct = (tr["flat_sum_hit"] / total_exp * 100) if total_exp else 0
        trav_pct = (tr["trav_hit"] / total_exp * 100) if total_exp else 0
        trav_exp_pct = (tr["trav_exp_hit"] / total_exp * 100) if total_exp else 0
        delta = trav_exp_pct - trav_pct
        print(
            f"{qtype:<14} {n:>6} "
            f"{flat_pct:>9.0f}% "
            f"{fs_pct:>13.0f}% "
            f"{trav_pct:>9.0f}% "
            f"{trav_exp_pct:>14.0f}% "
            f"{delta:>+6.0f}%"
        )

    # --- Per-level prune-error rates ---
    if per_level_errors:
        print("\n--- Per-Level Prune-Error Rates (traversal only) ---")
        print(f"{'Depth':>6} {'Errors':>7} {'Total':>7} {'Error Rate':>11}")
        print("-" * 35)
        for layer in sorted(per_level_errors.keys()):
            pe = per_level_errors[layer]
            rate = (pe["errors"] / pe["total"] * 100) if pe["total"] else 0
            print(f"{layer:>6} {pe['errors']:>7} {pe['total']:>7} {rate:>10.0f}%")
        print("\nInterpretation:")
        print("  - Low rate at depth 0 (coarse: software vs person) = expected")
        print("  - Rate increase at depth 1+ = subtler distinctions are harder")
        print("  - If aspect-level distinctions (depth 2+) degrade: stop deepening")

    # ── pruning-equality summary ──
    total_identical = sum(tr.get("prune_identical", 0) for tr in type_results.values())
    total_queries = sum(tr["count"] for tr in type_results.values())
    print(f"\nPruning identical (expansion on vs off): "
          f"{total_identical}/{total_queries}")
    if total_identical < total_queries:
        print("WARNING: expansion is influencing pruning — "
              "deltas may reflect pruner noise, not relation enrichment.")

    # ── baseline-noise run: two passes with expansion fixed OFF ──
    print("\n--- Baseline Noise (expansion=off, run twice) ---")
    noise_results = {}
    for q in queries:
        qid = q["id"]
        expected = q.get("expected_facts", [])
        t1_text, _ = traverse_recall(q["query"], graph, relation_expansion=False)
        t2_text, _ = traverse_recall(q["query"], graph, relation_expansion=False)
        h1 = compute_hit_k(expected, t1_text, graph)
        h2 = compute_hit_k(expected, t2_text, graph)
        noise_results[qid] = (h1, h2, h1 != h2)

    unstable = sum(1 for _, _, diff in noise_results.values() if diff)
    total = len(noise_results)
    print(f"Hits identical across runs: {total - unstable}/{total}")
    if unstable > 0:
        print(f"WARNING: {unstable} queries had different hits between runs — "
              f"pruner is non-deterministic. Expect ±{unstable} variance in any "
              f"single-run delta before attributing changes to expansion.")
        for qid, (h1, h2, _) in sorted(noise_results.items()):
            if h1 != h2:
                print(f"  {qid}: run1={h1}, run2={h2}")




if __name__ == "__main__":
    main()
