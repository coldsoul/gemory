#!/usr/bin/env python3
"""Calibration script: compute pairwise cosine similarities in a populated
memory.json to find the valley between true-duplicate and distinct-but-related
pairs.  Used to choose DEDUP_THRESHOLD and EDGE_THRESHOLD from real data.

With --topics-only, only analyzes topic nodes — useful for tuning
TOPIC_MATCH_THRESHOLD where topic strings embed tightly.

Skipped unless GEMORY_LIVE=1 is set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    if os.getenv("GEMORY_LIVE") != "1":
        print("Set GEMORY_LIVE=1 to run the calibration script.", file=sys.stderr)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Calibrate similarity thresholds")
    parser.add_argument("memory_path", nargs="?", default="memory.json",
                        help="Path to memory.json")
    parser.add_argument("embeddings_path", nargs="?", default="embeddings.json",
                        help="Path to embeddings.json")
    parser.add_argument("--topics-only", action="store_true",
                        help="Only analyze topic nodes (for tuning TOPIC_MATCH_THRESHOLD)")
    args = parser.parse_args()

    memory_path = args.memory_path
    embeddings_path = args.embeddings_path

    with open(memory_path) as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    if not nodes:
        print("No nodes in graph.", file=sys.stderr)
        sys.exit(0)

    with open(embeddings_path) as f:
        embeddings = json.load(f)

    # Filter to topic nodes if requested
    if args.topics_only:
        nodes = [
            n for n in nodes
            if n.get("kind") == "abstraction" and n.get("abstraction_kind") == "topic"
        ]
        if not nodes:
            print("No topic nodes found — run the server with a topic-aware extractor first.", file=sys.stderr)
            sys.exit(0)
        print(f"Analyzing {len(nodes)} topic nodes (for TOPIC_MATCH_THRESHOLD tuning)")

    # Build content + vector lists
    node_contents = []
    vectors = []
    for node in nodes:
        nid = node["id"]
        if nid in embeddings:
            node_contents.append((nid, node.get("content", "")))
            vectors.append(np.array(embeddings[nid], dtype=float))

    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    # Full pairwise cosine similarity
    sim_matrix = vectors @ vectors.T

    # Collect ranked pairs (above diagonal only)
    pairs = []
    for i in range(len(node_contents)):
        for j in range(i + 1, len(node_contents)):
            pairs.append((
                float(sim_matrix[i, j]),
                node_contents[i][1],
                node_contents[j][1],
            ))

    pairs.sort(key=lambda x: x[0], reverse=True)

    # Print ranked pairs
    print(f"\nPairwise cosine similarities ({len(pairs)} pairs):")
    print(f"{'cosine':>8}  content")
    print("-" * 80)
    for sim, content_a, content_b in pairs:
        bar = "#" * int(sim * 40)
        if sim > 0.90:
            label = "MERGE"
        elif sim > 0.75:
            label = "EDGE "
        else:
            label = "DIST "
        print(f"{sim:8.4f} [{label}] {bar}")
        print(f"         A: {content_a[:70]}")
        print(f"         B: {content_b[:70]}")
        print()

    # Summary buckets
    buckets = {"0.95-1.00": 0, "0.90-0.95": 0, "0.80-0.90": 0,
               "0.70-0.80": 0, "0.60-0.70": 0, "0.50-0.60": 0, "<0.50": 0}
    for sim, _, _ in pairs:
        if sim >= 0.95:
            buckets["0.95-1.00"] += 1
        elif sim >= 0.90:
            buckets["0.90-0.95"] += 1
        elif sim >= 0.80:
            buckets["0.80-0.90"] += 1
        elif sim >= 0.70:
            buckets["0.70-0.80"] += 1
        elif sim >= 0.60:
            buckets["0.60-0.70"] += 1
        elif sim >= 0.50:
            buckets["0.50-0.60"] += 1
        else:
            buckets["<0.50"] += 1

    print("Distribution:")
    for bucket, count in buckets.items():
        bar = "#" * count
        print(f"  {bucket}: {count:3d} {bar}")


if __name__ == "__main__":
    main()
