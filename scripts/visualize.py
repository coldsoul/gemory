#!/usr/bin/env python3
"""Visualize the Gemory memory graph as an interactive HTML page."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Gemory memory graph")
    parser.add_argument(
        "memory_path",
        nargs="?",
        default="memory.json",
        help="Path to memory.json (default: memory.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="graph.html",
        help="Output HTML file (default: graph.html)",
    )
    args = parser.parse_args()

    # Read memory.json
    with open(args.memory_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges") or data.get("links") or []

    if not nodes:
        print("No nodes in graph -- nothing to visualize.")
        sys.exit(0)

    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True, notebook=False)

    # Color nodes by confidence level
    for node in nodes:
        node_id = node["id"]
        content = node.get("content", "")
        confidence = node.get("confidence", 0.0)

        # Truncate long content for display labels
        label = content[:60] + "..." if len(content) > 60 else content
        title = (
            f"ID: {node_id[:8]}\n"
            f"Confidence: {confidence:.1f}\n"
            f"Content: {content}"
        )

        # Color: blue -> green -> orange -> red as confidence increases
        if confidence >= 3.0:
            color = "#e74c3c"  # red
        elif confidence >= 2.0:
            color = "#f39c12"  # orange
        elif confidence >= 1.0:
            color = "#2ecc71"  # green
        else:
            color = "#3498db"  # blue

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=max(10, confidence * 8),
        )

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 0.5)
        relation = edge.get("relation", "related")
        net.add_edge(
            source,
            target,
            title=f"{relation}\nweight: {weight:.3f}",
            value=weight,
        )

    # Physics options for better layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)

    net.save_graph(args.output)
    print(f"Graph saved to {args.output}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")


if __name__ == "__main__":
    main()
