"""Reach computation: transitive leaf count via parent_of traversal."""

from src.graph import GraphStore


def compute_reach(graph: GraphStore, node_ids: list[str]) -> int:
    """Return the number of distinct level-0 fact nodes reachable from
    *node_ids* by following ``parent_of`` edges downward.

    For a single node: count of distinct leaf descendants.
    For a cluster (set of nodes): count of distinct leaves across all members.
    """
    if not node_ids:
        return 0

    visited: set[str] = set()
    stack = list(node_ids)

    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)

        node = graph.get_node(nid)
        if node.level == 0:
            # Leaf fact -- count it, don't traverse further.
            continue

        # Traverse children via parent_of only.
        # get_children already filters by relation == "parent_of", so
        # relates_to edges never contribute to reach.
        children = graph.get_children(nid)
        stack.extend(children)

    # Count only level-0 nodes in the visited set.
    leaves = {nid for nid in visited if graph.get_node(nid).level == 0}
    return len(leaves)


def update_reach(graph: GraphStore, abstraction_id: str) -> int:
    """Compute and store the reach for an abstraction node.

    Returns the computed reach value.
    """
    children = graph.get_children(abstraction_id)
    r = compute_reach(graph, children)
    graph.set_node_attr(abstraction_id, "reach", r)
    return r


def backfill_reach(graph: GraphStore) -> int:
    """Recompute reach for all topic and abstraction nodes.

    Returns the number of nodes updated.
    """
    updated = 0
    for node in graph.all_nodes():
        if node.kind == "abstraction":
            r = compute_reach(graph, [node.id])
            if node.reach != r:
                graph.set_node_attr(node.id, "reach", r)
                updated += 1
    return updated
