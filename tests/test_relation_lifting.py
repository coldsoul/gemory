"""Tests for relation lifting on category creation."""

import pytest

from src.dreamer import _lift_relations
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lift_graph(tmp_graph_path):
    """Create a graph where user profile relates_to 3 project topics."""
    store = GraphStore(tmp_graph_path)

    # User profile.
    user = store.add_node(
        content="user profile", embedding=[0.5, 0.5],
        provenance={
            "source_id": "t1", "label": "user profile", "timestamp": "",
        },
        kind="abstraction", label="user profile",
        abstraction_kind="topic",
    )
    store.set_node_attr(user, "level", 1)

    # Category (simulating a newly created "software projects" theme).
    cat = store.add_node(
        content="software projects", embedding=[0.7, 0.3],
        provenance={
            "source_id": "dreamer:test", "label": "software projects",
            "timestamp": "",
        },
        kind="abstraction", label="software projects",
    )
    store.set_node_attr(cat, "level", 2)

    # 3 project topics with relates_to from user.
    members = []
    for name in ["Gemory", "MS Navigator", "honcho TUI"]:
        pid = store.add_node(
            content=name, embedding=[0.1, 0.9],
            provenance={
                "source_id": f"t-{name}", "label": name, "timestamp": "",
            },
            kind="abstraction", label=name,
            abstraction_kind="topic",
        )
        store.set_node_attr(pid, "level", 1)
        members.append(pid)

        # User profile -> project (stated relates_to).
        store._graph.add_edge(
            user, pid,
            relation="relates_to", provenance="stated", origin_fact="f1",
        )

    return store, user, cat, members


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRelationLifting:
    """Relation lifting on category creation."""

    def test_lift_when_all_members_have_relation(self, lift_graph):
        """When all members have relates_to from X, X->category is created."""
        store, user, cat, members = lift_graph
        lifted = _lift_relations(store, cat, members)
        assert lifted == 1

        derived = [
            (s, t, d) for s, t, d in store.get_edges_by_relation("relates_to")
            if d.get("provenance") == "derived"
        ]
        assert len(derived) == 1
        assert derived[0][0] == user
        assert derived[0][1] == cat

    def test_leaf_edges_preserved_after_lift(self, lift_graph):
        """Stated leaf edges remain -- lifting adds, never replaces."""
        store, user, cat, members = lift_graph
        _lift_relations(store, cat, members)

        stated = [
            (s, t) for s, t, d in store.get_edges_by_relation("relates_to")
            if d.get("provenance") == "stated"
        ]
        assert len(stated) == 3

    def test_lift_is_idempotent(self, lift_graph):
        """Running lift twice creates no duplicate derived edge."""
        store, user, cat, members = lift_graph
        _lift_relations(store, cat, members)
        _lift_relations(store, cat, members)  # second run

        derived = [
            (s, t) for s, t, d in store.get_edges_by_relation("relates_to")
            if d.get("provenance") == "derived"
        ]
        assert len(derived) == 1

    def test_no_lift_when_ratio_not_met(self, lift_graph):
        """If only some members have the relation, no lift occurs."""
        store, user, cat, members = lift_graph

        # Add a 4th member WITHOUT a relates_to from user.
        new_member = store.add_node(
            content="unrelated", embedding=[0.5, 0.5],
            provenance={
                "source_id": "new", "label": "unrelated", "timestamp": "",
            },
            kind="abstraction", label="unrelated",
            abstraction_kind="topic",
        )
        store.set_node_attr(new_member, "level", 1)
        # No relates_to from user to this member.

        partial_members = members + [new_member]  # 4 total, only 3 have relation

        cat2 = store.add_node(
            content="partial cat", embedding=[0.6, 0.4],
            provenance={
                "source_id": "d:test3", "label": "partial", "timestamp": "",
            },
            kind="abstraction", label="partial",
        )
        store.set_node_attr(cat2, "level", 2)

        # With RELATION_LIFT_RATIO=1.0 and 4 members: threshold=4, only 3/4.
        lifted = _lift_relations(store, cat2, partial_members)
        assert lifted == 0
