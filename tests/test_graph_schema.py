"""Tests for Node schema fields (kind/label) and new GraphStore methods."""

import json
import os

import pytest

from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Schema: kind / label
# ---------------------------------------------------------------------------

class TestNodeKindAndLabel:
    """``add_node`` must store *kind*, *label*, and *abstraction_kind* correctly."""

    def test_add_node_with_kind_and_label(self, empty_graph) -> None:
        nid = empty_graph.add_node(
            "abstraction content", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", label="test theme",
        )
        node = empty_graph.get_node(nid)
        assert node.kind == "abstraction"
        assert node.label == "test theme"
        assert node.abstraction_kind == ""

    def test_add_node_with_abstraction_kind(self, empty_graph) -> None:
        nid = empty_graph.add_node(
            "topic content", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", label="Gemory",
            abstraction_kind="topic",
        )
        node = empty_graph.get_node(nid)
        assert node.kind == "abstraction"
        assert node.abstraction_kind == "topic"
        assert node.label == "Gemory"

    def test_add_node_defaults_kind_and_label(self, empty_graph) -> None:
        nid = empty_graph.add_node(
            "fact content", [0.1, 0.2], {"source_id": "s1"},
        )
        node = empty_graph.get_node(nid)
        assert node.kind == "fact"
        assert node.label == ""
        assert node.abstraction_kind == ""


class TestLoadOldSchemaDefaults:
    """Nodes serialised before kind/label were added should get defaults on load."""

    def test_get_topic_nodes(self, empty_graph) -> None:
        t1 = empty_graph.add_node(
            "topic1", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", label="T1", abstraction_kind="topic",
        )
        t2 = empty_graph.add_node(
            "topic2", [0.3, 0.4], {"source_id": "s2"},
            kind="abstraction", label="T2", abstraction_kind="topic",
        )
        # A regular fact node — should not appear
        empty_graph.add_node("fact", [0.5, 0.6], {"source_id": "s3"})

        topics = empty_graph.get_topic_nodes()
        assert len(topics) == 2
        topic_ids = {n.id for n in topics}
        assert t1 in topic_ids
        assert t2 in topic_ids

    def test_load_old_schema_defaults_kind_label(self, tmp_path) -> None:
        from src.config import EMBEDDINGS_PATH
        mem_path = str(tmp_path / "memory.json")

        # Manually write a memory.json WITHOUT kind/label fields.
        old_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {
                    "id": "old-node-1",
                    "content": "old fact",
                    "confidence": 1.0,
                    "provenance": [{"source_id": "s1"}],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "level": 0,
                    # no "kind", no "label"
                },
            ],
            "edges": [],
        }
        with open(mem_path, "w") as f:
            json.dump(old_data, f)

        # Write a matching embeddings sidecar.
        emb_path = os.path.join(tmp_path, EMBEDDINGS_PATH)
        with open(emb_path, "w") as f:
            json.dump({"old-node-1": [0.1, 0.2, 0.3]}, f)

        graph = GraphStore(mem_path)
        graph.load()

        node = graph.get_node("old-node-1")
        assert node.kind == "fact"
        assert node.label == ""
        assert node.abstraction_kind == ""


# ---------------------------------------------------------------------------
# get_all_edges
# ---------------------------------------------------------------------------

class TestGetAllEdges:
    def test_get_all_edges(self, empty_graph) -> None:
        n1 = empty_graph.add_node("a", [0.1, 0.0], {"source_id": "s1"})
        n2 = empty_graph.add_node("b", [0.0, 0.1], {"source_id": "s2"})
        n3 = empty_graph.add_node("c", [0.5, 0.5], {"source_id": "s3"})

        empty_graph.add_edge(n1, n2, weight=0.8, relation="related")
        empty_graph.add_edge(n2, n3, weight=0.6, relation="supports")

        edges = empty_graph.get_all_edges()
        assert len(edges) == 2
        assert edges[0].source == n1
        assert edges[0].target == n2
        assert edges[0].weight == 0.8
        assert edges[0].relation == "related"
        assert edges[1].source == n2
        assert edges[1].target == n3


# ---------------------------------------------------------------------------
# get_parents / get_children
# ---------------------------------------------------------------------------

class TestGetParents:
    def test_get_parents(self, empty_graph) -> None:
        abstraction = empty_graph.add_node(
            "theme: Python", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", label="Python",
        )
        child1 = empty_graph.add_node("likes Python", [0.3, 0.4], {"source_id": "s2"})
        child2 = empty_graph.add_node("uses uv", [0.5, 0.6], {"source_id": "s3"})

        empty_graph.add_parent_edge(abstraction, child1)
        empty_graph.add_parent_edge(abstraction, child2)

        parents_1 = empty_graph.get_parents(child1)
        parents_2 = empty_graph.get_parents(child2)
        assert parents_1 == [abstraction]
        assert parents_2 == [abstraction]

    def test_get_parents_empty(self, empty_graph) -> None:
        nid = empty_graph.add_node("orphan", [0.1, 0.2], {"source_id": "s1"})
        assert empty_graph.get_parents(nid) == []


class TestGetChildren:
    def test_get_children(self, empty_graph) -> None:
        abstraction = empty_graph.add_node(
            "theme: Python", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", label="Python",
        )
        child1 = empty_graph.add_node("likes Python", [0.3, 0.4], {"source_id": "s2"})
        child2 = empty_graph.add_node("uses uv", [0.5, 0.6], {"source_id": "s3"})

        empty_graph.add_parent_edge(abstraction, child1)
        empty_graph.add_parent_edge(abstraction, child2)

        children = empty_graph.get_children(abstraction)
        assert len(children) == 2
        assert child1 in children
        assert child2 in children

    def test_get_children_empty(self, empty_graph) -> None:
        nid = empty_graph.add_node("leaf", [0.1, 0.2], {"source_id": "s1"})
        assert empty_graph.get_children(nid) == []


# ---------------------------------------------------------------------------
# set_node_attr
# ---------------------------------------------------------------------------

class TestReachField:
    """``reach`` field on Node."""

    def test_add_node_with_reach(self, empty_graph) -> None:
        nid = empty_graph.add_node(
            "abstraction", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction", reach=5,
        )
        node = empty_graph.get_node(nid)
        assert node.reach == 5

    def test_add_node_defaults_reach(self, empty_graph) -> None:
        nid = empty_graph.add_node("fact", [0.1, 0.2], {"source_id": "s1"})
        node = empty_graph.get_node(nid)
        assert node.reach == 0

    def test_load_migrates_missing_reach(self, tmp_path) -> None:
        from src.config import EMBEDDINGS_PATH
        mem_path = str(tmp_path / "memory.json")

        # Old JSON data without reach field.
        old_data = {
            "directed": True, "multigraph": False, "graph": {},
            "nodes": [
                {
                    "id": "n1", "content": "no reach", "confidence": 1.0,
                    "provenance": [{"source_id": "s1"}],
                    "created_at": "", "updated_at": "", "level": 0,
                    "kind": "fact", "label": "", "abstraction_kind": "",
                },
            ],
            "edges": [],
        }
        with open(mem_path, "w") as f:
            json.dump(old_data, f)
        emb_path = os.path.join(tmp_path, EMBEDDINGS_PATH)
        with open(emb_path, "w") as f:
            json.dump({"n1": [0.1, 0.2]}, f)

        graph = GraphStore(mem_path)
        graph.load()
        node = graph.get_node("n1")
        assert node.reach == 0


class TestSetNodeAttr:
    def test_set_node_attr(self, empty_graph) -> None:
        nid = empty_graph.add_node("test", [0.1, 0.2], {"source_id": "s1"})
        empty_graph.set_node_attr(nid, "level", 5)
        node = empty_graph.get_node(nid)
        assert node.level == 5

    def test_set_node_attr_missing_raises(self, empty_graph) -> None:
        with pytest.raises(KeyError):
            empty_graph.set_node_attr("nonexistent", "level", 5)


# ---------------------------------------------------------------------------
# add_parent_edge
# ---------------------------------------------------------------------------

class TestAddParentEdge:
    def test_add_parent_edge(self, empty_graph) -> None:
        parent = empty_graph.add_node(
            "parent", [0.1, 0.2], {"source_id": "s1"},
            kind="abstraction",
        )
        child = empty_graph.add_node("child", [0.3, 0.4], {"source_id": "s2"})

        empty_graph.add_parent_edge(parent, child)

        edges = empty_graph.get_all_edges()
        assert len(edges) == 1
        assert edges[0].source == parent
        assert edges[0].target == child
        assert edges[0].relation == "parent_of"

    def test_add_parent_edge_missing_parent_raises(self, empty_graph) -> None:
        child = empty_graph.add_node("child", [0.1, 0.2], {"source_id": "s1"})
        with pytest.raises(ValueError, match="Parent"):
            empty_graph.add_parent_edge("no-such-node", child)

    def test_add_parent_edge_missing_child_raises(self, empty_graph) -> None:
        parent = empty_graph.add_node(
            "parent", [0.1, 0.2], {"source_id": "s1"},
        )
        with pytest.raises(ValueError, match="Child"):
            empty_graph.add_parent_edge(parent, "no-such-child")
