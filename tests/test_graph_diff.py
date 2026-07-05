"""Tests for the graph-diff helper."""

from tests.graph_diff import GraphDiff, GraphSnapshot, diff, snapshot


def test_diff_no_changes():
    snap = GraphSnapshot(
        node_contents={"a": "fact a"},
        node_confidences={"a": 1.0},
        node_provenance={"a": [{"source_id": "src1", "label": "", "timestamp": "2024-01-01T00:00:00"}]},
        edge_set=set(),
        embedding_count=1,
    )
    d = diff(snap, snap)
    assert d.is_empty


def test_diff_node_added():
    before = GraphSnapshot({}, {}, {}, set(), 0)
    after = GraphSnapshot(
        node_contents={"a": "fact a"},
        node_confidences={"a": 1.0},
        node_provenance={"a": [{"source_id": "src1", "label": "", "timestamp": ""}]},
        edge_set=set(),
        embedding_count=1,
    )
    d = diff(before, after)
    assert len(d.nodes_added) == 1
    assert d.nodes_added[0][1] == "fact a"
    assert not d.is_empty


def test_diff_provenance_corroboration():
    before = GraphSnapshot(
        node_contents={"a": "fact a"},
        node_confidences={"a": 1.0},
        node_provenance={"a": [{"source_id": "src1", "label": "", "timestamp": ""}]},
        edge_set=set(),
        embedding_count=1,
    )
    after = GraphSnapshot(
        node_contents={"a": "fact a"},
        node_confidences={"a": 2.0},
        node_provenance={
            "a": [
                {"source_id": "src1", "label": "", "timestamp": ""},
                {"source_id": "src2", "label": "", "timestamp": ""},
            ]
        },
        edge_set=set(),
        embedding_count=1,
    )
    d = diff(before, after)
    assert len(d.provenance_added) == 1
    assert d.provenance_added[0][1] == "src2"
    assert len(d.confidence_changes) == 1


def test_diff_summary():
    d = GraphDiff(
        nodes_added=[("id1", "fact one")],
        provenance_added=[("id1", "src2")],
    )
    s = d.summary()
    assert "fact one" in s
    assert "src2" in s
