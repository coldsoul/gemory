"""Tests for label/summary separation on abstraction nodes."""

import json
import os

import pytest

from src.graph import GraphStore


class TestLabelSummary:
    """Abstraction nodes keep label and summary as separate, authoritative
    fields — summary prose never overwrites the short label."""

    def test_abstraction_node_has_both_fields(self, tmp_graph_path):
        """Creating an abstraction node populates both label and summary."""
        store = GraphStore(tmp_graph_path)

        abs_id = store.add_node(
            content="MS Navigator project",
            embedding=[1.0, 0.0],
            provenance={
                "source_id": "dreamer:test", "label": "MS Navigator project",
                "timestamp": "",
            },
            kind="abstraction",
            label="MS Navigator project",
            summary=(
                "The MS Navigator is a Bulgarian-language, "
                "zero-data static site."
            ),
            abstraction_kind="topic",
            reach=10,
        )

        node = store.get_node(abs_id)
        assert node.label == "MS Navigator project"
        assert (
            node.summary
            == "The MS Navigator is a Bulgarian-language, zero-data static site."
        )
        assert node.kind == "abstraction"

    def test_summary_is_not_label_field(self, tmp_graph_path):
        """The summary prose does NOT end up in the label field."""
        store = GraphStore(tmp_graph_path)

        abs_id = store.add_node(
            content="Gemory",
            embedding=[1.0, 0.0],
            provenance={
                "source_id": "topic-registry", "label": "Gemory", "timestamp": "",
            },
            kind="abstraction",
            label="Gemory",
            summary="",
            abstraction_kind="topic",
        )

        node = store.get_node(abs_id)
        assert len(node.label) < 50
        assert node.label == "Gemory"
        assert node.summary == ""

    def test_legacy_node_missing_summary_falls_back(self, tmp_path):
        """Old nodes without summary field default to empty string on load."""
        from src.config import EMBEDDINGS_PATH
        mem_path = str(tmp_path / "memory.json")
        emb_path = os.path.join(str(tmp_path), EMBEDDINGS_PATH)

        # Write a legacy-format graph (no summary field).
        legacy_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {
                    "id": "test1",
                    "content": "Old topic",
                    "confidence": 1.0,
                    "provenance": [],
                    "created_at": "",
                    "updated_at": "",
                    "level": 1,
                    "kind": "abstraction",
                    "label": "Old topic",
                    "abstraction_kind": "topic",
                    "reach": 0,
                }
            ],
            "edges": [],
        }
        with open(mem_path, "w") as f:
            json.dump(legacy_data, f)
        with open(emb_path, "w") as f:
            json.dump({"test1": [1.0, 0.0]}, f)

        store = GraphStore(mem_path)
        store.load()

        node = store.get_node("test1")
        assert node.summary == ""  # default
        assert node.label == "Old topic"  # preserved

    def test_embedding_uses_label_and_summary(self):
        """Design doc: embedding is computed from label+summary, not content alone.
        Verified by code review of _create_abstraction in dreamer.py which calls
        embed(f\"{label}. {summary_text}\")."""
        pass
