"""Regression: clusterer does NOT group nodes connected only by relates_to edges."""

import pytest

from src.consolidate import cluster_layer
from src.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def confab_graph(tmp_graph_path):
    """Create 4 unrelated topic nodes connected by relates_to edges from user
    profile.  Mirrors the real scenario where user profile relates_to all
    projects but they are genuinely unrelated."""
    store = GraphStore(tmp_graph_path)

    # Create user profile topic.
    user = store.add_node(
        content="user profile", embedding=[0.5, 0.5, 0.5],
        provenance={
            "source_id": "topic-registry", "label": "user profile", "timestamp": "",
        },
        kind="abstraction", label="user profile",
        abstraction_kind="topic",
    )
    store.set_node_attr(user, "level", 1)

    # Create 4 unrelated project topics with far-apart embeddings.
    projects = {
        "Gemory": [1.0, 0.0, 0.0],
        "MS Navigator": [0.0, 1.0, 0.0],
        "honcho TUI": [0.0, 0.0, 1.0],
        "Sofia transit": [-1.0, 0.0, 0.0],
    }
    project_ids = {}
    for name, emb in projects.items():
        pid = store.add_node(
            content=name, embedding=emb,
            provenance={
                "source_id": "topic-registry", "label": name, "timestamp": "",
            },
            kind="abstraction", label=name,
            abstraction_kind="topic", summary=f"Details about {name}.",
        )
        store.set_node_attr(pid, "level", 1)
        project_ids[name] = pid

        # Add relates_to from user to each project.
        store.add_relates_to_edge(user, pid, origin_fact="test")

        # Give each project 3 child facts.
        for i in range(3):
            fid = store.add_node(
                content=f"Fact about {name} #{i}",
                embedding=[float(i), 0.0, 0.0],
                provenance={
                    "source_id": f"{name}-{i}", "label": "", "timestamp": "",
                },
            )
            store.add_parent_edge(pid, fid)

    return store, user, project_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClustererIgnoresRelations:
    """Algorithm and LLM clusterers must NOT group nodes connected only by
    relates_to edges."""

    def test_algorithm_does_not_group_unrelated_projects(self, confab_graph):
        """Four unrelated projects with relates_to edges should NOT be grouped
        by the algorithm clusterer (which only uses embedding cosines and
        ``related`` edges, ignoring ``relates_to``)."""
        store, user, project_ids = confab_graph
        all_topic_ids = [user] + list(project_ids.values())

        clusters = cluster_layer(store, all_topic_ids, method="algorithm")

        # No cluster should contain all 4 projects.
        for cluster in clusters:
            proj_in_cluster = sum(
                1 for pid in project_ids.values() if pid in cluster
            )
            assert proj_in_cluster < 4, (
                f"Algorithm incorrectly grouped {proj_in_cluster} "
                f"unrelated projects together"
            )

    def test_llm_does_not_group_unrelated_projects(
        self, confab_graph, monkeypatch,
    ):
        """With LLM clusterer stubbed to return empty (correct answer for
        unrelated items), no cluster forms."""
        store, user, project_ids = confab_graph
        all_topic_ids = [user] + list(project_ids.values())

        monkeypatch.setattr("src.llm.cluster_by_llm", lambda x: [])

        clusters = cluster_layer(store, all_topic_ids, method="llm")

        assert len(clusters) == 0, (
            f"LLM should return no groups for unrelated topics, "
            f"got {len(clusters)}"
        )
