"""GraphStore: persistent DiGraph with an embedding sidecar.

Nodes carry content, confidence, provenance, and timestamps.
Embeddings live in a separate JSON sidecar file -- never on the node itself.
"""

import logging
from typing import Any

import json
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np

from src import config
from src.models import Edge, Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, data) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*.

    The file is first written to a ``.tmp`` sibling and then moved into
    place with :func:`os.replace` so that concurrent readers always see a
    complete file.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """Directed graph memory backed by two JSON files:

    * **memory.json** -- the graph structure (nodes + edges) via
      :func:`networkx.node_link_data`.
    * **embeddings.json** -- a flat dict ``{node_id: list[float]}`` sidecar.

    The sidecar is derived from :data:`config.EMBEDDINGS_PATH` and lives in
    the same directory as the memory file.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        dir_name = os.path.dirname(path)
        if dir_name:
            self._embeddings_path = os.path.join(dir_name, config.EMBEDDINGS_PATH)
        else:
            self._embeddings_path = config.EMBEDDINGS_PATH
        self._graph: nx.DiGraph = nx.DiGraph()
        self._embeddings: dict[str, list[float]] = {}

    # -- Persistence -------------------------------------------------------

    def load(self) -> None:
        """Load graph + embeddings from disk.

        If **memory.json** does not exist the graph starts empty (no error).
        If it does exist every node **must** have a matching embedding in the
        sidecar or a :class:`ValueError` is raised.  Orphan embeddings
        (embedding keys that do not correspond to any node) are silently
        discarded.
        """
        if not os.path.exists(self._path):
            logger.info("No memory file at %s, starting empty", self._path)
            self._graph = nx.DiGraph()
            self._embeddings = {}
            return

        with open(self._path) as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data, directed=True, multigraph=False)

        # Load embeddings sidecar.
        self._embeddings = {}
        if os.path.exists(self._embeddings_path):
            with open(self._embeddings_path) as f:
                raw = json.load(f)
            # Only keep embeddings whose node still exists in the graph --
            # orphan embeddings are silently ignored.
            node_ids = set(self._graph.nodes())
            orphans = len(raw) - len(node_ids & set(raw.keys()))
            if orphans:
                logger.info(
                    "Discarded %d orphan embeddings (no matching node)", orphans,
                )
            self._embeddings = {
                k: v for k, v in raw.items() if k in node_ids
            }

        # Every node MUST have an embedding.
        for node_id in self._graph.nodes():
            if node_id not in self._embeddings:
                logger.error(
                    "Node %r has no embedding in sidecar (%s)",
                    node_id,
                    self._embeddings_path,
                )
                raise ValueError(
                    f"Node {node_id!r} exists in the memory graph but has no "
                    f"embedding in the sidecar ({self._embeddings_path})"
                )

        # Migrate old schema: nodes created before newer fields.
        migrated = 0
        for nid in self._graph.nodes():
            attrs = self._graph.nodes[nid]
            if "kind" not in attrs:
                attrs["kind"] = "fact"
                migrated += 1
            if "label" not in attrs or not attrs["label"]:
                # Only backfill label for abstraction nodes; facts keep "".
                if attrs.get("kind") == "abstraction":
                    attrs["label"] = attrs.get("content", "")[:80]
                else:
                    attrs["label"] = ""
            if "abstraction_kind" not in attrs:
                attrs["abstraction_kind"] = ""
                migrated += 1
            if "reach" not in attrs:
                attrs["reach"] = 0
                migrated += 1
            if "summary" not in attrs:
                attrs["summary"] = ""
        if migrated:
            logger.info(
                "Migrated %d nodes missing schema fields", migrated,
            )

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def save(self) -> None:
        """Atomically write graph + embeddings to disk.

        Embeddings are **never** stored inside ``memory.json`` -- they are
        written exclusively to the sidecar file.
        """
        # Serialise graph structure via networkx (no embeddings here).
        data = nx.node_link_data(self._graph)

        # Safety: strip any stray embedding-related keys from node data.
        for node_entry in data.get("nodes", []):
            for key in list(node_entry.keys()):
                if "embedding" in key.lower():
                    del node_entry[key]

        # Write both files atomically.
        _atomic_write_json(self._path, data)
        _atomic_write_json(self._embeddings_path, self._embeddings)

        logger.info(
            "Saved graph: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # -- Node / edge operations --------------------------------------------

    def add_node(
        self,
        content: str,
        embedding: list[float],
        provenance: dict,
        kind: str = "fact",
        label: str = "",
        abstraction_kind: str = "",
        reach: int = 0,
        summary: str = "",
    ) -> str:
        """Create a new node in the graph.

        Parameters
        ----------
        content
            Fact or abstraction text (short label for abstractions).
        embedding
            Vector embedding of *content*.
        provenance
            Source metadata dict (must contain ``"source_id"``).
        kind
            ``"fact"`` (leaf, extracted) or ``"abstraction"`` (dreamer-created).
        label
            Short theme label for abstraction nodes; empty for facts.
        abstraction_kind
            ``""`` (not an abstraction), ``"topic"`` (level-1 topic), or
            ``"theme"`` (dreamer-created higher-level).
        reach
            Transitive leaf count; 0 for facts, computed for abstractions.
        summary
            1-2 sentence prose description (abstractions only).

        Returns the auto-generated UUID4 node id.
        """
        node_id = str(uuid.uuid4())
        now = _now_iso()
        self._graph.add_node(
            node_id,
            content=content,
            confidence=config.CONFIDENCE_BASE,
            provenance=[provenance],
            created_at=now,
            updated_at=now,
            level=0,
            kind=kind,
            label=label,
            abstraction_kind=abstraction_kind,
            reach=reach,
            summary=summary,
        )
        self._embeddings[node_id] = embedding
        logger.info(
            "Created node %s (confidence=%.1f, kind=%s, abstraction_kind=%s)",
            node_id, config.CONFIDENCE_BASE, kind, abstraction_kind,
        )
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        relation: str = "related",
    ) -> None:
        """Add a directed edge from *source* to *target*."""
        self._graph.add_edge(source, target, weight=weight, relation=relation)
        logger.info(
            "Added edge %s -> %s [weight=%.3f, relation=%s]",
            source, target, weight, relation,
        )

    # -- Query helpers -----------------------------------------------------

    def find_similar(
        self,
        embedding: list[float],
        threshold: float | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search against all stored embeddings.

        Parameters
        ----------
        embedding
            Query vector.
        threshold
            Minimum similarity to include in results.  When ``None`` (default)
            **all** nodes are returned (sorted descending).
        top_k
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(node_id, cosine_similarity)`` pairs sorted from most to least
            similar.
        """
        if not self._embeddings:
            logger.info("Similarity search against 0 embeddings, returning []")
            return []

        logger.info(
            "Similarity search against %d embeddings, threshold=%s, top_k=%d",
            len(self._embeddings), threshold, top_k,
        )

        query = np.array(embedding, dtype=float)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query_normalised = query / query_norm

        results: list[tuple[str, float]] = []
        for nid, emb in self._embeddings.items():
            emb_arr = np.array(emb, dtype=float)
            norm = np.linalg.norm(emb_arr)
            if norm == 0:
                continue
            similarity = float(np.dot(emb_arr / norm, query_normalised))
            results.append((nid, similarity))

        results.sort(key=lambda x: x[1], reverse=True)

        if threshold is not None:
            results = [(nid, sim) for nid, sim in results if sim >= threshold]

        logger.info("Found %d results", len(results))
        return results[:top_k]

    def get_node(self, node_id: str) -> Node:
        """Return the :class:`Node` dataclass for *node_id*.

        Raises :class:`KeyError` if the node does not exist.
        """
        if node_id not in self._graph:
            logger.error("Node %r not found in graph", node_id)
            raise KeyError(node_id)
        attrs = self._graph.nodes[node_id]
        return Node(id=node_id, **attrs)

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return the list of successor node ids (outgoing edges)."""
        return list(self._graph.successors(node_id))

    def get_roots(self) -> list[str]:
        """Return nodes with no incoming edges (in-degree == 0)."""
        return [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]

    def bump_confidence(self, node_id: str, provenance: dict) -> bool:
        """Corroborate a node from a new source.

        If ``provenance["source_id"]`` is already present in the node's
        provenance list the operation is idempotent -- returns ``False``.

        Otherwise the provenance dict is appended, confidence is incremented
        by :data:`config.CONFIDENCE_INCREMENT`, and ``updated_at`` is set to
        now.

        Returns
        -------
        bool
            ``True`` if the node was actually updated.
        """
        attrs = self._graph.nodes[node_id]
        existing_ids = {p["source_id"] for p in attrs["provenance"]}
        if provenance["source_id"] in existing_ids:
            logger.info(
                "Source %s already recorded for node %s, skipping",
                provenance["source_id"], node_id,
            )
            return False

        old_conf = attrs["confidence"]
        attrs["provenance"].append(provenance)
        attrs["confidence"] += config.CONFIDENCE_INCREMENT
        attrs["updated_at"] = _now_iso()
        logger.info(
            "Corroborated node %s (confidence: %.1f -> %.1f)",
            node_id, old_conf, attrs["confidence"],
        )
        return True

    # -- Schema-aware helpers ------------------------------------------------

    def get_all_edges(self) -> list[Edge]:
        """Return every edge in the graph as :class:`Edge` dataclass instances."""
        return [
            Edge(
                source=u,
                target=v,
                weight=data.get("weight", 1.0),
                relation=data.get("relation", "related"),
            )
            for u, v, data in self._graph.edges(data=True)
        ]

    def get_parents(self, node_id: str) -> list[str]:
        """Return nodes that have a ``parent_of`` edge pointing TO *node_id*
        (abstraction nodes above this node)."""
        return [
            u
            for u, v, data in self._graph.in_edges(node_id, data=True)
            if data.get("relation") == "parent_of"
        ]

    def get_children(self, node_id: str) -> list[str]:
        """Return nodes that receive a ``parent_of`` edge FROM *node_id*
        (members of this abstraction)."""
        return [
            v
            for u, v, data in self._graph.out_edges(node_id, data=True)
            if data.get("relation") == "parent_of"
        ]

    def set_node_attr(self, node_id: str, attr: str, value: Any) -> None:
        """Set an arbitrary attribute on a node.

        Raises :class:`KeyError` if the node does not exist.
        """
        if node_id not in self._graph:
            raise KeyError(node_id)
        self._graph.nodes[node_id][attr] = value

    def add_parent_edge(self, parent_id: str, child_id: str) -> None:
        """Add a directed ``parent_of`` edge from *parent_id* to *child_id*.

        Raises :class:`ValueError` if either node does not exist.
        """
        if parent_id not in self._graph:
            raise ValueError(f"Parent node {parent_id!r} not found")
        if child_id not in self._graph:
            raise ValueError(f"Child node {child_id!r} not found")
        self._graph.add_edge(
            parent_id, child_id, weight=1.0, relation="parent_of",
        )
        logger.info(
            "Added parent_of edge %s -> %s", parent_id, child_id,
        )

    def add_relates_to_edge(
        self, source: str, target: str, origin_fact: str = "",
    ) -> None:
        """Add a directed ``relates_to`` edge.

        Idempotent: if a ``relates_to`` edge already exists between the same
        pair it is not duplicated.  If a different edge type exists between
        them, a warning is logged and the call is skipped (DiGraph limitation
        — one edge per node pair).
        """
        if self._graph.has_edge(source, target):
            edge = self._graph.get_edge_data(source, target)
            if edge.get("relation") == "relates_to":
                return  # already exists
            logger.warning(
                "Cannot add relates_to edge %s->%s: edge exists with "
                "relation=%s", source[:8], target[:8], edge.get("relation"),
            )
            return
        self._graph.add_edge(
            source, target,
            relation="relates_to",
            provenance="stated",
            origin_fact=origin_fact,
        )
        logger.info(
            "Added relates_to edge %s -> %s", source[:8], target[:8],
        )

    def get_edges_by_relation(
        self, relation: str,
    ) -> list[tuple[str, str, dict]]:
        """Return all edges with the given relation type.

        Each element is ``(source, target, edge_data_dict)``.
        """
        result: list[tuple[str, str, dict]] = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("relation") == relation:
                result.append((u, v, data))
        return result

    def get_topic_nodes(self) -> list[Node]:
        """Return all nodes whose ``abstraction_kind`` is ``"topic"``."""
        return [
            n for n in self.all_nodes()
            if n.kind == "abstraction" and n.abstraction_kind == "topic"
        ]

    def get_node_embedding(self, node_id: str) -> list[float]:
        """Return the embedding vector for a node from the sidecar.

        Raises :class:`KeyError` if the node does not exist.
        """
        if node_id not in self._graph:
            raise KeyError(node_id)
        return self._embeddings.get(node_id, [])

    def all_nodes(self) -> list[Node]:
        """Return every :class:`Node` currently in the graph."""
        return [
            Node(id=nid, **attrs) for nid, attrs in self._graph.nodes(data=True)
        ]
