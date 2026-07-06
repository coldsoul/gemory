"""Configuration: environment variable loading and tunable constants.

All behavioural numbers that affect extraction, dedup, similarity, and
confidence live here as single named constants with documented meaning.
No behavioural number is hard-coded outside this module.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_PATH: str = os.getenv("MEMORY_PATH", "memory.json")
EMBEDDINGS_PATH: str = os.getenv("EMBEDDINGS_PATH", "embeddings.json")

# When set, server logs are written to this file in addition to stderr.
# Useful when the server is run by an MCP client (e.g. Claude Desktop) that
# hides stderr — tail -f this file to observe live logs.
GEMORY_LOG_FILE: str = os.getenv("GEMORY_LOG_FILE", "")


# ---------------------------------------------------------------------------
# DeepSeek / LLM
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_CHAT_MODEL: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "https://api.deepseek.com/v1")
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "deepseek-embed")


# ---------------------------------------------------------------------------
# Similarity thresholds (cosine similarity, range 0.0 – 1.0)
# ---------------------------------------------------------------------------

# Above this threshold, two facts are considered the same and are merged
# (duplicate detection). Start conservative/high: false duplicates are easy
# to spot and fix; wrongly-merged distinct facts are hard to notice.
DEDUP_THRESHOLD: float = float(os.getenv("DEDUP_THRESHOLD", "0.92"))

# Above this threshold (but below DEDUP_THRESHOLD), a new node is created
# AND an edge is added to link the close-but-distinct facts.
EDGE_THRESHOLD: float = float(os.getenv("EDGE_THRESHOLD", "0.75"))


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------

# Initial confidence value assigned to a newly created fact node.
CONFIDENCE_BASE: float = float(os.getenv("CONFIDENCE_BASE", "1.0"))

# Confidence added on each new-source corroboration (a previously unseen
# source_id confirming the same fact).
CONFIDENCE_INCREMENT: float = float(os.getenv("CONFIDENCE_INCREMENT", "1.0"))


# ---------------------------------------------------------------------------
# Dreamer (offline consolidation)
# ---------------------------------------------------------------------------

# Minimum cosine similarity for two nodes to be connected in the cluster graph.
# Start around EDGE_THRESHOLD neighbourhood (~0.75); tune on real data.
CLUSTER_SIM_THRESHOLD: float = float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.75"))

# Smallest cluster worth creating an abstraction for.
# Abstracting over one or two facts produces noise, not insight.
MIN_CLUSTER_SIZE: int = int(os.getenv("MIN_CLUSTER_SIZE", "3"))

# Largest cluster before forcing a split at a higher similarity threshold.
# An over-large cluster produces a uselessly-broad abstraction.
MAX_CLUSTER_SIZE: int = int(os.getenv("MAX_CLUSTER_SIZE", "12"))

# Jaccard overlap threshold for "same abstraction already exists".
# If a proposed abstraction covers >= this fraction of an existing
# abstraction's members, update the existing one rather than creating
# a parallel duplicate.
ABSTRACTION_OVERLAP: float = float(os.getenv("ABSTRACTION_OVERLAP", "0.8"))

# Safety cap on recursion depth — stop if the hierarchy reaches this many
# levels. Report if hit so a human can decide whether to increase.
MAX_LEVELS: int = int(os.getenv("MAX_LEVELS", "6"))


# ---------------------------------------------------------------------------
# Topic tagging
# ---------------------------------------------------------------------------

# Cosine threshold for matching proposed topic strings to existing canonical
# topic nodes. Set high (0.80-0.85) because topic strings are short and
# subject-focused — genuine same-topic phrasings score high.
TOPIC_MATCH_THRESHOLD: float = float(os.getenv("TOPIC_MATCH_THRESHOLD", "0.85"))
