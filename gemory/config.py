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
