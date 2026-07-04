# Gemory

Graph-based long-term memory for LLM agents via MCP (Model Context Protocol).

Gemory gives an LLM agent a persistent memory backed by a graph of atomic facts.
Facts are extracted from conversation transcripts, deduplicated via embedding
similarity, and connected into a knowledge graph.
Retrieval uses the same similarity search so the agent can recall relevant context at the start of a new conversation.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/coldsoul/gemory.git
cd gemory
uv sync --extra dev
```

### Configure

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Required variables in `.env`:

- `DEEPSEEK_API_KEY` — your DeepSeek API key for fact extraction
- `EMBEDDING_API_KEY` — your embeddings API key (can be the same key)

See `.env.example` for all configurable options (thresholds, model names, paths).

## Run

Start the MCP server on stdio:

```bash
uv run gemory
```

Or without the console script (works without `uv sync`):

```bash
uv run gemory/server.py
```

The server loads `memory.json` and `embeddings.json` from the current directory.
If they don't exist, it starts with an empty graph.

## MCP client registration

To register Gemory with an MCP client (Claude Desktop, etc.), add this to your
client's MCP server configuration:

```json
{
  "mcpServers": {
    "gemory": {
      "command": "uv",
      "args": [
        "run",
        "gemory/server.py"
      ],
      "cwd": "/absolute/path/to/gemory",
      "env": {
        "GEMORY_LOG_FILE": "/absolute/path/to/gemory/gemory.log"
      }
    }
  }
}
```

For Claude Desktop, this goes in `claude_desktop_config.json` (location varies by platform):

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

## Tools

### `remember`

Extract durable facts from a conversation transcript and store them in the memory graph.

**Input:**

| Parameter           | Type     | Required | Description                                       |
|---------------------|----------|----------|---------------------------------------------------|
| `transcript`        | string   | yes      | Full conversation transcript to extract facts from |
| `conversation_name` | string   | no       | Human-readable label for provenance tracking       |

**What it does:**
1. Calls DeepSeek to extract atomic, self-contained facts from the transcript
2. Computes a stable source identifier from the first exchange
3. For each fact: embeds it, checks for duplicates via cosine similarity, merges or creates a node, and connects related facts with edges
4. Saves the updated graph to disk

**Returns:** Summary with counts — facts extracted, new nodes created, existing facts corroborated, and duplicates skipped.

### `recall`

Search the memory graph for facts relevant to a query.
Call this at the start of a conversation to load context.

**Input:**

| Parameter | Type    | Required | Description                                   |
|-----------|---------|----------|-----------------------------------------------|
| `query`   | string  | yes      | Query to search for relevant memories           |
| `top_k`   | integer | no       | Maximum number of results (default: 5)          |

**Returns:** Ranked list of matching facts with similarity scores and confidence levels.

## Data model

The graph is stored as two JSON files:

- **`memory.json`** — the graph structure: nodes (facts with content, confidence, provenance timestamps) and edges (weighted relationships with relation types)
- **`embeddings.json`** — a sidecar mapping node IDs to embedding vectors (kept separate so `memory.json` stays human-readable and diff-friendly)

Both are gitignored — they contain runtime data and potentially sensitive extracted facts.

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

Tests use stubbed embeddings and mocked API calls — no network or API keys needed.

### Architecture

```
gemory/
├── server.py      # MCP server: remember + recall tools (stdio transport)
├── graph.py       # GraphStore: in-memory DiGraph + JSON persistence
├── llm.py         # DeepSeek wrapper: fact extraction + embeddings
├── extractor.py   # Store algorithm: source_id, dedup, edge creation
├── recall.py      # Query → embed → similarity search → formatted results
├── config.py      # Environment-driven constants and thresholds
├── models.py      # Node and Edge dataclasses
├── memory.json    # Human-readable graph (runtime, gitignored)
└── embeddings.json # Embedding sidecar (runtime, gitignored)
```

Strict isolation rules:

- `networkx` imports only in `graph.py`
- LLM/embedding API calls only in `llm.py`
- All other modules depend on these wrappers, never on the libraries directly
