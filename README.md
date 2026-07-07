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

Optional variables:

- `GEMORY_LOG_FILE` — write server logs to this file in addition to stderr.
  Critical when running under an MCP client that hides stderr (e.g. Claude Desktop).
  Set to an absolute path like `/Users/you/gemory/gemory.log` and `tail -f` it.

See `.env.example` for all configurable options (thresholds, model names, paths).

## Run

### Stdio (default)

```bash
uv run gemory/server.py
```

After running `uv sync --extra dev` the console script `uv run gemory` also works.

The server loads `memory.json` and `embeddings.json` from the `data/` directory.
If they don't exist, it starts with an empty graph.

### HTTP SSE (experimental)

```bash
uv run gemory/server.py --http
```

Starts on `http://127.0.0.1:8765/sse`.
Use `--host` and `--port` to change.
Note: Claude Desktop only accepts HTTPS URLs, so this transport is best for other MCP clients or local debugging.

## MCP client registration

### Claude Desktop

In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemory": {
      "command": "/absolute/path/to/gemory/.venv/bin/python",
      "args": [
        "/absolute/path/to/gemory/gemory/server.py"
      ],
      "env": {
        "GEMORY_LOG_FILE": "/absolute/path/to/gemory/gemory.log"
      }
    }
  }
}
```

Replace `/absolute/path/to/gemory` with the actual path (e.g. `/Users/you/Projects/gemory`).
Then restart Claude Desktop and `tail -f gemory.log` to watch live extraction logs.

The config file location:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Note: the `command` path above uses Unix-style absolute paths — adjust for Windows
(e.g. `C:\Users\you\gemory\.venv\Scripts\python.exe`).

### Other MCP clients

For clients that support URL-based registration, run the server in HTTP SSE mode:

```bash
uv run gemory/server.py --http
```

Then configure the client URL to `http://127.0.0.1:8765/sse`.

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

The graph is stored as two JSON files under `data/`:

- **`data/memory.json`** — the graph structure: nodes (facts with content, confidence, provenance timestamps) and edges (weighted relationships with relation types)
- **`data/embeddings.json`** — a sidecar mapping node IDs to embedding vectors (kept separate so `memory.json` stays human-readable and diff-friendly)

Both are gitignored — they contain runtime data and potentially sensitive extracted facts.

## Visualization

Render an interactive HTML graph of the stored facts:

```bash
uv run python scripts/visualize.py data/memory.json -o graph.html
open graph.html
```

Nodes are colored by confidence, sized by confidence, and show full fact content on hover.
Edges display relationship type and weight.

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

All 41 deterministic tests use stubbed embeddings — no network or API keys needed.

The test harness includes frozen transcript fixtures, controlled vector construction
via Cholesky decomposition, and a graph snapshot/diff helper for declarative assertions.
See [TESTING.md](docs/TESTING.md) for the full harness documentation.

Live sanity tests (real API calls, skipped by default):

```bash
GEMORY_LIVE=1 uv run pytest tests/live/ -v -s
```

### Architecture

```
gemory/
├── server.py       # MCP server: remember + recall tools (stdio + SSE)
├── graph.py        # GraphStore: in-memory DiGraph + JSON persistence
├── llm.py          # DeepSeek wrapper: fact extraction + embeddings
├── extractor.py    # Store algorithm: source_id, dedup, edge creation
├── recall.py       # Query → embed → similarity search → formatted results
├── config.py       # Environment-driven constants and thresholds
├── models.py       # Node and Edge dataclasses
scripts/
├── visualize.py    # pyvis interactive graph visualizer
data/                # Runtime data (gitignored)
├── memory.json
└── embeddings.json
docs/
├── DREAMER.md       # Dreamer documentation
├── TESTING.md       # Test harness documentation
└── instructions/    # Implementation specs
tests/
├── fixtures/       # Frozen transcripts + expectation files
├── live/           # Live sanity suite (GEMORY_LIVE=1, manual only)
└── *.py            # 41 deterministic unit + integration tests
```

Strict isolation rules:

- `networkx` imports only in `graph.py`
- LLM/embedding API calls only in `llm.py`
- All other modules depend on these wrappers, never on the libraries directly
