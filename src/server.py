"""MCP server for the Gemory memory system.

Exposes two tools over stdio transport:

* ``remember`` -- extract durable facts from a conversation transcript and
  store them in the memory graph.
* ``recall`` -- search the memory graph for facts relevant to a query.
"""

import asyncio
import logging
import os
import signal
import sys
import traceback

# Ensure the project root is on sys.path so `from src.*` imports resolve
# when running this script directly (e.g. `uv run src/server.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.config import (
    EMBEDDINGS_PATH,
    GEMORY_LOG_FILE as _RAW_GEMORY_LOG_FILE,
    MEMORY_PATH as _RAW_MEMORY_PATH,
)
from src.graph import GraphStore
from src.llm import extract_facts
from src.extractor import compute_source_id, store_facts
from src.recall import recall, traverse_recall

# Resolve relative paths against the project root so the server works
# regardless of what the calling process (e.g. Claude Desktop) sets as cwd.
def _resolve_path(path: str) -> str:
    """If *path* is relative, make it absolute against the project root."""
    if os.path.isabs(path) or not path:
        return path
    return os.path.join(_PROJECT_ROOT, path)

MEMORY_PATH = _resolve_path(_RAW_MEMORY_PATH)
GEMORY_LOG_FILE = _resolve_path(_RAW_GEMORY_LOG_FILE)

# ---------------------------------------------------------------------------
# Logging (always to stderr so stdout is reserved for MCP protocol)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

if GEMORY_LOG_FILE:
    _fh = logging.FileHandler(GEMORY_LOG_FILE)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    logging.getLogger("src.server").info("Logging to %s", GEMORY_LOG_FILE)
logger = logging.getLogger("src.server")

# ---------------------------------------------------------------------------
# Server / graph (module-level — initialised once in main())
# ---------------------------------------------------------------------------

server = Server("src")
graph = GraphStore(MEMORY_PATH)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Return the list of available tools."""
    return [
        types.Tool(
            name="remember",
            description=(
                "Extract durable facts from a conversation transcript and "
                "store them in the memory graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": (
                            "Full conversation transcript to extract facts from."
                        ),
                    },
                    "conversation_name": {
                        "type": "string",
                        "description": (
                            "Optional human-readable name for this conversation "
                            "(for provenance legibility only)."
                        ),
                    },
                },
                "required": ["transcript"],
            },
        ),
        types.Tool(
            name="recall",
            description=(
                "Search the memory graph for facts relevant to a query. "
                "Use this at the start of a conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for relevant memories.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "For 'flat' method: maximum number of results "
                            "(default: 5). For 'traverse' method: a budget -- "
                            "the largest set of facts to return. The pruned "
                            "region is returned unranked, grouped by branch."
                        ),
                        "default": 5,
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            "Recall method: 'flat' (vector similarity, default) "
                            "or 'traverse' (hierarchical descent)."
                        ),
                        "enum": ["flat", "traverse"],
                        "default": "flat",
                    },
                    "relation_expansion": {
                        "type": "boolean",
                        "description": (
                            "Follow relates_to edges one hop from kept branches "
                            "(default: true). Only for 'traverse' method."
                        ),
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    """Dispatch tool calls to the appropriate handler."""
    if name == "remember":
        return await _handle_remember(arguments)
    elif name == "recall":
        return await _handle_recall(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_remember(arguments: dict) -> list[types.TextContent]:
    """Extract facts from a transcript and store them in the graph."""
    transcript = arguments["transcript"]
    conversation_name = arguments.get("conversation_name")

    logger.info(
        "Remember called with transcript (%d chars)", len(transcript),
    )

    try:
        facts = extract_facts(transcript)
        topics_count = sum(1 for f in facts if f.get("topics"))
        logger.info("Extracted %d facts (%d with topics)", len(facts), topics_count)

        source_id = compute_source_id(transcript)
        logger.info("Computed source_id=%s", source_id)

        summary = store_facts(facts, source_id, conversation_name, graph)

        msg = (
            f"Stored {summary['facts_extracted']} facts "
            f"(new: {summary['new_nodes']}, "
            f"corroborated: {summary['corroborated']}, "
            f"skipped: {summary['skipped']})."
        )
        logger.info("Remember result: %s", msg)
        return [types.TextContent(type="text", text=msg)]

    except Exception:
        logger.exception("Remember failed")
        return [
            types.TextContent(
                type="text",
                text=f"Error during remember: {traceback.format_exc()}",
            ),
        ]


async def _handle_recall(arguments: dict) -> list[types.TextContent]:
    """Search the graph for facts relevant to a query."""
    query = arguments["query"]
    top_k = arguments.get("top_k", 5)
    method = arguments.get("method", "flat")

    logger.info(
        "Recall called with query (%d chars), top_k=%d, method=%s",
        len(query), top_k, method,
    )

    try:
        if method == "traverse":
            relation_expansion = arguments.get("relation_expansion", True)
            result_text, metrics = traverse_recall(
                query, graph, relation_expansion=relation_expansion,
            )
            logger.info(
                "Traverse recall: %d layers, %d facts, %d kept/%d pruned",
                metrics["layers_visited"], metrics["facts_collected"],
                metrics["branches_kept"], metrics["branches_pruned"],
            )
            return [types.TextContent(type="text", text=result_text)]
        else:
            result = recall(query, graph, top_k)
            return [types.TextContent(type="text", text=result)]

    except Exception:
        logger.exception("Recall failed")
        return [
            types.TextContent(
                type="text",
                text=f"Error during recall: {traceback.format_exc()}",
            ),
        ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main_async() -> None:
    """Load the memory graph and start the MCP server over stdio."""
    logger.info("Starting Gemory MCP server")

    try:
        graph.load()
        node_count = len(graph.all_nodes())
        logger.info("Loaded memory graph with %d nodes", node_count)
    except Exception as e:
        logger.error("Failed to load memory graph: %s", e)
        raise

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    try:
        async with stdio_server() as (read_stream, write_stream):
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                ),
            )

            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if shutdown_event.is_set():
                logger.info("Received shutdown signal, stopping server...")
                server_task.cancel()
                try:
                    await asyncio.wait_for(server_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            else:
                # Server finished on its own — cancel the shutdown watcher
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    except asyncio.CancelledError:
        pass
    except BaseExceptionGroup:
        # Suppress cascading cleanup errors from the stdio transport
        # (e.g. anyio ExceptionGroup on stdin closure during shutdown).
        logger.debug("Suppressed cleanup exception during shutdown")
    finally:
        logger.info("Gemory server stopped")


def main():
    """Sync entry point for console script and direct invocation."""
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
