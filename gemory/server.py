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

# Ensure the project root is on sys.path so `from gemory.*` imports resolve
# when running this script directly (e.g. `uv run gemory/server.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from gemory.config import GEMORY_LOG_FILE, MEMORY_PATH
from gemory.graph import GraphStore
from gemory.llm import extract_facts
from gemory.extractor import compute_source_id, store_facts
from gemory.recall import recall

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
    logging.getLogger("gemory.server").info("Logging to %s", GEMORY_LOG_FILE)
logger = logging.getLogger("gemory.server")

# ---------------------------------------------------------------------------
# Server / graph (module-level — initialised once in main())
# ---------------------------------------------------------------------------

server = Server("gemory")
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
                            "Maximum number of results to return (default: 5)."
                        ),
                        "default": 5,
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
        logger.info("Extracted %d facts", len(facts))

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

    logger.info(
        "Recall called with query (%d chars), top_k=%d", len(query), top_k,
    )

    try:
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
