"""Actual local stdio MCP server using the official SDK. No HTTP/model/credentials."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("agent-runtime-offline")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def fail() -> str:
    """Return a deliberate tool error for contract testing."""
    raise ValueError("fixture-private-error")


if __name__ == "__main__":
    server.run(transport="stdio")
