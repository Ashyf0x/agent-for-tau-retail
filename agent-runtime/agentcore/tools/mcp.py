"""Official MCP SDK transport, adapted into the same ToolSpec/ToolResult contract."""

from contextlib import asynccontextmanager
from datetime import timedelta

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from agentcore.core.errors import ToolFailure

from .executor import ToolRegistry, ToolResult, ToolSpec


@asynccontextmanager
async def connect_stdio(command, args, *, contracts=None, env=None, timeout=10):
    """Contracts are trusted application configuration, not server/model annotations.

    No SDK retry layer. Only stdio is supported here; credentials are not inherited wholesale.
    Keep the context open for the entire Agent run and close it in the creating task.
    """
    contracts = contracts or {}
    parameters = StdioServerParameters(command=command, args=list(args), env=env)
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=timeout)
        ) as session:
            await session.initialize()
            registry = ToolRegistry()
            cursor = None
            seen = set()
            while True:
                listing = await session.list_tools(cursor=cursor)

                def make_handler(name):
                    async def handler(**arguments):
                        try:
                            result = await session.call_tool(name, arguments)
                        except (McpError, BrokenResourceError, ClosedResourceError, EndOfStream):
                            raise ToolFailure(
                                "MCP transport/protocol failure; outcome unknown",
                                "transient",
                                outcome_known=False,
                            ) from None
                        if result.isError:
                            # Do not expose raw server exception text. Returned errors remain evidence.
                            return ToolResult(
                                "returned",
                                value=result.model_dump(
                                    mode="json", by_alias=True, exclude_none=True
                                ),
                                error="MCP tool returned an error",
                                category="business",
                            )
                        return result.model_dump(mode="json", by_alias=True, exclude_none=True)

                    return handler

                for remote in listing.tools:
                    contract = contracts.get(remote.name, {})
                    registry.register(
                        ToolSpec(
                            remote.name,
                            remote.description or "",
                            remote.inputSchema,
                            make_handler(remote.name),
                            timeout=timeout,
                            **contract,
                        )
                    )
                cursor = listing.nextCursor
                if cursor is None:
                    break
                if cursor in seen:
                    raise RuntimeError("MCP repeated a pagination cursor")
                seen.add(cursor)
            yield registry
