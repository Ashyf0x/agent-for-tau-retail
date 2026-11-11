import asyncio
import sys
import tempfile
from pathlib import Path

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel
from agentcore.storage import Store
from agentcore.tools.mcp import connect_stdio


async def main():
    async with connect_stdio(
        sys.executable,
        [str(Path(__file__).with_name("mcp_server.py"))],
        contracts={"add": {"read_only": True}},
    ) as tools:
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "agent.sqlite3")
            try:
                model = ScriptedModel(
                    [
                        ModelResponse(
                            tool_calls=[ToolCall("fixture", "add", {"a": 4, "b": 5})],
                            finish_reason="tool_calls",
                        ),
                        ModelResponse("4 + 5 = 9（本地 MCP 工具已返回）。"),
                    ]
                )
                result = await Agent(model, tools, store).run("Calculate 4 + 5")
                print(result.to_dict())
            finally:
                store.close()


if __name__ == "__main__":
    asyncio.run(main())
