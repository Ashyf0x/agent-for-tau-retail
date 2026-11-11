import argparse
import asyncio
import tempfile
from pathlib import Path

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel
from agentcore.memory import Candidate
from agentcore.observability import run_report
from agentcore.storage import Store
from agentcore.tools import ToolRegistry, ToolSpec


async def demo(path):
    store = Store(path)

    async def add(a, b):
        return a + b

    tools = ToolRegistry(
        [
            ToolSpec(
                "add",
                "Add two numbers",
                {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                add,
                read_only=True,
            )
        ]
    )
    model = ScriptedModel(
        [
            ModelResponse("我会记住你提供的偏好候选。"),
            ModelResponse(
                tool_calls=[ToolCall("fixture", "add", {"a": 2, "b": 3})],
                finish_reason="tool_calls",
            ),
            ModelResponse("2 + 3 = 5。"),
        ]
    )
    try:
        agent = Agent(model, tools, store)
        await agent.run("我偏好中文回答。", user_id="demo", session_id="preferences")
        source = store.messages("demo", "preferences")[0]["id"]
        agent.memory.commit(
            "demo", [Candidate("language", "用户偏好中文回答", [source], "preference")]
        )
        result = await agent.run("用中文计算 2 + 3。", user_id="demo", session_id="calculation")
        print(result.answer)
        print(run_report(store, result.run_id))
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Offline scripted Agent demo; no model endpoint or .env access"
    )
    parser.add_argument("--database", type=Path, help="Optional persistent SQLite file")
    args = parser.parse_args()
    if args.database:
        asyncio.run(demo(args.database))
    else:
        with tempfile.TemporaryDirectory(prefix="agent-runtime-") as directory:
            asyncio.run(demo(Path(directory) / "agent.sqlite3"))


if __name__ == "__main__":
    main()
