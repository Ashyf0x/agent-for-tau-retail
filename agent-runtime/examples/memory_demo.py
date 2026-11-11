"""Explicit and completed-turn memory triggers, using only scripted models."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel
from agentcore.memory.extraction import ExtractionConfig, MemoryExtractor
from agentcore.storage import Store
from agentcore.tools import ToolRegistry


def propose(payload):
    source = json.loads(payload["messages"][-1]["content"])["new_user_messages"][-1]
    return ModelResponse(
        tool_calls=[
            ToolCall(
                "fixture",
                "propose_memory",
                {
                    "candidates": [
                        {
                            "key": "language",
                            "content": "User prefers Chinese responses",
                            "kind": "preference",
                            "action": "upsert",
                            "sources": [source["id"]],
                        }
                    ]
                },
            )
        ],
        finish_reason="tool_calls",
    )


async def demo(path):
    store = Store(path)
    try:
        agent = Agent(
            ScriptedModel([ModelResponse("Noted") for _ in range(3)]), ToolRegistry(), store
        )
        extractor = MemoryExtractor(
            agent.memory,
            ScriptedModel([propose, propose]),
            config=ExtractionConfig(enabled=True, every_completed_turns=2, batch_turns=2),
        )
        first = await agent.run("I prefer Chinese responses", user_id="demo", session_id="chat")
        explicit = await extractor.extract(user_id="demo", session_id="chat")
        second = await agent.run(
            "Chinese remains my preferred language", user_id="demo", session_id="chat"
        )
        below_threshold = await extractor.after_run(second, user_id="demo", session_id="chat")
        third = await agent.run(
            "Please keep answering in Chinese", user_id="demo", session_id="chat"
        )
        triggered = await extractor.after_run(third, user_id="demo", session_id="chat")
        assert first.status == "completed" and explicit.changed == 1
        assert below_threshold.status == "skipped" and triggered.changed == 0
        print(
            json.dumps(
                {
                    "explicit": asdict(explicit),
                    "below_threshold": asdict(below_threshold),
                    "triggered": asdict(triggered),
                    "memories": agent.memory.retrieve("demo", "Chinese"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.database:
        asyncio.run(demo(args.database))
    else:
        with TemporaryDirectory(prefix="memory-demo-") as directory:
            asyncio.run(demo(Path(directory) / "memory.sqlite3"))


if __name__ == "__main__":
    main()
