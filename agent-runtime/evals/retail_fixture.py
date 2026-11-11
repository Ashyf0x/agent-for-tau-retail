"""Fixed interaction against public Retail data, not a benchmark task solution."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from agentcore import ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel
from agentcore.observability import run_report
from agentcore.storage import Store

from .retail import build_environment, load_retail, make_agent, scoring_status

USER_EMAIL = "noah.brown7922@example.com"  # public synthetic DB fixture, no task answer


def fixture_model():
    def lookup_from_observation(request):
        observation = json.loads(
            next(
                message["content"]
                for message in reversed(request["messages"])
                if message["role"] == "tool"
            )
        )
        # The second call derives its argument from the environment's actual first result.
        user_id = observation["value"]  # upstream preserves scalar strings without JSON quoting
        return ModelResponse(
            tool_calls=[ToolCall("fixture-2", "get_user_details", {"user_id": user_id})],
            finish_reason="tool_calls",
        )

    def reply_from_observation(request):
        observation = json.loads(request["messages"][-1]["content"])
        details = json.loads(observation["value"])
        return ModelResponse(f"Your account email is {details['email']}.")

    return ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall("fixture-1", "find_user_id_by_email", {"email": USER_EMAIL})],
                finish_reason="tool_calls",
            ),
            lookup_from_observation,
            reply_from_observation,
        ]
    )


async def run_fixture(root, database):
    lock, loaded, tasks = load_retail(root)
    environment = build_environment(loaded)
    before = environment.get_db_hash()
    store = Store(database)
    try:
        model = fixture_model()
        result = await make_agent(model, store, environment, loaded).run(
            f"My email is {USER_EMAIL}. Please find my account and confirm its email."
        )
        if result.status != "completed" or USER_EMAIL not in result.answer:
            raise AssertionError("Retail fixture did not complete through the Agent core")
        if before != environment.get_db_hash():
            raise AssertionError("Read-only fixture unexpectedly changed Retail DB")
        return {
            "kind": "offline_environment_wiring",
            "release": lock["release"],
            "commit": lock["commit"],
            "benchmark": lock["benchmark"],
            "language": "en",
            "domain": "retail",
            "available_tasks": len(tasks),
            "official_tools": len(environment.get_tools()),
            "fixture_tools": ["find_user_id_by_email", "get_user_details"],
            "read_only_db_unchanged": True,
            "agent_report": run_report(store, result.run_id),
            "real_model_calls": 0,
            "simulator_calls": 0,
            "judge_calls": 0,
            **scoring_status(),
        }
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(".vendor/tau2-bench"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="retail-fixture-") as directory:
        report = asyncio.run(run_fixture(args.checkout, Path(directory) / "fixture.sqlite3"))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
