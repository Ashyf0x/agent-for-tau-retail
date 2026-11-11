import json
import sqlite3

import pytest

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.errors import ToolFailure
from agentcore.core.model import ScriptedModel
from agentcore.observability.trajectory import export, reconstruct
from agentcore.tools import ToolRegistry, ToolSpec


async def recorded_run(store):
    attempts = []

    async def read(n):
        attempts.append(n)
        if len(attempts) == 1:
            raise ToolFailure("Temporary", "transient", outcome_known=False)
        return {"evidence": "original ``` evidence", "n": n}

    def response(arguments):
        return ModelResponse(
            tool_calls=[ToolCall("provider-id", "read", arguments)], finish_reason="tool_calls"
        )

    model = ScriptedModel([response({"n": "invalid"}), response({"n": 2}), ModelResponse("Done")])
    tools = ToolRegistry(
        [
            ToolSpec(
                "read",
                "Read",
                {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
                read,
                read_only=True,
                max_attempts=2,
            )
        ]
    )
    return await Agent(model, tools, store).run("Read evidence", user_id="alice")


async def test_readonly_exports_retain_actual_events_associations_and_evidence(store, tmp_path):
    result = await recorded_run(store)
    database = tmp_path / "agent.sqlite3"
    before = database.read_bytes()
    raw_events = [tuple(row) for row in store.db.execute("SELECT * FROM events")]
    trajectory = export(database, result.run_id, tmp_path / "trajectory.json", user_id="alice")
    export(database, result.run_id, tmp_path / "trajectory.md", format="markdown")
    assert database.read_bytes() == before
    assert [tuple(row) for row in store.db.execute("SELECT * FROM events")] == raw_events
    assert json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8")) == trajectory
    assert trajectory["messages"][0]["data"]["content"] == "Read evidence"
    assert len(trajectory["model_calls"]) == 3
    assert all(
        call["provider_id"] == "provider-id" and call["model_call_id"]
        for call in trajectory["tool_calls"]
    )
    second = trajectory["tool_calls"][1]
    assert len(second["attempts"]) == 2 and second["result_messages"] and second["observations"]
    assert trajectory["terminal_events"][0]["data"]["status"] == "completed"
    assert "original ``` evidence" in (tmp_path / "trajectory.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        export(database, result.run_id, tmp_path / "trajectory.json")
    with pytest.raises(KeyError):
        reconstruct(database, result.run_id, user_id="bob")


def test_missing_records_are_explicit_and_no_database_is_created(store, tmp_path):
    run = store.begin("u", "s", "pending input")
    store.event(run, "model_request", {"model_call_id": "pending", "body": {"messages": []}})
    trajectory = reconstruct(tmp_path / "agent.sqlite3", run)
    assert trajectory["run"]["status"] == "running" and trajectory["terminal_events"] == []
    assert any("terminal response/error not recorded" in gap for gap in trajectory["not_recorded"])
    assert any("run_end not recorded" in gap for gap in trajectory["not_recorded"])
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        reconstruct(missing, "unknown")
    assert not missing.exists()


async def test_legacy_correlation_is_not_guessed(store, tmp_path):
    result = await recorded_run(store)
    with store.transaction() as db:
        for row in db.execute(
            "SELECT seq,body FROM events WHERE kind='call_correlation'"
        ).fetchall():
            body = json.loads(row["body"])
            body.pop("model_call_id")
            db.execute("UPDATE events SET body=? WHERE seq=?", (json.dumps(body), row["seq"]))
    trajectory = reconstruct(tmp_path / "agent.sqlite3", result.run_id)
    assert all(call["model_call_id"] is None for call in trajectory["tool_calls"])
    assert any("no association guessed" in gap for gap in trajectory["not_recorded"])
