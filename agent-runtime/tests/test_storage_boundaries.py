import asyncio
import json
import sqlite3

import pytest

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.errors import ToolFailure
from agentcore.core.model import ScriptedModel
from agentcore.core.types import StorageError
from agentcore.tools import ToolRegistry, ToolResult, ToolSpec


def two_calls():
    return ModelResponse(tool_calls=[ToolCall("a", "write", {}), ToolCall("b", "write", {})], finish_reason="tool_calls")


async def test_archive_database_failure_preserves_known_write_and_stops_batch(store):
    calls = []
    async def write():
        calls.append(1)
        return {"receipt": "committed-receipt"}
    def authorize(action, table, *args):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == "observations" else sqlite3.SQLITE_OK
    store.db.set_authorizer(authorize)
    try:
        with pytest.raises(StorageError):
            await Agent(ScriptedModel([two_calls()]), ToolRegistry([
                ToolSpec("write", "Write", {"type": "object"}, write)
            ]), store).run("Write")
    finally:
        store.db.set_authorizer(None)
    assert calls == [1]
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "failed"
    assert store.unresolved("default", "default") == []
    results = [json.loads(r["message"]["content"]) for r in store.messages("default", "default") if r["message"]["role"] == "tool"]
    assert results[0]["status"] == "returned" and results[0]["value"]["receipt"] == "committed-receipt"
    assert results[1]["status"] == "not_dispatched"


async def test_query_result_persistence_failure_propagates_and_never_replays(store, monkeypatch):
    writes = []
    async def write():
        writes.append(1)
        raise ToolFailure("Response lost", "transient", outcome_known=False)
    async def query(operation_id):
        return ToolResult("returned", ok=True, value={"receipt": "known"})
    original = store.attempt
    def fail_resolution(*args):
        if args[-2] == "returned":
            raise StorageError("Resolution could not be saved")
        return original(*args)
    monkeypatch.setattr(store, "attempt", fail_resolution)
    with pytest.raises(StorageError):
        await Agent(ScriptedModel([two_calls()]), ToolRegistry([
            ToolSpec("write", "Write", {"type": "object"}, write, query_status=query)
        ]), store).run("Write")
    assert writes == [1]
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "failed"


async def test_cancellation_after_archive_keeps_returned_write_evidence(store, monkeypatch):
    calls = []
    async def write():
        calls.append(1)
        return {"receipt": "known"}
    original = store.archive
    def archive_and_cancel(*args):
        ref = original(*args)
        asyncio.current_task().cancel()
        return ref
    monkeypatch.setattr(store, "archive", archive_and_cancel)
    # A later await receives cancellation; the synchronous archive itself is atomic.
    class SlowModel(ScriptedModel):
        async def complete(self, request):
            if self.requests:
                await asyncio.Event().wait()
            return await super().complete(request)
    response = ModelResponse(tool_calls=[ToolCall("a", "write", {})], finish_reason="tool_calls")
    task = asyncio.create_task(Agent(SlowModel([response]), ToolRegistry([
        ToolSpec("write", "Write", {"type": "object"}, write)
    ]), store).run("Write"))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [1] and store.unresolved("default", "default") == []
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "cancelled"
    body = json.loads(store.messages("default", "default")[-1]["message"]["content"])
    assert body["status"] == "returned" and body["value"]["receipt"] == "known"
