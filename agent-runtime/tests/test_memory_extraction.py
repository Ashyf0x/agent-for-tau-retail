import asyncio
import json
import sqlite3

import pytest

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel
from agentcore.core.types import StorageError
from agentcore.memory import Candidate, Memory
from agentcore.memory.extraction import ExtractionConfig, MemoryExtractor
from agentcore.tools import ToolRegistry


async def turn(store, text, *, user="alice", session="chat"):
    return await Agent(
        ScriptedModel(
            [ModelResponse("Noted", provider_metadata={"reasoning_content": "not-a-user-fact"})]
        ),
        ToolRegistry(),
        store,
    ).run(text, user_id=user, session_id=session)


def proposal(*, key="drink", content="Prefers tea", action="upsert", source=None):
    def respond(payload):
        messages = json.loads(payload["messages"][-1]["content"])["new_user_messages"]
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    "proposal",
                    "propose_memory",
                    {
                        "candidates": [
                            {
                                "key": key,
                                "content": content,
                                "kind": "preference",
                                "action": action,
                                "sources": [source or messages[-1]["id"]],
                            }
                        ]
                    },
                )
            ],
            finish_reason="tool_calls",
        )

    return respond


def empty_proposal(payload):
    return ModelResponse(
        tool_calls=[ToolCall("proposal", "propose_memory", {"candidates": []})],
        finish_reason="tool_calls",
    )


def progress(store):
    return [tuple(row) for row in store.db.execute("SELECT * FROM memory_progress")]


async def test_explicit_extraction_is_sourced_scoped_and_not_repeated(store):
    await turn(store, "I prefer tea")
    await turn(store, "Other user's preference", user="bob")
    model = ScriptedModel([proposal()])
    memory = Memory(store)
    extractor = MemoryExtractor(memory, model)
    result = await extractor.extract(user_id="alice", session_id="chat")
    assert result.status == "completed" and result.changed == 1
    assert memory.retrieve("alice", "tea")[0]["sources"] == [
        store.messages("alice", "chat")[0]["id"]
    ]
    assert memory.retrieve("bob", "tea") == []
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "skipped"
    assert len(model.requests) == 1
    assert "not-a-user-fact" not in str(model.requests) and "Other user's preference" not in str(
        model.requests
    )
    row = store.db.execute("SELECT status,body FROM memory_batches").fetchone()
    assert row[0] == "completed" and json.loads(row[1])["purpose"] == "memory"


async def test_default_off_and_completed_turn_trigger(store):
    first = await turn(store, "I prefer tea")
    model = ScriptedModel([proposal()])
    extractor = MemoryExtractor(Memory(store), model)
    assert (
        await extractor.after_run(first, user_id="alice", session_id="chat")
    ).status == "disabled"
    assert model.requests == [] and progress(store) == []
    enabled = MemoryExtractor(
        Memory(store),
        model,
        config=ExtractionConfig(enabled=True, every_completed_turns=2, batch_turns=2),
    )
    assert (await enabled.after_run(first, user_id="alice", session_id="chat")).status == "skipped"
    second = await turn(store, "Tea remains my preference")
    result = await enabled.after_run(second, user_id="alice", session_id="chat")
    assert result.status == "completed" and result.processed_turns == 2
    assert (await enabled.after_run(second, user_id="alice", session_id="chat")).status == "skipped"
    assert len(model.requests) == 1


async def test_explicit_batches_are_bounded_and_empty_proposals_advance(store):
    for index in range(3):
        await turn(store, f"Temporary request {index}")
    model = ScriptedModel([empty_proposal, empty_proposal])
    extractor = MemoryExtractor(
        Memory(store), model, config=ExtractionConfig(every_completed_turns=2, batch_turns=2)
    )
    first = await extractor.extract(user_id="alice", session_id="chat")
    second = await extractor.extract(user_id="alice", session_id="chat")
    assert [first.processed_turns, second.processed_turns] == [2, 1]
    assert first.changed == second.changed == 0
    assert store.db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "skipped"


async def test_uncompleted_runs_do_not_trigger_or_supply_sources(store):
    from agentcore import AgentConfig

    failed = await Agent(
        ScriptedModel([ModelResponse("", finish_reason="length")]),
        ToolRegistry(),
        store,
        config=AgentConfig(max_protocol_corrections=0),
    ).run("temporary", user_id="alice", session_id="chat")
    model = ScriptedModel([])
    extractor = MemoryExtractor(
        Memory(store), model, config=ExtractionConfig(enabled=True, every_completed_turns=1)
    )
    assert (
        await extractor.after_run(failed, user_id="alice", session_id="chat")
    ).status == "skipped"
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "skipped"
    assert model.requests == []


async def test_invalid_foreign_sources_fail_without_progress_then_retry_once(store):
    await turn(store, "I prefer tea")
    await turn(store, "Foreign", user="bob")
    foreign = store.messages("bob", "chat")[0]["id"]
    model = ScriptedModel([proposal(source=foreign), proposal()])
    extractor = MemoryExtractor(Memory(store), model)
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "failed"
    assert progress(store) == [] and Memory(store).retrieve("alice", "tea") == []
    assert (await extractor.extract(user_id="alice", session_id="chat")).changed == 1
    assert store.db.execute("SELECT count(*) FROM memory_changes").fetchone()[0] == 1


async def test_duplicate_content_is_not_inserted_under_multiple_keys(store):
    await turn(store, "I prefer tea")

    def duplicates(payload):
        response = proposal()(payload)
        first = response.tool_calls[0].arguments["candidates"][0]
        response.tool_calls[0].arguments["candidates"].append({**first, "key": "duplicate"})
        return response

    result = await MemoryExtractor(Memory(store), ScriptedModel([duplicates])).extract(
        user_id="alice", session_id="chat"
    )
    assert result.changed == 1
    assert store.db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


async def test_correction_and_delete_need_new_evidence_and_trusted_key_approval(store):
    await turn(store, "I prefer tea")
    model = ScriptedModel(
        [
            proposal(),
            proposal(content="Prefers coffee"),
            proposal(content="Prefers coffee"),
            proposal(content="", action="delete"),
            proposal(content="", action="delete"),
        ]
    )
    memory = Memory(store)
    extractor = MemoryExtractor(memory, model)
    await extractor.extract(user_id="alice", session_id="chat")
    await turn(store, "Actually, I prefer coffee now")
    saved = progress(store)
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "failed"
    assert progress(store) == saved and memory.retrieve("alice", "tea")
    assert (
        await extractor.extract(user_id="alice", session_id="chat", corrections={"drink"})
    ).changed == 1
    assert memory.retrieve("alice", "coffee") and not memory.retrieve("alice", "tea")
    await turn(store, "Forget my drink preference")
    saved = progress(store)
    assert (await extractor.extract(user_id="alice", session_id="chat")).status == "failed"
    assert progress(store) == saved
    assert (
        await extractor.extract(user_id="alice", session_id="chat", deletions={"drink"})
    ).changed == 1
    assert memory.retrieve("alice", "coffee") == []


async def test_progress_write_failure_rolls_back_memory_and_retry_does_not_duplicate(store):
    await turn(store, "I prefer tea")
    extractor = MemoryExtractor(Memory(store), ScriptedModel([proposal(), proposal()]))

    def authorize(action, table, *args):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_INSERT and table == "memory_progress"
            else sqlite3.SQLITE_OK
        )

    store.db.set_authorizer(authorize)
    try:
        with pytest.raises(StorageError):
            await extractor.extract(user_id="alice", session_id="chat")
    finally:
        store.db.set_authorizer(None)
    assert progress(store) == []
    assert store.db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert store.db.execute("SELECT count(*) FROM memory_changes").fetchone()[0] == 0
    assert (await extractor.extract(user_id="alice", session_id="chat")).changed == 1
    assert store.db.execute("SELECT count(*) FROM memory_changes").fetchone()[0] == 1


async def test_cancelled_extraction_never_advances_or_changes_agent_terminal(store):
    result = await turn(store, "I prefer tea")
    started = asyncio.Event()

    class SlowModel(ScriptedModel):
        async def complete(self, request):
            started.set()
            await asyncio.Event().wait()

    extractor = MemoryExtractor(Memory(store), SlowModel([]))
    task = asyncio.create_task(extractor.extract(user_id="alice", session_id="chat"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert progress(store) == []
    assert store.db.execute("SELECT status FROM memory_batches").fetchone()[0] == "cancelled"
    assert (
        store.db.execute("SELECT status FROM runs WHERE id=?", (result.run_id,)).fetchone()[0]
        == "completed"
    )


async def test_two_extractors_cannot_apply_same_batch_twice(store):
    await turn(store, "I prefer tea")
    ready = asyncio.Event()
    count = 0

    class GateModel(ScriptedModel):
        async def complete(self, request):
            nonlocal count
            count += 1
            if count == 2:
                ready.set()
            await ready.wait()
            return await super().complete(request)

    extractors = [MemoryExtractor(Memory(store), GateModel([proposal()])) for _ in range(2)]
    results = await asyncio.gather(
        *(e.extract(user_id="alice", session_id="chat") for e in extractors)
    )
    assert sorted(r.status for r in results) == ["completed", "failed"]
    assert store.db.execute("SELECT count(*) FROM memory_changes").fetchone()[0] == 1


async def test_old_evidence_cannot_overwrite_even_with_key_approval(store):
    await turn(store, "I prefer tea")
    source = store.messages("alice", "chat")[0]["id"]
    memory = Memory(store)
    memory.commit("alice", [Candidate("drink", "Prefers tea", [source], "preference")])
    await turn(store, "Actually, I prefer coffee")
    extractor = MemoryExtractor(
        memory, ScriptedModel([proposal(content="Prefers coffee", source=source)])
    )
    result = await extractor.extract(user_id="alice", session_id="chat", corrections={"drink"})
    assert result.status == "failed" and progress(store) == []
    assert memory.retrieve("alice", "tea") and not memory.retrieve("alice", "coffee")


async def test_protocol_failure_and_input_bound_never_advance(store):
    from agentcore.core.types import ContextBudgetError

    await turn(store, "I prefer tea")
    model = ScriptedModel([ModelResponse('{"candidates":[]}')])
    result = await MemoryExtractor(Memory(store), model).extract(user_id="alice", session_id="chat")
    assert result.status == "failed" and len(model.requests) == 1 and progress(store) == []
    bounded = ScriptedModel([])
    extractor = MemoryExtractor(
        Memory(store), bounded, config=ExtractionConfig(max_request_bytes=1)
    )
    with pytest.raises(ContextBudgetError):
        await extractor.extract(user_id="alice", session_id="chat")
    assert bounded.requests == [] and progress(store) == []
