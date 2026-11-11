import json

import pytest

from agentcore import Agent, AgentConfig, ModelResponse, ToolCall
from agentcore.context import ContextManager
from agentcore.core.model import ScriptedModel
from agentcore.core.types import ContextBudgetError, dumps
from agentcore.memory import Candidate, Memory
from agentcore.tools import ToolRegistry, ToolSpec


def seed(store, user="alice", session="a", text="I prefer tea"):
    run = store.begin(user, session, text)
    from agentcore.core.types import RunResult

    store.finish(RunResult(run, status="completed"))
    return store.messages(user, session)[-1]["id"]


def test_memory_retrieval_correction_deletion_and_isolation(store):
    memory = Memory(store)
    source = seed(store)
    memory.commit("alice", [Candidate("drink", "I prefer tea", [source], "preference")])
    assert memory.retrieve("alice", "tea")[0]["sources"] == [source]
    assert memory.retrieve("alice", "unrelated bicycle") == []
    assert memory.retrieve("bob", "tea") == []
    correction = seed(store, session="b", text="Actually I prefer coffee")
    memory.commit("alice", [Candidate("drink", "I prefer coffee", [correction], "preference")])
    assert memory.retrieve("alice", "tea") == []
    assert memory.retrieve("alice", "coffee")[0]["updated_at"]
    memory.commit("alice", [Candidate("drink", "", [correction], action="delete")])
    assert memory.retrieve("alice", "coffee") == []
    assert "coffee" not in str(
        [tuple(r) for r in store.db.execute("SELECT body FROM memory_changes")]
    )
    assert store.source("alice", correction) is not None  # original session is a separate record


def test_memory_batch_is_atomic_and_sources_are_scoped(store):
    source = seed(store)
    memory = Memory(store)
    with pytest.raises(ValueError):
        memory.commit(
            "alice", [Candidate("valid", "tea", [source]), Candidate("invalid", "tea", ["missing"])]
        )
    assert memory.retrieve("alice", "tea") == []
    with pytest.raises(ValueError):
        memory.commit("bob", [Candidate("stolen", "tea", [source])])


def test_event_memory_can_reference_tool_evidence_but_preferences_cannot(store):
    run = store.begin("a", "s", "Inspect")
    tool = store.append(
        "a",
        "s",
        run,
        {
            "role": "tool",
            "tool_call_id": "x",
            "content": dumps({"status": "returned", "ok": False}),
        },
    )
    memory = Memory(store)
    memory.commit("a", [Candidate("inspection", "Inspection returned a failure", [tool], "event")])
    assert memory.retrieve("a", "inspection")[0]["semantic_status"] == "candidate_unverified"
    with pytest.raises(ValueError):
        memory.commit("a", [Candidate("preference", "Prefers failures", [tool], "preference")])


async def test_context_masks_view_preserves_raw_and_latest_requirement(store):
    async def document():
        return "Original evidence " * 2000

    tools = ToolRegistry(
        [
            ToolSpec(
                "document", "Read", {"type": "object", "properties": {}}, document, read_only=True
            )
        ]
    )
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall("id", "document", {})], finish_reason="tool_calls"),
            ModelResponse("Read it"),
            ModelResponse("Use your corrected limit"),
        ]
    )
    agent = Agent(model, tools, store, config=AgentConfig(max_request_bytes=6000))
    assert (await agent.run("Inspect document")).status == "completed"
    assert (await agent.run("Correction: budget is 17, not 70; no writes.")).status == "completed"
    request = model.requests[-1]
    assert "budget is 17, not 70; no writes" in str(request)
    assert "Full evidence archived" in str(request)
    messages = store.messages("default", "default")
    assert len(messages[2]["message"]["content"]) > 30000
    ref = json.loads(messages[2]["message"]["content"])["observation_ref"]
    page = store.read_observation("default", "default", ref, 0, 80)
    next_page = store.read_observation("default", "default", ref, page["next_offset"], 80)
    assert next_page["offset"] == 80
    with pytest.raises(KeyError):
        store.read_observation("other", "default", ref)
    with pytest.raises(KeyError):
        store.read_observation("default", "other", ref)


def test_summary_failure_preserves_records_and_current_request():
    records = [
        {"id": str(i), "message": {"role": "user", "content": f"Request {i}"}} for i in range(6)
    ]
    original = dumps(records)

    def broken(records):
        records.clear()
        raise RuntimeError("summary failed")

    manager = ContextManager("Rules", summary_builder=broken)
    request, info = manager.prepare(ScriptedModel([]), records, [], [], {}, 8000)
    assert dumps(records) == original and info["summary_failed"]
    assert "Request 5" in request.body.decode()


def test_failed_tool_summary_does_not_become_success():
    from agentcore.context.manager import history_summary

    summary = history_summary(
        [
            {
                "id": "source",
                "message": {"role": "tool", "content": dumps({"ok": False, "status": "unknown"})},
            }
        ]
    )
    assert summary[0]["ok"] is False and summary[0]["status"] == "unknown"


async def test_input_bound_prevents_send_without_truncating_current_user(store):
    model = ScriptedModel([ModelResponse("unused")])
    result = await Agent(
        model, ToolRegistry(), store, config=AgentConfig(max_request_bytes=100)
    ).run("Unabridged request" * 100)
    assert result.reason == "ContextBudgetError" and model.requests == []
    assert (
        store.messages("default", "default")[0]["message"]["content"] == "Unabridged request" * 100
    )


def test_context_rejects_orphan_result():
    with pytest.raises(ContextBudgetError):
        ContextManager.validate_groups([{"role": "tool", "tool_call_id": "missing"}])


async def test_retrieved_page_reaches_model_without_recursive_masking(store):
    ref = store.archive("default", "default", "original", {"text": "a" * 6000})
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall("page", "read_observation", {"ref": ref, "limit": 3000})],
                finish_reason="tool_calls",
            ),
            ModelResponse("Received page"),
        ]
    )
    result = await Agent(model, ToolRegistry(), store).run("Retrieve the evidence")
    body = json.loads(model.requests[-1]["messages"][-1]["content"])
    assert result.status == "completed" and len(body["value"]["text"]) == 3000
    assert "observation_ref" not in body
