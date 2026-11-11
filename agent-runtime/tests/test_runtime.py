import asyncio
import json

import pytest

from agentcore import Agent, AgentConfig, ModelResponse, ToolCall
from agentcore.core.errors import ToolFailure
from agentcore.core.model import ScriptedModel
from agentcore.core.types import ModelError, SessionBusy, StorageError, Usage
from agentcore.observability import run_report
from agentcore.tools import ToolRegistry, ToolResult, ToolSpec

SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


def call(name="work", args=None, id="provider-id"):
    return ModelResponse(
        tool_calls=[ToolCall(id, name, {"n": 1} if args is None else args)],
        finish_reason="tool_calls",
    )


async def test_conversation_isolation_and_natural_answers(store):
    model = ScriptedModel(
        [ModelResponse("Which file?"), ModelResponse("Understood"), ModelResponse("Hello")]
    )
    agent = Agent(model, ToolRegistry(), store)
    first = await agent.run("Please help", user_id="alice")
    second = await agent.run("Use draft.txt", user_id="alice")
    await agent.run("Hi", user_id="bob")
    assert first.status == second.status == "completed"
    # No tool was registered, so no mutating call could have been dispatched. The runtime
    # reports that observation instead of leaving the field at its never-assigned default.
    assert first.goal_status == "no_state_change_attempted"
    assert "Please help" in str(model.requests[1])
    assert "Please help" not in str(model.requests[2])
    assert len(store.messages("alice", "default")) == 4


async def test_tool_evidence_and_program_owned_ids(store):
    async def work(n):
        return {"n": n, "text": "complete original evidence"}

    model = ScriptedModel([call(), ModelResponse("Done")])
    result = await Agent(
        model, ToolRegistry([ToolSpec("work", "work", SCHEMA, work, read_only=True)]), store
    ).run("Work")
    messages = [r["message"] for r in store.messages("default", "default")]
    assert result.status == "completed"
    assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"] != "provider-id"
    body = json.loads(messages[2]["content"])
    assert body["value"]["text"] == "complete original evidence"
    page = store.read_observation("default", "default", body["observation_ref"])
    assert "complete original evidence" in page["text"]
    report = run_report(store, result.run_id)
    assert report["model_calls"] == 2 and report["unterminated_model_calls"] == []


async def test_parameter_correction_has_no_dispatch_and_is_bounded(store):
    dispatched = []

    async def work(n):
        dispatched.append(n)

    model = ScriptedModel(
        [call(args={"n": "bad"}), call(args={"n": "bad"}), ModelResponse("unused")]
    )
    result = await Agent(
        model,
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work)]),
        store,
        config=AgentConfig(max_argument_corrections=1),
    ).run("Work")
    assert result.reason == "argument_correction_limit" and dispatched == []
    assert len(model.requests) == 2
    assert all(
        row[0] == "not_dispatched" for row in store.db.execute("SELECT status FROM attempts")
    )


async def test_read_only_retry_and_model_redecision_are_separate(store):
    attempts = []

    async def work(n):
        attempts.append(n)
        if len(attempts) < 3:
            raise ToolFailure("Temporary service failure", "transient", outcome_known=False)
        return n

    model = ScriptedModel([call(args={"n": "bad"}), call(), ModelResponse("Done")])
    result = await Agent(
        model,
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work, read_only=True, max_attempts=3)]),
        store,
    ).run("Work")
    report = run_report(store, result.run_id)
    assert result.status == "completed" and attempts == [1, 1, 1]
    assert report["automatic_retries"] == 2 and report["argument_corrections"] == 1


async def test_unknown_write_blocks_batch_and_future_turn(store):
    count = 0

    async def work(n):
        nonlocal count
        count += 1
        await asyncio.sleep(1)

    response = call()
    response.tool_calls.append(ToolCall("second", "work", {"n": 2}))
    model = ScriptedModel([response, ModelResponse("must not be called")])
    agent = Agent(
        model,
        ToolRegistry(
            [ToolSpec("work", "work", SCHEMA, work, timeout=0.01, max_attempts=3, idempotent=True)]
        ),
        store,
    )
    result = await agent.run("Write")
    again = await agent.run("Try again")
    assert result.status == again.status == "waiting_user"
    assert count == len(model.requests) == 1
    assert again.reason == "unresolved_write"
    messages = [r["message"] for r in store.messages("default", "default")]
    assert json.loads(messages[2]["content"])["status"] == "unknown"
    assert json.loads(messages[3]["content"])["status"] == "not_dispatched"


async def test_status_query_resolves_without_replaying_write(store):
    writes, queries = [], []

    async def work(n, operation_id):
        writes.append(operation_id)
        raise ToolFailure("lost response", "transient", outcome_known=False)

    async def query(operation_id):
        queries.append(operation_id)
        return ToolResult("returned", ok=True, value={"recorded": True})

    spec = ToolSpec(
        "work", "work", SCHEMA, work, query_status=query, operation_id_parameter="operation_id"
    )
    result = await Agent(
        ScriptedModel([call(), ModelResponse("Recorded")]), ToolRegistry([spec]), store
    ).run("Write")
    assert writes == queries and len(writes) == 1
    assert result.status == "completed" and store.unresolved("default", "default") == []


@pytest.mark.parametrize("category", ["authorization", "missing_input"])
async def test_precondition_waits_without_execution(store, category):
    calls = []

    async def work(n):
        calls.append(n)

    async def check(n):
        raise ToolFailure("User action is needed", category)

    result = await Agent(
        ScriptedModel([call()]),
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work, precondition=check)]),
        store,
    ).run("Work")
    assert result.status == "waiting_user" and calls == []


async def test_clarification_resumes_with_user_reply(store):
    model = ScriptedModel(
        [call("ask_user", {"question": "Which folder?"}), ModelResponse("Using documents")]
    )
    agent = Agent(model, ToolRegistry(), store)
    first = await agent.run("Find my file")
    second = await agent.run("documents")
    assert first.status == "waiting_user" and first.answer == "Which folder?"
    assert second.status == "completed" and "Which folder?" in str(model.requests[-1])


@pytest.mark.parametrize(
    ("question_length", "expected_run_status", "expected_tool_status"),
    [(240, "waiting_user", "returned"), (241, "failed", "not_dispatched")],
    ids=["240-characters-accepted", "241-characters-rejected"],
)
async def test_ask_user_question_length_boundary(
    store, question_length, expected_run_status, expected_tool_status
):
    # 准备：脚本模型只提出一次澄清；不允许参数纠正，便于观察首次调用的结果。
    question = "a" * question_length
    model = ScriptedModel([call("ask_user", {"question": question})])
    agent = Agent(model, ToolRegistry(), store, config=AgentConfig(max_argument_corrections=0))

    # 执行：走真实 Agent 主循环和工具参数校验，不调用真实模型。
    result = await agent.run("Ask me for the missing information")

    # 验证：分别检查运行终态、持久化的工具结果，以及是否发生过派发。
    assert result.status == expected_run_status
    assert len(model.requests) == 1
    tool_message = store.messages("default", "default")[-1]["message"]
    assert tool_message["role"] == "tool"
    tool_result = json.loads(tool_message["content"])
    assert tool_result["status"] == expected_tool_status
    dispatches = [
        event
        for event in store.events(result.run_id)
        if event["kind"] == "tool_attempt" and event.get("phase") == "dispatch_intent"
    ]
    if question_length == 240:
        assert result.answer == question
        assert result.reason == "clarification"
        assert tool_result["ok"] is True
        assert len(dispatches) == 1
    else:
        assert result.reason == "argument_correction_limit"
        assert tool_result["category"] == "arguments"
        assert tool_result["ok"] is False
        assert dispatches == []


async def test_cancelled_write_is_durable_unknown(store):
    started = asyncio.Event()

    async def work(n):
        started.set()
        await asyncio.Event().wait()

    agent = Agent(
        ScriptedModel([call()]), ToolRegistry([ToolSpec("work", "work", SCHEMA, work)]), store
    )
    task = asyncio.create_task(agent.run("Write"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "cancelled"
    assert len(store.unresolved("default", "default")) == 1
    assert (
        json.loads(store.messages("default", "default")[-1]["message"]["content"])["status"]
        == "unknown"
    )


async def test_cancel_during_model_has_terminal_call_and_run(store):
    started = asyncio.Event()

    class SlowModel(ScriptedModel):
        async def complete(self, request):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(Agent(SlowModel([]), ToolRegistry(), store).run("Hello"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = store.db.execute("SELECT id FROM runs").fetchone()[0]
    report = run_report(store, run)
    assert report["terminal"]["status"] == "cancelled"
    assert report["unterminated_model_calls"] == [] and report["unknown_usage_calls"] == 1


async def test_internal_tool_defect_terminates_and_hides_exception_text(store):
    async def work(n):
        raise RuntimeError("private-api-key")

    result = await Agent(
        ScriptedModel([call()]), ToolRegistry([ToolSpec("work", "work", SCHEMA, work)]), store
    ).run("Work")
    assert result.status == "failed" and result.reason == "internal"
    assert "private-api-key" not in str(store.messages("default", "default"))
    assert "RuntimeError" in str(store.events(result.run_id))


async def test_storage_failure_before_dispatch_does_not_execute(store, monkeypatch):
    calls = []

    async def work(n):
        calls.append(n)

    def fail(*args):
        raise StorageError("test failure")

    monkeypatch.setattr(store, "attempt", fail)
    with pytest.raises(StorageError):
        await Agent(
            ScriptedModel([call()]), ToolRegistry([ToolSpec("work", "work", SCHEMA, work)]), store
        ).run("Write")
    assert calls == []
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "failed"


async def test_terminal_storage_failure_cannot_return_completed(store, monkeypatch):
    def fail(result):
        raise StorageError("disk unavailable")

    monkeypatch.setattr(store, "finish", fail)
    with pytest.raises(StorageError):
        await Agent(ScriptedModel([ModelResponse("Hello")]), ToolRegistry(), store).run("Hi")
    assert store.db.execute("SELECT status FROM runs").fetchone()[0] == "running"


@pytest.mark.parametrize("known,expected_calls", [(True, 2), (False, 1)])
async def test_bounded_protocol_correction_requires_known_usage(store, known, expected_calls):
    error = ModelError("bad protocol", kind="protocol_error", usage=Usage(4, 2) if known else None)
    model = ScriptedModel([error, error, ModelResponse("unused")])
    result = await Agent(model, ToolRegistry(), store).run("Hello")
    assert result.status == "failed" and len(model.requests) == expected_calls
    report = run_report(store, result.run_id)
    assert report["known_input_tokens"] == (8 if known else 0)


def test_crash_recovery_preserves_unknown_and_prevents_implicit_resume(store):
    run = store.begin("u", "s", "write")
    store.append(
        "u",
        "s",
        run,
        {"role": "assistant", "tool_calls": [ToolCall("c", "write", {}).to_message()]},
    )
    store.attempt(run, "u", "s", "c", "write", False, 1, "unknown", {})
    with pytest.raises(SessionBusy):
        store.begin("u", "s", "concurrent")
    assert store.recover("u", "s") == 1
    assert store.recover("u", "s") == 0
    assert store.unresolved("u", "s")[0]["call"] == "c"
    assert store.events(run)[-1]["status"] == "interrupted"


async def test_precondition_bug_is_not_offered_as_argument_correction(store):
    calls = []

    async def work(n):
        calls.append(n)

    async def broken(n):
        raise ValueError("program defect")

    result = await Agent(
        ScriptedModel([call()]),
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work, precondition=broken)]),
        store,
    ).run("Work")
    assert result.status == "failed" and result.reason == "internal_error"
    assert calls == []
    assert run_report(store, result.run_id)["argument_corrections"] == 0


async def test_cancelling_retry_backoff_does_not_dispatch_another_attempt(store):
    finished_attempt = asyncio.Event()
    count = 0

    async def work(n):
        nonlocal count
        count += 1
        finished_attempt.set()
        raise ToolFailure("Temporary", "transient", outcome_known=False)

    agent = Agent(
        ScriptedModel([call()]),
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work, read_only=True, max_attempts=3)]),
        store,
    )
    task = asyncio.create_task(agent.run("Read"))
    await finished_attempt.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert count == 1


async def test_status_query_failure_keeps_write_unknown_without_retry(store):
    calls = []

    async def work(n):
        calls.append(n)
        raise ToolFailure("lost response", "transient", outcome_known=False)

    async def query(operation_id):
        raise RuntimeError("private-query-body")

    result = await Agent(
        ScriptedModel([call()]),
        ToolRegistry([ToolSpec("work", "work", SCHEMA, work, max_attempts=3, query_status=query)]),
        store,
    ).run("Write")
    assert result.status == "waiting_user" and calls == [1]
    assert len(store.unresolved("default", "default")) == 1
    assert "private-query-body" not in str(store.events(result.run_id))


def test_recovery_retains_result_committed_before_message_append(store):
    run = store.begin("u", "s", "write")
    store.append(
        "u",
        "s",
        run,
        {"role": "assistant", "tool_calls": [ToolCall("c", "write", {}).to_message()]},
    )
    store.attempt(
        run,
        "u",
        "s",
        "c",
        "write",
        False,
        1,
        "returned",
        {"status": "returned", "ok": True, "value": {"receipt": "original-receipt"}},
    )
    store.recover("u", "s")
    body = json.loads(store.messages("u", "s")[-1]["message"]["content"])
    assert body["value"]["receipt"] == "original-receipt" and body["ok"]
    assert store.unresolved("u", "s") == []
