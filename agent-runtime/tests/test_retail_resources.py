"""Resource-only behavior through actual dispatch seams; all HTTP is mocked."""

import asyncio
import json
from contextlib import closing
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from agentcore import Agent, AgentConfig
from agentcore.context import ContextManager
from agentcore.core.types import dumps
from agentcore.storage import Store
from agentcore.tools import ToolRegistry, ToolSpec
from evals.pilot_calls import Calls, PilotStop
from evals.retail_pilot import build_config, run_pilot, validate_config

ROOT = Path(__file__).parents[1] / ".vendor/tau2-bench"
pytestmark = pytest.mark.retail


def config(**kwargs):
    return build_config(ROOT, ["6"], model="offline-fixture", base_url="https://fixture.invalid/v1",
                        max_request_bytes=131072, agent_max_completion_tokens=9060, **kwargs)


def response(*, inputs=100, outputs=20, tool=False):
    message = {"content": "offline reply"}
    if tool:
        message = {"tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "work", "arguments": "{}"}},
            {"id": "two", "type": "function", "function": {"name": "work", "arguments": "{}"}},
        ]}
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "tool_calls" if tool else "stop", "message": message}],
        "usage": {"prompt_tokens": inputs, "completion_tokens": outputs},
    })


async def dispatch(calls, role):
    model = calls.model(role)
    return await model.complete(model.prepare([{"role": "user", "content": "offline"}], [], 131072))


def records(path):
    return [json.loads(line) for line in (path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("history,compaction", [("recent", "legacy"), ("capacity", "legacy"),
                                               ("capacity", "task")])
def test_profiles_snapshots_core_wiring_and_unchanged_requests(tmp_path, monkeypatch, history, compaction):
    from evals.retail import load_retail
    load_retail(ROOT)
    import evals.retail_interaction as interaction
    original = interaction.build_interaction
    actual_core_steps = []

    def capture(*args, **kwargs):
        orch = original(*args, **kwargs)
        actual_core_steps.append(orch.agent.agent.config.max_steps)
        return orch

    monkeypatch.setattr(interaction, "build_interaction", capture)
    requests = []
    for profile, count, steps, budget in (("legacy", 40, 12, "1.00"), ("relaxed", 80, 24, "3.00")):
        plan = config(resource_profile=profile, context_history=history, context_compaction=compaction)
        validate_config(ROOT, plan)
        assert plan["resources"] == {"profile": profile, "max_calls_per_task": count,
                                     "core_max_steps": steps, "task_budget_cny": budget}
        assert plan["limits"]["max_calls"] == count and plan["budget_cny"] == budget
        assert plan["max_steps"] == 60 and plan["seed"] == 42
        output = tmp_path / profile
        summary = run_pilot(ROOT, plan, output, offline=True)
        assert summary["all_tasks_finished"] and summary["official_score"] is None
        saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert saved == plan
        assert (output / "source-snapshot/evals/pilot_config.py").exists()
        simulation = json.loads((output / "task-6/official/simulation.json").read_text(encoding="utf-8"))
        assert simulation["termination_reason"] == "user_stop"
        requests.append([(r["role"], r["request"]) for r in records(output) if r["event"] == "dispatch_intent"])
    assert actual_core_steps == [12, 24]
    # Authorized exception: Core already includes its actual step limit in run_state.
    for role, request in requests[1]:
        if role == "agent":
            system = request["messages"][0]
            prefix, state_json = system["content"].rsplit("\n", 1)
            state = json.loads(state_json)
            assert state["run_state"]["limit"] == 24
            state["run_state"]["limit"] = 12
            system["content"] = prefix + "\n" + dumps(state)
    # Every other byte/field, including the full Context view, must match.
    assert requests[0] == requests[1]


@pytest.mark.parametrize("profile,cap", [("relaxed", 80), ("legacy", 40)])
@pytest.mark.parametrize("blocked_at", ["core", "official_user"])
def test_all_role_cap_blocks_next_dispatch_at_both_seams(tmp_path, profile, cap, blocked_at):
    plan = config(resource_profile=profile)
    seen = []

    def factory(role):
        def handler(request):
            seen.append(role)
            return response()
        return httpx.MockTransport(handler)

    calls = Calls(tmp_path, plan, transport_factory=factory)
    calls.task_id = "6"

    async def warmup():
        for i in range(cap):
            await dispatch(calls, ("agent", "user", "communicate", "nl")[i % 4])

    try:
        asyncio.run(warmup())
        assert calls.count == calls.task_counts["6"] == len(seen) == cap
        assert {role: seen.count(role) for role in plan["roles"]} == dict.fromkeys(plan["roles"], cap // 4)
        assert calls.reserved == 0
        if blocked_at == "core":
            with closing(Store(tmp_path / "core.sqlite3")) as store:
                calls.core_store = store
                agent = Agent(calls.model("agent"), ToolRegistry(), store,
                              config=AgentConfig(max_steps=plan["resources"]["core_max_steps"],
                                                 max_request_bytes=131072))
                result = asyncio.run(agent.run("offline", user_id="retail", session_id="call"))
                assert (result.status, result.reason, result.steps) == ("failed", "internal_error", 1)
        else:
            import tau2.user.user_simulator as user
            from tau2.data_model.message import UserMessage
            with calls.official_transport(), pytest.raises(PilotStop, match="before dispatch"):
                user.generate(model=plan["roles"]["user"]["model"],
                              messages=[UserMessage(role="user", content="offline")], seed=42)
        assert calls.stop_reason == "call_limit"
        assert len(seen) == calls.count == cap and calls.reserved == 0
    finally:
        calls.close()
    assert len([r for r in records(tmp_path) if r["event"] == "dispatch_intent"]) == cap
    assert records(tmp_path)[-1]["task_calls"] == cap


@pytest.mark.parametrize("profile,steps", [("relaxed", 24), ("legacy", 12)])
async def test_core_steps_count_iterations_not_tools_or_all_roles(tmp_path, profile, steps):
    plan = config(resource_profile=profile)
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(
        lambda request: response(tool=role == "agent")))
    calls.task_id = "6"
    tools = []

    async def work():
        tools.append("work")
        return "known result"

    try:
        await dispatch(calls, "user")
        with closing(Store(tmp_path / "core.sqlite3")) as store:
            calls.core_store = store
            agent = Agent(calls.model("agent"), ToolRegistry([
                ToolSpec("work", "fixture", {"type": "object", "properties": {}}, work, read_only=True)
            ]), store, config=AgentConfig(max_steps=plan["resources"]["core_max_steps"], max_request_bytes=131072),
                context=ContextManager("fixture", **plan["context"]))
            result = await agent.run("offline", user_id="retail", session_id="call")
            assert (result.status, result.reason, result.steps) == ("failed", "step_limit", steps)
            # The fixture tool is declared read_only, so no mutating call was ever eligible.
            assert result.goal_status == "no_state_change_attempted"
            assert len(tools) == 2 * steps
            assert calls.count == steps + 1 and calls.stop_reason is None
    finally:
        calls.close()


@pytest.mark.parametrize("scope,budget,expected", [("task", "6.00", "task_budget_limit"),
                                                  ("run", "0.50", "run_budget_limit")])
async def test_reserve_settle_and_lower_run_budget(tmp_path, scope, budget, expected):
    plan = build_config(ROOT, ["6", "8"] if scope == "run" else ["6"],
                        model="offline-fixture", base_url="https://fixture.invalid/v1", budget=budget,
                        max_request_bytes=131072, agent_max_completion_tokens=9060)
    seen, held = [], []
    calls = None

    def handler(request):
        seen.append(request)
        held.append(calls.reserved)
        return response(inputs=133120, outputs=9060)

    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    calls.task_id = "6"
    reserve, charge = Decimal("0.20328"), Decimal("0.203232")
    count = 14 if scope == "task" else 2
    try:
        for i in range(count):
            calls.task_id = "8" if scope == "run" and i == 1 else "6"
            await dispatch(calls, "agent")
            assert calls.reserved == 0
        assert held == [reserve] * count
        assert calls.spent == sum(calls.task_spent.values()) == count * charge
        if scope == "run":
            assert calls.task_spent == {"6": charge, "8": charge}
            calls.task_id = "6"
        assert calls.spent + reserve > (Decimal("3") if scope == "task" else Decimal(budget))
        with pytest.raises(PilotStop, match="before dispatch"):
            await dispatch(calls, "agent")
        assert calls.stop_reason == expected and len(seen) == count
        assert calls.reserved == 0
    finally:
        calls.close()


async def test_usage_overrun_settles_then_stops_all_roles(tmp_path):
    calls = Calls(tmp_path, config(), transport_factory=lambda role: httpx.MockTransport(
        lambda request: response(inputs=133121)))
    calls.task_id = "6"
    try:
        with pytest.raises(PilotStop, match="usage exceeds"):
            await dispatch(calls, "agent")
        assert calls.spent == calls.task_spent["6"] == calls.cost(133121, 20)
        assert calls.reserved == 0 and calls.stop_reason == "usage_bound"
        with pytest.raises(PilotStop):
            await dispatch(calls, "user")
        assert calls.count == 1
    finally:
        calls.close()


async def test_task_counts_do_not_borrow_another_tasks_allowance(tmp_path):
    plan = build_config(ROOT, ["6", "8"], model="offline-fixture", base_url="https://fixture.invalid/v1")
    validate_config(ROOT, plan)
    assert plan["limits"]["max_calls"] == 160 and Decimal(plan["budget_cny"]) == 3  # not 6
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(lambda r: response()))
    try:
        calls.task_id = "6"
        for _ in range(79):
            await dispatch(calls, "agent")
        calls.task_id = "8"
        await dispatch(calls, "user")
        calls.task_id = "6"
        await dispatch(calls, "nl")
        with pytest.raises(PilotStop):
            await dispatch(calls, "user")
        assert calls.count == 81 and calls.task_counts == {"6": 80, "8": 1}
        assert calls.stop_reason == "call_limit"
    finally:
        calls.close()


@pytest.mark.parametrize("field,value", [("max_calls_per_task", 81), ("core_max_steps", 25),
                                         ("task_budget_cny", "4.00"), ("profile", "unknown")])
def test_snapshot_resource_drift_rejected(field, value):
    plan = config()
    plan["resources"][field] = value
    with pytest.raises(ValueError, match="resource profile"):
        validate_config(ROOT, plan)


@pytest.mark.parametrize("failure", ["core_cap", "user_cap", "core_steps", "runtime"])
def test_resource_and_runtime_failures_keep_official_unscored(tmp_path, monkeypatch, failure):
    import evals.pilot_calls as pilot_calls
    from evals import retail_pilot
    plan = config()
    original = pilot_calls.Calls

    class FixtureCalls(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Seed the aggregate counter only to place the boundary in the real
            # run_pilot path. The separate cap test actually dispatches all 80.
            if failure in {"core_cap", "user_cap"}:
                self.count = 79 if failure == "core_cap" else 78

    monkeypatch.setattr(pilot_calls, "Calls", FixtureCalls)
    if failure == "runtime":
        monkeypatch.setattr(retail_pilot, "offline_transport", lambda: lambda role: httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"invalid-json")))
    if failure == "core_steps":
        # Read-only tool calls; 24 iterations finish before the official 60-step cap.
        from evals.retail_fixture import USER_EMAIL
        def fixture_response(request):
            if not json.loads(request.content)["enable_thinking"]:
                return response()
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "tool_calls": [{"id": "page", "type": "function", "function": {
                    "name": "find_user_id_by_email", "arguments": json.dumps({"email": USER_EMAIL})}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        monkeypatch.setattr(retail_pilot, "offline_transport",
                            lambda: lambda role: httpx.MockTransport(fixture_response))
    output = tmp_path / "run"
    with pytest.raises((RuntimeError, PilotStop)):
        run_pilot(ROOT, plan, output, offline=True)
    partial = json.loads((output / "task-6/official/simulation.unscored.json").read_text(encoding="utf-8"))
    assert partial["termination_reason"] == "infrastructure_error"
    assert partial.get("reward_info") is None
    assert not (output / "task-6/official/simulation.json").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_score"] is None and summary["stopped"] and not summary["all_tasks_finished"]
    if failure in {"core_cap", "user_cap"}:
        assert summary["stop_reason"] == "call_limit"
        assert summary["calls"] == (1 if failure == "core_cap" else 2)
    elif failure == "core_steps":
        import sqlite3
        with sqlite3.connect(output / "task-6/core.sqlite3") as db:
            result = json.loads(db.execute("SELECT result FROM runs").fetchone()[0])
        assert (result["reason"], result["steps"]) == ("step_limit", 24)
