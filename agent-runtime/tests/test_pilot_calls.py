import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from evals.pilot_calls import Calls, PilotStop
from evals.retail_pilot import build_config, run_pilot, validate_config

ROOT = Path(__file__).parents[1] / ".vendor/tau2-bench"
pytestmark = pytest.mark.retail


@pytest.fixture
def plan():
    if not ROOT.exists():
        pytest.skip("Pinned Retail checkout not installed")
    return build_config(ROOT, ["6"], model="qwen3.7-flash", base_url="https://fixture.invalid/v1")


def reply(*, known=True):
    data = {"choices": [{"finish_reason": "stop", "message": {"content": "offline reply"}}]}
    if known:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    return httpx.Response(200, json=data)


async def test_agent_output_cap_reaches_wire_and_reservation_without_changing_other_roles(tmp_path):
    current = build_config(ROOT, ["78"], model="fixture-model", base_url="https://fixture.invalid/v1",
                     max_request_bytes=131072, agent_max_completion_tokens=9060)
    validate_config(ROOT, current)
    seen = []
    def factory(role):
        def handler(request):
            body = json.loads(request.content)
            expected = 9060 if role == "agent" else 2048
            assert body["max_completion_tokens"] == expected and "max_tokens" not in body
            assert body["enable_thinking"] == body["preserve_thinking"] == (role == "agent")
            assert "thinking_budget" not in body and "reasoning_effort" not in body
            seen.append(role)
            return reply()
        return httpx.MockTransport(handler)
    calls = Calls(tmp_path, current, transport_factory=factory)
    try:
        for role in current["roles"]:
            model = calls.model(role)
            await model.complete(model.prepare([{"role": "user", "content": "offline"}], [], 131072))
    finally:
        calls.close()
    assert seen == list(current["roles"])
    intents = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(line)["event"] == "dispatch_intent"]
    assert Decimal(intents[0]["reserved_cny"]) == Decimal('0.20328')
    assert all(Decimal(r["reserved_cny"]) == Decimal('0.1696224') for r in intents[1:])


async def test_budget_gate_uses_new_output_reservation_before_network(tmp_path):
    current = build_config(ROOT, ["78"], model="fixture-model", base_url="https://fixture.invalid/v1",
                     budget='0.19', max_request_bytes=131072, agent_max_completion_tokens=9060)
    def forbidden(request):
        raise AssertionError('New output reservation must reject before HTTP')
    calls = Calls(tmp_path, current, transport_factory=lambda role: httpx.MockTransport(forbidden))
    try:
        model = calls.model('agent')
        with pytest.raises(PilotStop, match='before dispatch'):
            await model.complete(model.prepare([{"role": "user", "content": "offline"}], [], 131072))
        assert calls.count == 0
    finally:
        calls.close()


def test_agent_output_cap_cannot_exceed_explicit_bound():
    current = build_config(ROOT, ["78"], model="fixture-model", base_url="https://fixture.invalid/v1",
                     agent_max_completion_tokens=9060)
    current['limits']['agent_max_output_tokens'] = 9061
    with pytest.raises(ValueError, match='Agent output'):
        validate_config(ROOT, current)


async def test_shared_usage_log_and_no_credential_material(tmp_path, plan):
    seen = []
    def factory(role):
        def handler(request):
            seen.append(role)
            assert request.headers["authorization"] == "Bearer fixture-auth"
            return reply()
        return httpx.MockTransport(handler)
    calls = Calls(tmp_path, plan, key="fixture-auth", transport_factory=factory)
    try:
        for role in plan["roles"]:
            model = calls.model(role)
            await model.complete(model.prepare([{"role": "user", "content": "offline"}], [], 28000))
        assert seen == ["agent", "user", "communicate", "nl"]
        assert calls.count == 4 and calls.reserved == 0
        assert calls.spent == 4 * calls.cost(100, 20)
    finally:
        calls.close()
    data = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "fixture-auth" not in data
    records = [json.loads(line) for line in data.splitlines()]
    assert len(records) == 16  # intent, decoded HTTP body, parsed response, settlement
    assert all(row["usage_known"] for row in records if row["event"] == "response")
    import base64
    bodies = [json.loads(base64.b64decode(r["body_base64"])) for r in records if r["event"] == "http_response"]
    assert len(bodies) == 4 and bodies[0]["choices"][0]["message"]["content"] == "offline reply"


async def test_unknown_usage_stops_other_roles_and_retains_reservation(tmp_path, plan):
    seen = []
    def handler(request):
        seen.append(request)
        return reply(known=False)
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    try:
        for role in ("user", "agent"):
            model = calls.model(role)
            with pytest.raises(PilotStop):
                await model.complete(model.prepare([{"role": "user", "content": "offline"}], [], 28000))
        assert len(seen) == 1 and calls.reserved > 0 and calls.stopped
    finally:
        calls.close()


@pytest.mark.parametrize("reason", ["budget", "cap", "stop", "oversize"])
async def test_total_guard_refuses_before_dispatch(tmp_path, plan, reason):
    def forbidden(request):
        raise AssertionError("No HTTP dispatch permitted")
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(forbidden))
    if reason == "budget":
        calls.spent = Decimal(plan["budget_cny"])
    elif reason == "cap":
        calls.count = plan["limits"]["max_calls"]
    elif reason == "stop":
        (tmp_path / "STOP").touch()
    model = calls.model("agent")
    try:
        request = model.prepare([{"role": "user", "content": "x" * (29000 if reason == "oversize" else 1)}], [], 40000)
        with pytest.raises(PilotStop):
            await model.complete(request)
    finally:
        calls.close()
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(r["event"] == "dispatch_intent" for r in records)
    assert records[-1]["event"] == "stop"


def test_official_user_and_core_end_to_end_without_network(tmp_path, plan):
    summary = run_pilot(ROOT, plan, tmp_path / "run", offline=True)
    assert summary["all_tasks_finished"] and summary["official_score"] is None
    assert summary["calls"] == 3
    output = tmp_path / "run/task-6"
    assert (output / "official/simulation.json").exists()
    assert (output / "official/simulation.unscored.json").exists()
    assert (output / "core.sqlite3").exists()
    assert list(output.glob("trajectory-*.json")) and list(output.glob("trajectory-*.md"))
    records = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["role"] for r in records} == {"user", "agent"}
    agent_input = next(r["request"] for r in records if r["role"] == "agent" and r["event"] == "dispatch_intent")
    assert "evaluation_criteria" not in json.dumps(agent_input)
    agent_intent = next(r for r in records if r["role"] == "agent" and r["event"] == "dispatch_intent")
    assert agent_intent["run_id"] and agent_intent["model_call_id"]
    assert summary["cost_complete"] and summary["per_role"]["user"]["calls"] == 2
    assert json.loads((tmp_path / "run/config.json").read_text(encoding="utf-8")) == plan
    assert not (tmp_path / "run/plan.json").exists()


@pytest.mark.parametrize("history,compaction", [("recent", "legacy"), ("capacity", "legacy"),
                                               ("capacity", "task"), ("recent", "task")])
def test_context_variants_reach_core_without_changing_role_configuration(tmp_path, history, compaction):
    import sqlite3
    current = build_config(ROOT, ["6"], model="fixture-model", base_url="https://fixture.invalid/v1",
                           agent_max_completion_tokens=9060, max_request_bytes=131072,
                           context_history=history, context_compaction=compaction)
    summary = run_pilot(ROOT, current, tmp_path / "run", offline=True)
    assert summary["mode"] == "offline_http_fixture" and summary["cost_complete"]
    with sqlite3.connect((tmp_path / "run/task-6/core.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        requests = [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE kind='model_request'")]
    assert requests
    assert all(r["history_policy"] == history and r["compaction_policy"] == compaction for r in requests)
    assert all(r["body"]["max_completion_tokens"] == 9060 for r in requests)
    assert all(r["output_token_reserve"] == 9060 for r in requests)
    assert all(r["body"]["enable_thinking"] is r["body"]["preserve_thinking"] is True for r in requests)


def test_call_limits_cannot_be_bypassed(plan):
    validate_config(ROOT, plan)
    plan["limits"]["max_calls"] = 999
    with pytest.raises(ValueError, match="limits"):
        validate_config(ROOT, plan)


@pytest.mark.parametrize("request_cap", [64000, 131072])
async def test_explicit_larger_request_cap_raises_input_reservation_together(tmp_path, request_cap):
    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    plan = build_config(ROOT, ["6"], model="fixture-model", base_url="https://fixture.invalid/v1",
                  max_request_bytes=request_cap)
    assert plan["limits"]["input_token_proxy_ceiling"] == request_cap + 2048
    validate_config(ROOT, plan)
    seen = []
    def handler(request):
        seen.append(request)
        return reply()
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    try:
        model = calls.model("agent")
        request = model.prepare([{"role":"user", "content":"x" * (request_cap - 1000)}], [], request_cap)
        await model.complete(request)
        assert len(seen) == 1
    finally:
        calls.close()
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    intent = next(r for r in records if r["event"] == "dispatch_intent")
    assert Decimal(intent["reserved_cny"]) == calls.cost(request_cap + 2048, 2058)
    plan["limits"]["input_token_proxy_ceiling"] = 32000
    with pytest.raises(ValueError, match="limits"):
        validate_config(ROOT, plan)


def test_request_cap_above_authorized_128k_ceiling_is_rejected():
    with pytest.raises(ValueError, match="131072"):
        build_config(ROOT, ["6"], model="fixture", base_url="https://fixture.invalid/v1", max_request_bytes=131073)


def test_official_judge_transport_uses_configured_model_not_upstream_default(tmp_path, plan):
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.data_model.message import SystemMessage, UserMessage
    old = nl.DEFAULT_LLM_NL_ASSERTIONS
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return reply()
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    try:
        with calls.official_transport():
            nl.generate(model=nl.DEFAULT_LLM_NL_ASSERTIONS,
                        messages=[SystemMessage(role="system", content="fixture"),
                                  UserMessage(role="user", content="fixture")],
                        **nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
        assert nl.DEFAULT_LLM_NL_ASSERTIONS == old
        assert seen[0]["model"] == plan["roles"]["nl"]["model"]
        assert seen[0]["enable_thinking"] is False
    finally:
        calls.close()


async def test_malformed_http_body_is_retained_and_blocks_all_roles(tmp_path, plan):
    from agentcore.core.types import ModelError
    from evals.pilot_calls import inspect_calls
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"not-json")
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    try:
        agent = calls.model("agent")
        with pytest.raises(ModelError):
            await agent.complete(agent.prepare([{"role":"user", "content":"fixture"}], [], 28000))
        judge = calls.model("nl")
        with pytest.raises(PilotStop):
            await judge.complete(judge.prepare([{"role":"user", "content":"fixture"}], [], 28000))
        assert len(seen) == 1
    finally:
        calls.close()
    report = inspect_calls(tmp_path)
    assert not report["cost_complete"] and report["unknown_usage_call_ids"]
    assert Decimal(report["unresolved_reservation_cny"]) > 0


async def test_durable_intent_failure_prevents_dispatch(tmp_path, plan, monkeypatch):
    import os
    seen = []
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(lambda r: seen.append(r)))
    def fail(fd):
        raise OSError("offline storage failure")
    monkeypatch.setattr(os, "fsync", fail)
    try:
        model = calls.model("agent")
        with pytest.raises(OSError):
            await model.complete(model.prepare([{"role":"user", "content":"fixture"}], [], 28000))
        assert not seen and calls.stopped
    finally:
        calls.close()


def test_official_full_scoring_all_roles_use_shared_log(tmp_path, plan):
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.data_model.tasks import EvaluationCriteria, RewardType

    from evals.retail import load_retail
    from evals.retail_interaction import build_interaction, run_and_score
    _, loaded, tasks = load_retail(ROOT)
    task = tasks["6"].model_copy(deep=True)
    task.evaluation_criteria = EvaluationCriteria(
        actions=[], reward_basis=[RewardType.DB, RewardType.COMMUNICATE, RewardType.NL_ASSERTION],
        communicate_info=["fixture fact"], nl_assertions=["fixture assertion"])
    count = {}
    def factory(role):
        def handler(request):
            count[role] = count.get(role, 0) + 1
            if role == "user":
                assert json.loads(request.content)["seed"] == 42
                content = "Help me" if count[role] == 1 else "###STOP###"
            elif role == "agent":
                content = "fixture fact"
            elif role == "communicate":
                content = '{"reasoning":"fixture","met":true}'
            else:
                content = json.dumps({"results":[{"expectedOutcome":"fixture assertion",
                                                  "metExpectation":False,"reasoning":"fixture"}]})
            data = json.loads(reply().content)
            data["choices"][0]["message"]["content"] = content
            return httpx.Response(200, json=data)
        return httpx.MockTransport(handler)
    calls = Calls(tmp_path, plan, transport_factory=factory)
    try:
        with calls.official_transport():
            orch = build_interaction(loaded, task, calls.model("agent"), tmp_path / "core.sqlite3",
                                     user_model=plan["roles"]["user"]["model"], user_args={},
                                     communicate_scoring=CommunicateScoring(judge={"model":plan["roles"]["communicate"]["model"],"model_args":{}}))
            result = run_and_score(orch, loaded, tmp_path / "official")
        assert result.reward_info.reward == 0
        assert count == {"user":2, "agent":1, "communicate":1, "nl":1}
        assert calls.count == 5
    finally:
        calls.close()
