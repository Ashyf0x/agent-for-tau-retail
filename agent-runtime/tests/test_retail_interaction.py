import json
from pathlib import Path

import pytest

from agentcore import ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel

pytestmark = pytest.mark.retail
ROOT = Path(__file__).parents[1] / ".vendor/tau2-bench"


@pytest.fixture
def retail():
    if not ROOT.exists():
        pytest.skip("Pinned Retail checkout not installed")
    from evals.retail import load_retail
    return load_retail(ROOT)


def build(tmp_path, retail, model, *, task=None, max_steps=100):
    from tau2.data_model.message import UserMessage
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.user.user_simulator_base import HalfDuplexUser

    from evals.retail_fixture import USER_EMAIL
    from evals.retail_interaction import build_interaction

    class User(HalfDuplexUser):
        def get_init_state(self, message_history=None):
            return 0

        def generate_next_message(self, message, state):
            content = f"My email is {USER_EMAIL}. Find my account." if state == 0 else "###STOP###"
            return UserMessage(role="user", content=content), state + 1

        @classmethod
        def is_stop(cls, message):
            return message.content == "###STOP###"

    _, loaded, tasks = retail
    task = task or tasks["6"]
    return build_interaction(loaded, task, model, tmp_path / "agent.sqlite3",
                             user_model="offline-user", user_args={},
                             communicate_scoring=CommunicateScoring(), max_steps=max_steps,
                             user_factory=lambda *args: User())


def test_official_orchestration_replay_and_our_core(tmp_path, retail):
    from tau2.data_model.simulation import SimulationRun

    from evals.retail_fixture import USER_EMAIL, fixture_model
    from evals.retail_interaction import run_and_score
    model = fixture_model()
    orch = build(tmp_path, retail, model)
    result = run_and_score(orch, retail[1], tmp_path / "result")
    assert result.termination_reason.value == "user_stop"
    calls = [call for message in result.messages for call in (getattr(message, "tool_calls", None) or [])]
    assert [call.name for call in calls] == ["find_user_id_by_email", "get_user_details"]
    assert len(orch.agent.results) == 1 and orch.agent.closed
    assert USER_EMAIL in orch.agent.results[0].answer
    assert len(model.requests) == 3
    assert "evaluation_criteria" not in json.dumps(model.requests)
    saved = SimulationRun.model_validate_json((tmp_path / "result/simulation.json").read_text(encoding="utf-8"))
    assert saved.reward_info is not None
    assert saved.language_context.task_sha256 == result.language_context.task_sha256
    # L3 evidence: the official declaration is copied into the same trial store,
    # without adding a model request (the three requests above are unchanged).
    from agentcore.storage import Store
    probe = Store(tmp_path / "agent.sqlite3")
    try:
        terminal = probe.session_terminal("retail", "call")
    finally:
        probe.close()
    assert terminal["reason"] == "user_stop"
    assert terminal["source"] == "tau2_orchestrator"
    assert terminal["run_count"] == 1
    assert terminal["terminal_run"] == orch.agent.results[0].run_id


def test_l3_terminal_record_is_invisible_to_the_model(tmp_path, retail):
    """A/B on the official path: enabling L3 evidence changes no model request."""
    import hashlib
    from unittest.mock import patch

    from agentcore.storage import Store
    from evals import retail_interaction
    from evals.retail_fixture import fixture_model

    def shape(request):
        """Deterministic projection; tool_call_id / observation_ref are fresh UUIDs."""
        def message(item):
            shaped = {"role": item["role"]}
            content = item.get("content")
            if item["role"] == "tool" and isinstance(content, str):
                try:
                    body = json.loads(content)
                except ValueError:
                    body = None
                if isinstance(body, dict):
                    body.pop("observation_ref", None)
                    content = json.dumps(body, sort_keys=True)
            shaped["content"] = content
            if item.get("tool_calls"):
                shaped["tool_calls"] = [
                    {"name": call["function"]["name"],
                     "arguments": call["function"]["arguments"]}
                    for call in item["tool_calls"]
                ]
            return shaped
        return {"tools": request.get("tools"),
                "messages": [message(m) for m in request["messages"]]}

    def once(name, *, record):
        root = tmp_path / name
        root.mkdir()
        model = fixture_model()
        orch = build(root, retail, model)
        if record:
            result = retail_interaction.run_and_score(orch, retail[1], root / "result")
        else:
            with patch.object(retail_interaction, "_record_trial_terminal",
                              lambda *args, **kwargs: None):
                result = retail_interaction.run_and_score(orch, retail[1], root / "result")
        probe = Store(root / "agent.sqlite3")
        try:
            terminal = probe.session_terminal("retail", "call")
        finally:
            probe.close()
        wire = [hashlib.sha256(json.dumps(shape(r), sort_keys=True).encode()).hexdigest()
                for r in model.requests]
        return result, wire, terminal

    with_record, wire_with, terminal_with = once("with_record", record=True)
    without_record, wire_without, terminal_without = once("without_record", record=False)

    assert wire_with == wire_without, "L3 evidence must not alter any model request"
    assert with_record.termination_reason == without_record.termination_reason
    assert with_record.reward_info.reward == without_record.reward_info.reward
    assert terminal_with["reason"] == "user_stop"
    assert terminal_without is None


def test_judge_failure_retains_unscored_simulation(tmp_path, retail, monkeypatch):
    import tau2.evaluator.evaluator as evaluator

    from evals.retail_interaction import run_and_score
    model = ScriptedModel([ModelResponse("I cannot complete this request.")])
    orch = build(tmp_path, retail, model)
    def fail(**kwargs):
        raise RuntimeError("offline judge failure")
    monkeypatch.setattr(evaluator, "evaluate_simulation", fail)
    with pytest.raises(RuntimeError, match="offline judge failure"):
        run_and_score(orch, retail[1], tmp_path / "result")
    assert (tmp_path / "result/simulation.unscored.json").exists()
    assert not (tmp_path / "result/simulation.json").exists()


def test_full_reward_basis_calls_official_judges(tmp_path, retail, monkeypatch):
    import tau2.evaluator.evaluator_communicate as comm
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.tasks import EvaluationCriteria, RewardType

    from evals.retail_interaction import run_and_score
    task = retail[2]["6"].model_copy(deep=True)
    # Synthetic wiring task, never a released benchmark score.
    task.evaluation_criteria = EvaluationCriteria(
        reward_basis=[RewardType.DB, RewardType.COMMUNICATE, RewardType.NL_ASSERTION],
        actions=[], communicate_info=["fixture information"], nl_assertions=["fixture expectation"])
    seen = []
    def comm_reply(**kwargs):
        seen.append(kwargs["call_name"])
        return AssistantMessage(role="assistant", content='{"reasoning":"fixture","met":true}')
    def nl_reply(**kwargs):
        seen.append(kwargs["call_name"])
        return AssistantMessage(role="assistant", content=json.dumps({"results":[{
            "expectedOutcome":"fixture expectation", "reasoning":"fixture", "metExpectation":False}]}))
    monkeypatch.setattr(comm, "generate", comm_reply)
    monkeypatch.setattr(nl, "generate", nl_reply)
    orch = build(tmp_path, retail, ScriptedModel([ModelResponse("fixture information")]), task=task)
    result = run_and_score(orch, retail[1], tmp_path / "result")
    assert seen == ["communicate_info_eval", "nl_assertions_eval"]
    assert result.reward_info.reward == 0
    assert result.reward_info.reward_breakdown[RewardType.DB] == 1


def test_step_limit_cancels_pending_core_without_replaying_write(tmp_path, retail):
    from evals.retail_interaction import run_and_score
    model = ScriptedModel([ModelResponse(tool_calls=[ToolCall("one", "cancel_pending_order", {
        "order_id":"#fixture", "reason":"no longer needed"})], finish_reason="tool_calls")])
    orch = build(tmp_path, retail, model, max_steps=2)
    result = run_and_score(orch, retail[1], tmp_path / "result")
    assert result.reward_info.reward == 0
    assert orch.agent.closed
    # Upstream checks its step limit after executing the tool. Cleanup must retain
    # its known result and must not dispatch another model call or retry the write.
    assert len([m for m in result.messages if m.role == "tool"]) == 1
    import sqlite3
    with sqlite3.connect(tmp_path / "agent.sqlite3") as db:
        assert db.execute("SELECT status FROM attempts").fetchall() == [("returned",)]
    assert len(model.requests) == 1


def test_local_clarification_continues_same_session(tmp_path, retail):
    from tau2.data_model.message import UserMessage
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("ask", "ask_user", {"question":"Which account?"})],
                      finish_reason="tool_calls"), ModelResponse("Thank you.")])
    bridge = build(tmp_path, retail, model).agent
    try:
        state = bridge.get_init_state()
        first, state = bridge.generate_next_message(UserMessage(role="user", content="Help me"), state)
        second, state = bridge.generate_next_message(UserMessage(role="user", content="My account"), state)
        assert first.content == "Which account?" and not first.tool_calls
        assert second.content == "Thank you."
        assert len(bridge.results) == 2
        assert "Which account?" in json.dumps(model.requests[-1])
    finally:
        bridge.stop()


def test_write_executes_once_and_replays_officially(tmp_path, retail, monkeypatch):
    from evals.retail import build_environment
    from evals.retail_interaction import run_and_score
    order = next(o for o in build_environment(retail[1]).tools.db.orders.values() if o.status == "pending")
    # Public DB fixture only; not a solution trajectory for the released task.
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("cancel", "cancel_pending_order", {
            "order_id": order.order_id, "reason": "no longer needed"})], finish_reason="tool_calls"),
        ModelResponse("The order is cancelled.")])
    orch = build(tmp_path, retail, model)
    original = orch.environment.get_response
    seen = []
    def record(call):
        seen.append(call)
        return original(call)
    monkeypatch.setattr(orch.environment, "get_response", record)
    run_and_score(orch, retail[1], tmp_path / "result")
    assert len(seen) == 1
    assert orch.environment.tools.db.orders[order.order_id].status == "cancelled"


def test_pilot_inventory_is_offline_and_unapproved(retail):
    from evals.retail_pilot import prepare
    plan = prepare(ROOT)
    assert len(plan["tasks"]) == 3 and not plan["authorized_for_real_calls"]
    assert plan["agent"]["model"] is None and plan["official_score"] is None
    with pytest.raises(ValueError):
        prepare(ROOT, ["not-a-locked-task"])


def test_incomplete_nl_reply_does_not_save_vacuous_success(tmp_path, retail, monkeypatch):
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.tasks import EvaluationCriteria, RewardType

    from evals.retail_interaction import run_and_score
    task = retail[2]["6"].model_copy(deep=True)
    task.evaluation_criteria = EvaluationCriteria(actions=[], nl_assertions=["Required assertion"],
                                                 reward_basis=[RewardType.DB, RewardType.NL_ASSERTION])
    monkeypatch.setattr(nl, "generate", lambda **kwargs: AssistantMessage(role="assistant", content='{"results":[]}'))
    orch = build(tmp_path, retail, ScriptedModel([ModelResponse("Fixture answer")]), task=task)
    with pytest.raises(ValueError, match="do not cover"):
        run_and_score(orch, retail[1], tmp_path / "result")
    assert (tmp_path / "result/simulation.unscored.json").exists()
    assert not (tmp_path / "result/simulation.json").exists()
