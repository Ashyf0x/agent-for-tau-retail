"""Offline preparation and model routing checks, not model capability evaluations."""

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from agentcore.core.model import HTTPModelClient
from agentcore.core.types import ModelError
from evals.decision_snapshots import decision_request, digest, prepare, read_json
from evals.pilot_config import resolve_roles
from evals.retail_pilot import build_config, validate_config

PROJECT = Path(__file__).parents[1]
LEGACY = Path(__file__).parent / "fixtures/legacy"
STUDY = LEGACY / "decision-study"
AUDIT = LEGACY / "decision-audit-requests.json"
RUNS = LEGACY / "decision-source-runs"
CANDIDATE = read_json(STUDY / "candidate.json")
pytestmark = pytest.mark.retail


def test_agent_override_uses_actual_configuration_priority_without_changing_other_roles(tmp_path):
    settings = tmp_path / "public-settings.txt"
    settings.write_text('AGENT_MODEL=qwen3.7-flash\nAGENT_BASE_URL=https://fixture.invalid/v1\n'
                        'RETAIL_AGENT_MODEL=old-agent\nRETAIL_NL_MODEL=qwen3.7-flash\n'
                        'RETAIL_NL_BASE_URL=https://judge-fixture.invalid/v1\n', encoding="utf-8")
    original = resolve_roles(env_file=settings)
    selected = resolve_roles(env_file=settings, agent_model=CANDIDATE["agent_model"])
    assert selected["agent"] == {**original["agent"], "model": CANDIDATE["agent_model"],
                                 "provider": "bailian_deepseek", "preserve_thinking": False,
                                 "model_source": "explicit --agent-model"}
    for role in ("user", "nl", "communicate"):
        assert selected[role] == original[role]
        assert selected[role]["model"] == "qwen3.7-flash"
    plan = build_config(PROJECT / ".vendor/tau2-bench", ["6"], env_file=settings,
                        agent_model=CANDIDATE["agent_model"], resource_profile="relaxed",
                        max_request_bytes=131072, agent_max_completion_tokens=9060,
                        input_price="2", output_price="8")  # proposed explicit rates, not billed usage
    validate_config(PROJECT / ".vendor/tau2-bench", plan)
    assert plan["roles"] == selected
    assert plan["resources"] == {"profile": "relaxed", "max_calls_per_task": 80,
                                 "core_max_steps": 24, "task_budget_cny": "3.00"}
    assert plan["limits"]["agent_max_output_tokens"] == 9060
    assert plan["limits"]["max_output_tokens"] == 2048
    assert plan["context"] == CANDIDATE["context"]


async def test_mock_native_tools_and_reasoning_replay_keep_all_parameters():
    seen = []
    tool = {"type": "function", "function": {"name": "fixture_read", "description": "fixture",
            "parameters": {"type": "object", "properties": {}}}}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["model"] == CANDIDATE["agent_model"]
        assert body["enable_thinking"] is True and "preserve_thinking" not in body
        assert body["max_completion_tokens"] == 9060 and body["temperature"] == 0
        assert body["stream"] is False and body["tools"] == [tool]
        assert "thinking" not in body and "reasoning_effort" not in body and "max_tokens" not in body
        message = ({"content": None, "reasoning_content": "mock reasoning", "tool_calls": [{
            "id": "provider-one", "type": "function", "function": {"name": "fixture_read", "arguments": "{}"}}]}
            if len(seen) == 1 else {"content": "mock answer", "reasoning_content": "mock continuation"})
        return httpx.Response(200, json={"choices": [{"message": message,
            "finish_reason": "tool_calls" if len(seen) == 1 else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    client = HTTPModelClient("https://fixture.invalid/v1", CANDIDATE["agent_model"], provider="bailian_deepseek",
                             thinking=True, max_completion_tokens=9060,
                             transport=httpx.MockTransport(handler))
    history = [{"role": "user", "content": "offline fixture"}]
    first = await client.complete(client.prepare(history, [tool], 131072))
    assert first.tool_calls[0].name == "fixture_read" and first.usage_known
    history += [{"role": "assistant", "content": None, "_provider_metadata": first.provider_metadata,
                 "tool_calls": [{"id": "provider-one", "type": "function",
                                 "function": {"name": "fixture_read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "provider-one", "content": "fixture-only result"}]
    final = await client.complete(client.prepare(history, [tool], 131072))
    assert final.content == "mock answer"
    assert seen[1]["messages"][1]["reasoning_content"] == "mock reasoning"
    assert seen[1]["messages"][2]["tool_call_id"] == "provider-one"


async def test_rejected_parameter_does_not_fallback_or_retry():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "fixture unsupported parameter"}})
    client = HTTPModelClient("https://fixture.invalid/v1", CANDIDATE["agent_model"], provider="bailian_deepseek",
                             thinking=True, max_completion_tokens=9060,
                             transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError) as exc:
        await client.complete(client.prepare([{"role": "user", "content": "fixture"}], [], 131072))
    assert exc.value.kind == "capability"
    assert len(seen) == 1 and seen[0]["enable_thinking"] is True and "preserve_thinking" not in seen[0]


def test_prepared_snapshots_keep_original_inputs_and_separate_rubrics(tmp_path):
    result = prepare(PROJECT, STUDY / "candidate.json", tmp_path / "prepared",
                     audit_path=AUDIT, runs_root=RUNS)
    assert len(result["snapshots"]) == 6 and len({s["task_id"] for s in result["snapshots"]}) == 6
    assert result["planned_calls"]["agent_decisions"] == 36 and result["planned_calls"]["executed"] == 0
    assert result["cost_proposal"]["per_call_reserve_cny"] == "0.3388"
    assert Decimal(result["cost_proposal"]["combined_worst_reserve_cny"]) == Decimal("12.8744")
    assert result["authorized_for_real_calls"] is False
    assert result["roles"]["agent"]["model"] == CANDIDATE["agent_model"]
    for role in ("user", "nl", "communicate"):
        assert result["roles"][role]["model"] == "qwen3.7-flash"
    for item in result["snapshots"]:
        folder = tmp_path / "prepared" / item["id"]
        original = (folder / "original-request.json").read_bytes()
        assert digest(original) == item["original_request_sha256"]
        old = json.loads(original)
        requests = {arm: read_json(folder / f"{arm}.request.json") for arm in ("A", "B", "C")}
        expected = deepcopy(old)
        expected["model"] = CANDIDATE["agent_model"]
        prefix, state_json = expected["messages"][0]["content"].rsplit("\n", 1)
        state = json.loads(state_json)
        state["run_state"]["limit"] = 24
        from agentcore.core.types import dumps
        expected["messages"][0]["content"] = prefix + "\n" + dumps(state)
        assert requests["A"] == expected
        for arm in ("B", "C"):
            intervention = requests[arm]["messages"].pop(1)
            assert intervention["role"] == "system"
            assert digest(intervention["content"].encode("utf-8")) == result["arm_text_sha256"][arm]
            assert requests[arm] == requests["A"]
        prefix_rows = read_json(folder / "source-prefix.json")
        assert prefix_rows[-1]["seq"] == item["source_message_seq_max"]
        assert [r["id"] for r in prefix_rows] == item["source_message_ids"]
        assert all("reward_info" not in req and "evaluation_criteria" not in req for req in requests.values())
        for arm in ("A", "B", "C"):
            assert (folder / f"{arm}.request.json").stat().st_size <= 131072 - 2048
    with pytest.raises(FileExistsError):
        prepare(PROJECT, STUDY / "candidate.json", tmp_path / "prepared",
                audit_path=AUDIT, runs_root=RUNS)


def test_rubric_evidence_is_visible_and_evaluation_only():
    evidence = read_json(STUDY / "evidence-classification.json")
    assert evidence["agent_input"] is False and not evidence["independent_human_gold"]
    for case in evidence["cases"]:
        request = read_json(STUDY / "prepared" / case["id"] / "original-request.json")
        for fact in case["facts"]:
            for source in fact["actual_visible_sources"]:
                message = request["messages"][source["request_message_index"]]
                assert message["role"] == fact["role"]
                assert message["content"][source["offset"]:].startswith(fact["needle"])
                assert source["source_message_ids"]
        assert all(text in request["messages"][0]["content"] for text in case["policy_needles"])


def test_intervention_cannot_expand_effective_context_limit():
    original = read_json(STUDY / "prepared/known-payment-constraint/original-request.json")
    with pytest.raises(ValueError, match="byte limit"):
        decision_request(original, agent_model="fixture-model", core_steps=24, intervention="x" * 131072)


def test_actual_transfer_marker_reaches_official_stop_check():
    from evals.retail import load_retail
    load_retail(PROJECT / ".vendor/tau2-bench")
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator
    report = read_json(STUDY / "transfer-diagnosis.json")
    assert len(report["chains"]) == 3
    for chain in report["chains"]:
        assert chain["user_simulator_input_last_message"]["content"] == chain["agent_notification"]["message"]["content"]
        assert "###TRANSFER###" in chain["user_simulator_system_transfer_instruction"]
        output = UserMessage(role="user", content=chain["simulator_output"])
        assert UserSimulator.is_stop(output) == chain["transfer_marker_returned"]
    assert [c["transfer_marker_returned"] for c in report["chains"]] == [False, False, True]
