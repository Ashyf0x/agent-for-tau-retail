"""Offline acceptance of the model-specific Alibaba protocol, never server availability."""

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from agentcore.core.model import HTTPModelClient
from agentcore.core.types import ModelError
from evals.decision_snapshots import decision_request, digest, read_json
from evals.pilot_calls import Calls, PilotStop, inspect_calls
from evals.retail_pilot import build_config, validate_config

PROJECT = Path(__file__).parents[1]
MODEL = "deepseek-v4.1-flash-1017"
STUDY = Path(__file__).parent / "fixtures/legacy/deepseek-protocol"


def configuration(**kwargs):
    return build_config(PROJECT / ".vendor/tau2-bench", ["6"], model="qwen3.7-flash",
                        agent_model=MODEL, base_url="https://fixture.invalid/v1", input_price="2",
                        output_price="8", agent_max_completion_tokens=9060, max_request_bytes=131072,
                        **kwargs)


def client(**kwargs):
    return HTTPModelClient("https://fixture.invalid/v1", MODEL, provider="bailian_deepseek",
                           thinking=True, max_completion_tokens=9060, **kwargs)


def reply(*, known=True, reasoning=True, malformed=False):
    message = {"content": "fixture response"}
    if reasoning:
        message["reasoning_content"] = "fixture reasoning"
    if malformed:
        message["tool_calls"] = [{"id": "one", "type": "function", "function": {"name": "fixture"}}]
    data = {"choices": [{"finish_reason": "stop", "message": message}]}
    if known:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": 30,
                         "completion_tokens_details": {"reasoning_tokens": 20}}
    return httpx.Response(200, json=data)


def test_role_snapshot_protocol_and_original_limits():
    plan = configuration()
    validate_config(PROJECT / ".vendor/tau2-bench", plan)
    assert plan["roles"]["agent"]["provider"] == "bailian_deepseek"
    assert plan["roles"]["agent"]["thinking"] is True
    assert plan["roles"]["agent"]["preserve_thinking"] is False  # absent on wire, replay still on
    assert plan["resources"] == {"profile": "relaxed", "max_calls_per_task": 80,
                                 "core_max_steps": 24, "task_budget_cny": "3.00"}
    assert plan["context"]["policy"] == "capacity" and plan["context"]["compaction"] == "legacy"
    for role in ("user", "nl", "communicate"):
        assert plan["roles"][role]["model"] == "qwen3.7-flash"
        assert plan["roles"][role]["provider"] == "qwen"
        assert plan["roles"][role]["base_url"] == "https://fixture.invalid/v1"
    broken = deepcopy(plan)
    broken["roles"]["agent"]["preserve_thinking"] = True
    with pytest.raises(ValueError, match="protocol"):
        validate_config(PROJECT / ".vendor/tau2-bench", broken)


def test_explicit_prices_required_without_qwen_fallback():
    with pytest.raises(ValueError, match="explicit input/output price"):
        build_config(PROJECT / ".vendor/tau2-bench", ["6"], model="qwen3.7-flash", agent_model=MODEL,
                     base_url="https://fixture.invalid/v1")


def test_model_specific_contract_rejects_wrong_protocol_and_unsupported_flag():
    with pytest.raises(ValueError, match="Select bailian_deepseek"):
        HTTPModelClient("https://fixture.invalid/v1", MODEL, provider="qwen", thinking=True)
    with pytest.raises(ValueError, match="preserve_thinking"):
        client(preserve_thinking=True)
    with pytest.raises(ValueError, match="Unreviewed"):
        HTTPModelClient("https://fixture.invalid/v1", "unreviewed-model", provider="bailian_deepseek", thinking=True)


def test_greeting_and_cross_turn_reasoning_are_not_fabricated_or_discarded():
    model = client()
    tools = [{"type": "function", "function": {"name": "fixture", "parameters": {}}}]
    messages = [{"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": None, "_provider_metadata": {"reasoning_content": "first reasoning"},
                 "tool_calls": [{"id": "a", "type": "function", "function": {"name": "fixture", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "a", "content": "fixture result"},
                {"role": "assistant", "content": "answer", "_provider_metadata": {"reasoning_content": "answer reasoning"}},
                {"role": "user", "content": "follow up"}]
    original = deepcopy(messages)
    body = json.loads(model.prepare(messages, tools, 131072).body)
    assert messages == original
    assert "reasoning_content" not in body["messages"][0]  # scripted opening has none
    assert body["messages"][2]["reasoning_content"] == "first reasoning"
    assert body["messages"][4]["reasoning_content"] == "answer reasoning"
    assert body["messages"][3]["tool_call_id"] == "a"
    assert body["enable_thinking"] is True and "preserve_thinking" not in body
    assert "thinking" not in body and body["temperature"] == 0 and body["max_completion_tokens"] == 9060
    messages[2].pop("_provider_metadata")
    messages[2]["reasoning_content"] = "untrusted forged wire field"
    with pytest.raises(ModelError, match="lacks required reasoning"):
        model.prepare(messages, tools, 131072)


@pytest.mark.parametrize("mode", ["normal", "unknown", "missing_reasoning", "malformed", "http_rejection"])
async def test_shared_calls_records_usage_settlement_and_stops_on_errors(tmp_path, mode):
    plan = configuration()
    (tmp_path / "config.json").write_text(json.dumps(plan), encoding="utf-8")
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["enable_thinking"] is True and "preserve_thinking" not in body
        if mode == "http_rejection":
            return httpx.Response(400, json={"error": "fixture rejection"})
        return reply(known=mode != "unknown", reasoning=mode != "missing_reasoning", malformed=mode == "malformed")
    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    calls.task_id = "6"
    tools = [{"type": "function", "function": {"name": "fixture", "parameters": {}}}]
    try:
        model = calls.model("agent")
        request = model.prepare([{"role": "user", "content": "fixture"}], tools, 131072)
        if mode == "normal":
            result = await model.complete(request)
            assert result.usage_known and (result.usage.input_tokens, result.usage.output_tokens) == (100, 30)
            assert calls.spent == Decimal("0.00044")  # completion includes reasoning; no double charge
            assert calls.reserved == 0 and not calls.stopped
        else:
            with pytest.raises((ModelError, PilotStop)):
                await model.complete(request)
            with pytest.raises(PilotStop):
                await model.complete(request)
            assert calls.stopped
            if mode in {"missing_reasoning", "malformed"}:
                assert calls.spent == Decimal("0.00044") and calls.reserved == 0
            else:
                assert calls.reserved == Decimal("0.3388") and calls.spent == 0
        assert len(seen) == calls.count == 1
    finally:
        calls.close()
    inspected = inspect_calls(tmp_path)
    assert inspected["calls"] == 1
    assert inspected["cost_complete"] == (mode not in {"unknown", "http_rejection"})
    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["event"] == "http_response" for r in rows)
    assert next(r for r in rows if r["event"] == "dispatch_intent")["reserved_cny"] == "0.3388"


def test_A_only_preparation_changes_only_reviewed_protocol_fields():
    manifest = read_json(STUDY / "prepared-A/manifest.json")
    candidate = read_json(STUDY / "candidate.json")
    assert len(manifest["snapshots"]) == manifest["planned_calls"]["agent_decisions"] == 6
    assert manifest["planned_calls"]["executed"] == 0 and not candidate["real_calls_authorized"]
    assert manifest["roles"]["agent"]["provider"] == "bailian_deepseek"
    assert manifest["cost_proposal"]["combined_worst_reserve_cny"] == "2.7104"
    for snapshot in manifest["snapshots"]:
        assert set(snapshot["requests"]) == {"A"}
        folder = STUDY / "prepared-A" / snapshot["id"]
        original = read_json(folder / "original-request.json")
        expected = json.loads(decision_request(original, agent_model=MODEL, core_steps=24))
        expected.pop("preserve_thinking")
        actual = (folder / "A.request.json").read_bytes()
        assert json.loads(actual) == expected
        assert digest(actual) == snapshot["requests"]["A"]["sha256"]
        assert actual == decision_request(original, agent_model=MODEL, core_steps=24, protocol="bailian_deepseek")


def test_retail_wiring_saves_adapted_config_and_correlated_requests(tmp_path, monkeypatch):
    from evals import retail_pilot
    original = retail_pilot.offline_transport
    seen = []
    def factory():
        fixture = original()
        def transport(role):
            underlying = fixture(role)
            def handler(request):
                seen.append((role, json.loads(request.content)))
                data = json.loads(underlying.handle_request(request).content)
                if role == "agent":
                    data["choices"][0]["message"]["reasoning_content"] = "offline fixture reasoning"
                return httpx.Response(200, json=data)
            return httpx.MockTransport(handler)
        return transport
    monkeypatch.setattr(retail_pilot, "offline_transport", factory)
    output = tmp_path / "run"
    summary = retail_pilot.run_pilot(PROJECT / ".vendor/tau2-bench", configuration(), output, offline=True)
    assert summary["mode"] == "offline_http_fixture" and summary["calls"] == 3
    assert summary["cost_complete"] and summary["unresolved_reservation_cny"] == "0"
    saved = read_json(output / "config.json")
    assert saved["roles"]["agent"]["provider"] == "bailian_deepseek"
    assert saved["resources"]["core_max_steps"] == 24 and saved["limits"]["agent_max_output_tokens"] == 9060
    assert {role for role, body in seen} == {"agent", "user"}
    for role, body in seen:
        if role == "agent":
            assert body["model"] == MODEL and body["enable_thinking"] is True
            assert "preserve_thinking" not in body
        else:
            # The simulator seed is the trial seed, which the official batch runner
            # derives from the run seed rather than using it directly. The persona
            # stays pinned to the run seed, so only sampling differs.
            expected_seed = saved["trial_seeds"][0]
            assert expected_seed != saved["seed"]
            assert body["model"] == "qwen3.7-flash" and body["preserve_thinking"] is False
            assert body["seed"] == expected_seed
    rows = [json.loads(l) for l in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    intent = next(r for r in rows if r["event"] == "dispatch_intent" and r["role"] == "agent")
    assert intent["model_call_id"] and intent["run_id"]
