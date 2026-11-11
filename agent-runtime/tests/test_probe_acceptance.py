"""Offline regression of the user-approved final-reasoning acceptance revision."""

import base64
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

import agentcore
import agentcore.core
import agentcore.core.model
import agentcore.core.types
from agentcore.core.types import ModelResponse, ToolCall, Usage
from evals.pilot_calls import Calls, inspect_calls

ORIGINAL = Path(__file__).parent / "fixtures/legacy/deepseek-probe"

# The frozen 2026-11-06 probe runner is read-only evidence and still imports the
# retired package name. Do not rewrite it: alias the old name to the current
# package inside this test process only, so the fixture stays byte-identical.
for _retired, _current in {
    "research_agent": agentcore,
    "research_agent.core": agentcore.core,
    "research_agent.core.model": agentcore.core.model,
    "research_agent.core.types": agentcore.core.types,
}.items():
    sys.modules.setdefault(_retired, _current)

spec = importlib.util.spec_from_file_location("reviewed_probe_runner", ORIGINAL / "run_probe.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)  # Imports only; never invokes main, public preflight or authentication.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_recorded_final_response_passes_revised_gate_without_rewriting_original():
    summary_bytes = (ORIGINAL / "run/summary.json").read_bytes()
    receipt_bytes = (ORIGINAL / "receipt.json").read_bytes()
    ledger_bytes = (ORIGINAL / "run/calls.jsonl").read_bytes()
    rows = [json.loads(line) for line in ledger_bytes.decode("utf-8").splitlines()]
    final = [r for r in rows if r["event"] == "response"][-1]["response"]
    response = ModelResponse(final["content"], [], final["finish_reason"],
                             Usage(**final["usage"]), final["usage_known"], final["provider_metadata"])
    fixture = read(ORIGINAL / "run/synthetic-tool-result.json")["content"]
    assert response.provider_metadata["reasoning_content"] == ""
    runner.validate_final_response(response, fixture)
    assert json.loads(summary_bytes)["probe_passed"] is False
    assert json.loads(receipt_bytes)["result"]["original_runner_probe_passed"] is False
    assert (ORIGINAL / "run/summary.json").read_bytes() == summary_bytes
    assert (ORIGINAL / "receipt.json").read_bytes() == receipt_bytes
    assert (ORIGINAL / "run/calls.jsonl").read_bytes() == ledger_bytes


@pytest.mark.parametrize("reasoning", ["", "valid reasoning"])
def test_final_string_reasoning_allowed(reasoning):
    response = ModelResponse("Value 7, tag fixture-tag", usage_known=True,
                             provider_metadata={"reasoning_content": reasoning})
    runner.validate_final_response(response, {"verification_tag": "fixture-tag"})


@pytest.mark.parametrize("defect", ["missing", "null", "number", "list", "unknown_usage", "tool_call",
                                  "length", "empty_answer", "wrong_tag", "wrong_value"])
def test_other_final_response_failures_remain_rejected(defect):
    response = ModelResponse("Value 7, tag fixture-tag", usage_known=True,
                             provider_metadata={"reasoning_content": ""})
    if defect == "missing":
        response.provider_metadata = {}
    elif defect in {"null", "number", "list"}:
        response.provider_metadata["reasoning_content"] = {"null": None, "number": 1, "list": []}[defect]
    elif defect == "unknown_usage":
        response.usage_known = False
    elif defect == "tool_call":
        response.tool_calls = [ToolCall("extra", "protocol_echo", {"value": 7})]
    elif defect == "length":
        response.finish_reason = "length"
    elif defect == "empty_answer":
        response.content = "  "
    elif defect == "wrong_tag":
        response.content = "Value 7, tag wrong"
    else:
        response.content = "Value 8, tag fixture-tag"
    with pytest.raises(RuntimeError):
        runner.validate_final_response(response, {"verification_tag": "fixture-tag"})


@pytest.mark.parametrize("defect", [None, "first_empty_reasoning", "wrong_tool", "wrong_args",
                                  "two_calls", "bad_call_id", "unknown_final_usage", "HTTP_rejection"])
async def test_actual_probe_flow_with_mock_only_keeps_guards_and_call_association(tmp_path, defect):
    plan = deepcopy(read(ORIGINAL / "run/config.json"))
    for settings in plan["roles"].values():
        settings["base_url"] = "https://fixture.invalid/v1"
    plan["real_calls_authorized"] = False
    plan["kind"] = "offline_acceptance_regression_fixture"
    (tmp_path / "config.json").write_text(json.dumps(plan), encoding="utf-8")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["enable_thinking"] is True and "preserve_thinking" not in body
        if defect == "HTTP_rejection":
            return httpx.Response(400, json={"error": "fixture rejection"})
        if len(seen) == 1:
            assert body["tool_choice"] == {"type": "function", "function": {"name": "protocol_echo"}}
            calls = [{"id": "server-call" if defect != "bad_call_id" else "", "type": "function",
                      "function": {"name": "other" if defect == "wrong_tool" else "protocol_echo",
                                   "arguments": '{"value":8}' if defect == "wrong_args" else '{"value":7}'}}]
            if defect == "two_calls":
                calls.append({**deepcopy(calls[0]), "id": "server-call-2"})
            message = {"content": "", "tool_calls": calls,
                       "reasoning_content": "" if defect == "first_empty_reasoning" else "fixture reasoning"}
            finish = "tool_calls"
        else:
            assert body["messages"][-2]["reasoning_content"] == "fixture reasoning"
            assert body["messages"][-2]["tool_calls"][0]["id"] == "server-call"
            assert body["messages"][-1]["tool_call_id"] == "server-call"
            fixture = json.loads(body["messages"][-1]["content"])
            message = {"content": "Value 7, tag " + fixture["verification_tag"], "reasoning_content": ""}
            finish = "stop"
        payload = {"choices": [{"finish_reason": finish, "message": message}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 30}}
        if defect == "unknown_final_usage" and len(seen) == 2:
            payload.pop("usage")
        return httpx.Response(200, json=payload)

    calls = Calls(tmp_path, plan, transport_factory=lambda role: httpx.MockTransport(handler))
    calls.task_id = "synthetic-only"
    try:
        if defect is None:
            result = await runner.probe(calls, tmp_path)
            assert result["probe_passed"] is True and result["probe_acceptance_version"] == runner.ACCEPTANCE_VERSION
        else:
            with pytest.raises(RuntimeError):
                await runner.probe(calls, tmp_path)
        assert calls.count == len(seen) == (2 if defect in {None, "unknown_final_usage"} else 1)
        if defect in {"unknown_final_usage", "HTTP_rejection"}:
            assert calls.stopped and calls.reserved > 0
        else:
            assert calls.reserved == 0
    finally:
        calls.close()
    report = inspect_calls(tmp_path)
    assert report["calls"] == len(seen)
    assert report["cost_complete"] == (defect not in {"unknown_final_usage", "HTTP_rejection"})


def test_original_raw_request_and_response_association_still_valid():
    rows = [json.loads(line) for line in (ORIGINAL / "run/calls.jsonl").read_text(encoding="utf-8").splitlines()]
    intents = [r for r in rows if r["event"] == "dispatch_intent"]
    raw = [r for r in rows if r["event"] == "http_response"]
    assert len(intents) == len(raw) == 2
    for intent, response in zip(intents, raw, strict=True):
        assert intent["call_id"] == response["call_id"]
        assert hashlib.sha256(base64.b64decode(response["body_base64"])).hexdigest() == response["body_sha256"]
    first = json.loads(base64.b64decode(raw[0]["body_base64"]))["choices"][0]["message"]
    assert intents[1]["request"]["messages"][-2]["reasoning_content"] == first["reasoning_content"]
    assert intents[1]["request"]["messages"][-1]["tool_call_id"] == first["tool_calls"][0]["id"]
