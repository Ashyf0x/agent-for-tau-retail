"""One newly authorized protocol probe: at most two calls, no Retail execution."""

import argparse
import asyncio
import base64
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from evals.pilot_calls import Calls, inspect_calls
from evals.pilot_config import resolve_roles
from research_agent.core.model import HTTPModelClient
from research_agent.core.types import PreparedRequest

ACCEPTANCE_VERSION = "protocol-probe-v2-final-reasoning-string"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def public_preflight():
    prior = PROJECT / "experiments/retail-deepseek-protocol-20261106"
    stages = read(prior / "stages.json")
    frozen = read(prior / "source-sha256.json")
    if any(sha((PROJECT / name).read_bytes()) != expected for name, expected in frozen.items()):
        raise RuntimeError("Frozen runtime changed; stop before authentication")
    roles = resolve_roles(agent_model=stages["roles"]["agent"]["model"], env_file=PROJECT / ".env")
    fields = ("model", "base_url", "provider", "thinking", "preserve_thinking", "temperature", "provider_seed")
    if any(roles[r][f] != stages["roles"][r][f] for r in roles for f in fields):
        raise RuntimeError("Effective public configuration differs from the reviewed stage plan")
    template = read(prior / "protocol-mock/config.json")
    plan = {**template, "kind": "authorized_real_protocol_probe", "roles": roles,
            "real_calls_authorized": True, "context": stages["context"],
            "authorization": {"scope": "protocol probe only", "max_calls": 2, "budget_cny": "0.70",
                              "A_snapshots_authorized": False, "other_role_calls_authorized": False,
                              "business_tool_execution_authorized": False},
            "source_sha256": frozen, "runner_sha256": sha(Path(__file__).read_bytes()),
            "probe_acceptance_version": ACCEPTANCE_VERSION,
            "tool_choice_first": {"type": "function", "function": {"name": "protocol_echo"}},
            "tool_choice_second": "default auto; no forced second tool call",
            "created_utc": datetime.now(timezone.utc).isoformat()}
    assert plan["budget_cny"] == "0.70" and plan["limits"]["max_calls"] == 2
    assert plan["limits"]["agent_max_output_tokens"] == 9060
    assert plan["price_assumption"] == {"input_cny_per_million": "2", "output_cny_per_million": "8"}
    return plan


TOOLS = [{"type": "function", "function": {
    "name": "protocol_echo", "description": "Synthetic protocol fixture only; no business effect.",
    "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                   "required": ["value"], "additionalProperties": False}}}]
OPENING = "Call protocol_echo exactly once with value 7. After receiving its tool result, report the returned value and verification_tag. This is a synthetic protocol fixture, not a retail operation."


def prepare(model, messages, *, first=False):
    request = model.prepare(messages, TOOLS, 131072)
    if first:
        body = json.loads(request.body)
        body["tool_choice"] = {"type": "function", "function": {"name": "protocol_echo"}}
        request = PreparedRequest(json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"), 131072)
    request.validate()
    if len(request.body) > 129024:
        raise RuntimeError("Existing effective Context boundary exceeded")
    return request


def validate_final_response(second, fixture):
    """Final natural replies require a string reasoning field, which may be empty.

    User-approved acceptance revision; first tool reply and all other gates stay
    strict. Historical summaries and the original frozen runner are unchanged.
    """
    if not second.usage_known or not isinstance(second.provider_metadata.get("reasoning_content"), str):
        raise RuntimeError("second_response_local_reasoning_or_usage_contract_failed")
    if second.finish_reason != "stop" or second.tool_calls or not (second.content or "").strip():
        raise RuntimeError("second_response_final_answer_contract_failed")
    if fixture["verification_tag"] not in second.content or "7" not in second.content:
        raise RuntimeError("second_response_fixture_result_not_demonstrated")


async def probe(calls, output):
    model = calls.model("agent", call_name="authorized-synthetic-protocol-probe")
    messages = [{"role": "user", "content": OPENING}]
    first = await model.complete(prepare(model, messages, first=True))
    if not first.usage_known or not first.provider_metadata.get("reasoning_content"):
        raise RuntimeError("first_response_local_reasoning_or_usage_contract_failed")
    if first.finish_reason != "tool_calls" or len(first.tool_calls) != 1:
        raise RuntimeError("first_response_native_tool_contract_failed")
    call = first.tool_calls[0]
    arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
    if call.name != "protocol_echo" or arguments != {"value": 7}:
        raise RuntimeError("first_response_tool_name_or_arguments_failed")
    print(json.dumps({"stage": "first_response_checked", "calls": calls.count,
                      "usage": asdict(first.usage), "reasoning_present": True,
                      "native_tool_valid": True}), flush=True)
    fixture = {"fixture_only": True, "value": 7, "verification_tag": "probe-" + uuid4().hex}
    save(output / "synthetic-tool-result.json", {"tool_call_id": call.id, "content": fixture,
                                                 "note": "Program-supplied fixture; no function or business tool executed"})
    messages += [{"role": "assistant", "content": first.content,
                  "_provider_metadata": first.provider_metadata, "tool_calls": [call.to_message()]},
                 {"role": "tool", "tool_call_id": call.id,
                  "content": json.dumps(fixture, ensure_ascii=False)}]
    second = await model.complete(prepare(model, messages))
    validate_final_response(second, fixture)
    return {"probe_passed": True, "native_tool_call_checked": True, "reasoning_fields_checked": True,
            "probe_acceptance_version": ACCEPTANCE_VERSION,
            "actual_first_reasoning_replayed": True, "tool_result_call_id_matched": True,
            "new_fixture_tag_reflected_in_final_answer": True, "usage_known_both": True}


def raw_response_diagnostics(output):
    rows = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    result = []
    for row in rows:
        if row["event"] != "http_response":
            continue
        entry = {"call_id": row["call_id"], "http_status": row["status_code"],
                 "body_sha256": row["body_sha256"]}
        try:
            data = json.loads(base64.b64decode(row["body_base64"]))
            message = data["choices"][0]["message"]
            reasoning = message.get("reasoning_content")
            entry.update(native_tool_calls_present=bool(message.get("tool_calls")),
                         reasoning_field_present="reasoning_content" in message,
                         reasoning_type=type(reasoning).__name__,
                         reasoning_nonempty=isinstance(reasoning, str) and bool(reasoning),
                         usage_present=isinstance(data.get("usage"), dict),
                         reported_model=data.get("model"))
            if message.get("tool_calls") and not isinstance(reasoning, str):
                entry["classification"] = "native_tool_call_present_but_local_reasoning_contract_mismatch"
                entry["does_not_disprove_model_tool_support"] = True
        except (KeyError, IndexError, TypeError, ValueError):
            entry["classification"] = "HTTP_or_response_shape_rejected; inspect preserved raw body"
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = public_preflight()
    if not args.execute:
        settings = plan["roles"]["agent"]
        client = HTTPModelClient(settings["base_url"], settings["model"], provider=settings["provider"],
                                 thinking=True, max_completion_tokens=9060)
        request = prepare(client, [{"role": "user", "content": OPENING}], first=True)
        save(ROOT / "public-preflight.json", {"configuration_matches_reviewed_stage": True,
             "source_hashes_match": True, "auth_loaded": False, "requests_sent": 0,
             "first_request_sha256": sha(request.body), "first_request_bytes": len(request.body),
             "first_tool_choice": plan["tool_choice_first"],
             "models": {r: s["model"] for r, s in plan["roles"].items()},
             "budget_cny": "0.70", "max_calls": 2})
        print(json.dumps(read(ROOT / "public-preflight.json")), flush=True)
        return
    output = ROOT / "run"
    output.mkdir(exist_ok=False)  # Never resume, overwrite or auto-repeat a probe.
    save(output / "authorization.json", plan["authorization"])
    save(output / "config.json", plan)
    for name, expected in plan["source_sha256"].items():
        data = (PROJECT / name).read_bytes()
        if sha(data) != expected:
            raise RuntimeError("Source drift before authentication")
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / "source-snapshot/run_probe.py").write_bytes(Path(__file__).read_bytes())
    from research_agent.app.smoke import _load_key
    key = _load_key(PROJECT / ".env")
    calls = Calls(output, plan, key=key)
    calls.task_id = "synthetic-protocol-fixture"
    result = {"probe_passed": False}
    try:
        result = asyncio.run(probe(calls, output))
    except BaseException as exc:
        calls.stopped = True
        calls.stop_reason = calls.stop_reason or getattr(exc, "kind", type(exc).__name__)
        calls.record(event="probe_stopped", reason=calls.stop_reason,
                     error_type=type(exc).__name__, diagnostics=getattr(exc, "diagnostics", {}))
        result["failure_type"] = type(exc).__name__
        result["failure_category"] = getattr(exc, "kind", "local_probe_contract")
        # These locally generated contract labels contain no server text or credentials.
        if type(exc) is RuntimeError and str(exc).startswith(("first_response_", "second_response_")):
            result["failure_contract"] = str(exc)
    finally:
        calls.close()
        summary = {"mode": "authorized_real_protocol_probe", **result, **inspect_calls(output),
                   "stopped": calls.stopped, "stop_reason": calls.stop_reason,
                   "A_snapshots_executed": 0, "A_snapshots_authorized": False,
                   "simulator_or_judge_calls": 0, "business_tools_executed": 0,
                   "official_score": None, "raw_response_diagnostics": raw_response_diagnostics(output)}
        save(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("probe_passed", "calls", "estimated_cny",
              "unresolved_reservation_cny", "cost_complete", "stop_reason", "A_snapshots_executed")}), flush=True)


if __name__ == "__main__":
    main()
