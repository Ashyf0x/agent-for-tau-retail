"""Prepare fixed-input decision research artifacts only; no execution/credential path."""

import argparse
import hashlib
import json
import sqlite3
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .pilot_config import RESOURCE_PROFILES


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_bytes(encode(value))


def decision_request(original, *, agent_model, core_steps, intervention="", max_bytes=131072,
                     byte_margin=2048, protocol=None):
    """Keep historical Context; optionally apply the explicitly selected model protocol."""
    request = deepcopy(original)
    if request["messages"][0]["role"] != "system":
        raise ValueError("Expected the recorded Retail system message")
    prefix, state_json = request["messages"][0]["content"].rsplit("\n", 1)
    state = json.loads(state_json)
    state["run_state"]["limit"] = core_steps
    # Match Context's compact JSON, without rebuilding or enriching its contents.
    from research_agent.core.types import dumps
    request["messages"][0]["content"] = prefix + "\n" + dumps(state)
    request["model"] = agent_model
    if intervention:
        request["messages"].insert(1, {"role": "system", "content": intervention})
    if protocol is not None:
        from research_agent.core.model import bailian_protocol, chat_request_bytes
        if protocol != "bailian_deepseek" or protocol != bailian_protocol(agent_model):
            raise ValueError("Unreviewed decision snapshot protocol")
        expected = {"model", "messages", "temperature", "max_completion_tokens", "tools",
                    "enable_thinking", "preserve_thinking", "stream"}
        if set(request) - expected or request.get("enable_thinking") is not True:
            raise ValueError("Unexpected source options; refusing silent protocol changes")
        # These wire fields come from the SHA-verified historical dispatch, not
        # arbitrary message content. Restore trusted adapter metadata for replay.
        for message in request["messages"]:
            if message["role"] == "assistant" and "reasoning_content" in message:
                message["_provider_metadata"] = {"reasoning_content": message.pop("reasoning_content")}
        body = chat_request_bytes(agent_model, request["messages"], request.get("tools"),
                                  request["max_completion_tokens"], request["temperature"],
                                  provider=protocol, thinking=True, preserve_thinking=False,
                                  max_completion_tokens=request["max_completion_tokens"])
    else:
        body = encode(request)  # Read-only reproduction of the earlier preparation.
    if len(body) > max_bytes - byte_margin:
        raise ValueError("Intervention exceeds the existing effective Context byte limit")
    return body


def prepare(project, candidate_path, output):
    project, candidate_path, output = Path(project), Path(candidate_path), Path(output)
    candidate = read_json(candidate_path)
    if candidate["real_calls_authorized"] or candidate["business_tool_execution"]:
        raise ValueError("This preparer only accepts an offline, non-executing study")
    resources = RESOURCE_PROFILES[candidate["resource_profile"]]
    if len({s["job_id"].split("-")[1] for s in candidate["snapshots"]}) != len(candidate["snapshots"]):
        raise ValueError("This minimal study selects one decision per independent task")
    arm_text = {arm: (candidate_path.parent / name).read_text(encoding="utf-8").strip() if name else ""
                for arm, name in candidate["arms"].items()}
    audit = read_json(project / "experiments/retail-context-behavior-audit-20261106/requests.json")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "prepared_decision_snapshots", "authorized_for_real_calls": False,
                "source_candidate_sha256": digest(candidate_path.read_bytes()),
                "resources": {"profile": candidate["resource_profile"], **resources},
                "context": candidate["context"], "snapshots": [],
                "arm_text_sha256": {arm: digest(text.encode("utf-8")) for arm, text in arm_text.items()},
                "note": "Fixed Qwen-generated historical prefixes; no future replies, rubric or scores in inputs"}
    for selection in candidate["snapshots"]:
        job, event = selection["job_id"], selection["request_event"]
        item = next(r for r in audit if r["job_id"] == job and r["request_event"] == event)
        task_id = job.split("-")[1]
        source = project / "experiments/retail-context-abc-20261106/runs" / job
        config = read_json(source / "config.json")
        if config["context"] != candidate["context"]:
            raise ValueError("Source Context must equal the fixed study Context")
        if (config["max_steps"] != candidate["official_max_steps"]
                or config["limits"]["max_request_bytes"] != candidate["max_request_bytes"]
                or config["limits"]["max_output_tokens"] != candidate["other_max_completion_tokens"]):
            raise ValueError("Source non-intervention limits differ")
        if any(r["model"] != candidate["other_models_required"] for r in config["roles"].values()):
            raise ValueError("Unexpected historical role model")
        route = deepcopy(config["roles"])
        route["agent"]["model"] = candidate["agent_model"]
        route["agent"]["model_source"] = "explicit --agent-model"
        if candidate.get("agent_protocol"):
            route["agent"]["provider"] = candidate["agent_protocol"]
            route["agent"]["preserve_thinking"] = False
        if "roles" in manifest and manifest["roles"] != route:
            raise ValueError("Role configuration changed across snapshots")
        manifest["roles"] = route
        intents = [r for line in (source / "calls.jsonl").read_text(encoding="utf-8").splitlines()
                   if (r := json.loads(line))["event"] == "dispatch_intent"]
        intent = next(r for r in intents if r.get("model_call_id") == item["model_call_id"])
        raw = encode(intent["request"])
        expected = {"max_completion_tokens": candidate["agent_max_completion_tokens"],
                    "temperature": candidate["temperature"], "stream": False,
                    "enable_thinking": candidate["agent_thinking"],
                    "preserve_thinking": True}  # Frozen historical Qwen request, not the new protocol.
        if any(intent["request"].get(k) != v for k, v in expected.items()):
            raise ValueError("Source model protocol/output differs")
        if digest(raw) != intent["request_sha256"] or digest(raw) != item["actual_sha256"]:
            raise ValueError("Reconstructed original request does not match actual dispatch SHA")
        if intent["request"] != item["actual"]:
            raise ValueError("Audit request differs from original dispatch")
        database = source / f"task-{task_id}/core.sqlite3"
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            event_row = db.execute("SELECT run,body FROM events WHERE seq=?", (event,)).fetchone()
            evidence = json.loads(event_row["body"])
            if evidence["sha256"] != digest(raw) or evidence["model_call_id"] != item["model_call_id"]:
                raise ValueError("Core and dispatch correlation differ")
            prefix = [dict(r) for r in db.execute("SELECT seq,id,run,body FROM messages ORDER BY seq")]
            prefix = prefix[:item["history_records"]]
            for row in prefix:
                row["message"] = json.loads(row.pop("body"))
        folder = output / selection["id"]
        folder.mkdir()
        (folder / "original-request.json").write_bytes(raw)
        save(folder / "source-prefix.json", prefix)
        save(folder / "historical-response.evaluation-only.json", item["response"])
        paths = {}
        for arm, text in arm_text.items():
            data = decision_request(intent["request"], agent_model=candidate["agent_model"],
                                    core_steps=resources["core_max_steps"], intervention=text,
                                    max_bytes=candidate["max_request_bytes"],
                                    byte_margin=candidate["context"]["byte_margin"],
                                    protocol=candidate.get("agent_protocol"))
            (folder / f"{arm}.request.json").write_bytes(data)
            paths[arm] = {"file": f"{selection['id']}/{arm}.request.json", "sha256": digest(data), "bytes": len(data)}
        original_sources = {str(p.relative_to(project)).replace("\\", "/"): digest(p.read_bytes())
                            for p in (source / "config.json", source / "calls.jsonl", database)}
        manifest["snapshots"].append({**selection, "task_id": task_id, "run_id": event_row["run"],
            "model_call_id": item["model_call_id"], "ledger_call_id": intent["call_id"],
            "response_event": item["response_event"], "original_request_sha256": digest(raw),
            "original_sources": original_sources, "source_message_ids": [r["id"] for r in prefix],
            "source_message_seq_max": prefix[-1]["seq"], "requests": paths})
    count = len(manifest["snapshots"]) * len(arm_text) * candidate["repetitions"]
    price = candidate["pricing"]["proposed_agent_cny_per_million"]
    reserve = (Decimal(candidate["max_request_bytes"] + 2048) * Decimal(price["input"])
               + Decimal(candidate["agent_max_completion_tokens"] + 10) * Decimal(price["output"])) / 1000000
    manifest["planned_calls"] = {"agent_decisions": count, "optional_protocol_probe": 2,
                                 "user": 0, "nl": 0, "communicate": 0, "business_tools": 0,
                                 "executed": 0, "repetitions_are_independent_tasks": False}
    manifest["cost_proposal"] = {"per_call_reserve_cny": str(reserve),
        "decision_worst_reserve_cny": str(count * reserve), "probe_worst_reserve_cny": str(2 * reserve),
        "combined_worst_reserve_cny": str((count + 2) * reserve), "approved": False,
        "basis": "Proposed Beijing busy-hour uncached list rates, not Qwen historical prices or actual bill"}
    save(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(Path(__file__).parents[1], args.candidate, args.output)
    print(json.dumps({"snapshots": len(result["snapshots"]), "planned_calls": result["planned_calls"],
                      "cost_proposal": result["cost_proposal"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
