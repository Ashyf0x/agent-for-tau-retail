"""Run Retail directly; save effective configuration and evidence automatically."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .retail import load_retail


def prepare(checkout, task_ids=None):
    lock, loaded, tasks = load_retail(checkout)
    from tau2.evaluator.evaluator_nl_assertions import (
        DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS,
    )
    from tau2.multilingual.runtime import task_digest, resolve_task

    if task_ids is None:
        # Deterministic wiring sample, no selection based on model outcomes.
        task_ids = sorted(tasks, key=int)[:3]
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(t not in tasks for t in task_ids):
        raise ValueError("Choose distinct task IDs from the locked 50-task subset")
    return {
        "kind": "offline_pilot_preparation", "release": lock["release"],
        "commit": lock["commit"], "benchmark": lock["benchmark"],
        "language": "en", "domain": "retail", "seed": 42, "trials": 1,
        "max_steps": 100, "selection": "explicit IDs or first three numeric IDs; no outcome selection",
        "tasks": [{"id": t, "sha256": task_digest(tasks[t]),
                   "reward_basis": tasks[t].evaluation_criteria.reward_basis,
                   "communicate_facts": len(tasks[t].evaluation_criteria.communicate_info or []),
                   "nl_assertions": len(tasks[t].evaluation_criteria.nl_assertions or []),
                   "language_context": resolve_task(loaded, tasks[t], 42).model_dump(mode="json")}
                  for t in task_ids],
        "agent": {"model": None, "base_url": None, "provider": None,
                  "note": "Keep user-selected effective values; no defaults inferred from smoke or .env"},
        "user_simulator": {"implementation": "official UserSimulator", "model": None, "model_args": None},
        "communicate_judge": {"scorer": "semantic", "model": None, "model_args": None},
        "nl_judge": {"upstream_model": DEFAULT_LLM_NL_ASSERTIONS,
                     "upstream_args": DEFAULT_LLM_NL_ASSERTIONS_ARGS,
                     "note": "Upstream module constants, independent of Agent; do not silently override"},
        "budget": None, "authorized_for_real_calls": False,
        "real_model_calls": 0, "real_simulator_calls": 0, "real_judge_calls": 0,
        "official_score": None,
        "not_ready_for_execution": [
            "Freeze effective model/address/protocol and price assumptions for all roles",
            "Authorize NL judge default explicitly or implement an explicit authorized transport adapter",
            "Provide request/usage evidence and one shared dispatch/budget stop boundary across all roles",
            "Obtain fresh authorization for Agent, simulator and all applicable judges",
        ],
    }


def build_config(checkout, task_ids, *, model=None, base_url=None, agent_model=None, budget=None,
           env_file=Path(".env"), input_price="1.2", output_price="4.8",
           max_request_bytes=28000, agent_max_completion_tokens=None,
           context_history="capacity", context_compaction="legacy", resource_profile="relaxed"):
    from .pilot_config import RESOURCE_PROFILES, resolve_roles
    from research_agent.context import ContextManager

    if context_history not in {"recent", "capacity"}:
        raise ValueError("Retail history must be recent or capacity")
    ContextManager("", policy=context_history, compaction=context_compaction)

    task_ids = task_ids or ["6"]
    if len(task_ids) > 2:
        raise ValueError("Minimal Pilot accepts only one or two tasks")
    if type(max_request_bytes) is not int or not 1 <= max_request_bytes <= 131072:
        raise ValueError("Pilot request-byte limit must be 1..131072 (128 KiB)")
    if agent_max_completion_tokens is not None and (type(agent_max_completion_tokens) is not int
                                                   or not 1 <= agent_max_completion_tokens <= 9060):
        raise ValueError("Agent output limit must be 1..9060")
    if resource_profile not in RESOURCE_PROFILES:
        raise ValueError("Unknown Retail resource profile")
    resources = {"profile": resource_profile, **RESOURCE_PROFILES[resource_profile]}
    # A run-wide ceiling, not a per-task multiplier or a batch authorization.
    amount = Decimal(resources["task_budget_cny"] if budget is None else budget)
    if not amount.is_finite() or not Decimal(0) < amount <= Decimal(10):
        raise ValueError("Pilot estimate budget must be positive and at most 10 CNY")
    roles = resolve_roles(model=model, base_url=base_url, agent_model=agent_model, env_file=env_file)
    for price in (input_price, output_price):
        if not Decimal(price).is_finite() or Decimal(price) <= 0:
            raise ValueError("Price assumptions must be finite positive CNY per million tokens")
    config = prepare(checkout, task_ids)
    project = Path(__file__).parents[1]
    sources = (list((project / "research_agent").rglob("*.py"))
               + list((project / "evals").glob("*.py")) + [project / "evals/retail.lock.json"])
    config.update(kind="retail_run_configuration", version="retail-direct-pilot-v3-resources", roles=roles, budget_cny=str(amount),
                resources=resources,
                context={"policy": context_history, "compaction": context_compaction,
                         "byte_margin": 2048, "handoff_max_bytes": 16384},
                configuration_inputs={"model": model, "base_url": base_url},
                max_steps=60, limits={"max_calls": resources["max_calls_per_task"] * len(task_ids), "max_request_bytes": max_request_bytes,
                                      "input_token_proxy_ceiling": max(32000, max_request_bytes + 2048),
                                      "max_output_tokens": 2048},
                price_assumption={"currency": "CNY", "input_cny_per_million": str(input_price),
                                  "output_cny_per_million": str(output_price), "historical_date": "2026-11-05",
                                  "source": "Explicit flat price assumptions; defaults are historical smoke highest rates, not reverified",
                                  "limits": "Byte proxy is not a tokenizer; estimates cannot guarantee account bill"},
                authentication={"source": "--env-file, execution only", "key_name": "DASHSCOPE_API_KEY",
                                "environment_fallback": False},
                source_sha256={str(p.relative_to(project)).replace("\\", "/"):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
                transport="Existing HTTPModelClient, POST /chat/completions, nonstreaming, zero retries",
                memory_extraction=False, automatic_retries=0)
    # Replace obsolete unspecified inventory fields; one authoritative role table.
    for key in ("agent", "user_simulator", "communicate_judge", "nl_judge", "budget",
                "authorized_for_real_calls", "not_ready_for_execution"):
        config.pop(key, None)
    if agent_model is not None:
        config["configuration_inputs"]["agent_model"] = agent_model
    if agent_max_completion_tokens is not None:
        config["limits"]["agent_max_output_tokens"] = agent_max_completion_tokens
    config.pop("real_model_calls", None)
    config.pop("real_simulator_calls", None)
    config.pop("real_judge_calls", None)
    config.pop("nl_judge", None)
    return config


def validate_config(checkout, config):
    from .retail import load_retail
    from .pilot_config import RESOURCE_PROFILES
    lock, loaded, tasks = load_retail(checkout)
    from tau2.multilingual.runtime import task_digest, resolve_task
    if any(config[k] != lock[k] for k in ("commit", "release", "benchmark")):
        raise ValueError("Official version differs")
    if config["version"] != "retail-direct-pilot-v3-resources" or len(config["tasks"]) not in (1, 2):
        raise ValueError("Not a minimal Pilot configuration")
    resources = config["resources"]
    profile = resources.get("profile")
    if profile not in RESOURCE_PROFILES or resources != {"profile": profile, **RESOURCE_PROFILES[profile]}:
        raise ValueError("Invalid Retail resource profile")
    expected_context = {"policy": config["context"]["policy"],
                        "compaction": config["context"]["compaction"],
                        "byte_margin": 2048, "handoff_max_bytes": 16384}
    if (config["context"] != expected_context
            or expected_context["policy"] not in {"recent", "capacity"}
            or expected_context["compaction"] not in {"legacy", "task"}):
        raise ValueError("Invalid Retail context configuration")
    if len({t["id"] for t in config["tasks"]}) != len(config["tasks"]):
        raise ValueError("Duplicate Pilot tasks")
    for task in config["tasks"]:
        if task_digest(tasks[task["id"]]) != task["sha256"]:
            raise ValueError("Task changed")
        if resolve_task(loaded, tasks[task["id"]], 42).model_dump(mode="json") != task["language_context"]:
            raise ValueError("Language/persona differs")
    project = Path(__file__).parents[1]
    current = (list((project / "research_agent").rglob("*.py"))
               + list((project / "evals").glob("*.py")) + [project / "evals/retail.lock.json"])
    hashes = {str(p.relative_to(project)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in current}
    if hashes != config["source_sha256"]:
        raise ValueError("Source changed while preparing the run")
    # Keep fixed resource and protocol bounds for direct API callers as well as the CLI.
    request_cap = config["limits"]["max_request_bytes"]
    if type(request_cap) is not int or not 1 <= request_cap <= 131072:
        raise ValueError("Invalid request-byte limit")
    expected_limits = {"max_calls": resources["max_calls_per_task"] * len(config["tasks"]), "max_request_bytes": request_cap,
                       "input_token_proxy_ceiling": max(32000, request_cap + 2048), "max_output_tokens": 2048}
    if "agent_max_output_tokens" in config["limits"]:
        output_cap = config["limits"]["agent_max_output_tokens"]
        if type(output_cap) is not int or not 1 <= output_cap <= 9060:
            raise ValueError("Invalid Agent output limit")
        expected_limits["agent_max_output_tokens"] = output_cap
    if config["limits"] != expected_limits:
        raise ValueError("Pilot limits differ")
    if not Decimal(config["budget_cny"]).is_finite() or not Decimal(0) < Decimal(config["budget_cny"]) <= Decimal(10):
        raise ValueError("Invalid Pilot budget")
    if config["max_steps"] != 60 or config["seed"] != 42 or config["trials"] != 1:
        raise ValueError("Unexpected Pilot run limits")
    for key in ("input_cny_per_million", "output_cny_per_million"):
        price = Decimal(config["price_assumption"][key])
        if not price.is_finite() or price <= 0:
            raise ValueError("Invalid price assumption")
    from research_agent.core.model import HTTPModelClient
    if set(config["roles"]) != {"agent", "user", "communicate", "nl"}:
        raise ValueError("All Pilot roles must be explicit")
    for role, settings in config["roles"].items():
        if (settings["provider"] != "qwen" or settings["temperature"] != 0
                or settings["thinking"] != (role == "agent")
                or settings["preserve_thinking"] != (role == "agent")):
            raise ValueError("Qwen protocol differs")
        HTTPModelClient(settings["base_url"], settings["model"], provider="qwen", thinking=settings["thinking"])
    return tasks


def run_pilot(checkout, config, output, *, key="", offline=False):
    """Explicit execution only. Offline mode uses local MockTransport for every role."""
    import sqlite3
    from datetime import datetime, timezone
    from .pilot_calls import Calls, inspect_calls
    from .retail_interaction import build_interaction, run_and_score, save_new
    from research_agent.core.agent import AgentConfig
    from research_agent.observability.trajectory import export
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.data_model.simulation import SimulationRun, TerminationReason

    validate_config(checkout, config)
    _, loaded, tasks = load_retail(checkout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    project = Path(__file__).parents[1]
    for name, expected in config["source_sha256"].items():
        source = (project / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Source changed while saving the execution snapshot")
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source)
    factory = offline_transport() if offline else None
    calls = Calls(output, config, key=key, transport_factory=factory)
    completed = []
    try:
        with calls.official_transport():
            for task_plan in config["tasks"]:
                calls.task_id = task_plan["id"]
                directory = output / ("task-" + calls.task_id)
                directory.mkdir()
                database = directory / "core.sqlite3"
                orch = build_interaction(
                    loaded, tasks[calls.task_id], calls.model("agent"), database,
                    user_model=config["roles"]["user"]["model"], user_args={},
                    communicate_scoring=CommunicateScoring(judge={"model": config["roles"]["communicate"]["model"], "model_args": {}}),
                    seed=config["seed"], max_steps=config["max_steps"],
                    context_options={**config["context"],
                                     "output_reserve_tokens": calls.output_limit("agent")},
                    config=AgentConfig(max_steps=config["resources"]["core_max_steps"],
                                       max_request_bytes=config["limits"]["max_request_bytes"],
                                       max_argument_corrections=0, max_protocol_corrections=0))
                calls.core_store = orch.agent.store
                started = datetime.now(timezone.utc)
                try:
                    simulation = run_and_score(orch, loaded, directory / "official")
                    completed.append({"task_id": calls.task_id,
                                      "reward": simulation.reward_info.reward,
                                      "termination_reason": simulation.termination_reason.value})
                except BaseException:
                    # If interaction itself failed, retain the partial official schema
                    # without manufacturing a score. Judge failure already has unscored.
                    target = directory / "official" / "simulation.unscored.json"
                    if not target.exists():
                        target.parent.mkdir(exist_ok=True)
                        now = datetime.now(timezone.utc)
                        partial = SimulationRun(
                            id=orch.simulation_id, task_id=calls.task_id,
                            start_time=started.isoformat(), end_time=now.isoformat(),
                            duration=(now-started).total_seconds(), seed=config["seed"],
                            termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                            messages=orch.get_trajectory(), language_context=orch.call_context.language,
                            policy=orch.environment.get_policy())
                        save_new(target, partial)
                    raise
                finally:
                    orch.agent.stop()
                    calls.core_store = None
                    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
                        run_ids = [row[0] for row in db.execute("SELECT id FROM runs")]
                    for run_id in run_ids:
                        export(database, run_id, directory / f"trajectory-{run_id}.json")
                        export(database, run_id, directory / f"trajectory-{run_id}.md", format="markdown")
                if calls.stopped:
                    break
    finally:
        if len(completed) != len(config["tasks"]):
            calls.stopped = True
            calls.stop_reason = calls.stop_reason or "interaction_or_scoring_failed"
        calls.close()
        summary = {"mode": "offline_http_fixture" if offline else "authorized_real_pilot",
                   "finished_tasks": completed, "calls": calls.count,
                   "estimated_cny": str(calls.spent), "unresolved_reservation_cny": str(calls.reserved),
                   "stopped": calls.stopped, "all_tasks_finished": len(completed) == len(config["tasks"]),
                   "official_score": None, "note": "Fixture scores are wiring only; small real Pilot is not a baseline"}
        summary.update(inspect_calls(output), stop_reason=calls.stop_reason)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def offline_transport():
    """Scripted replies on the actual HTTP adapter; never a benchmark solution."""
    import httpx
    counts = {}

    def factory(role):
        def handler(request):
            counts[role] = counts.get(role, 0) + 1
            if role == "user":
                content = "My email is noah.brown7922@example.com. Confirm my account email." if counts[role] % 2 else "###STOP###"
            elif role == "agent":
                content = "I cannot complete this request in this offline fixture."
            elif role == "communicate":
                content = '{"reasoning":"offline fixture","met":false}'
            else:
                content = '{"results":[]}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 100, "completion_tokens": 30}})
        return httpx.MockTransport(handler)
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(".vendor/tau2-bench"))
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--agent-model", help="Override only the Agent model; all role URLs stay configured")
    parser.add_argument("--agent-max-completion-tokens", type=int,
                        help="Agent-only output cap (1..9060); other roles retain 2048")
    parser.add_argument("--base-url")
    parser.add_argument("--context-history", choices=("recent", "capacity"), default="capacity",
                        help="Agent history retention: recent=original three user turns; capacity=budget-aware")
    parser.add_argument("--context-compaction", choices=("legacy", "task"), default="legacy",
                        help="Old-history summary used only when history is omitted")
    from .pilot_config import RESOURCE_PROFILES
    parser.add_argument("--resource-profile", choices=tuple(RESOURCE_PROFILES), default="relaxed",
                        help="Project resource caps: relaxed=80 calls/24 Core steps/3 CNY per task; legacy=40/12/1")
    parser.add_argument("--budget-yuan", help="Run-wide estimate cap (not multiplied by tasks); defaults to profile task cap")
    parser.add_argument("--max-request-bytes", type=int,
                        help="Explicit local request cap (up to 131072 / 128 KiB); input reservation grows with it")
    parser.add_argument("--input-price", help="Flat estimate CNY per million input tokens")
    parser.add_argument("--output-price", help="Flat estimate CNY per million output tokens")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true", help="Real calls for all applicable roles; user execution only")
    mode.add_argument("--offline-check", action="store_true", help="Scripted HTTP wiring only; not a benchmark score")
    mode.add_argument("--check-config", action="store_true", help="Show effective public role settings; no requests or authentication")
    mode.add_argument("--inspect", action="store_true", help="Read persisted call evidence only; never resumes")
    args = parser.parse_args()
    from .pilot_config import resolve_roles
    if args.check_config:
        print(json.dumps({"roles": resolve_roles(model=args.model, base_url=args.base_url,
                                                 agent_model=args.agent_model,
                                                 env_file=args.env_file),
                          "credentials_loaded": False, "requests_sent": 0}, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --check-config is used")
    if args.inspect:
        from .pilot_calls import inspect_calls
        print(json.dumps(inspect_calls(args.output), ensure_ascii=False, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("Execution output already exists; inspect, do not resume")
    if args.offline_check and (args.model is None or args.base_url is None):
        parser.error("--offline-check requires explicit --model and --base-url; it never reads .env")
    config = build_config(
        args.checkout, args.task_id, model=args.model, agent_model=args.agent_model,
        base_url=args.base_url, budget=args.budget_yuan, env_file=args.env_file,
        resource_profile=args.resource_profile,
        input_price=args.input_price or "1.2", output_price=args.output_price or "4.8",
        context_history=args.context_history, context_compaction=args.context_compaction,
        agent_max_completion_tokens=args.agent_max_completion_tokens,
        max_request_bytes=args.max_request_bytes if args.max_request_bytes is not None else 28000)
    validate_config(args.checkout, config)
    key = ""
    if args.execute:
        # Authentication is loaded only after public settings and bounds are checked.
        from research_agent.app.smoke import _load_key
        key = _load_key(args.env_file)
    print(json.dumps(run_pilot(args.checkout, config, args.output, key=key,
                               offline=args.offline_check), ensure_ascii=False))


if __name__ == "__main__":
    main()
