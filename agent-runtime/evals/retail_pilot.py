"""Run Retail directly; save effective configuration and evidence automatically."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .retail import load_retail


def trial_seeds(seed, trials):
    """Official per-trial seeds: batch.py draws them from the run-level seed.

    Upstream `runner/batch.py` does `random.seed(config.seed)` and then draws
    `num_trials` values, so the trial list is a deterministic function of the
    run seed. Reproducing that here keeps our multi-trial layout identical to
    the official one instead of inventing a different convention.
    """
    import random
    random.seed(seed)
    return [random.randint(0, 1000000) for _ in range(trials)]


def prepare(checkout, task_ids=None, *, seed=42, trials=1):
    lock, loaded, tasks = load_retail(checkout)
    from tau2.evaluator.evaluator_nl_assertions import (
        DEFAULT_LLM_NL_ASSERTIONS,
        DEFAULT_LLM_NL_ASSERTIONS_ARGS,
    )
    from tau2.multilingual.runtime import resolve_task, task_digest

    if task_ids is None:
        # Deterministic wiring sample, no selection based on model outcomes.
        task_ids = sorted(tasks, key=int)[:3]
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(t not in tasks for t in task_ids):
        raise ValueError("Choose distinct task IDs from the locked 50-task subset")
    # Upstream separates the two: persona/language are resolved once per run from
    # the run-level seed (runner/helpers.py), while each trial gets its own seed
    # for the orchestrator (runner/batch.py). Keeping only one seed would silently
    # change the caller persona between trials, which is not a repeated measurement.
    return {
        "kind": "offline_pilot_preparation", "release": lock["release"],
        "commit": lock["commit"], "benchmark": lock["benchmark"],
        "language": "en", "domain": "retail", "seed": seed, "trials": trials,
        "trial_seeds": trial_seeds(seed, trials),
        "max_steps": 100, "selection": "explicit IDs or first three numeric IDs; no outcome selection",
        "tasks": [{"id": t, "sha256": task_digest(tasks[t]),
                   "reward_basis": tasks[t].evaluation_criteria.reward_basis,
                   "communicate_facts": len(tasks[t].evaluation_criteria.communicate_info or []),
                   "nl_assertions": len(tasks[t].evaluation_criteria.nl_assertions or []),
                   "language_context": resolve_task(loaded, tasks[t], seed).model_dump(mode="json")}
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
           env_file=Path(".env"), input_price=None, output_price=None,
           max_request_bytes=28000, agent_max_completion_tokens=None,
           context_history="capacity", context_compaction="legacy", resource_profile="relaxed",
           failure_reason_chars=0, role_models=None, max_steps=60, role_thinking=None,
           role_prices=None, role_max_tokens=None, timeout_seconds=60, seed=42, trials=1):
    from agentcore.context import ContextManager

    from .pilot_config import RESOURCE_PROFILES, resolve_roles

    if type(seed) is not int or seed < 0:
        raise ValueError("Run seed must be a non-negative integer")
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("Trials must be 1..10; more repeats are a different study, not this Pilot")

    if context_history not in {"recent", "capacity"}:
        raise ValueError("Retail history must be recent or capacity")
    ContextManager("", policy=context_history, compaction=context_compaction,
                   failure_reason_chars=failure_reason_chars)

    task_ids = task_ids or ["6"]
    if len(task_ids) > 30:
        raise ValueError("Retail batch accepts at most the 30 development tasks per invocation")
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
    if not amount.is_finite() or not Decimal(0) < amount <= Decimal(30):
        raise ValueError("Pilot estimate budget must be positive and at most 30 CNY")
    if role_models or role_thinking:
        roles = resolve_roles(model=model, base_url=base_url, agent_model=agent_model,
                              role_models=role_models, role_thinking=role_thinking,
                              env_file=env_file)
    else:
        roles = resolve_roles(model=model, base_url=base_url, agent_model=agent_model, env_file=env_file)
    if any(r["provider"] == "bailian_deepseek" for r in roles.values()) and (input_price is None or output_price is None):
        raise ValueError("Bailian DeepSeek requires explicit input/output price assumptions; Qwen defaults do not apply")
    input_price = "1.2" if input_price is None else input_price
    output_price = "4.8" if output_price is None else output_price
    for price in (input_price, output_price):
        if not Decimal(price).is_finite() or Decimal(price) <= 0:
            raise ValueError("Price assumptions must be finite positive CNY per million tokens")
    if type(max_steps) is not int or not 1 <= max_steps <= 200:
        raise ValueError("max_steps must be 1..200 (official default is 200)")
    per_role_prices = {}
    if role_prices:
        unknown_roles = set(role_prices) - {"agent", "user", "communicate", "nl"}
        if unknown_roles:
            raise ValueError(f"Unknown role(s) for price override: {sorted(unknown_roles)}")
        for role, pair in role_prices.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"price for {role} must be (input, output) CNY per million")
            for value in pair:
                if not Decimal(str(value)).is_finite() or Decimal(str(value)) <= 0:
                    raise ValueError(f"Role price for {role} must be finite positive")
            per_role_prices[role] = {"input_cny_per_million": str(pair[0]),
                                     "output_cny_per_million": str(pair[1])}
    role_output_caps = {}
    if role_max_tokens:
        unknown_roles = set(role_max_tokens) - {"agent", "user", "communicate", "nl"}
        if unknown_roles:
            raise ValueError(f"Unknown role(s) for output cap: {sorted(unknown_roles)}")
        for role, cap in role_max_tokens.items():
            if type(cap) is not int or not 1 <= cap <= 32000:
                raise ValueError(f"Output cap for {role} must be 1..32000")
            role_output_caps[role] = cap
    # The client default of 60s was tuned for short non-thinking replies. With
    # thinking on, a long user-simulator prompt can exceed it and the whole batch
    # stops on a transport timeout that says nothing about model capability.
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be 1..600")
    config = prepare(checkout, task_ids, seed=seed, trials=trials)
    project = Path(__file__).parents[1]
    sources = (list((project / "agentcore").rglob("*.py"))
               + list((project / "evals").glob("*.py")) + [project / "evals/retail.lock.json"])
    # Opt-in only: when the option is off, the context block stays byte-identical to
    # the frozen decision-study snapshot. A new key is added only when actually used.
    context = {"policy": context_history, "compaction": context_compaction,
               "byte_margin": 2048, "handoff_max_bytes": 16384}
    if failure_reason_chars:
        context["failure_reason_chars"] = failure_reason_chars
    config.update(kind="retail_run_configuration", version="retail-direct-pilot-v3-resources", roles=roles, budget_cny=str(amount),
                resources=resources,
                context=context,
                configuration_inputs={"model": model, "base_url": base_url,
                                      "role_models": dict(role_models) if role_models else None,
                                      "role_thinking": dict(role_thinking) if role_thinking else None},
                max_steps=max_steps, limits={**{"max_calls": resources["max_calls_per_task"] * len(task_ids) * trials, "max_request_bytes": max_request_bytes,
                                      "input_token_proxy_ceiling": max(32000, max_request_bytes + 2048),
                                      "max_output_tokens": 2048},
                                      **({"role_max_output_tokens": role_output_caps} if role_output_caps else {})},
                timeout_seconds=timeout_seconds,
                price_assumption={**{"currency": "CNY", "input_cny_per_million": str(input_price),
                                  "output_cny_per_million": str(output_price), "historical_date": "2026-11-05",
                                  "source": "Explicit flat price assumptions; defaults are historical smoke highest rates, not reverified",
                                  "limits": "Byte proxy is not a tokenizer; estimates cannot guarantee account bill"},
                                  **({"per_role": per_role_prices} if per_role_prices else {})},
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
    from .pilot_config import RESOURCE_PROFILES
    from .retail import load_retail
    lock, loaded, tasks = load_retail(checkout)
    from tau2.multilingual.runtime import resolve_task, task_digest
    if any(config[k] != lock[k] for k in ("commit", "release", "benchmark")):
        raise ValueError("Official version differs")
    if config["version"] != "retail-direct-pilot-v3-resources" or not 1 <= len(config["tasks"]) <= 30:
        raise ValueError("Not a minimal Pilot configuration")
    # Multi-trial is opt-in and self-consistent: the seed list must be exactly what
    # the frozen run seed produces, so an edited config cannot smuggle in extra runs.
    # Configs written before multi-trial existed have no trial_seeds at all; they are
    # single-trial evidence and stay replayable rather than failing a check they predate.
    if type(config.get("trials")) is not int or not 1 <= config["trials"] <= 10:
        raise ValueError("Unexpected trial count")
    if config.get("trial_seeds") != trial_seeds(config["seed"], config["trials"]):
        if not ("trial_seeds" not in config and config["trials"] == 1):
            raise ValueError("Trial seeds differ from the frozen run seed")
    resources = config["resources"]
    profile = resources.get("profile")
    if profile not in RESOURCE_PROFILES or resources != {"profile": profile, **RESOURCE_PROFILES[profile]}:
        raise ValueError("Invalid Retail resource profile")
    expected_context = {"policy": config["context"]["policy"],
                        "compaction": config["context"]["compaction"],
                        "byte_margin": 2048, "handoff_max_bytes": 16384}
    if "failure_reason_chars" in config["context"]:
        expected_context["failure_reason_chars"] = config["context"]["failure_reason_chars"]
    if (config["context"] != expected_context
            or expected_context["policy"] not in {"recent", "capacity"}
            or expected_context["compaction"] not in {"legacy", "task"}):
        raise ValueError("Invalid Retail context configuration")
    if len({t["id"] for t in config["tasks"]}) != len(config["tasks"]):
        raise ValueError("Duplicate Pilot tasks")
    for task in config["tasks"]:
        if task_digest(tasks[task["id"]]) != task["sha256"]:
            raise ValueError("Task changed")
        if resolve_task(loaded, tasks[task["id"]], config["seed"]).model_dump(mode="json") != task["language_context"]:
            raise ValueError("Language/persona differs")
    project = Path(__file__).parents[1]
    current = (list((project / "agentcore").rglob("*.py"))
               + list((project / "evals").glob("*.py")) + [project / "evals/retail.lock.json"])
    hashes = {str(p.relative_to(project)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in current}
    if hashes != config["source_sha256"]:
        raise ValueError("Source changed while preparing the run")
    # Keep fixed resource and protocol bounds for direct API callers as well as the CLI.
    request_cap = config["limits"]["max_request_bytes"]
    if type(request_cap) is not int or not 1 <= request_cap <= 131072:
        raise ValueError("Invalid request-byte limit")
    expected_limits = {"max_calls": resources["max_calls_per_task"] * len(config["tasks"]) * config["trials"], "max_request_bytes": request_cap,
                       "input_token_proxy_ceiling": max(32000, request_cap + 2048), "max_output_tokens": 2048}
    if "agent_max_output_tokens" in config["limits"]:
        output_cap = config["limits"]["agent_max_output_tokens"]
        if type(output_cap) is not int or not 1 <= output_cap <= 9060:
            raise ValueError("Invalid Agent output limit")
        expected_limits["agent_max_output_tokens"] = output_cap
    if "role_max_output_tokens" in config["limits"]:
        caps = config["limits"]["role_max_output_tokens"]
        if not isinstance(caps, dict) or not caps:
            raise ValueError("Invalid per-role output caps")
        for role, cap in caps.items():
            if role not in {"agent", "user", "communicate", "nl"} or type(cap) is not int or not 1 <= cap <= 32000:
                raise ValueError("Invalid per-role output cap")
        expected_limits["role_max_output_tokens"] = caps
    if config["limits"] != expected_limits:
        raise ValueError("Pilot limits differ")
    if type(config.get("timeout_seconds")) is not int or not 1 <= config["timeout_seconds"] <= 600:
        raise ValueError("Invalid request timeout")
    if not Decimal(config["budget_cny"]).is_finite() or not Decimal(0) < Decimal(config["budget_cny"]) <= Decimal(30):
        raise ValueError("Invalid Pilot budget")
    if not isinstance(config["max_steps"], int) or not 1 <= config["max_steps"] <= 200:
        raise ValueError("Unexpected Pilot run limits")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Unexpected Pilot run limits")
    for key in ("input_cny_per_million", "output_cny_per_million"):
        price = Decimal(config["price_assumption"][key])
        if not price.is_finite() or price <= 0:
            raise ValueError("Invalid price assumption")
    for role, rates in (config["price_assumption"].get("per_role") or {}).items():
        if role not in {"agent", "user", "communicate", "nl"}:
            raise ValueError("Unknown role in price table")
        for key in ("input_cny_per_million", "output_cny_per_million"):
            price = Decimal(rates[key])
            if not price.is_finite() or price <= 0:
                raise ValueError("Invalid per-role price assumption")
    from agentcore.core.model import HTTPModelClient, bailian_protocol
    if set(config["roles"]) != {"agent", "user", "communicate", "nl"}:
        raise ValueError("All Pilot roles must be explicit")
    # thinking is a per-role frozen field: a role table may deliberately give the
    # simulator more reasoning than the caller to reproduce the official order.
    for role, settings in config["roles"].items():
        expected_provider = bailian_protocol(settings["model"])
        if (settings["provider"] != expected_provider or settings["temperature"] != 0
                or type(settings["thinking"]) is not bool
                or settings["preserve_thinking"] != (role == "agent" and expected_provider == "qwen"
                                                     and settings["thinking"])):
            raise ValueError("Model protocol differs")
        HTTPModelClient(settings["base_url"], settings["model"], provider=settings["provider"],
                        thinking=settings["thinking"], preserve_thinking=settings["preserve_thinking"])
    return tasks


def run_pilot(checkout, config, output, *, key="", offline=False):
    """Explicit execution only. Offline mode uses local MockTransport for every role."""
    import sqlite3
    from datetime import datetime, timezone

    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.data_model.simulation import SimulationRun, TerminationReason

    from agentcore.core.agent import AgentConfig
    from agentcore.observability.trajectory import export

    from .pilot_calls import Calls, inspect_calls
    from .retail_interaction import build_interaction, run_and_score, save_new

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
    plan = [(trial, config["trial_seeds"][trial], task_plan)
            for trial in range(config["trials"]) for task_plan in config["tasks"]]
    # Tasks within one trial keep their order; trials are the outer loop, matching
    # upstream's `for trial in range(num_trials): for task in tasks`.
    completed = []
    try:
        with calls.official_transport():
            for trial, trial_seed, task_plan in plan:
                calls.task_id = task_plan["id"]
                calls.trial_seed = trial_seed
                calls.task_key = (calls.task_id if config["trials"] == 1
                                  else f"{calls.task_id}#t{trial + 1}")
                directory = output / ("task-" + calls.task_id)
                if config["trials"] > 1:
                    directory = directory / f"trial-{trial + 1}"
                directory.mkdir(parents=True, exist_ok=config["trials"] > 1)
                database = directory / "core.sqlite3"
                orch = build_interaction(
                    loaded, tasks[calls.task_id], calls.model("agent"), database,
                    user_model=config["roles"]["user"]["model"], user_args={},
                    communicate_scoring=CommunicateScoring(judge={"model": config["roles"]["communicate"]["model"], "model_args": {}}),
                    # The trial seed drives the orchestrator and the simulator's
                    # per-request seed; the persona stays pinned to config["seed"],
                    # because build_interaction resolves the language context with
                    # the run seed, exactly as upstream runner/helpers.py does.
                    seed=trial_seed, persona_seed=config["seed"], max_steps=config["max_steps"],
                    context_options={**config["context"],
                                     "output_reserve_tokens": calls.output_limit("agent")},
                    config=AgentConfig(max_steps=config["resources"]["core_max_steps"],
                                       max_request_bytes=config["limits"]["max_request_bytes"],
                                       # The Core default of 60s wraps the HTTP timeout and
                                       # fires first, so raising only the HTTP timeout does
                                       # nothing. Keep both in step.
                                       model_timeout=config["timeout_seconds"],
                                       max_argument_corrections=0, max_protocol_corrections=0))
                calls.core_store = orch.agent.store
                started = datetime.now(timezone.utc)
                try:
                    simulation = run_and_score(orch, loaded, directory / "official")
                    completed.append({"task_id": calls.task_id, "trial": trial + 1,
                                      "seed": trial_seed,
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
                            duration=(now-started).total_seconds(), seed=trial_seed,
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
        if len(completed) != len(plan):
            calls.stopped = True
            calls.stop_reason = calls.stop_reason or "interaction_or_scoring_failed"
        calls.close()
        summary = {"mode": "offline_http_fixture" if offline else "authorized_real_pilot",
                   "trials": config["trials"], "trial_seeds": config["trial_seeds"],
                   "planned_simulations": len(plan),
                   "finished_tasks": completed, "calls": calls.count,
                   "estimated_cny": str(calls.spent), "unresolved_reservation_cny": str(calls.reserved),
                   "stopped": calls.stopped, "all_tasks_finished": len(completed) == len(plan),
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
    parser.add_argument("--user-model", help="Override only the user simulator model")
    parser.add_argument("--nl-model", help="Override only the NL-assertion judge model")
    parser.add_argument("--communicate-model", help="Override only the semantic communication judge model")
    parser.add_argument("--role-thinking", action="append", default=None, metavar="ROLE=BOOL",
                        help="Explicit per-role thinking flag, repeatable (e.g. user=true). "
                             "Frozen into the plan and re-validated; default keeps caller-only thinking")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Maximum conversation turns (official default 200)")
    parser.add_argument("--timeout-seconds", type=int, default=60,
                        help="Per-request HTTP timeout (1..600). The 60s client default is short "
                             "for thinking replies with long prompts and surfaces as a transport "
                             "error, not as a model failure")
    parser.add_argument("--agent-max-completion-tokens", type=int,
                        help="Agent-only output cap (1..9060); other roles retain 2048")
    parser.add_argument("--role-max-tokens", action="append", default=None, metavar="ROLE=N",
                        help="Per-role output cap (1..32000), repeatable (e.g. user=8192). "
                             "Needed when a role runs with thinking on, since reasoning tokens "
                             "count against this cap and can exhaust it before any content is emitted")
    parser.add_argument("--base-url")
    parser.add_argument("--context-history", choices=("recent", "capacity"), default="capacity",
                        help="Agent history retention: recent=original three user turns; capacity=budget-aware")
    parser.add_argument("--context-compaction", choices=("legacy", "task"), default="legacy",
                        help="Old-history summary used only when history is omitted")
    parser.add_argument("--failure-reason-chars", type=int, default=0,
                        help="0 keeps the default lossy mask (failed value hidden entirely); "
                             "1..240 adds a bounded untrusted excerpt of the rejection reason")
    from .pilot_config import RESOURCE_PROFILES
    parser.add_argument("--resource-profile", choices=tuple(RESOURCE_PROFILES), default="relaxed",
                        help="Project resource caps: relaxed=80 calls/24 Core steps/3 CNY per task; legacy=40/12/1")
    parser.add_argument("--budget-yuan", help="Run-wide estimate cap (not multiplied by tasks); defaults to profile task cap")
    parser.add_argument("--seed", type=int, default=42,
                        help="Run-level seed: fixes the caller persona and derives the per-trial seeds "
                             "the way the official batch runner does (default 42)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Repeat each task N times with per-trial seeds (default 1). The persona and "
                             "temperature stay fixed, so repeats measure sampling variance, not a new setup")
    parser.add_argument("--max-request-bytes", type=int,
                        help="Explicit local request cap (up to 131072 / 128 KiB); input reservation grows with it")
    parser.add_argument("--input-price", help="Flat estimate CNY per million input tokens")
    parser.add_argument("--output-price", help="Flat estimate CNY per million output tokens")
    parser.add_argument("--role-price", action="append", default=None, metavar="ROLE=IN,OUT",
                        help="Per-role rate in CNY per million tokens, repeatable "
                             "(e.g. agent=0.96,5.76). Overrides the flat assumption for that role.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true", help="Real calls for all applicable roles; user execution only")
    mode.add_argument("--offline-check", action="store_true", help="Scripted HTTP wiring only; not a benchmark score")
    mode.add_argument("--check-config", action="store_true", help="Show effective public role settings; no requests or authentication")
    mode.add_argument("--inspect", action="store_true", help="Read persisted call evidence only; never resumes")
    args = parser.parse_args()
    from .pilot_config import resolve_roles
    _thinking = {}
    for item in args.role_thinking or []:
        _role, _, _raw = item.partition("=")
        if _raw.lower() in {"true", "false"}:
            _thinking[_role] = _raw.lower() == "true"
    if args.check_config:
        print(json.dumps({"roles": resolve_roles(model=args.model, base_url=args.base_url,
                                                 agent_model=args.agent_model,
                                                 role_models={role: value for role, value in
                                                              (("user", args.user_model), ("nl", args.nl_model),
                                                               ("communicate", args.communicate_model)) if value} or None,
                                                 role_thinking=_thinking or None,
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
    role_models = {role: value for role, value in
                   (("user", args.user_model), ("nl", args.nl_model),
                    ("communicate", args.communicate_model)) if value}
    role_thinking = {}
    for item in args.role_thinking or []:
        role, _, raw = item.partition("=")
        if role not in {"agent", "user", "communicate", "nl"} or raw.lower() not in {"true", "false"}:
            parser.error("--role-thinking expects ROLE=true|false")
        role_thinking[role] = raw.lower() == "true"
    role_prices = {}
    for item in args.role_price or []:
        role, _, raw = item.partition("=")
        parts = raw.split(",")
        if role not in {"agent", "user", "communicate", "nl"} or len(parts) != 2:
            parser.error("--role-price expects ROLE=IN,OUT (CNY per million tokens)")
        try:
            role_prices[role] = (Decimal(parts[0].strip()), Decimal(parts[1].strip()))
        except Exception:
            parser.error("--role-price values must be decimal numbers")
    role_max_tokens = {}
    for item in args.role_max_tokens or []:
        role, _, raw = item.partition("=")
        try:
            cap = int(raw)
        except Exception:
            parser.error("--role-max-tokens expects ROLE=N")
        if role not in {"agent", "user", "communicate", "nl"} or not 1 <= cap <= 32000:
            parser.error("--role-max-tokens expects ROLE=N with N in 1..32000")
        role_max_tokens[role] = cap
    config = build_config(
        args.checkout, args.task_id, model=args.model, agent_model=args.agent_model,
        base_url=args.base_url, budget=args.budget_yuan, env_file=args.env_file,
        resource_profile=args.resource_profile,
        input_price=args.input_price, output_price=args.output_price,
        context_history=args.context_history, context_compaction=args.context_compaction,
        failure_reason_chars=args.failure_reason_chars,
        agent_max_completion_tokens=args.agent_max_completion_tokens,
        role_models=role_models or None, role_thinking=role_thinking or None,
        role_prices=role_prices or None,
        role_max_tokens=role_max_tokens or None,
        timeout_seconds=args.timeout_seconds,
        max_steps=args.max_steps,
        seed=args.seed, trials=args.trials,
        max_request_bytes=args.max_request_bytes if args.max_request_bytes is not None else 28000)
    validate_config(args.checkout, config)
    key = ""
    if args.execute:
        # Authentication is loaded only after public settings and bounds are checked.
        from agentcore.app.smoke import _load_key
        key = _load_key(args.env_file)
    print(json.dumps(run_pilot(args.checkout, config, args.output, key=key,
                               offline=args.offline_check), ensure_ascii=False))


if __name__ == "__main__":
    main()
