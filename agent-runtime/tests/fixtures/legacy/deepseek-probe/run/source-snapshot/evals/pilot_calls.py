"""Single-process Pilot call log and total estimate guard, shared by all roles."""

import asyncio
import base64
import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from research_agent.core.model import HTTPModelClient
from research_agent.core.types import PreparedRequest
import httpx


class PilotStop(RuntimeError):
    pass


def inspect_calls(output):
    """Read durable evidence after completion/crash; never resumes or reads .env."""
    output = Path(output)
    config_path = output / "config.json"
    if not config_path.exists():
        config_path = output / "plan.json"  # Read-only compatibility with historical runs.
    plan = json.loads(config_path.read_text(encoding="utf-8"))
    raw = (output / "calls.jsonl").read_bytes()
    rows, incomplete_last_line = [], False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                incomplete_last_line = True
            else:
                raise ValueError("Call evidence contains an invalid complete record") from None
    intents = {r["call_id"]: r for r in rows if r["event"] == "dispatch_intent"}
    terminals = {r["call_id"]: r for r in rows if r["event"] in {"response", "error"}}
    unknown, unfinished, spent, held = [], [], Decimal(0), Decimal(0)
    per_role = {r: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "unknown_usage": 0}
                for r in plan["roles"]}
    price = plan["price_assumption"]
    for call_id, intent in intents.items():
        stats = per_role[intent["role"]]
        stats["calls"] += 1
        terminal = terminals.get(call_id)
        usage = None
        if terminal and terminal["usage_known"]:
            usage = (terminal["response"]["usage"] if terminal["event"] == "response" else terminal["usage"])
        if usage is None:
            unknown.append(call_id)
            stats["unknown_usage"] += 1
            held += Decimal(intent["reserved_cny"])
        else:
            stats["input_tokens"] += usage["input_tokens"]
            stats["output_tokens"] += usage["output_tokens"]
            spent += (Decimal(usage["input_tokens"]) * Decimal(price["input_cny_per_million"])
                      + Decimal(usage["output_tokens"]) * Decimal(price["output_cny_per_million"])) / 1000000
        if terminal is None:
            unfinished.append(call_id)
    return {"calls": len(intents), "per_role": per_role, "estimated_cny": str(spent),
            "unresolved_reservation_cny": str(held), "unknown_usage_call_ids": unknown,
            "unterminated_call_ids": unfinished, "incomplete_last_line": incomplete_last_line,
            "cost_complete": not unknown and not incomplete_last_line,
            "absolute_bill_guaranteed": False}


class Calls:
    def __init__(self, output, plan, key="", transport_factory=None):
        self.output, self.plan, self.key = Path(output), plan, key
        self.transport_factory = transport_factory
        self.stream = (self.output / "calls.jsonl").open("x", encoding="utf-8")
        self.spent = Decimal(0)
        self.reserved = Decimal(0)
        self.count = 0
        self.task_counts = {}
        self.task_spent = {}
        self.stopped = False
        self.task_id = None
        self.core_store = None
        self.stop_reason = None

    def record(self, **entry):
        entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            self.stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException:
            self.stopped = True
            self.stop_reason = "evidence_storage_failure"
            raise

    def stop(self, reason):
        self.stopped = True
        self.stop_reason = self.stop_reason or reason
        self.record(event="stop", task_id=self.task_id, reason=self.stop_reason,
                    estimated_cny=str(self.spent), held_cny=str(self.reserved),
                    calls=self.count, task_calls=self.task_counts.get(self.task_id, 0),
                    task_estimated_cny=str(self.task_spent.get(self.task_id, Decimal(0))))

    def close(self):
        self.stream.close()

    def cost(self, inputs, outputs):
        price = self.plan["price_assumption"]
        return (Decimal(inputs) * Decimal(price["input_cny_per_million"])
                + Decimal(outputs) * Decimal(price["output_cny_per_million"])) / 1000000

    def client(self, role):
        config = self.plan["roles"][role]
        return HTTPModelClient(config["base_url"], config["model"], self.key,
                               provider=config["provider"], thinking=config["thinking"],
                               preserve_thinking=config["preserve_thinking"],
                               temperature=config["temperature"], timeout=60,
                               max_completion_tokens=self.output_limit(role))

    def output_limit(self, role):
        limits = self.plan["limits"]
        return limits.get("agent_max_output_tokens", limits["max_output_tokens"]) if role == "agent" else limits["max_output_tokens"]

    def model(self, role, call_name=None):
        owner = self
        client = self.client(role)

        class Model:
            def prepare(self, *args, **kwargs):
                return client.prepare(*args, **kwargs)

            async def complete(self, request):
                return await owner.complete(role, client, request, call_name=call_name)

        return Model()

    async def complete(self, role, client, request, *, call_name=None):
        limits = self.plan["limits"]
        output_limit = self.output_limit(role)
        reserve = self.cost(limits["input_token_proxy_ceiling"], output_limit + 10)
        resources = self.plan.get("resources")  # Historical evidence/configs retain their old bounds.
        task_count = self.task_counts.get(self.task_id, 0)
        task_spent = self.task_spent.get(self.task_id, Decimal(0))
        reason = None
        if self.stopped or self.reserved or (self.output / "STOP").exists():
            reason = "pre_dispatch_guard"
        elif self.count >= limits["max_calls"] or (resources and task_count >= resources["max_calls_per_task"]):
            reason = "call_limit"
        elif self.spent + reserve > Decimal(self.plan["budget_cny"]):
            reason = "run_budget_limit"
        elif resources and task_spent + reserve > Decimal(resources["task_budget_cny"]):
            reason = "task_budget_limit"
        if reason:
            self.stop(reason)
            raise PilotStop("Pilot stopped before dispatch: stop flag, unresolved call, call cap or budget")
        request.validate()
        if (len(request.body) > limits["max_request_bytes"]
                or len(request.body) + 2048 > limits["input_token_proxy_ceiling"]):
            self.stop("input_bound")
            raise PilotStop("Pilot request exceeds configured input bound")
        call_id = uuid4().hex
        body = json.loads(request.body)
        config = self.plan["roles"][role]
        protocol_matches = (client.provider == config["provider"]
                            and ("preserve_thinking" not in body and "thinking" not in body
                                 if config["provider"] == "bailian_deepseek"
                                 else body.get("preserve_thinking") == config["preserve_thinking"]))
        if (body["model"] != config["model"] or body.get("stream") is not False
                or body.get("enable_thinking") != config["thinking"]
                or not protocol_matches
                or body.get("max_completion_tokens") != output_limit
                or body.get("temperature") != config["temperature"]
                or client.url != config["base_url"].rstrip("/") + "/chat/completions"):
            self.stop("configuration_drift")
            raise PilotStop("Effective client differs from run configuration")
        self.count += 1
        self.task_counts[self.task_id] = task_count + 1
        self.reserved = reserve
        correlation = {}
        if role == "agent" and self.core_store is not None:
            row = self.core_store.db.execute(
                "SELECT run,body FROM events WHERE kind='model_request' ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            event = json.loads(row["body"]) if row else {}
            if event.get("sha256") != hashlib.sha256(request.body).hexdigest():
                self.stop("core_correlation_missing")
                raise PilotStop("Core request correlation missing")
            correlation = {"run_id": row["run"], "model_call_id": event["model_call_id"]}
        # Durable intent before the only network dispatch. No headers/credentials.
        self.record(event="dispatch_intent", call_id=call_id, task_id=self.task_id,
                    role=role, request=body, request_sha256=hashlib.sha256(request.body).hexdigest(),
                    base_url=config["base_url"], reserved_cny=str(reserve),
                    call_name=call_name, **correlation)
        try:
            underlying = (self.transport_factory(role) if self.transport_factory else
                          httpx.AsyncHTTPTransport(retries=0, trust_env=False))
            owner = self

            class Capture(httpx.AsyncBaseTransport):
                async def handle_async_request(self, request):
                    result = await underlying.handle_async_request(request)
                    try:
                        raw = bytearray()
                        async for chunk in result.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 16 * 1024 * 1024:
                                raise PilotStop("HTTP response exceeds evidence bound")
                        owner.record(event="http_response", call_id=call_id, role=role,
                                     task_id=owner.task_id, status_code=result.status_code,
                                     body_base64=base64.b64encode(raw).decode("ascii"),
                                     decoded_body=True, body_sha256=hashlib.sha256(raw).hexdigest())
                        # aread/aiter_bytes decode content encoding. Return decoded
                        # bytes without encoding headers to avoid double decoding.
                        return httpx.Response(result.status_code, content=bytes(raw), request=request)
                    finally:
                        await result.aclose()

                async def aclose(self):
                    await underlying.aclose()

            client.transport = Capture()
            response = await client.complete(request)
            usage = response.usage if response.usage_known else None
            self.record(event="response", call_id=call_id, task_id=self.task_id, role=role,
                        response=asdict(response), usage_known=usage is not None)
        except BaseException as exc:
            self.stopped = True
            self.stop_reason = self.stop_reason or getattr(exc, "kind", type(exc).__name__)
            usage = getattr(exc, "usage", None)
            self.record(event="error", call_id=call_id, task_id=self.task_id, role=role,
                        category=getattr(exc, "kind", type(exc).__name__),
                        usage=asdict(usage) if usage is not None else None,
                        usage_known=usage is not None)
            if usage is not None:
                charge = self.cost(usage.input_tokens, usage.output_tokens)
                self.spent += charge
                self.task_spent[self.task_id] = task_spent + charge
                self.reserved = Decimal(0)
            self.record(event="settlement", call_id=call_id, role=role, task_id=self.task_id,
                        estimated_cny=str(self.spent), held_cny=str(self.reserved),
                        stopped=True, reason=self.stop_reason)
            raise
        if usage is None:
            self.stop("usage_unknown")  # keep full reservation; no later roles dispatch
            raise PilotStop("Provider usage is unknown; stop all roles")
        charge = self.cost(usage.input_tokens, usage.output_tokens)
        self.spent += charge
        self.task_spent[self.task_id] = task_spent + charge
        self.reserved = Decimal(0)
        self.record(event="settlement", call_id=call_id, role=role, task_id=self.task_id,
                    call_estimated_cny=str(self.cost(usage.input_tokens, usage.output_tokens)),
                    estimated_cny=str(self.spent), held_cny="0", usage=asdict(usage))
        if (usage.input_tokens > limits["input_token_proxy_ceiling"]
                or usage.output_tokens > output_limit + 10
                or self.spent > Decimal(self.plan["budget_cny"])
                or (resources and self.task_spent[self.task_id] > Decimal(resources["task_budget_cny"]))):
            self.stop("usage_bound")
            raise PilotStop("Reported usage exceeds frozen estimate bounds")
        return response

    @contextmanager
    def official_transport(self):
        """Scoped, explicit generate seam; official prompts and evaluators unchanged.

        This small Pilot is sequential. No global LiteLLM config or vendor edits.
        NL's module constants are temporarily set to the explicitly frozen role.
        """
        import tau2.user.user_simulator as user
        import tau2.evaluator.evaluator_communicate as communicate
        import tau2.evaluator.evaluator_nl_assertions as nl
        from tau2.data_model.message import AssistantMessage, ToolCall
        from tau2.utils.llm_utils import to_litellm_messages

        def generator(role):
            def generate(*, model, messages, tools=None, call_name=None, **kwargs):
                # Official UserSimulator.set_seed adds this per-call argument.
                # Preserve it on the wire rather than silently discarding it.
                expected_args = {"seed": self.plan["seed"]} if role == "user" else {}
                if model != self.plan["roles"][role]["model"] or kwargs != expected_args:
                    self.stopped = True
                    raise PilotStop("Official caller configuration differs from frozen transport")
                model_client = self.model(role, call_name=call_name)
                schemas = [tool.openai_schema for tool in tools or []]
                wire = to_litellm_messages(messages)
                for message in wire:
                    if message.get("tool_calls") is None:
                        message.pop("tool_calls", None)
                request = model_client.prepare(wire, schemas, self.plan["limits"]["max_request_bytes"])
                if role == "user":
                    body = json.loads(request.body)
                    body["seed"] = kwargs["seed"]
                    request = PreparedRequest(json.dumps(body, ensure_ascii=False).encode("utf-8"), request.max_bytes)
                response = asyncio.run(model_client.complete(request))
                usage = response.usage
                return AssistantMessage(
                    role="assistant", content=response.content,
                    tool_calls=[ToolCall(id=c.id, name=c.name, arguments=c.arguments)
                                for c in response.tool_calls] or None,
                    usage={"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens},
                    # Official cost fields use no explicit currency. CNY stays in our log.
                    cost=None, raw_data={"provider_metadata": response.provider_metadata,
                                         "pilot_role": role, "call_name": call_name},
                )
            return generate

        with ExitStack() as stack:
            for module, role in ((user, "user"), (communicate, "communicate"), (nl, "nl")):
                stack.enter_context(patch.object(module, "generate", generator(role)))
            stack.enter_context(patch.object(nl, "DEFAULT_LLM_NL_ASSERTIONS", self.plan["roles"]["nl"]["model"]))
            stack.enter_context(patch.object(nl, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", {}))
            yield
