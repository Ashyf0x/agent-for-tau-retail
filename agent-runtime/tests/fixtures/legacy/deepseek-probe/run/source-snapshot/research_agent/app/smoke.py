"""One fixed Qwen smoke experiment. Default: freeze a plan, without credentials or network."""

import argparse
import asyncio
import hashlib
import io
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from research_agent import Agent, AgentConfig
from research_agent.context import ContextManager
from research_agent.core.model import HTTPModelClient
from research_agent.core.types import ModelError, Usage, dumps
from research_agent.observability import run_report
from research_agent.storage import Store
from research_agent.tools import ToolRegistry, ToolSpec

MODEL = "qwen3.7-flash"
PRICE_URL = "https://help.aliyun.com/zh/model-studio/qwen3-7-flash"
PROTOCOL_URL = "https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions"
MAX_CALLS, MAX_BYTES, FRAMING_BYTES = 2, 16000, 1024
MAX_INPUT, MAX_OUTPUT, OUTPUT_SLACK = 32000, 2048, 10
PROMPT = "Use the local add tool to calculate 19 + 23, then answer using its returned result."
SYSTEM = "You are a tool-using assistant. Use the provided add tool for the requested calculation. Do not invent tool results."
TIERS = ((32000, "0.2", "0.8"), (256000, "0.6", "2.4"), (1000000, "1.2", "4.8"))


def estimate_cost(input_tokens, output_tokens):
    for ceiling, price_in, price_out in TIERS:
        if input_tokens <= ceiling:
            return (
                Decimal(input_tokens) * Decimal(price_in)
                + Decimal(output_tokens) * Decimal(price_out)
            ) / Decimal(1000000)
    raise ValueError("Usage outside frozen price card")


RESERVATION = estimate_cost(MAX_INPUT, MAX_OUTPUT + OUTPUT_SLACK)


def frozen_plan(budget="0.05"):
    limit = Decimal(str(budget))
    if not limit.is_finite() or not Decimal(0) < limit <= Decimal("0.05"):
        raise ValueError("Smoke budget must be positive and at most 0.05 CNY")
    return {
        "version": "qwen-smoke-v1-ws",
        "model": MODEL,
        "region": "cn-beijing",
        "price_checked": "2026-11-05",
        "price_source": PRICE_URL,
        "protocol_source": PROTOCOL_URL,
        "price_tiers_cny_per_million": [list(tier) for tier in TIERS],
        "snapshot_family_price_assumption": True,
        "max_calls": MAX_CALLS,
        "max_request_bytes": MAX_BYTES,
        "framing_proxy_bytes": FRAMING_BYTES,
        "input_token_proxy_ceiling": MAX_INPUT,
        "max_completion_tokens": MAX_OUTPUT,
        "allowed_output_slack_tokens": OUTPUT_SLACK,
        "thinking": True,
        "preserve_thinking": True,
        "budget_cny": str(limit),
        "reservation_per_call_cny": str(RESERVATION),
        "two_call_reservation_cny": str(RESERVATION * MAX_CALLS),
        "tool_names": ["add"],
        "automatic_retries": 0,
        "protocol_corrections": 0,
        "argument_corrections": 0,
        "memory_extraction": False,
        "prompt": PROMPT,
        "limits": "Request-byte proxy plus framing is not the provider tokenizer. Price-card estimates and output tolerance cannot guarantee the absolute account bill.",
    }


def make_tools():
    async def add(a, b):
        return a + b

    return ToolRegistry(
        [
            ToolSpec(
                "add",
                "Add two integers",
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                add,
                read_only=True,
                max_attempts=1,
            )
        ]
    )


class _AddOnlyAgent(Agent):
    def _tools(self, user, session):
        # Reuse the core loop, exposing no ask_user/read_observation tools in this fixed smoke.
        return self.tools


def make_client(base_url, api_key=""):
    return HTTPModelClient(
        base_url,
        MODEL,
        api_key,
        provider="qwen",
        thinking=True,
        preserve_thinking=True,
        max_completion_tokens=MAX_OUTPUT,
        timeout=60,
    )


class SmokeModel:
    """Single-experiment reservation and dispatch owner; no retries, resume, or pricing registry."""

    def __init__(self, client, store, plan, *, user, session):
        self.client, self.store, self.plan = client, store, plan
        self.user, self.session = user, session
        if plan != plan_preview(plan["budget_cny"], plan["base_url"]):
            raise ValueError("Smoke plan differs from frozen configuration")
        host = urlsplit(client.url).hostname
        if (
            host not in {"127.0.0.1", "localhost", "::1"}
            and client.url != plan["base_url"].rstrip("/") + "/chat/completions"
        ):
            raise ValueError("Client endpoint differs from the frozen Beijing endpoint")
        if (
            client.model != MODEL
            or client.provider != "qwen"
            or client.thinking is not True
            or client.preserve_thinking is not True
            or client.max_completion_tokens != MAX_OUTPUT
            or client.transport is not None
        ):
            raise ValueError("Client differs from frozen smoke configuration")
        with store.transaction() as db:
            db.execute(
                "CREATE TABLE smoke_state (id INTEGER PRIMARY KEY, plan TEXT, stop_reason TEXT)"
            )
            db.execute("CREATE TABLE smoke_calls (id TEXT PRIMARY KEY, status TEXT, body TEXT)")
            db.execute("INSERT INTO smoke_state VALUES (1,?,NULL)", (dumps(plan),))

    def prepare(self, messages, tools, max_bytes):
        return self.client.prepare(messages, tools, min(max_bytes, MAX_BYTES))

    def _stop(self, reason):
        with self.store.transaction() as db:
            db.execute(
                "UPDATE smoke_state SET stop_reason=COALESCE(stop_reason,?) WHERE id=1", (reason,)
            )

    def _reject(self, reason):
        self._stop(reason)
        raise ModelError(
            "Smoke dispatch refused",
            kind="smoke_guard",
            diagnostics={"reason": reason, "dispatched": False},
            usage=Usage(),
        )

    async def complete(self, request):
        stopped = self.store.db.execute(
            "SELECT stop_reason FROM smoke_state WHERE id=1"
        ).fetchone()[0]
        if stopped:
            self._reject("experiment_already_stopped")
        existing = self.store.db.execute("SELECT status,body FROM smoke_calls").fetchall()
        if any(row[0] not in {"returned", "error_known"} for row in existing):
            self._reject("unresolved_request")
        if len(existing) >= MAX_CALLS:
            self._reject("request_limit")
        payload = json.loads(request.body)
        if (
            len(request.body) > min(request.max_bytes, MAX_BYTES)
            or len(request.body) + FRAMING_BYTES > MAX_INPUT
        ):
            self._reject("input_bound")
        required = {
            "model": MODEL,
            "enable_thinking": True,
            "preserve_thinking": True,
            "max_completion_tokens": MAX_OUTPUT,
            "stream": False,
            "temperature": 0,
        }
        if (
            any(payload.get(key) != value for key, value in required.items())
            or "max_tokens" in payload
        ):
            self._reject("request_configuration_drift")
        if payload.get("tools") != make_tools().definitions() or set(payload) != {
            *required,
            "tools",
            "messages",
        }:
            self._reject("tool_or_request_configuration_drift")
        row = self.store.db.execute(
            "SELECT e.run,e.body FROM events e JOIN runs r ON r.id=e.run "
            "WHERE r.user=? AND r.session=? AND e.kind='model_request' ORDER BY e.seq DESC LIMIT 1",
            (self.user, self.session),
        ).fetchone()
        digest = hashlib.sha256(request.body).hexdigest()
        if not existing and digest != self.plan["initial_request_sha256"]:
            self._reject("initial_request_drift")
        if row is None or json.loads(row["body"]).get("sha256") != digest:
            self._reject("core_request_correlation_missing")
        call_id = json.loads(row["body"])["model_call_id"]
        spent = sum(
            (Decimal(json.loads(item[1]).get("estimated_cost_cny") or "0") for item in existing),
            Decimal(0),
        )
        if spent + RESERVATION > Decimal(self.plan["budget_cny"]):
            self._reject("budget_insufficient")
        detail = {
            "model_call_id": call_id,
            "run_id": row["run"],
            "request_sha256": digest,
            "request_bytes": len(request.body),
            "input_proxy_with_framing": len(request.body) + FRAMING_BYTES,
            "reserved_cny": str(RESERVATION),
            "estimated_cost_cny": None,
            "usage_known": False,
            "dispatch_intent": False,
        }
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO smoke_calls VALUES (?,?,?)", (call_id, "reserved", dumps(detail))
            )
        # Mark a conservative intent before handing off to HTTP; crash here is unresolved, never resumed.
        detail["dispatch_intent"] = True
        with self.store.transaction() as db:
            db.execute(
                "UPDATE smoke_calls SET status='dispatch_intent',body=? WHERE id=?",
                (dumps(detail), call_id),
            )
        try:
            response = await self.client.complete(request)
            detail["response"] = asdict(response)
            if not response.usage_known:
                raise ModelError("Smoke response usage is unknown", kind="usage_unknown")
            usage = response.usage
            self._record_usage(detail, usage)
            if usage.input_tokens > MAX_INPUT or usage.output_tokens > MAX_OUTPUT + OUTPUT_SLACK:
                raise ModelError(
                    "Smoke response exceeded frozen usage bounds", kind="usage_bound", usage=usage
                )
            Agent._validate_response(response)
            if (
                any(call.name != "add" for call in response.tool_calls)
                or len(response.tool_calls) > 1
            ):
                raise ModelError(
                    "Smoke only permits one add call per response",
                    kind="protocol_error",
                    usage=usage,
                )
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE smoke_calls SET status='returned',body=? WHERE id=?",
                    (dumps(detail), call_id),
                )
            return response
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, ModelError) and exc.usage is not None:
                self._record_usage(detail, exc.usage)
            detail["error"] = {
                "type": type(exc).__name__,
                "category": getattr(
                    exc,
                    "kind",
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failure",
                ),
            }
            status = "error_known" if detail["usage_known"] else "unknown"
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE smoke_calls SET status=?,body=? WHERE id=?",
                    (status, dumps(detail), call_id),
                )
                db.execute(
                    "UPDATE smoke_state SET stop_reason=COALESCE(stop_reason,?) WHERE id=1",
                    (detail["error"]["category"],),
                )
            raise

    @staticmethod
    def _record_usage(detail, usage):
        detail.update(usage=asdict(usage), usage_known=True)
        try:
            detail["estimated_cost_cny"] = str(
                estimate_cost(usage.input_tokens, usage.output_tokens)
            )
        except ValueError:
            detail["cost_status"] = "outside_frozen_price_card"


def budget_report(db):
    db.row_factory = sqlite3.Row
    state = db.execute("SELECT plan,stop_reason FROM smoke_state WHERE id=1").fetchone()
    plan = json.loads(state["plan"])
    calls = [
        {"id": row["id"], "status": row["status"], **json.loads(row["body"])}
        for row in db.execute("SELECT * FROM smoke_calls ORDER BY rowid")
    ]
    unknown = [
        call["id"] for call in calls if call.get("dispatch_intent") and not call["usage_known"]
    ]
    unterminated = [
        call["id"] for call in calls if call["status"] in {"reserved", "dispatch_intent"}
    ]
    spent = sum(
        (
            Decimal(call["estimated_cost_cny"])
            for call in calls
            if call["estimated_cost_cny"] is not None
        ),
        Decimal(0),
    )
    held = sum(
        (
            Decimal(call["reserved_cny"])
            for call in calls
            if call["status"] in {"reserved", "dispatch_intent", "unknown"}
        ),
        Decimal(0),
    )
    return {
        "plan": plan,
        "stop_reason": state["stop_reason"],
        "calls": calls,
        "dispatch_intents": sum(call.get("dispatch_intent", False) for call in calls),
        "unknown_usage_call_ids": unknown,
        "unterminated_call_ids": unterminated,
        "known_estimated_cost_cny": str(spent),
        "held_reservations_cny": str(held),
        "cost_complete": not unknown
        and not unterminated
        and all(call["estimated_cost_cny"] is not None for call in calls),
        "known_estimate_exceeds_budget": spent > Decimal(plan["budget_cny"]),
        "absolute_bill_guaranteed": False,
    }


async def run_smoke(directory, client, plan):
    directory = Path(directory)
    path = directory / "smoke.sqlite3"
    # Exclusive creation ensures one fresh session and rejects reuse/resume, including old unresolved runs.
    with path.open("xb"):
        pass
    store = Store(path)
    user = session = uuid4().hex
    result = None
    guarded = None
    try:
        guarded = SmokeModel(client, store, plan, user=user, session=session)
        agent = _AddOnlyAgent(
            guarded,
            make_tools(),
            store,
            system_prompt=SYSTEM,
            config=AgentConfig(
                max_steps=MAX_CALLS,
                max_request_bytes=MAX_BYTES,
                model_timeout=60,
                max_argument_corrections=0,
                max_protocol_corrections=0,
            ),
        )
        result = await agent.run(PROMPT, user_id=user, session_id=session)
        if result.status != "completed":
            guarded._stop(result.reason or result.status)
        return result
    except (Exception, asyncio.CancelledError) as exc:
        if guarded is not None:
            guarded._stop(
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "execution_failure"
            )
        raise
    finally:
        try:
            if guarded is not None:
                _write_summary(store, directory, client, user, session, result)
        finally:
            store.close()


def _write_summary(store, directory, client, user, session, result):
    report = budget_report(store.db)
    report["mode"] = (
        "offline_http_fixture"
        if urlsplit(client.url).hostname in {"127.0.0.1", "localhost", "::1"}
        else "real_endpoint"
    )
    row = store.db.execute(
        "SELECT id FROM runs WHERE user=? AND session=? ORDER BY rowid DESC LIMIT 1",
        (user, session),
    ).fetchone()
    report["run_report"] = run_report(store, row[0]) if row is not None else None
    tools = [
        json.loads(row[0])
        for row in store.db.execute(
            "SELECT body FROM messages WHERE user=? AND session=? AND json_extract(body,'$.role')='tool'",
            (user, session),
        )
    ]
    outcomes = [json.loads(message["content"]) for message in tools]
    report["tool_roundtrip_observed"] = bool(
        result
        and result.status == "completed"
        and report["dispatch_intents"] == 2
        and len(outcomes) == 1
        and outcomes[0].get("ok")
        and outcomes[0].get("value") == 42
    )
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def plan_preview(budget="0.05", base_url=None, *, env_file=Path(".env")):
    base_url = _resolve_base_url(base_url, env_file)
    plan = frozen_plan(budget)
    _validate_beijing_url(base_url)
    plan["base_url"] = base_url
    client = make_client(base_url)
    request, info = ContextManager(SYSTEM).prepare(
        client,
        [{"id": "preview", "message": {"role": "user", "content": PROMPT}}],
        make_tools().definitions(),
        [],
        {"step": 1, "limit": MAX_CALLS, "feedback": ""},
        MAX_BYTES,
    )
    plan["initial_request"] = json.loads(request.body)
    plan["initial_request_sha256"] = hashlib.sha256(request.body).hexdigest()
    plan["initial_request_bytes"] = info["wire_bytes"]
    return plan


def prepare_plan(directory, budget, base_url=None, *, env_file=Path(".env")):
    directory = Path(directory)
    plan = plan_preview(budget, base_url, env_file=env_file)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    return plan


def _load_key(path):
    # Imported and invoked only in the explicit execution branch, never during plan or inspection.
    from dotenv import dotenv_values

    key = dotenv_values(path, interpolate=False).get("DASHSCOPE_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("DASHSCOPE_API_KEY is missing from the explicit authentication file")
    return key


def _resolve_base_url(explicit, env_file):
    if explicit is not None:
        return explicit  # Explicit override never reads the configuration file.
    from dotenv import dotenv_values

    path = Path(env_file)
    if not path.is_file():
        raise ValueError("Configuration file is missing; set AGENT_BASE_URL or pass --base-url")
    # Only endpoint assignments reach the dotenv parser; never parse or print keys
    # while inspecting configuration. No process-env fallback or interpolation.
    with path.open(encoding="utf-8-sig") as stream:
        selected = "".join(line for line in stream if re.match(
            r"^\s*(?:export\s+)?AGENT_BASE_URL\s*=", line))
    value = dotenv_values(stream=io.StringIO(selected), interpolate=False).get("AGENT_BASE_URL")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("AGENT_BASE_URL is missing or empty; no default endpoint is configured")
    return value.strip()


def _validate_beijing_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is not None
    ):
        raise ValueError("An official Beijing HTTPS endpoint is required")
    if parsed.path.rstrip("/") != "/compatible-mode/v1" or not (
        parsed.hostname == "dashscope.aliyuncs.com"
        or re.fullmatch(r"[a-zA-Z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com", parsed.hostname or "")
    ):
        raise ValueError("An official Beijing endpoint is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-yuan", default="0.05")
    parser.add_argument("--base-url", help="Override AGENT_BASE_URL from --env-file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Real requests; requires explicit user authorization before use",
    )
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="Read stored ledger, including unresolved calls; no credentials/network",
    )
    mode.add_argument("--check-config", action="store_true",
                      help="Show the actual configured endpoint; no authentication or requests")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if not args.check_config and args.output is None:
        parser.error("--output is required unless --check-config is used")
    if args.check_config:
        base_url = _resolve_base_url(args.base_url, args.env_file)
        _validate_beijing_url(base_url)
        print(json.dumps({"base_url": base_url,
                          "source": "--base-url" if args.base_url is not None else "AGENT_BASE_URL",
                          "env_file": str(args.env_file.resolve()) if args.base_url is None else None,
                          "model": MODEL, "credentials_loaded": False, "requests_sent": 0},
                         ensure_ascii=False, indent=2))
        return
    if args.inspect:
        with closing(
            sqlite3.connect(
                (args.output / "smoke.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as db:
            report = budget_report(db)
    elif not args.execute:
        report = prepare_plan(args.output, args.budget_yuan, args.base_url, env_file=args.env_file)
    else:
        plan = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
        base_url = _resolve_base_url(args.base_url, args.env_file)
        expected = plan_preview(args.budget_yuan, base_url)
        if plan != expected:
            raise ValueError("Frozen plan differs from execution configuration")
        if any((args.output / name).exists() for name in ("smoke.sqlite3", "summary.json")):
            raise ValueError("Experiment already started; inspect it, do not resume")
        client = make_client(base_url, _load_key(args.env_file))
        asyncio.run(run_smoke(args.output, client, plan))
        report = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    # No credential or HTTP headers appear in these artifacts. Raw responses stay in local evidence.
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "calls"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
