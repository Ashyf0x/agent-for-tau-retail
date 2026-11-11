"""Single retry owner for local and MCP tools. Handlers never choose retry policy."""

import asyncio
import inspect
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

from jsonschema import Draft202012Validator

from agentcore.core.errors import Decision, ErrorInfo, ToolFailure, decide
from agentcore.core.types import dumps


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable
    read_only: bool = False
    idempotent: bool = False
    max_attempts: int = 1
    timeout: float = 10
    # Trusted, read-only lookup by runtime operation ID, returns known result or None.
    query_status: Callable | None = None
    # Optional trusted check, run before any dispatch, may raise ToolFailure.
    precondition: Callable | None = None
    # Pass the stable operation ID to a handler implementing an idempotency contract.
    operation_id_parameter: str | None = None

    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }


@dataclass
class ToolResult:
    status: str
    ok: bool = False
    value: object = None
    error: str | None = None
    category: str | None = None
    decision: str = "feedback"
    attempts: int = 0
    error_info: dict | None = None

    def to_dict(self):
        return asdict(self)


async def invoke(handler, **kwargs):
    if inspect.iscoroutinefunction(handler):
        return await handler(**kwargs)
    value = await asyncio.to_thread(handler, **kwargs)
    return await value if inspect.isawaitable(value) else value


class ToolRegistry:
    def __init__(self, specs=()):
        self.specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec):
        if not spec.name or spec.name in self.specs:
            raise ValueError("Tool names must be nonempty and unique")
        if (
            type(spec.max_attempts) is not int
            or not 1 <= spec.max_attempts <= 5
            or not math.isfinite(spec.timeout)
            or spec.timeout <= 0
        ):
            raise ValueError("Invalid tool execution limits")
        if spec.parameters.get("type") != "object":
            raise ValueError("Tool parameters must be an object")
        Draft202012Validator.check_schema(spec.parameters)
        if spec.operation_id_parameter in spec.parameters.get("properties", {}):
            raise ValueError("The model must not supply the operation ID")
        self.specs[spec.name] = spec

    def definitions(self):
        return [s.definition() for s in self.specs.values()]

    async def execute(self, call, record, *, backoff=0.02):
        """record(attempt,status,body) must durably save intent before invocation."""
        spec = self.specs.get(call.name)
        try:
            if spec is None:
                raise ToolFailure("Unknown tool", "arguments")
            args = (
                json.loads(call.arguments)
                if isinstance(call.arguments, str)
                else deepcopy(call.arguments)
            )
            dumps(args)  # reject NaN and non-JSON values
            errors = list(Draft202012Validator(spec.parameters).iter_errors(args))
            if errors:
                # Schema path only: validation messages can echo secrets from argument values.
                field = ".".join(map(str, errors[0].absolute_path)) or "arguments"
                if errors[0].validator == "required" and isinstance(args, dict):
                    field = next(
                        (key for key in errors[0].validator_value if key not in args), field
                    )
                raise ToolFailure(f"Invalid field {field} ({errors[0].validator})", "arguments")
        except (ValueError, TypeError):
            result = ToolResult(
                "not_dispatched", error="Arguments must be valid JSON", category="arguments"
            )
            record(0, result.status, result.to_dict())
            return result
        except ToolFailure as exc:
            decision = decide(ErrorInfo("validation", exc.category, False, True, 0), 1)
            result = ToolResult(
                "not_dispatched", error=str(exc), category=exc.category, decision=decision
            )
            record(0, result.status, result.to_dict())
            return result

        if spec.precondition:
            try:
                await asyncio.wait_for(invoke(spec.precondition, **args), spec.timeout)
            except ToolFailure as exc:
                info = ErrorInfo(
                    "precondition",
                    exc.category,
                    False,
                    True,
                    0,
                    spec.read_only,
                    spec.idempotent,
                    spec.query_status is not None,
                )
                result = ToolResult(
                    "not_dispatched",
                    error=str(exc),
                    category=exc.category,
                    decision=decide(info, 1),
                    error_info=asdict(info),
                )
                record(0, result.status, result.to_dict())
                return result
            # Unexpected precondition errors are program defects, not argument corrections.

        if spec.operation_id_parameter:
            args[spec.operation_id_parameter] = call.id
        for attempt in range(1, spec.max_attempts + 1):
            # Conservative intent: a crash between this commit and invocation is still unknown.
            record(attempt, "unknown", {"phase": "dispatch_intent"})
            try:
                value = await asyncio.wait_for(invoke(spec.handler, **args), spec.timeout)
                if isinstance(value, ToolResult):
                    if value.status != "returned":
                        raise ValueError("A returned tool result must have known outcome")
                    dumps(value.to_dict())
                    result = value
                    result.attempts = attempt
                else:
                    dumps(value)
                    result = ToolResult("returned", ok=True, value=value, attempts=attempt)
            except asyncio.CancelledError:
                record(
                    attempt, "unknown", {"category": "cancelled", "diagnostic": "CancelledError"}
                )
                raise
            except (ToolFailure, TimeoutError) as exc:
                known = isinstance(exc, ToolFailure) and exc.outcome_known
                category = exc.category if isinstance(exc, ToolFailure) else "transient"
                info = ErrorInfo(
                    "execution",
                    category,
                    True,
                    known,
                    attempt,
                    spec.read_only,
                    spec.idempotent,
                    spec.query_status is not None,
                )
                decision = decide(info, spec.max_attempts)
                result = ToolResult(
                    "returned" if known else "unknown",
                    error=str(exc)
                    if isinstance(exc, ToolFailure)
                    else "Tool timed out; outcome unknown",
                    category=category,
                    decision=decision,
                    attempts=attempt,
                    error_info=asdict(info),
                )
            except Exception as exc:
                result = ToolResult(
                    "unknown",
                    error="Internal tool error",
                    category="internal",
                    decision=Decision.TERMINATE,
                    attempts=attempt,
                )
                record(
                    attempt, result.status, {**result.to_dict(), "diagnostic": type(exc).__name__}
                )
                return result
            record(attempt, result.status, result.to_dict())
            if result.decision == Decision.QUERY:
                # Query once, never resubmit the write. Query failure preserves unknown.
                try:
                    resolved = await asyncio.wait_for(
                        invoke(spec.query_status, operation_id=call.id), spec.timeout
                    )
                    if resolved is not None:
                        if not isinstance(resolved, ToolResult) or resolved.status != "returned":
                            raise TypeError("Status contract must return a known ToolResult")
                        dumps(resolved.to_dict())
                        resolved.attempts = attempt
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    record(
                        attempt,
                        "unknown",
                        {**result.to_dict(), "query_diagnostic": type(exc).__name__},
                    )
                    resolved = None
                if resolved is not None:
                    # Persistence is outside the query-error boundary: storage failures must abort.
                    record(attempt, "returned", {**resolved.to_dict(), "resolution": "status_query"})
                    return resolved
                result.decision = Decision.WAIT
                return result
            if result.decision != Decision.RETRY:
                return result
            await asyncio.sleep(backoff * attempt)
        raise AssertionError("Retry policy exceeded its bound")
