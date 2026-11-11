import asyncio
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from uuid import uuid4

from agentcore.context import ContextManager
from agentcore.core.errors import Decision, ToolFailure
from agentcore.core.types import (
    ContextBudgetError,
    ModelError,
    ModelResponse,
    RunResult,
    StorageError,
    ToolCall,
    dumps,
)
from agentcore.memory import Memory
from agentcore.tools import ToolRegistry, ToolSpec


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 12
    max_request_bytes: int = 64000
    model_timeout: float = 60
    max_argument_corrections: int = 2
    max_protocol_corrections: int = 1

    def __post_init__(self):
        if (
            self.max_steps < 1
            or self.max_request_bytes < 1
            or not math.isfinite(self.model_timeout)
            or self.model_timeout <= 0
        ):
            raise ValueError("Run limits must be positive")
        if min(self.max_argument_corrections, self.max_protocol_corrections) < 0:
            raise ValueError("Correction limits must be nonnegative")


DEFAULT_PROMPT = (
    "Help the user using the available evidence and tools. Ask for missing information or authorization. "
    "Report incomplete work and uncertainty honestly. Answer naturally; no action JSON is required."
)


class Agent:
    """One loop; identities, persistence and execution policy belong to the program."""

    def __init__(
        self,
        model,
        tools,
        store,
        *,
        config=None,
        context=None,
        memory=None,
        system_prompt=DEFAULT_PROMPT,
    ):
        self.model, self.tools, self.store = model, tools, store
        self.config = config or AgentConfig()
        self.context = context or ContextManager(system_prompt)
        self.memory = memory or Memory(store)
        if {"ask_user", "read_observation"} & tools.specs.keys():
            raise ValueError("ask_user and read_observation are reserved runtime tools")

    def _tools(self, user, session):
        tools = ToolRegistry(self.tools.specs.values())

        async def ask_user(question):
            return {"question": question}

        async def read_observation(ref, offset=0, limit=2000):
            try:
                return self.store.read_observation(
                    user,
                    session,
                    ref,
                    offset,
                    limit,
                    project=lambda body: self.context.project_evidence(body),
                )
            except (KeyError, ValueError):
                raise ToolFailure(
                    "Observation reference or page is invalid in this session", "arguments"
                ) from None

        tools.register(
            ToolSpec(
                "ask_user",
                "Pause and ask the user for missing information or authorization.",
                {
                    "type": "object",
                    "properties": {"question": {"type": "string", "minLength": 1, "maxLength": 240}},
                    "required": ["question"],
                    "additionalProperties": False,
                },
                ask_user,
                read_only=True,
            )
        )
        tools.register(
            ToolSpec(
                "read_observation",
                "Read a page of archived evidence in this session (character offsets).",
                {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 4000},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
                read_observation,
                read_only=True,
            )
        )
        return tools

    async def run(self, message: str, *, user_id="default", session_id="default") -> RunResult:
        if not all(isinstance(x, str) and x.strip() for x in (message, user_id, session_id)):
            raise ValueError("User, session and message must be nonempty strings")
        run = self.store.begin(user_id, session_id, message)
        result = RunResult(run)
        tools = self._tools(user_id, session_id)
        corrections = protocol_corrections = 0
        feedback = ""
        try:
            unresolved = self.store.unresolved(user_id, session_id)
            if unresolved:
                result.status, result.reason = "waiting_user", "unresolved_write"
                result.answer = (
                    "A previous write has an unknown result. Check its status before continuing."
                )
                return result
            memories = self.memory.retrieve(user_id, message)
            for step in range(self.config.max_steps):
                result.steps = step + 1
                request, info = self.context.prepare(
                    self.model,
                    self.store.messages(user_id, session_id),
                    tools.definitions(),
                    memories,
                    {"step": result.steps, "limit": self.config.max_steps, "feedback": feedback},
                    self.config.max_request_bytes,
                )
                model_call = uuid4().hex
                self.store.event(
                    run,
                    "model_request",
                    {
                        "model_call_id": model_call,
                        **info,
                        "sha256": hashlib.sha256(request.body).hexdigest(),
                        "body": json.loads(request.body),
                    },
                )
                try:
                    response = await asyncio.wait_for(
                        self.model.complete(request), self.config.model_timeout
                    )
                    self._validate_response(response)
                except (Exception, asyncio.CancelledError) as exc:
                    usage = getattr(exc, "usage", None)
                    self.store.event(
                        run,
                        "model_error",
                        {
                            "model_call_id": model_call,
                            "category": getattr(exc, "kind", type(exc).__name__),
                            "diagnostics": getattr(exc, "diagnostics", {}),
                            "usage": asdict(usage) if usage is not None else None,
                            "usage_known": usage is not None,
                        },
                    )
                    if (
                        isinstance(exc, ModelError)
                        and exc.kind == "protocol_error"
                        and usage is not None
                        and protocol_corrections < self.config.max_protocol_corrections
                    ):
                        protocol_corrections += 1
                        feedback = "The response violated the provider protocol. Use native tool calls or ordinary text. Prior tools must not be replayed."
                        self.store.event(
                            run, "protocol_correction", {"number": protocol_corrections}
                        )
                        continue
                    raise
                self.store.event(
                    run, "model_response", {"model_call_id": model_call, **asdict(response)}
                )
                feedback = ""
                assistant_message = {"role": "assistant", "content": response.content}
                if response.provider_metadata:
                    assistant_message["_provider_metadata"] = response.provider_metadata
                if not response.tool_calls:
                    self.store.append(
                        user_id, session_id, run, assistant_message
                    )
                    result.status, result.answer = "completed", response.content
                    break
                calls = [
                    ToolCall(uuid4().hex, call.name, call.arguments) for call in response.tool_calls
                ]
                self.store.event(
                    run,
                    "call_correlation",
                    {
                        "model_call_id": model_call,
                        "calls": [
                            {"provider_id": original.id, "call_id": call.id}
                            for original, call in zip(response.tool_calls, calls, strict=True)
                        ]
                    },
                )
                self.store.append(
                    user_id,
                    session_id,
                    run,
                    {
                        **assistant_message,
                        "tool_calls": [call.to_message() for call in calls],
                    },
                )
                completed = set()
                try:
                    for call in calls:
                        spec = tools.specs.get(call.name)

                        def record(attempt, status, body, call=call, spec=spec):
                            self.store.attempt(
                                run,
                                user_id,
                                session_id,
                                call.id,
                                call.name,
                                spec.read_only if spec else True,
                                attempt,
                                status,
                                body,
                            )
                            self.store.event(
                                run,
                                "tool_attempt",
                                {
                                    "call_id": call.id,
                                    "name": call.name,
                                    "attempt": attempt,
                                    "status": status,
                                    **body,
                                },
                            )

                        outcome = await tools.execute(call, record)
                        body = outcome.to_dict()
                        # A retrieved page must remain readable, not become another archive preview.
                        if call.name != "read_observation":
                            body["observation_ref"] = self.store.archive(
                                user_id, session_id, call.id, body
                            )
                        self.store.append(
                            user_id,
                            session_id,
                            run,
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": dumps(body),
                            },
                        )
                        completed.add(call.id)
                        if outcome.category == "arguments":
                            corrections += 1
                            self.store.event(
                                run,
                                "argument_correction",
                                {"call_id": call.id, "number": corrections},
                            )
                            if corrections > self.config.max_argument_corrections:
                                result.status, result.reason = "failed", "argument_correction_limit"
                                break
                        if call.name == "ask_user" and outcome.ok:
                            result.status, result.answer = "waiting_user", outcome.value["question"]
                            result.reason = "clarification"
                            break
                        if outcome.decision == Decision.WAIT:
                            result.status, result.reason = "waiting_user", outcome.category
                            result.answer = (
                                outcome.error or "User input or operation status is required."
                            )
                            break
                        if outcome.decision == Decision.TERMINATE:
                            result.status, result.reason = "failed", outcome.category
                            break
                finally:
                    # Every emitted call gets a result, even if a prior call stops/cancels a batch.
                    for call in calls:
                        if call.id in completed:
                            continue
                        row = self.store.db.execute(
                            "SELECT status,body FROM attempts WHERE call=? ORDER BY attempt DESC LIMIT 1",
                            (call.id,),
                        ).fetchone()
                        body = json.loads(row["body"]) if row else {}
                        body.update(
                            status=row["status"] if row else "not_dispatched",
                            error="Run stopped; inspect execution evidence",
                        )
                        self.store.append(
                            user_id,
                            session_id,
                            run,
                            {"role": "tool", "tool_call_id": call.id, "content": dumps(body)},
                        )
                if result.status != "running":
                    break
            else:
                result.status, result.reason = "failed", "step_limit"
        except asyncio.CancelledError:
            result.status, result.reason = "cancelled", "cancelled"
            raise
        except (StorageError, sqlite3.Error) as exc:
            result.status, result.reason = "failed", "storage_error"
            raise StorageError("Persistence failed; terminal durability is not guaranteed") from exc
        except (ModelError, ContextBudgetError, TimeoutError) as exc:
            result.status, result.reason = "failed", getattr(exc, "kind", type(exc).__name__)
        except Exception as exc:
            result.status, result.reason = "failed", "internal_error"
            self.store.event(run, "internal_error", {"type": type(exc).__name__})
        finally:
            # Structural only: counts settled mutating dispatches recorded by the executor.
            # Deliberately does not read the answer text, the user's wording, or the goal.
            result.goal_status = self._goal_status(run)
            self.store.finish(result)
        return result

    def _goal_status(self, run):
        """Report what the runtime observed about state change, never that a goal was met.

        Why this is not "did the agent finish?" -- that question belongs to the user. The
        program cannot tell a correctly read-only request from a run that stopped short,
        and inferring intent is exactly what this runtime refuses to do elsewhere
        (`not_inferred` in task_handoff). So the field states an observation:

          no_state_change_attempted  no mutating call ever reached a handler
          state_change_unconfirmed   mutating calls ran, none of them reported success
          state_change_observed      at least one mutating call reported success

        In experiments/retail-qwen-tier-order-20261108 these separate sharply on the
        official score (35 failures / 115 passes): runs that never dispatched a write
        failed 62% of the time, runs with an observed success 12%. A run may still be
        wrong after an observed change, so this is a signal, not a verdict.
        """
        attempted = confirmed = 0
        for row in self.store.db.execute(
            "SELECT read_only,status,body FROM attempts WHERE run=? ORDER BY rowid", (run,)
        ):
            if row["read_only"]:
                continue
            try:
                body = json.loads(row["body"])
            except (ValueError, TypeError):
                body = {}
            # dispatch_intent is the conservative pre-dispatch marker, not an execution.
            if body.get("phase") == "dispatch_intent":
                continue
            if row["status"] not in {"returned", "unknown"}:
                continue
            attempted += 1
            if row["status"] == "returned" and body.get("ok") is True:
                confirmed += 1
        if not attempted:
            return "no_state_change_attempted"
        return "state_change_observed" if confirmed else "state_change_unconfirmed"

    @staticmethod
    def _validate_response(response):
        if not isinstance(response, ModelResponse):
            raise ModelError("Adapter returned invalid type", kind="protocol_error")
        valid = response.finish_reason == ("tool_calls" if response.tool_calls else "stop")
        ids = [call.id for call in response.tool_calls]
        if not valid or len(ids) != len(set(ids)) or any(not x for x in ids):
            raise ModelError(
                "Invalid response termination/correlation",
                kind="protocol_error",
                usage=response.usage if response.usage_known else None,
            )
        if not response.tool_calls and not (response.content or "").strip():
            raise ModelError(
                "Empty response",
                kind="protocol_error",
                usage=response.usage if response.usage_known else None,
            )
