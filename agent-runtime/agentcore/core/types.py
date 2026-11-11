import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    # A provider correlation ID, replaced by the runtime before execution/replay.
    id: str
    name: str
    arguments: dict[str, Any] | str

    def to_message(self) -> dict:
        args = self.arguments if isinstance(self.arguments, str) else dumps(self.arguments)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": args},
        }


@dataclass
class ModelResponse:
    content: str | None = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    usage_known: bool = False
    # Opaque replay data owned by the model adapter, never treated as facts or an answer.
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRequest:
    """Immutable exact wire body: measured by Context and sent unchanged by the adapter."""

    body: bytes
    max_bytes: int

    def validate(self) -> None:
        if len(self.body) > self.max_bytes:
            raise ContextBudgetError("Required input exceeds the configured request byte limit")


class ModelClient(Protocol):
    def prepare(
        self, messages: list[dict], tools: list[dict], max_bytes: int
    ) -> PreparedRequest: ...
    async def complete(self, request: PreparedRequest) -> ModelResponse: ...


@dataclass
class RunResult:
    run_id: str
    status: Literal[
        "running", "completed", "waiting_user", "failed", "cancelled", "interrupted"
    ] = "running"
    answer: str = ""
    # What the runtime observed about state change -- never a claim that the goal was met.
    # A run that only read data and a run that changed nothing look identical from here;
    # distinguishing them needs the user's intent, which the program must not infer.
    goal_status: Literal[
        "unverified",
        "no_state_change_attempted",
        "state_change_unconfirmed",
        "state_change_observed",
    ] = "unverified"
    reason: str | None = None
    steps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ModelError(RuntimeError):
    def __init__(self, message: str, *, kind="model_error", diagnostics=None, usage=None):
        super().__init__(message)
        self.kind, self.diagnostics, self.usage = kind, diagnostics or {}, usage


class ContextBudgetError(RuntimeError):
    pass


class StorageError(RuntimeError):
    """A run cannot claim a durable terminal state if storage failed."""


class SessionBusy(RuntimeError):
    pass
