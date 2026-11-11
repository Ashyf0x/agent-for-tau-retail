from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    RETRY = "retry"
    FEEDBACK = "feedback"
    WAIT = "wait_user"
    QUERY = "query_status"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class ErrorInfo:
    stage: str
    category: str
    dispatched: bool
    outcome_known: bool
    attempt: int
    read_only: bool = False
    idempotent: bool = False
    queryable: bool = False


def decide(error: ErrorInfo, max_attempts: int) -> Decision:
    if error.category in {"internal", "storage", "cancelled"}:
        return Decision.TERMINATE
    if error.category in {"missing_input", "authorization"}:
        return Decision.WAIT
    if error.dispatched and not error.outcome_known and not error.read_only:
        return Decision.QUERY if error.queryable else Decision.WAIT
    if error.category == "transient" and error.attempt < max_attempts:
        if error.read_only or error.idempotent or not error.dispatched:
            return Decision.RETRY
    return Decision.FEEDBACK


class ToolFailure(Exception):
    """Trusted handler contract; message must be safe for model/user display."""

    def __init__(self, message: str, category="business", *, outcome_known=True):
        super().__init__(message)
        self.category, self.outcome_known = category, outcome_known
