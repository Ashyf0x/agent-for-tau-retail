"""L3 session/trial terminal state: declared by an external authority, never read by the loop.

The last test is the important one. It locks in the property that makes this layer
safe to add to a fixed evaluation: writing a terminal declaration cannot change a
single byte of what prepare() sends to the model.
"""

import hashlib

import pytest

from agentcore.context import ContextManager
from agentcore.core.model import ScriptedModel
from agentcore.core.types import RunResult


def _projection(store, user, session):
    manager = ContextManager("system")
    model = ScriptedModel([])
    request, info = manager.prepare(
        model, store.messages(user, session), [], [],
        {"step": 1, "limit": 12, "feedback": ""}, 64000,
    )
    return request.body, info


def test_finish_session_records_only_structural_facts(store):
    first = store.begin("retail", "call", "first turn")
    store.finish(RunResult(first, status="completed", answer="a"))
    second = store.begin("retail", "call", "second turn")
    store.append("retail", "call", second, {"role": "assistant", "content": "done"})

    store.finish_session("retail", "call", reason="user_stop",
                         source="tau2_orchestrator")

    row = store.session_terminal("retail", "call")
    assert row["reason"] == "user_stop"
    assert row["source"] == "tau2_orchestrator"
    assert row["terminal_run"] == second
    assert row["run_count"] == 2
    assert row["created"]


def test_caller_cannot_assert_false_structural_facts(store):
    """run_count and terminal_run come from runs, not from the declaration."""
    store.finish_session("retail", "call", reason="user_stop", source="official")
    assert store.session_terminal("retail", "call")["run_count"] == 0
    assert store.session_terminal("retail", "call")["terminal_run"] is None

    run = store.begin("retail", "call", "one turn")
    store.finish_session("retail", "call", reason="user_stop", source="official")
    row = store.session_terminal("retail", "call")
    assert row["run_count"] == 1
    assert row["terminal_run"] == run


def test_declarations_are_append_only(store):
    store.finish_session("retail", "call", reason="user_stop", source="official")
    store.finish_session("retail", "call", reason="max_steps", source="official")
    assert store.session_terminal("retail", "call")["reason"] == "max_steps"
    rows = store.db.execute(
        "SELECT reason FROM session_terminations ORDER BY seq"
    ).fetchall()
    assert [row["reason"] for row in rows] == ["user_stop", "max_steps"]


def test_empty_or_missing_declaration_is_rejected(store):
    for kwargs in (
        {"reason": "", "source": "official"},
        {"reason": None, "source": "official"},
        {"reason": "   ", "source": "official"},
        {"reason": "user_stop", "source": ""},
        {"reason": "user_stop", "source": None},
    ):
        with pytest.raises(ValueError):
            store.finish_session("retail", "call", **kwargs)
    assert store.session_terminal("retail", "call") is None


def test_other_sessions_are_untouched(store):
    store.finish_session("retail", "call", reason="user_stop", source="official")
    assert store.session_terminal("retail", "other") is None
    assert store.session_terminal("other", "call") is None


def test_terminal_record_does_not_change_model_visible_context(store):
    """L3 evidence is not an input to prepare(); the wire body must be identical."""
    store.begin("retail", "call", "hello")
    body_before, info_before = _projection(store, "retail", "call")
    messages_before = store.messages("retail", "call")

    store.finish_session("retail", "call", reason="user_stop",
                         source="tau2_orchestrator")
    store.finish_session("retail", "call", reason="max_steps", source="official")

    body_after, info_after = _projection(store, "retail", "call")
    assert store.messages("retail", "call") == messages_before
    assert body_after == body_before
    assert info_after == info_before
    assert hashlib.sha256(body_after).hexdigest() == hashlib.sha256(body_before).hexdigest()
