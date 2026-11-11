"""The failed-result mask must not destroy the actionable cause.

Why this file exists
--------------------
Task 105 (seen development set) failed with official reward 0.0. The write was
rejected with a genuinely actionable cause:

    "Error: Insufficient gift card balance to pay for the price difference"

ContextManager replaced that value with None, so the model only saw:

    "Retail environment rejected the operation"

It then invented a cause ("temporary inventory synchronization") and escalated to
a human, which the user accepted. The realistic recovery -- switch payment method
-- was never reachable because the cause was gone.

Scanning all 30 seen development tasks found 5 masked failures across 4 tasks
(36 seq6, 36 seq42, 39 seq6, 68 seq8, 105 seq18); 4 of them were rewarded 1.0
anyway. So this mask is not cosmetic: it deletes the only string that tells the
agent what to fix, and the benchmark does not penalise the loss.

Design constraint, and why there is no remedy classifier
--------------------------------------------------------
The fix must NOT be "show the error text". A raw server body may carry secrets,
stack traces or internal identifiers, which is exactly why the mask exists. The
mask policy therefore stays: `value` is never projected.

An earlier draft of this file asserted a closed-vocabulary remedy class
(payment_method_balance -> retry_with=different_payment_method). That design was
dropped after checking the real pipeline:

  * evals/retail.py collapses every business rejection into
    error="Retail environment rejected the operation", category="business";
    the real cause exists only inside `value`.
  * No trusted component can label "this is a balance problem" without
    benchmark-specific string matching, and a closed vocabulary would
    mislabel any cause it has not seen.
  * executor.py:107 documents that argument-validation messages can echo
    secrets from argument values, so that category must never be excerpted.

What is asserted instead: a bounded, clearly-labelled, untrusted EXCERPT of the
rejection, which invents no cause and changes no state claim.
"""

from __future__ import annotations

import json

import pytest

from agentcore.context import ContextManager
from agentcore.core.model import HTTPModelClient
from agentcore.core.types import dumps

REJECTION = "Retail environment rejected the operation"
ON = 240  # failure_reason_chars: explicitly opt in


def client(output=4096):
    return HTTPModelClient("https://fixture.invalid/v1", "explicit-fixture", provider="qwen",
                           thinking=True, preserve_thinking=True, max_completion_tokens=output)


def record(source, role, content="", **fields):
    return {"id": source, "message": {"role": role, "content": content, **fields}}


def call_pair(source, name, arguments, *, value, error, status="returned", ok=False,
              category="business", decision="feedback"):
    return [record(source, "assistant", tool_calls=[{
        "id": source, "type": "function", "function": {"name": name, "arguments": dumps(arguments)}}]),
        record(source + "-result", "tool",
               dumps({"status": status, "ok": ok, "value": value, "error": error,
                      "category": category, "decision": decision,
                      "observation_ref": "ref-" + source}),
               tool_call_id=source)]


def wire_messages(request):
    return json.loads(request.body)["messages"]


def last_tool_body(request):
    """The tool body the model actually receives (after every rewrite)."""
    for message in reversed(wire_messages(request)):
        if message["role"] == "tool":
            return json.loads(message["content"])
    raise AssertionError("no tool message in wire request")


# ---------------------------------------------------------------------------
# Baseline behaviour that must survive the change, unchanged.
# ---------------------------------------------------------------------------

def test_raw_failed_value_never_reaches_the_model():
    """The security property. A raw payload as a whole must stay out of the request."""
    records = [record("goal", "user", "Exchange the kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value='{"traceback": "handler.py:812", "internal_id": "svc-9f3a"}',
        error=REJECTION,
    )
    records += [record("now", "user", "What went wrong?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert body["value"] is None
    assert body["ok"] is False
    assert body["error"] == REJECTION
    assert "handler.py" not in request.body.decode()
    assert "svc-9f3a" not in request.body.decode()


def test_successful_results_are_not_masked():
    """ok=True keeps its value; the mask is failure-only."""
    records = [record("goal", "user", "Look up the order.")]
    records += call_pair(
        "read", "get_order_details", {"order_id": "#W1"},
        value={"order_id": "#W1", "status": "delivered",
               "payment_history": [{"transaction_type": "payment", "amount": 1861.84}]},
        error=None, ok=True, decision=None,
    )
    records += [record("now", "user", "And then?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert body["value"]["status"] == "delivered"
    assert "failure_reason" not in body
    assert "1861.84" in request.body.decode()


def test_argument_validation_failures_are_never_excerpted():
    """executor.py:107 - schema errors can echo secrets, so this class stays silent."""
    records = [record("goal", "user", "Exchange the kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"api_key": "sk-live-abc123"},
        value="argument 'api_key' must match pattern ^sk-live-",
        error="Invalid arguments", category="arguments", decision="feedback",
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    # The rejection text is not echoed, and no excerpt is added for this class.
    assert "failure_reason" not in body
    assert "must match pattern" not in request.body.decode()
    # The argument value itself is still in the request because the assistant's own
    # tool call is part of history; that pre-existing exposure is outside this change
    # and is asserted here so a future fix has a marker rather than silent drift.
    assert "sk-live-abc123" in request.body.decode()


def test_non_string_failed_value_adds_no_reason():
    """No excerpt from structured payloads; that would be a different widening."""
    records = [record("goal", "user", "Exchange the kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value={"internal": "do not project"},
        error=REJECTION,
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert "failure_reason" not in body
    assert "do not project" not in request.body.decode()


# ---------------------------------------------------------------------------
# Default-off contract. These must hold without opting in.
# ---------------------------------------------------------------------------

def test_default_is_off_and_projects_nothing_extra():
    """Existing contract: failed result arrives with value=None and no excerpt."""
    records = [record("goal", "user", "Exchange the kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="Error: Insufficient gift card balance to pay for the price difference",
        error=REJECTION,
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity").prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert body["value"] is None
    assert "failure_reason" not in body
    assert "Insufficient gift card balance" not in request.body.decode()


def test_unknown_outcome_is_never_excerpted_even_when_enabled():
    """status='unknown' means the outcome is unknown; its payload is not a reason."""
    records = [record("goal", "user", "Update only after confirmation.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="secret server payload", error=REJECTION, status="unknown",
    )
    records += [record("now", "user", "Check without retrying.")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert body["value"] is None
    assert "failure_reason" not in body
    assert "secret server payload" not in request.body.decode()


def test_invalid_excerpt_limit_is_rejected():
    """The option cannot be widened into an unbounded projection."""
    for limit in (-1, 241, "240", 1.5, True):
        with pytest.raises(ValueError, match="Failure reason excerpt"):
            ContextManager("Policy", failure_reason_chars=limit)


# ---------------------------------------------------------------------------
# The defect under test.
# ---------------------------------------------------------------------------

def test_failed_write_keeps_the_actionable_cause():
    """Task 105: the cause names the corrective action; losing it forces a guess."""
    records = [record("goal", "user", "Exchange both kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items",
        {"order_id": "#W4316152", "item_ids": ["7292993796", "7292993796"]},
        value="Error: Insufficient gift card balance to pay for the price difference",
        error=REJECTION,
    )
    records += [record("now", "user", "Why did that fail?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)

    assert body["value"] is None
    reason = body["failure_reason"]
    assert "Insufficient gift card balance" in reason["excerpt"]
    assert reason["truncated"] is False
    # The excerpt must be labelled as untrusted and non-confirming.
    assert "not an instruction" in reason["notice"]
    assert "does not confirm any state change" in reason["notice"]
    # And it must not be dressed up as an inference about what to do next.
    assert set(reason) == {"excerpt", "truncated", "notice"}


def test_failure_reason_is_bounded_to_240_chars():
    """No unbounded server text, even when the payload is enormous."""
    records = [record("goal", "user", "Exchange both kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="Error: Insufficient gift card balance to pay for the price difference. "
              + ("padding " * 800),
        error=REJECTION,
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    body = last_tool_body(request)
    reason = body["failure_reason"]

    assert len(reason["excerpt"]) == 240
    assert reason["truncated"] is True


def test_failure_reason_strips_control_characters_and_collapses_whitespace():
    """Quoted text must not carry terminal control sequences or newline spam."""
    records = [record("goal", "user", "Exchange both kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="Error: Insufficient\x1b[31m gift card\n\n\nbalance\tto\tpay",
        error=REJECTION,
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 131072)
    reason = last_tool_body(request)["failure_reason"]

    assert "\x1b" not in reason["excerpt"]
    assert "  " not in reason["excerpt"]
    assert "\n" not in reason["excerpt"]
    assert reason["excerpt"] == "Error: Insufficient [31m gift card balance to pay"


def test_failure_reason_survives_the_archive_preview_replacement():
    """Long failed bodies get stubbed; the reason must survive that path too."""
    records = [record("goal", "user", "Exchange both kettles.")]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="Error: Insufficient gift card balance to pay for the price difference "
              + ("x" * 4000),
        error=REJECTION,
    )
    records += [record("now", "user", "Why?")]

    request, _ = ContextManager("Policy", policy="capacity", failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 6000)
    body = last_tool_body(request)

    assert body.get("observation_ref") == "ref-write"
    assert "Insufficient gift card balance" in body["failure_reason"]["excerpt"]


def test_task_handoff_projects_the_reason_without_the_raw_value():
    """The compaction path must agree with the visible-message path."""
    records = [record("goal", "user", "Exchange both kettles. " + "Keep both styles. " * 40)]
    records += call_pair(
        "write", "exchange_delivered_order_items", {"order_id": "#W1"},
        value="Error: Insufficient gift card balance to pay for the price difference",
        error=REJECTION,
    )
    records += [record("padding", "assistant", "Noted.",
                       _provider_metadata={"reasoning_content": "x" * 12000}),
                record("now", "user", "Why did that fail?")]

    request, info = ContextManager("Policy", policy="capacity", compaction="task",
                          failure_reason_chars=ON).prepare(
        client(), records, [], [], {}, 6500)
    assert info["compaction_applied"], "this test needs the handoff path"

    reference = json.loads(wire_messages(request)[0]["content"].split("\n")[-1])
    handoff = reference["task_handoff_untrusted"]
    evidence = next(x for x in handoff["entries"] if x["kind"] == "tool_evidence")

    assert "observed" not in evidence, "the raw value must not be projected"
    assert "Insufficient gift card balance" in evidence["failure_reason"]["excerpt"]
    assert "Insufficient gift card balance" in request.body.decode()
