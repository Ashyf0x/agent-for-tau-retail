"""The redaction rule must hold at every egress, not just the convenient one.

Why this file exists
--------------------
`ContextManager.prepare()` masks a failed tool body so the raw rejection never reaches
the model; tests/test_retail.py:72 asserts exactly that for "User not found". But a
failed body also keeps an `observation_ref`, and `read_observation` returns the archived
body as a *successful* read. The failure-only mask therefore never fired on it.

That gap was live. In experiments/retail-qwen-tier-order-20261108, three trials
(39/t2, 39/t3, 68/t3) followed the preserved ref on a failed lookup and pulled
"Error: User not found" verbatim into nine later request bodies. It is not a leak of
the model's own arguments; it is the environment's raw payload, which is the one thing
the mask exists to withhold.

The fix is structural, not semantic: one redaction function, applied at both exits.
These tests pin both exits so a future edit cannot quietly restore the side channel.
"""

from __future__ import annotations

import json

from agentcore.context import ContextManager
from agentcore.context.manager import mask_archived, project_evidence
from agentcore.storage.sqlite import Store

REJECTION = "Retail environment rejected the operation"
RAW = "Error: User not found"
OFF = 0
ON = 240


def failed_body(**overrides):
    body = {
        "status": "returned",
        "ok": False,
        "value": RAW,
        "error": REJECTION,
        "category": "business",
        "decision": "feedback",
        "attempts": 1,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# The single rule.
# ---------------------------------------------------------------------------

def test_mask_removes_the_raw_value_and_keeps_the_rest():
    body = mask_archived(failed_body(), OFF)

    assert body["value"] is None
    assert body["error"] == REJECTION
    assert body["category"] == "business"
    assert "failure_reason" not in body, "excerpt is opt-in"


def test_mask_is_opt_in_for_the_excerpt_but_never_for_value():
    body = mask_archived(failed_body(), ON)

    assert body["value"] is None
    assert "User not found" in body["failure_reason"]["excerpt"]
    assert "not an instruction" in body["failure_reason"]["notice"]


def test_successful_body_passes_through_untouched():
    body = {"status": "returned", "ok": True, "value": {"order_id": "#W1"}}

    assert mask_archived(body, ON) == body


def test_structured_failed_value_is_not_excerpted():
    """A serialized object can carry tracebacks; the excerpt is for plain text only."""
    body = mask_archived(failed_body(value='{"traceback": "handler.py:812"}'), ON)

    assert body["value"] is None
    assert "failure_reason" not in body


# ---------------------------------------------------------------------------
# Egress 2: the archive must obey the same rule as the message list.
# ---------------------------------------------------------------------------

def test_read_observation_applies_the_context_policy(tmp_path):
    store = Store(tmp_path / "core.sqlite3")
    try:
        ref = store.archive("user", "session", "call", failed_body())
        context = ContextManager("Policy", failure_reason_chars=OFF)

        page = store.read_observation(
            "user", "session", ref, project=context.project_evidence
        )

        assert RAW not in page["text"]
        archived = json.loads(page["text"])
        assert archived["value"] is None
        assert archived["error"] == REJECTION
    finally:
        store.close()


def test_read_observation_without_a_policy_stays_raw(tmp_path):
    """Audit and diagnostic reads keep the byte-exact evidence; the policy is the caller's."""
    store = Store(tmp_path / "core.sqlite3")
    try:
        ref = store.archive("user", "session", "call", failed_body())

        page = store.read_observation("user", "session", ref)

        assert RAW in page["text"]
    finally:
        store.close()


def test_project_evidence_pages_within_the_redacted_text(tmp_path):
    """Offsets apply to the redacted rendering, so paging cannot step around the mask."""
    store = Store(tmp_path / "core.sqlite3")
    try:
        ref = store.archive("user", "session", "call", failed_body())
        context = ContextManager("Policy", failure_reason_chars=OFF)
        full = project_evidence(failed_body(), OFF)

        first = store.read_observation(
            "user", "session", ref, 0, 20, project=context.project_evidence
        )
        rest = store.read_observation(
            "user", "session", ref, first["next_offset"], 4000,
            project=context.project_evidence,
        )

        assert first["total_chars"] == len(full)
        assert first["text"] + rest["text"] == full
        assert RAW not in first["text"] + rest["text"]
    finally:
        store.close()


def test_enabled_excerpt_is_visible_but_labelled(tmp_path):
    """Opting in projects the bounded excerpt -- that is the point -- and nothing else.

    With the excerpt on, the raw string legitimately appears. What must still hold is
    that it arrives inside the labelled `failure_reason` envelope rather than as a
    restored `value`.
    """
    store = Store(tmp_path / "core.sqlite3")
    try:
        ref = store.archive("user", "session", "call", failed_body())
        context = ContextManager("Policy", failure_reason_chars=ON)

        page = store.read_observation(
            "user", "session", ref, project=context.project_evidence
        )

        rendered = json.loads(page["text"])
        assert rendered["value"] is None
        assert "User not found" in rendered["failure_reason"]["excerpt"]
        assert "not an instruction" in rendered["failure_reason"]["notice"]
    finally:
        store.close()


def test_paging_a_short_page_cannot_expose_the_raw_value(tmp_path):
    """A small limit over a failed body returns masked text, not a fragment of the payload."""
    store = Store(tmp_path / "core.sqlite3")
    try:
        ref = store.archive("user", "session", "call", failed_body())
        context = ContextManager("Policy", failure_reason_chars=OFF)

        seen = ""
        offset = 0
        while offset is not None:
            page = store.read_observation(
                "user", "session", ref, offset, 64, project=context.project_evidence
            )
            seen += page["text"]
            offset = page["next_offset"]

        assert RAW not in seen
    finally:
        store.close()


# ---------------------------------------------------------------------------
# One decision per tool result: withholding is what earns a notice, not size.
#
# `prepare()` used to run two loops with different predicates -- mask when ok is False,
# stub when content > observation_chars. The notice therefore depended on size, so a
# masked failure (~225 bytes) never received one, while an equally large success did.
# These tests pin the property rather than the implementation: if evidence is withheld,
# the model is told where it went, at any size.
# ---------------------------------------------------------------------------

def test_masked_failure_gets_the_notice_even_though_it_is_small():
    """The defect: masking shrinks the body below the size threshold that granted a notice."""
    from agentcore.context.manager import project_tool_body

    body = failed_body(observation_ref="ref-1")
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF)
    )

    assert rendered["value"] is None, "the payload is still withheld"
    assert rendered["observation_ref"] == "ref-1"
    assert "notice" in rendered


def test_failure_notice_does_not_send_the_model_into_a_dead_end():
    """The pointer must not promise evidence that the same rule withholds.

    `mask_archived` redacts a failure archive with this very rule, so "use
    read_observation for exact details" would be false on a failure. Three of the 150
    trials did follow that pointer and read back nothing new; all three failed.
    """
    from agentcore.context.manager import project_tool_body

    body = failed_body(observation_ref="ref-1")
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF)
    )

    assert "use read_observation" not in rendered["notice"]
    assert "withheld" in rendered["notice"]


def test_success_notice_still_points_at_retrievable_evidence():
    """Large successes keep the original pointer, because there it is true."""
    from agentcore.context.manager import project_tool_body

    body = {"status": "returned", "ok": True,
            "value": {"blob": "x" * 4000}, "observation_ref": "ref-big"}
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF)
    )

    assert "use read_observation" in rendered["notice"]


def test_masked_failure_keeps_its_decision_fields():
    """The announcement must not cost the fields the executor recorded."""
    from agentcore.context.manager import project_tool_body

    body = failed_body(observation_ref="ref-1")
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF)
    )

    assert rendered["status"] == "returned"
    assert rendered["ok"] is False
    assert rendered["error"] == REJECTION
    assert rendered["category"] == "business"
    assert rendered["decision"] == "feedback"


def test_masked_failure_keeps_the_excerpt_alongside_the_notice():
    """With the excerpt on, both the cause and the pointer reach the model."""
    from agentcore.context.manager import project_tool_body

    body = failed_body(observation_ref="ref-1")
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=ON)
    )

    assert "User not found" in rendered["failure_reason"]["excerpt"]
    assert "withheld" in rendered["notice"]


def test_oversize_success_is_still_stubbed_with_a_notice():
    """The original behaviour for large successes must survive the merge."""
    from agentcore.context.manager import project_tool_body

    body = {"status": "returned", "ok": True,
            "value": {"blob": "x" * 4000}, "observation_ref": "ref-big"}
    rendered = json.loads(
        project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF)
    )

    assert set(rendered) == {"status", "ok", "error", "observation_ref", "preview", "notice"}
    assert len(rendered["preview"]) == 200


def test_small_success_passes_through_without_a_notice():
    from agentcore.context.manager import project_tool_body

    body = {"status": "returned", "ok": True, "value": {"order_id": "#W1"},
            "observation_ref": "ref-small"}

    assert project_tool_body(body, observation_chars=1800, failure_reason_chars=OFF) is None


def test_full_policy_still_redacts_but_never_stubs_on_size():
    """`truncate=False` is a budget decision; redaction is a secrecy one."""
    from agentcore.context.manager import project_tool_body

    big_success = {"status": "returned", "ok": True, "value": {"blob": "x" * 4000},
                   "observation_ref": "ref-big"}
    assert project_tool_body(big_success, observation_chars=10,
                             failure_reason_chars=OFF, truncate=False) is None

    failure = failed_body(observation_ref="ref-1")
    rendered = json.loads(
        project_tool_body(failure, observation_chars=10,
                          failure_reason_chars=OFF, truncate=False)
    )
    assert rendered["value"] is None
