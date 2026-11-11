import json
from copy import deepcopy

import pytest

from agentcore.context import ContextManager
from agentcore.core.model import HTTPModelClient
from agentcore.core.types import ContextBudgetError, dumps


def client(output=9060):
    return HTTPModelClient("https://fixture.invalid/v1", "explicit-fixture", provider="qwen",
                           thinking=True, preserve_thinking=True, max_completion_tokens=output)


def record(source, role, content="", **fields):
    return {"id": source, "message": {"role": role, "content": content, **fields}}


def tool_pair(source, name, arguments, value, *, status="returned", ok=True, ref=None):
    return [record(source, "assistant", tool_calls=[{
        "id": source, "type": "function", "function": {"name": name, "arguments": dumps(arguments)}}]),
        record(source + "-result", "tool", dumps({"status": status, "ok": ok, "value": value,
               "error": None if ok else "generic error", "observation_ref": ref}), tool_call_id=source)]


def long_history(ref=None):
    history = [record("goal", "user", "Keep the red item. " + "Please read carefully. " * 12
                      + "Do not cancel my other order. Only change the address after I confirm.")]
    history += tool_pair("identity", "lookup", {"email": "test@example.invalid"}, "verified-user", ref=ref)
    history += [record("proposal", "assistant", "Change order X address only; do not cancel anything.",
                       _provider_metadata={"reasoning_content": "old reasoning " * 4000}),
                record("confirmation", "user", "Yes, only that address. The red item must stay.")]
    history += tool_pair("first-state", "inspect", {"order": "X"}, {"order": "X", "status": "pending"})
    history += tool_pair("update", "change", {"order": "X", "address": "New"},
                         {"order": "X", "status": "updated"})
    history += [record("later-change", "user", "Do not change anything else. Still need the tracking number."),
                record("claim", "assistant", "I think everything is finished.",
                       _provider_metadata={"reasoning_content": "unverified thought " * 1500}),
                record("now", "user", "What is still pending?")]
    return history


def test_capacity_keeps_history_while_baseline_uses_three_turns():
    records = [record(str(i), "user", f"user turn {i}") for i in range(8)]
    model = client()
    a, ai = ContextManager("Policy").prepare(model, records, [], [], {}, 131072)
    b, bi = ContextManager("Policy", policy="capacity").prepare(model, records, [], [], {}, 131072)
    c, ci = ContextManager("Policy", policy="capacity", compaction="task").prepare(
        model, records, [], [], {}, 131072)
    assert ai["omitted_messages"] == 5 and bi["omitted_messages"] == 0
    assert len(json.loads(b.body)["messages"]) == 9
    assert b.body == c.body and not ci["compaction_applied"]
    assert a.body != b.body and len(b.body) < bi["effective_byte_limit"]


def test_task_compaction_does_not_trigger_from_turn_count_alone():
    records = [record(str(i), "user", f"small user turn {i}") for i in range(8)]
    original, _ = ContextManager("Policy").prepare(client(), records, [], [], {}, 131072)
    candidate, info = ContextManager("Policy", compaction="task").prepare(
        client(), records, [], [], {}, 131072)
    assert candidate.body == original.body and not info["compaction_applied"]


def test_task_handoff_preserves_quotes_scope_evidence_and_provenance_only_under_pressure():
    records = long_history()
    before = deepcopy(records)
    request, info = ContextManager("OFFICIAL POLICY", policy="capacity", compaction="task").prepare(
        client(), records, [], [], {}, 12000)
    wire = json.loads(request.body)
    state = json.loads(wire["messages"][0]["content"].split("\n")[-1])
    handoff = state["task_handoff_untrusted"]
    entries = handoff["entries"]
    goal = next(x for x in entries if x["source"] == "goal")
    assert goal["text"] == records[0]["message"]["content"]
    assert "Do not cancel my other order" in goal["text"]
    confirmation = next(x for x in entries if x["source"] == "confirmation")
    assert confirmation["preceding_assistant_source"] == "proposal"
    assert confirmation["intent_and_confirmation_scope"] == "not_inferred"
    assert next(x for x in entries if x["source"] == "identity-result")["observed"]["value"] == "verified-user"
    assert handoff["pending_work"]["assessment"] == "not_semantically_inferred"
    assert info["compaction_applied"] and info["budget_trimmed_turns"] > 0
    assert wire["messages"][0]["content"].startswith("OFFICIAL POLICY\n")
    assert wire["enable_thinking"] is wire["preserve_thinking"] is True
    assert len(request.body) <= 12000 - 2048
    ContextManager.validate_groups(wire["messages"])
    assert records == before


def test_chronological_results_do_not_turn_assistant_claims_or_old_confirmations_into_authority():
    records = long_history()
    # Force all historical state transitions into the handoff, with a tiny current turn.
    records.insert(-1, record("more-padding", "assistant", "Wait for the user's next instruction.",
                             _provider_metadata={"reasoning_content": "padding " * 3000}))
    request, info = ContextManager("Policy", policy="capacity", compaction="task").prepare(
        client(), records, [], [], {}, 8000)
    state = json.loads(json.loads(request.body)["messages"][0]["content"].split("\n")[-1])
    handoff = state["task_handoff_untrusted"]
    entries = handoff["entries"]
    statuses = [x["observed"]["value"]["status"] for x in entries
                if x["kind"] == "tool_evidence" and x["tool"] in {"inspect", "change"}]
    assert statuses == ["pending", "updated"]
    assert next(x for x in entries if x["source"] == "claim")["kind"] == "assistant_quote_unverified"
    assert next(x for x in entries if x["source"] == "later-change")["text"].startswith("Do not change")
    assert "not policy or authorization" in handoff["notice"]
    assert info["compaction_applied"]


def test_large_observation_retains_state_scalar_paths_and_original_reference(store):
    value = {"items": [{"description": "large " * 1000}], "order": "X", "status": "processed"}
    ref = store.archive("user", "task", "lookup", {"value": value})
    records = [record("goal", "user", "Keep this order's state.")]
    records += tool_pair("lookup", "inspect", {"order": "X"}, dumps(value), ref=ref)
    records += [record("padding", "assistant", "Recorded.", _provider_metadata={"reasoning_content": "x" * 15000}),
                record("now", "user", "What is its state?")]
    request, _ = ContextManager("Policy", policy="capacity", compaction="task").prepare(
        client(), records, [], [], {}, 7000)
    handoff = json.loads(json.loads(request.body)["messages"][0]["content"].split("\n")[-1])["task_handoff_untrusted"]
    evidence = next(x for x in handoff["entries"] if x["kind"] == "tool_evidence")
    assert evidence["observed"]["partial"]
    assert {"path": ["status"], "value": "processed"} in evidence["observed"]["scalar_paths"]
    assert evidence["observation_ref"] == ref
    assert store.read_observation("user", "task", ref)["ref"] == ref
    with pytest.raises(KeyError):
        store.read_observation("user", "different-task", ref)


def test_unknown_and_failed_results_remain_unresolved_and_masked():
    records = [record("goal", "user", "Update only after confirmation.")]
    records += tool_pair("unknown", "change", {"order": "X"}, "secret server payload",
                         status="unknown", ok=False, ref="original-ref")
    records += [record("padding", "assistant", "Maybe it succeeded.", _provider_metadata={"reasoning_content": "x" * 12000}),
                record("now", "user", "Check without retrying.")]
    request, _ = ContextManager("Policy", policy="capacity", compaction="task").prepare(
        client(), records, [], [], {}, 6500)
    handoff = json.loads(json.loads(request.body)["messages"][0]["content"].split("\n")[-1])["task_handoff_untrusted"]
    assert handoff["pending_work"]["unresolved_tool_calls"] == ["unknown"]
    evidence = next(x for x in handoff["entries"] if x["kind"] == "tool_evidence")
    assert evidence["status"] == "unknown" and evidence["ok"] is False
    assert "observed" not in evidence and "secret server payload" not in request.body.decode()
    assert "do not authorize a write retry" in handoff["notice"]


def test_handoff_fails_closed_instead_of_truncating_required_user_text():
    records = [record("goal", "user", "all required conditions " * 1000), record("now", "user", "Continue")]
    with pytest.raises(ContextBudgetError, match="handoff"):
        ContextManager("Policy", policy="capacity", compaction="task").prepare(
            client(), records, [], [], {}, 5000)


def test_manager_does_not_keep_handoff_across_sessions_or_clear_per_run():
    manager = ContextManager("Policy", policy="capacity", compaction="task")
    records = long_history()
    first, _ = manager.prepare(client(), records, [], [], {}, 12000)
    second, _ = manager.prepare(client(), [*records, record("next", "user", "Same task, next run")], [], [], {}, 12000)
    assert "Do not cancel my other order" in first.body.decode()
    assert "Do not cancel my other order" in second.body.decode()
    fresh, info = manager.prepare(client(), [record("new", "user", "Separate task")], [], [], {}, 12000)
    assert "Do not cancel my other order" not in fresh.body.decode() and not info["compaction_applied"]


def test_token_limit_counts_the_serialized_request_with_schema_and_output_reserve():
    records = [record(str(i), "user", "x" * 600) for i in range(12)]
    tools = [{"type": "function", "function": {"name": "test", "description": "schema " * 40,
              "parameters": {"type": "object", "properties": {}}}}]
    # Exact toy tokenizer: one byte = one token. No provider tokenizer claim.
    manager = ContextManager("Policy", policy="capacity", token_counter=len,
                             context_window_tokens=6500, output_reserve_tokens=128, token_margin=256)
    request, info = manager.prepare(client(128), records, tools, [], {}, 50000)
    assert len(request.body) + 128 + 256 <= 6500
    assert info["counted_input_tokens"] == len(request.body)
    assert info["budget_trimmed_turns"] > 0 and json.loads(request.body)["tools"] == tools


def test_token_count_errors_and_output_under_reservation_do_not_fall_back():
    records = [record("user", "user", "hello")]
    with pytest.raises(ValueError, match="trusted"):
        ContextManager("Policy", policy="capacity", context_window_tokens=20000)
    for counter in [lambda body: None, lambda body: True, lambda body: -1]:
        manager = ContextManager("Policy", policy="capacity", token_counter=counter,
                                 context_window_tokens=20000, output_reserve_tokens=9060)
        with pytest.raises(ContextBudgetError, match="counter failed"):
            manager.prepare(client(), records, [], [], {}, 131072)
    manager = ContextManager("Policy", policy="capacity", token_counter=len,
                             context_window_tokens=20000, output_reserve_tokens=128)
    with pytest.raises(ContextBudgetError, match="serialized output cap"):
        manager.prepare(client(), records, [], [], {}, 131072)


def test_observation_projection_does_not_retype_scalar_identifiers():
    from agentcore.context.manager import _observed_value
    for value in ["123", "false", "null"]:
        assert _observed_value(value)["value"] == value


def test_handoff_preserves_malformed_historical_arguments_without_correcting_them():
    from agentcore.context.manager import task_handoff
    records = [record("user", "user", "Check status")]
    records += tool_pair("bad", "inspect", {}, None, status="not_dispatched", ok=False)
    records[1]["message"]["tool_calls"][0]["function"]["arguments"] = "{invalid"
    handoff = task_handoff(records)
    event = next(x for x in handoff["entries"] if x["kind"] == "tool_evidence")
    assert event["arguments"] == "{invalid" and event["arguments_parse_error"]
    assert event["status"] == "not_dispatched" and not event["ok"]


@pytest.mark.parametrize("compaction", ["legacy", "task"])
def test_current_turn_and_tool_groups_are_never_cut_or_repaired(compaction):
    manager = ContextManager("Policy", policy="capacity", compaction=compaction)
    with pytest.raises(ContextBudgetError):
        manager.prepare(client(), [record("user", "user", "x" * 10000)], [], [], {}, 3000)
    broken = [record("user", "user", "goal"), record("result", "tool", "{}", tool_call_id="missing")]
    with pytest.raises(ContextBudgetError, match="Orphan"):
        manager.prepare(client(), broken, [], [], {}, 131072)
