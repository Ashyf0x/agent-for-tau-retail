def run_report(store, run_id):
    events = store.events(run_id)
    requests = [e for e in events if e["kind"] == "model_request"]
    ends = [e for e in events if e["kind"] in {"model_response", "model_error"}]
    terminal_ids = {e["model_call_id"] for e in ends}
    known = [e["usage"] for e in ends if e.get("usage_known")]
    return {
        "run_id": run_id,
        "terminal": next((e for e in reversed(events) if e["kind"] == "run_end"), None),
        "model_calls": len(requests),
        "unknown_usage_calls": sum(not e.get("usage_known", False) for e in ends),
        "unterminated_model_calls": [
            e["model_call_id"] for e in requests if e["model_call_id"] not in terminal_ids
        ],
        "known_input_tokens": sum(u["input_tokens"] for u in known),
        "known_output_tokens": sum(u["output_tokens"] for u in known),
        "automatic_retries": sum(
            e["kind"] == "tool_attempt" and e.get("phase") == "dispatch_intent" and e["attempt"] > 1
            for e in events
        ),
        "argument_corrections": sum(e["kind"] == "argument_correction" for e in events),
        "protocol_corrections": sum(e["kind"] == "protocol_correction" for e in events),
        "goal_assessment": "unverified",
    }
