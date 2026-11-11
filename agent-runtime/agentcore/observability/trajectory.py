"""Read-only reconstruction of recorded evidence, without executing or judging a run."""

import argparse
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def _rows(db, query, parameters):
    rows = []
    for row in db.execute(query, parameters):
        item = dict(row)
        if "body" in item:
            item["data"] = json.loads(item["body"])
        rows.append(item)
    return rows


def reconstruct(database, run_id, *, user_id=None):
    """Read a consistent snapshot. Missing records are gaps, never synthesized events."""
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if run is None or user_id is not None and run["user"] != user_id:
            raise KeyError("Run not found in requested scope")
        run = dict(run)
        run["result"] = json.loads(run["result"])
        messages = _rows(
            db,
            "SELECT * FROM messages WHERE run=? AND user=? AND session=? ORDER BY seq",
            (run_id, run["user"], run["session"]),
        )
        events = _rows(db, "SELECT * FROM events WHERE run=? ORDER BY seq", (run_id,))
        attempts = _rows(db, "SELECT * FROM attempts WHERE run=? ORDER BY call,attempt", (run_id,))
        all_observations = _rows(
            db,
            "SELECT * FROM observations WHERE user=? AND session=?",
            (run["user"], run["session"]),
        )

    calls = {}
    for message in messages:
        for call in message["data"].get("tool_calls", []):
            calls[call["id"]] = {
                "call_id": call["id"],
                "definition": call,
                "message_id": message["id"],
                "provider_id": None,
                "model_call_id": None,
                "result_messages": [],
                "attempts": [],
                "attempt_events": [],
                "observations": [],
            }
    gaps = [
        "Raw HTTP request bytes/headers and raw HTTP responses are not recorded; request JSON, byte count/hash and parsed responses are exported.",
        "Message seq and event seq order their own tables; no global cross-table timestamp order is recorded.",
        "Unreturned model reasoning, user-goal correctness and failure causes are not inferred.",
    ]
    for message in messages:
        call = calls.get(message["data"].get("tool_call_id"))
        if call is not None:
            call["result_messages"].append(message["id"])
    model_calls = {}
    for event in events:
        data = event["data"]
        model_id = data.get("model_call_id")
        if event["kind"] in {"model_request", "model_response", "model_error"}:
            entry = model_calls.setdefault(
                model_id,
                {"model_call_id": model_id, "request": None, "responses": [], "errors": []},
            )
            if event["kind"] == "model_request":
                entry["request"] = event["seq"]
            else:
                entry["responses" if event["kind"] == "model_response" else "errors"].append(
                    event["seq"]
                )
        if event["kind"] == "call_correlation":
            for mapping in data["calls"]:
                if mapping["call_id"] in calls:
                    calls[mapping["call_id"]].update(
                        provider_id=mapping["provider_id"], model_call_id=model_id
                    )
        if event["kind"] == "tool_attempt" and data.get("call_id") in calls:
            calls[data["call_id"]]["attempt_events"].append(event["seq"])
    for attempt in attempts:
        if attempt["call"] in calls:
            calls[attempt["call"]]["attempts"].append(attempt)
    observations = [row for row in all_observations if row["call"] in calls]
    for observation in observations:
        calls[observation["call"]]["observations"].append(observation["ref"])
    for call in calls.values():
        if call["model_call_id"] is None:
            gaps.append(
                f"{call['call_id']}: model_call_id not recorded in correlation; no association guessed."
            )
        if not call["result_messages"]:
            gaps.append(f"{call['call_id']}: tool result message not recorded.")
    for entry in model_calls.values():
        if entry["request"] is None or not (entry["responses"] or entry["errors"]):
            gaps.append(
                f"{entry['model_call_id']}: model request or terminal response/error not recorded."
            )
    ends = [event for event in events if event["kind"] == "run_end"]
    if not ends:
        gaps.append(
            "run_end not recorded; the stored run status is shown without inventing a terminal event."
        )
    if not any(message["data"].get("role") == "user" for message in messages):
        gaps.append("User input not recorded for this run.")
    return {
        "version": "trajectory-v1",
        "run": run,
        "messages": messages,
        "events": events,
        "model_calls": list(model_calls.values()),
        "tool_calls": list(calls.values()),
        "observations": observations,
        "terminal_events": ends,
        "not_recorded": gaps,
    }


def _json_block(value):
    content = json.dumps(value, ensure_ascii=False, indent=2)
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", content)), default=0))
    return f"{fence}json\n{content}\n{fence}\n"


def to_markdown(trajectory):
    run = trajectory["run"]
    parts = [
        f"# Recorded trajectory: {run['id']}\n",
        f"Recorded run status: `{run['status']}`. Goal assessment: unverified.\n",
        "## Gaps and interpretation limits\n",
        *[f"- {gap}\n" for gap in trajectory["not_recorded"]],
        "\n## Run and tool/model associations\n",
        _json_block(run),
        _json_block(
            {"model_calls": trajectory["model_calls"], "tool_calls": trajectory["tool_calls"]}
        ),
        "## Original messages (message sequence)\n",
    ]
    for message in trajectory["messages"]:
        parts.extend([f"### Message {message['seq']} / {message['id']}\n", _json_block(message)])
    parts.append("## Recorded events (event sequence)\n")
    for event in trajectory["events"]:
        parts.extend([f"### Event {event['seq']} / {event['kind']}\n", _json_block(event)])
    parts.extend(["## Archived evidence\n", _json_block(trajectory["observations"])])
    return "\n".join(parts)


def export(database, run_id, output, *, format="json", user_id=None):
    if format not in {"json", "markdown"}:
        raise ValueError("Export format must be json or markdown")
    trajectory = reconstruct(database, run_id, user_id=user_id)
    content = (
        to_markdown(trajectory)
        if format == "markdown"
        else json.dumps(trajectory, ensure_ascii=False, indent=2)
    )
    # Never overwrite an existing export or database, and never mutate the source DB.
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--user-id")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.database, args.run_id, args.output, format=args.format, user_id=args.user_id)
    print(f"Exported recorded trajectory to {args.output}")


if __name__ == "__main__":
    main()
