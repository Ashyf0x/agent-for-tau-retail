import asyncio
import hashlib
import json
import sqlite3
import sys
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentcore.app import smoke
from agentcore.core.types import ModelError, PreparedRequest
from agentcore.storage import Store

# Explicit synthetic endpoint for deterministic tests; never a production default.
FIXTURE_URL = "https://ws-offline-fixture.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
OTHER_URL = "https://ws-other-fixture.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def response(*, tool=False, input_tokens=100, output_tokens=50, known=True):
    message = {
        "content": None if tool else "19 + 23 = 42",
        "reasoning_content": "fixture reasoning",
    }
    if tool:
        message["tool_calls"] = [
            {
                "id": "provider-add",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":19,"b":23}'},
            }
        ]
    value = {"choices": [{"finish_reason": "tool_calls" if tool else "stop", "message": message}]}
    if known:
        value["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
    return value


@pytest.fixture
def server():
    state = {
        "requests": [],
        "responses": [],
        "arrived": threading.Event(),
        "release": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append(raw)
            state["arrived"].set()
            payload = state["responses"].pop(0)
            if callable(payload):
                payload = payload()
            if payload == "wait":
                state["release"].wait(3)
                payload = response()
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(json.dumps(payload).encode())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{instance.server_port}/v1"
    try:
        yield state
    finally:
        state["release"].set()
        instance.shutdown()
        instance.server_close()
        thread.join()


async def run(tmp_path, server, *, budget="0.05", timeout=None):
    directory = tmp_path / "experiment"
    plan = smoke.prepare_plan(directory, budget, FIXTURE_URL)
    client = smoke.make_client(server["url"], api_key="fixture-not-a-real-key")
    if timeout is not None:
        client.timeout = timeout
    result = await smoke.run_smoke(directory, client, plan)
    return result, json.loads((directory / "summary.json").read_text(encoding="utf-8"))


async def test_full_native_roundtrip_uses_total_output_limit_and_correlated_budget(
    tmp_path, server
):
    reservations_seen = []

    def inspect_reservation(tool):
        with sqlite3.connect(
            (tmp_path / "experiment/smoke.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            rows = db.execute("SELECT status,body FROM smoke_calls ORDER BY rowid").fetchall()
            reservations_seen.append((rows[-1][0], json.loads(rows[-1][1])["reserved_cny"]))
        return response(tool=tool)

    server["responses"] = [lambda: inspect_reservation(True), lambda: inspect_reservation(False)]
    result, report = await run(tmp_path, server)
    assert result.status == "completed" and report["tool_roundtrip_observed"]
    assert report["dispatch_intents"] == len(server["requests"]) == 2
    assert report["unknown_usage_call_ids"] == report["unterminated_call_ids"] == []
    assert report["held_reservations_cny"] == "0" and report["cost_complete"]
    assert reservations_seen == [("dispatch_intent", str(smoke.RESERVATION))] * 2
    assert Decimal(report["known_estimated_cost_cny"]) == smoke.estimate_cost(200, 100)
    first, second = [json.loads(body) for body in server["requests"]]
    for payload in (first, second):
        assert payload["model"] == smoke.MODEL
        assert payload["max_completion_tokens"] == 2048 and "max_tokens" not in payload
        assert payload["enable_thinking"] is payload["preserve_thinking"] is True
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["add"]
    assistant = next(m for m in second["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "fixture reasoning"
    with sqlite3.connect(tmp_path / "experiment/smoke.sqlite3") as db:
        events = [
            json.loads(row[0])
            for row in db.execute("SELECT body FROM events WHERE kind='model_request'")
        ]
        assert [item["model_call_id"] for item in events] == [
            item["id"] for item in report["calls"]
        ]
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM memory_batches").fetchone()[0] == 0
    assert "fixture-not-a-real-key" not in (tmp_path / "experiment/summary.json").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("budget,expected", [("0.001", 0), ("0.01", 1)])
async def test_budget_refuses_before_dispatch_including_second_request(
    tmp_path, server, budget, expected
):
    server["responses"] = [response(tool=True, input_tokens=32000, output_tokens=2058), response()]
    result, report = await run(tmp_path, server, budget=budget)
    assert result.status == "failed" and report["stop_reason"] == "budget_insufficient"
    assert len(server["requests"]) == report["dispatch_intents"] == expected
    assert report["unknown_usage_call_ids"] == []


@pytest.mark.parametrize(
    "failure", ["unknown", "protocol", "length", "output_bound", "input_bound"]
)
async def test_unknown_usage_and_failures_stop_without_a_second_request(tmp_path, server, failure):
    value = response(tool=True, known=failure != "unknown")
    if failure == "protocol":
        value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {}
    if failure == "length":
        value["choices"][0]["finish_reason"] = "length"
    if failure == "output_bound":
        value["usage"]["completion_tokens"] = 2059
    if failure == "input_bound":
        value["usage"]["prompt_tokens"] = 500000
    server["responses"] = [value, response()]
    result, report = await run(tmp_path, server)
    assert result.status == "failed" and len(server["requests"]) == 1
    assert report["stop_reason"]
    if failure == "unknown":
        assert len(report["unknown_usage_call_ids"]) == 1 and not report["cost_complete"]
        assert Decimal(report["held_reservations_cny"]) == smoke.RESERVATION
    else:
        assert report["cost_complete"] and Decimal(report["known_estimated_cost_cny"]) > 0
    if failure == "input_bound":
        assert report[
            "known_estimate_exceeds_budget"
        ]  # actual out-of-bound usage is not hidden or capped


async def test_timeout_keeps_unknown_reservation_and_stops(tmp_path, server):
    server["responses"] = ["wait"]
    result, report = await run(tmp_path, server, timeout=0.03)
    assert result.status == "failed" and len(server["requests"]) == 1
    assert report["unknown_usage_call_ids"] and not report["cost_complete"]


async def test_cancellation_records_agent_terminal_and_unknown_cost(tmp_path, server):
    server["responses"] = ["wait"]
    directory = tmp_path / "experiment"
    plan = smoke.prepare_plan(directory, "0.05", FIXTURE_URL)
    task = asyncio.create_task(smoke.run_smoke(directory, smoke.make_client(server["url"]), plan))
    assert await asyncio.to_thread(server["arrived"].wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert len(server["requests"]) == 1 and report["unknown_usage_call_ids"]
    assert report["run_report"]["terminal"]["status"] == "cancelled"


async def test_call_limit_and_persisted_unterminated_gate(tmp_path, server):
    plan = smoke.plan_preview(base_url=FIXTURE_URL)
    store = Store(tmp_path / "limit.sqlite3")
    try:
        model = smoke.SmokeModel(
            smoke.make_client(server["url"]), store, plan, user="u", session="s"
        )
        run_id = store.begin("u", "s", smoke.PROMPT)
        request = PreparedRequest(
            json.dumps(plan["initial_request"], ensure_ascii=False).encode(), smoke.MAX_BYTES
        )
        server["responses"] = [response(), response()]
        for index in range(3):
            store.event(
                run_id,
                "model_request",
                {"model_call_id": str(index), "sha256": hashlib.sha256(request.body).hexdigest()},
            )
            if index < 2:
                await model.complete(request)
            else:
                with pytest.raises(ModelError):
                    await model.complete(request)
        assert (
            len(server["requests"]) == 2
            and smoke.budget_report(store.db)["stop_reason"] == "request_limit"
        )
    finally:
        store.close()
    interrupted = Store(tmp_path / "interrupted.sqlite3")
    try:
        model = smoke.SmokeModel(
            smoke.make_client(server["url"]), interrupted, plan, user="u", session="s"
        )
        detail = {
            "dispatch_intent": True,
            "usage_known": False,
            "estimated_cost_cny": None,
            "reserved_cny": str(smoke.RESERVATION),
        }
        with interrupted.transaction() as db:
            db.execute(
                "INSERT INTO smoke_calls VALUES ('interrupted','dispatch_intent',?)",
                (json.dumps(detail),),
            )
        report = smoke.budget_report(interrupted.db)
        assert (
            report["unterminated_call_ids"] == report["unknown_usage_call_ids"] == ["interrupted"]
        )
        with pytest.raises(ModelError):
            await model.complete(request)
        assert len(server["requests"]) == 2
    finally:
        interrupted.close()


def test_plan_mode_never_loads_credentials_and_freeze_drift_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def forbidden(*args):
        raise AssertionError("Credentials or network must not be used")

    monkeypatch.setattr(smoke, "_load_key", forbidden)
    directory = tmp_path / "plan"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(directory), "--base-url", FIXTURE_URL])
    smoke.main()
    assert (directory / "plan.json").exists() and not (directory / "smoke.sqlite3").exists()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    plan["price_tiers_cny_per_million"][0][1] = "0"
    (directory / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(directory), "--execute", "--base-url", FIXTURE_URL])
    with pytest.raises(ValueError, match="differs"):
        smoke.main()


async def test_execution_cannot_reuse_an_experiment_or_accept_wrong_region(tmp_path, server):
    server["responses"] = [response()]
    await run(tmp_path, server)
    with pytest.raises(FileExistsError):
        await smoke.run_smoke(
            tmp_path / "experiment", smoke.make_client(server["url"]), smoke.plan_preview(base_url=FIXTURE_URL)
        )
    with pytest.raises(ValueError, match="Beijing"):
        smoke.plan_preview(base_url="https://ws-fixture.cn-shanghai.maas.aliyuncs.com/compatible-mode/v1")
    store = Store(tmp_path / "wrong-region.sqlite3")
    try:
        with pytest.raises(ValueError, match="Beijing"):
            smoke.SmokeModel(
                smoke.make_client(OTHER_URL),
                store,
                smoke.plan_preview(base_url=FIXTURE_URL),
                user="u",
                session="s",
            )
    finally:
        store.close()


async def test_oversize_actual_input_is_refused_before_reservation(tmp_path, server):
    store = Store(tmp_path / "bound.sqlite3")
    try:
        model = smoke.SmokeModel(
            smoke.make_client(server["url"]), store, smoke.plan_preview(base_url=FIXTURE_URL), user="u", session="s"
        )
        request = PreparedRequest(
            json.dumps({"messages": [{"role": "user", "content": "x" * 16001}]}).encode(), 20000
        )
        with pytest.raises(ModelError):
            await model.complete(request)
        assert server["requests"] == [] and smoke.budget_report(store.db)["calls"] == []
    finally:
        store.close()


def test_cli_execution_branch_uses_only_injected_local_http_and_no_key_file(
    tmp_path, server, monkeypatch, capsys
):
    directory = tmp_path / "cli"
    base_url = "https://ws-offline-fixture.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    config = tmp_path / "config.txt"
    config.write_text(f'AGENT_BASE_URL="{base_url}"\n', encoding="utf-8")
    smoke.prepare_plan(directory, "0.05", base_url)
    server["responses"] = [response(tool=True), response()]
    original = smoke.make_client
    configured_urls = []

    def fixture_client(base_url=None, api_key=""):
        if api_key:
            configured_urls.append(base_url)
        return original(server["url"], api_key)

    monkeypatch.setattr(
        smoke,
        "make_client",
        fixture_client,
    )
    # Fixture text, not a credential file: no .env or real key is read by this test.
    monkeypatch.setattr(smoke, "_load_key", lambda path: "offline-fixture-auth")
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", "--output", str(directory), "--execute", "--env-file", str(config)],
    )
    smoke.main()
    assert len(server["requests"]) == 2
    assert configured_urls == [base_url]
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert report["mode"] == "offline_http_fixture" and report["tool_roundtrip_observed"]
    assert "offline-fixture-auth" not in capsys.readouterr().out

    def forbidden(path):
        raise AssertionError("Completed experiment must be rejected before authentication")

    monkeypatch.setattr(smoke, "_load_key", forbidden)
    with pytest.raises(ValueError, match="already started"):
        smoke.main()


def test_cli_reads_only_endpoint_setting_and_refuses_config_drift(tmp_path, monkeypatch, capsys):
    import dotenv

    config = tmp_path / "config.txt"
    base_url = "https://ws-offline-fixture.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    config.write_text(
        f'export AGENT_BASE_URL="{base_url}"\n'
        "AGENT_MODEL=qwen3.7-flash\nDASHSCOPE_API_KEY=not-a-real-key\n",
        encoding="utf-8",
    )
    original = dotenv.dotenv_values

    def selected_config_only(*, stream, interpolate):
        assert stream.getvalue() == f'export AGENT_BASE_URL="{base_url}"\n'
        assert interpolate is False
        return original(stream=stream, interpolate=interpolate)

    monkeypatch.setattr(dotenv, "dotenv_values", selected_config_only)

    def forbidden(*args):
        raise AssertionError("Configuration must be checked before authentication or dispatch")

    monkeypatch.setattr(smoke, "_load_key", forbidden)
    monkeypatch.setattr(smoke, "run_smoke", forbidden)
    directory = tmp_path / "plan"
    argv = ["smoke", "--output", str(directory), "--env-file", str(config)]
    monkeypatch.setattr(sys, "argv", argv)
    smoke.main()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    assert plan["base_url"] == base_url
    assert plan["model"] == smoke.MODEL == "qwen3.7-flash"
    assert not (directory / "smoke.sqlite3").exists()
    assert "not-a-real-key" not in capsys.readouterr().out
    # A different endpoint cannot execute an existing frozen plan, even within Beijing.
    monkeypatch.setattr(sys, "argv", argv + ["--execute", "--base-url", OTHER_URL])
    with pytest.raises(ValueError, match="differs"):
        smoke.main()


def test_endpoint_explicit_override_and_missing_config_do_not_read_file(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit endpoint must not read the configuration file")

    monkeypatch.setattr(Path, "open", forbidden)
    assert smoke._resolve_base_url(OTHER_URL, tmp_path / ".env") == OTHER_URL
    with pytest.raises(ValueError, match="missing"):
        smoke._resolve_base_url(None, tmp_path / "missing-config.txt")


def test_check_config_reads_selected_file_without_auth_network_or_output_dir(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.txt"
    config.write_text(f'AGENT_BASE_URL="{OTHER_URL}"\nDASHSCOPE_API_KEY=not-a-real-key\n', encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Config inspection cannot load authentication or construct a client")

    monkeypatch.setattr(smoke, "_load_key", forbidden)
    monkeypatch.setattr(smoke, "make_client", forbidden)
    monkeypatch.setattr(sys, "argv", ["smoke", "--check-config", "--env-file", str(config)])
    smoke.main()
    raw = capsys.readouterr().out
    report = json.loads(raw)
    assert report["base_url"] == OTHER_URL and report["source"] == "AGENT_BASE_URL"
    assert report["credentials_loaded"] is False and report["requests_sent"] == 0
    assert "not-a-real-key" not in raw


@pytest.mark.parametrize("contents", ["", "AGENT_BASE_URL=\n", "AGENT_BASE_URL\n", "AGENT_BASE_URL='  '\n"])
def test_endpoint_missing_or_empty_has_no_fallback(tmp_path, monkeypatch, contents):
    config = tmp_path / "config.txt"
    config.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("AGENT_BASE_URL", OTHER_URL)
    with pytest.raises(ValueError, match="missing or empty"):
        smoke._resolve_base_url(None, config)


def test_endpoint_resolution_disables_interpolation_and_ignores_other_fields(tmp_path, monkeypatch):
    config = tmp_path / "config.txt"
    config.write_text('DASHSCOPE_API_KEY=not-a-real-key\nexport AGENT_BASE_URL="${HOST}/v1"\n', encoding="utf-8")
    monkeypatch.setenv("HOST", OTHER_URL)
    assert smoke._resolve_base_url(None, config) == "${HOST}/v1"


async def test_reservation_storage_failure_prevents_http_dispatch(tmp_path, server):
    from agentcore.core.types import StorageError

    store = Store(tmp_path / "disk.sqlite3")
    try:
        plan = smoke.plan_preview(base_url=FIXTURE_URL)
        model = smoke.SmokeModel(
            smoke.make_client(server["url"]), store, plan, user="u", session="s"
        )
        run_id = store.begin("u", "s", smoke.PROMPT)
        request = PreparedRequest(
            json.dumps(plan["initial_request"], ensure_ascii=False).encode(), smoke.MAX_BYTES
        )
        store.event(
            run_id,
            "model_request",
            {"model_call_id": "c", "sha256": hashlib.sha256(request.body).hexdigest()},
        )

        def authorize(action, table, *args):
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_INSERT and table == "smoke_calls"
                else sqlite3.SQLITE_OK
            )

        store.db.set_authorizer(authorize)
        with pytest.raises(StorageError):
            await model.complete(request)
        store.db.set_authorizer(None)
        assert server["requests"] == []
    finally:
        store.db.set_authorizer(None)
        store.close()
