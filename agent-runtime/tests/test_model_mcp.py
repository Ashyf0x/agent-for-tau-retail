import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentcore import Agent, ModelResponse, ToolCall
from agentcore.core.model import HTTPModelClient, ScriptedModel
from agentcore.core.types import ContextBudgetError, ModelError, Usage
from agentcore.tools import ToolRegistry, ToolSpec
from agentcore.tools.mcp import connect_stdio


def test_native_parser_retains_usage_and_safe_diagnostics():
    data = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": {"private": "secret"}},
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4},
    }
    with pytest.raises(ModelError) as error:
        HTTPModelClient._parse(data)
    assert error.value.usage == Usage(9, 4)
    assert "secret" not in str(error.value.diagnostics)


async def test_http_mock_server_gets_exact_prepared_body_and_native_tools():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            response = {
                "choices": [{"finish_reason": "stop", "message": {"content": "Plain answer"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HTTPModelClient(f"http://127.0.0.1:{server.server_port}/v1", "fixture")
        definition = ToolSpec("f", "Function", {"type": "object"}, lambda: None).definition()
        request = client.prepare([{"role": "user", "content": "Hello"}], [definition], 4000)
        result = await client.complete(request)
        assert result.content == "Plain answer" and received == [request.body]
        assert json.loads(received[0])["tools"] == [definition]
        too_small = client.prepare([], [], 1)
        with pytest.raises(ContextBudgetError):
            await client.complete(too_small)
        assert len(received) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unsupported_native_tools_fail_explicitly():
    client = HTTPModelClient("http://127.0.0.1:1/v1", "fixture", supports_tools=False)
    with pytest.raises(ModelError, match="does not support"):
        client.prepare([], [{"type": "function"}], 4000)


async def test_real_stdio_mcp_and_local_tools_share_execution_contract(store):
    server = Path(__file__).parents[1] / "examples" / "mcp_server.py"
    async with connect_stdio(
        sys.executable, [str(server)], contracts={"add": {"read_only": True}}, timeout=5
    ) as registry:
        model = ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall("x", "add", {"a": 3, "b": 4})], finish_reason="tool_calls"
                ),
                ModelResponse("7"),
            ]
        )
        result = await Agent(model, registry, store).run("Add 3 and 4")
        assert result.status == "completed"
        remote = json.loads(store.messages("default", "default")[2]["message"]["content"])
        assert remote["status"] == "returned" and remote["ok"]
        assert remote["value"]["structuredContent"]["result"] == 7
        bad = await registry.execute(ToolCall("b", "add", {"a": "wrong", "b": 4}), lambda *a: None)
        assert bad.status == "not_dispatched"
        failure = await registry.execute(ToolCall("e", "fail", {}), lambda *a: None)
        assert failure.status == "returned" and not failure.ok
        assert "fixture-private-error" not in failure.error
        local = ToolRegistry(
            [
                ToolSpec(
                    "add",
                    "Add",
                    registry.specs["add"].parameters,
                    lambda a, b: a + b,
                    read_only=True,
                )
            ]
        )
        known = await local.execute(ToolCall("l", "add", {"a": 3, "b": 4}), lambda *a: None)
        assert known.value == remote["value"]["structuredContent"]["result"]
        assert known.status == remote["status"]
