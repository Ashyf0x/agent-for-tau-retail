"""Offline contracts from the official Qwen/DeepSeek Chat Completions documentation."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentcore import Agent
from agentcore.core.model import HTTPModelClient
from agentcore.core.types import ContextBudgetError, ModelError, Usage, dumps
from agentcore.tools import ToolRegistry, ToolSpec


def tool_response(reasoning="fixture-reasoning"):
    return {
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "reasoning_content": reasoning,
            "tool_calls": [{"id": "provider-call", "type": "function",
                            "function": {"name": "read", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }


def test_reasoning_is_received_without_becoming_the_answer():
    response = HTTPModelClient._parse(tool_response())
    assert response.provider_metadata == {"reasoning_content": "fixture-reasoning"}
    assert response.content is None and response.tool_calls[0].name == "read"


@pytest.fixture
def chat_server():
    state = {"received": [], "responses": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            state["received"].append(raw)
            response = state["responses"].pop(0)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def readonly_tools():
    async def read():
        return {"evidence": "actual local handler result"}
    return ToolRegistry([ToolSpec("read", "Read evidence", {"type": "object", "properties": {}}, read, read_only=True)])


@pytest.mark.parametrize("provider,preserve", [("qwen", False), ("qwen", True), ("deepseek", False),
                                             ("bailian_deepseek", False)])
async def test_thinking_tool_roundtrip_and_followup(store, chat_server, provider, preserve):
    final = {"choices": [{"finish_reason": "stop", "message": {
        "content": "Evidence received", "reasoning_content": "fixture-final-reasoning",
    }}], "usage": {"prompt_tokens": 20, "completion_tokens": 9}}
    chat_server["responses"] = [tool_response(), final, final]
    model = "deepseek-v4.1-flash-1017" if provider == "bailian_deepseek" else "fixture-model"
    client = HTTPModelClient(chat_server["url"], model, provider=provider,
                             thinking=True, preserve_thinking=preserve)
    agent = Agent(client, readonly_tools(), store)
    first = await agent.run("Inspect the evidence")
    second = await agent.run("Explain it again")
    assert first.status == second.status == "completed"
    assert first.answer == "Evidence received"
    records = store.messages("default", "default")
    assert records[1]["message"]["_provider_metadata"]["reasoning_content"] == "fixture-reasoning"
    assert records[3]["message"]["_provider_metadata"]["reasoning_content"] == "fixture-final-reasoning"
    requests = [json.loads(body) for body in chat_server["received"]]
    replay = provider in {"deepseek", "bailian_deepseek"} or preserve
    for request in requests:
        assert request["stream"] is False
        assert "tool_choice" not in request  # no forced tool call under Qwen thinking
        assert "_provider_metadata" not in json.dumps(request)
        if provider == "qwen":
            assert request["enable_thinking"] is True
            assert request["preserve_thinking"] is preserve
            assert "thinking" not in request
        elif provider == "bailian_deepseek":
            assert request["enable_thinking"] is True
            assert "thinking" not in request and "preserve_thinking" not in request
        else:
            assert request["thinking"] == {"type": "enabled"}
            assert "enable_thinking" not in request and "preserve_thinking" not in request
    assistants = [m for m in requests[-1]["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 2
    if replay:
        assert [m["reasoning_content"] for m in assistants] == ["fixture-reasoning", "fixture-final-reasoning"]
    else:
        assert all("reasoning_content" not in m for m in assistants)
    events = store.events(first.run_id) + store.events(second.run_id)
    prepared = [e for e in events if e["kind"] == "model_request"]
    assert [e["wire_bytes"] for e in prepared] == [len(b) for b in chat_server["received"]]


async def test_reasoning_counts_toward_context_limit_and_is_not_truncated(store, chat_server):
    from agentcore import AgentConfig
    chat_server["responses"] = [tool_response("z" * 10000)]
    client = HTTPModelClient(chat_server["url"], "fixture-model", provider="qwen",
                             thinking=True, preserve_thinking=True)
    result = await Agent(client, readonly_tools(), store, config=AgentConfig(max_request_bytes=6000)).run("Inspect")
    assert result.reason == "ContextBudgetError"
    assert len(chat_server["received"]) == 1  # second request refused locally
    assistant = store.messages("default", "default")[1]["message"]
    assert assistant["_provider_metadata"]["reasoning_content"] == "z" * 10000


@pytest.mark.parametrize("reasoning", ["", None])
def test_empty_or_null_reasoning_is_retained_as_metadata(reasoning):
    response = HTTPModelClient._parse(tool_response(reasoning))
    assert response.provider_metadata == {"reasoning_content": reasoning}
    assert response.content is None


def test_invalid_reasoning_keeps_known_usage_without_echoing_value():
    with pytest.raises(ModelError) as caught:
        HTTPModelClient._parse(tool_response({"private-fixture": "do-not-echo"}))
    assert caught.value.usage == Usage(10, 8)
    assert caught.value.diagnostics["field"] == "choices[0].message.reasoning_content"
    assert "do-not-echo" not in str(caught.value.diagnostics)


@pytest.mark.parametrize("provider,expected", [("qwen", {"enable_thinking": False}), ("deepseek", {"thinking": {"type": "disabled"}})])
def test_thinking_off_is_an_explicit_wire_setting(provider, expected):
    client = HTTPModelClient("http://127.0.0.1:1/v1", "fixture", provider=provider, thinking=False)
    body = json.loads(client.prepare([{"role": "user", "content": "Hi"}], [], 5000).body)
    for key, value in expected.items():
        assert body[key] == value


def test_context_drops_whole_old_turn_but_preserves_retained_reasoning():
    from agentcore.context import ContextManager
    records = []
    for index in range(4):
        records.extend([
            {"id": f"user-{index}", "message": {"role": "user", "content": f"request-{index}"}},
            {"id": f"assistant-{index}", "message": {"role": "assistant", "content": "answer",
             "_provider_metadata": {"reasoning_content": f"完整推理-{index}"}}},
        ])
    original = dumps(records)
    client = HTTPModelClient("http://127.0.0.1:1/v1", "fixture", provider="qwen", thinking=True, preserve_thinking=True)
    request, info = ContextManager("Rules").prepare(client, records, [], [], {}, 20000)
    messages = json.loads(request.body)["messages"]
    assert [m["reasoning_content"] for m in messages if m["role"] == "assistant"] == ["完整推理-1", "完整推理-2", "完整推理-3"]
    assert info["omitted_messages"] == 2 and dumps(records) == original
    assert "完整推理" not in messages[0]["content"]  # protocol data never becomes a summary/fact


async def test_complete_refuses_one_byte_over_boundary_including_reasoning(chat_server):
    client = HTTPModelClient(chat_server["url"], "fixture", provider="qwen", thinking=True, preserve_thinking=True)
    messages = [{"role": "assistant", "content": "answer", "_provider_metadata": {"reasoning_content": "原样保留"}}]
    measured = client.prepare(messages, [], 10000)
    exact = client.prepare(messages, [], len(measured.body))
    exact.validate()
    assert exact.body == measured.body
    with pytest.raises(ContextBudgetError):
        await client.complete(client.prepare(messages, [], len(measured.body) - 1))
    assert chat_server["received"] == []


def test_deepseek_missing_history_reasoning_is_refused_without_fabrication():
    client = HTTPModelClient("http://127.0.0.1:1/v1", "fixture", provider="deepseek", thinking=True)
    with pytest.raises(ModelError, match="lacks required"):
        client.prepare([{"role": "assistant", "content": "old answer"}], readonly_tools().definitions(), 5000)

