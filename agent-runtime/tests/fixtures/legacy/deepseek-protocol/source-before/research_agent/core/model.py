import json
import math
from copy import deepcopy
from urllib.parse import urlsplit

import httpx

from .types import ModelError, ModelResponse, PreparedRequest, ToolCall, Usage, dumps

_MISSING = object()
_KNOWN_FINISH_REASONS = {"stop", "tool_calls", "length", "content_filter", "function_call", "error"}


def chat_request_bytes(model, messages, tools, maximum, temperature, *,
                       provider="compatible", thinking=None, preserve_thinking=False,
                       max_completion_tokens=None):
    """One serializer shared by actual HTTP and public budget/preflight guards."""
    messages = deepcopy(messages)
    replay_reasoning = (provider == "qwen" and preserve_thinking) or (provider == "deepseek" and thinking)
    for message in messages:
        metadata = message.pop("_provider_metadata", {})
        # Never accept wire fields independently of the retained adapter metadata.
        message.pop("reasoning_content", None)
        if message.get("role") != "assistant" or not replay_reasoning:
            continue
        reasoning = metadata.get("reasoning_content")
        if provider == "deepseek" and tools and not isinstance(reasoning, str):
            raise ModelError("Thinking-tool history lacks required reasoning metadata; use a new session",
                             kind="capability")
        if reasoning is not None:
            message["reasoning_content"] = reasoning
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    payload["max_tokens" if max_completion_tokens is None else "max_completion_tokens"] = (
        maximum if max_completion_tokens is None else max_completion_tokens)
    if tools:
        payload["tools"] = tools
    if provider == "qwen":
        payload.update(enable_thinking=thinking, preserve_thinking=preserve_thinking, stream=False)
    elif provider == "deepseek":
        payload.update(thinking={"type": "enabled" if thinking else "disabled"}, stream=False)
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _actual_type(value) -> str:
    return "missing" if value is _MISSING else type(value).__name__


def _extract_usage(data) -> tuple[Usage | None, tuple | None]:
    """先独立核对用量，后面解析工具失败时仍能保留已发生的成本。"""
    if not isinstance(data, dict) or data.get("usage") is None:
        return None, None
    raw = data["usage"]
    if not isinstance(raw, dict):
        return None, ("usage", raw, "dict", "invalid_type")
    counts = []
    for key in ("prompt_tokens", "completion_tokens"):
        value = raw.get(key, _MISSING)
        if type(value) is not int:
            return None, (
                f"usage.{key}",
                value,
                "nonnegative int",
                "missing_field" if value is _MISSING else "invalid_type",
            )
        if value < 0:
            return None, (f"usage.{key}", value, "nonnegative int", "negative_value")
        counts.append(value)
    return Usage(*counts), None


class HTTPModelClient:
    """一个非流式 Chat Completions 适配器，只负责协议转换。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        max_output_tokens: int = 1024,
        temperature: float = 0,
        timeout: float = 60,
        *,
        supports_tools: bool = True,
        transport=None,
        provider: str = "compatible",
        thinking: bool | None = None,
        preserve_thinking: bool = False,
        max_completion_tokens: int | None = None,
    ):
        if max_completion_tokens is not None and (type(max_completion_tokens) is not int or max_completion_tokens < 1):
            raise ValueError("max_completion_tokens must be a positive integer")
        if provider not in {"compatible", "qwen", "deepseek"}:
            raise ValueError("Unsupported provider protocol")
        if type(preserve_thinking) is not bool:
            raise ValueError("preserve_thinking must be boolean")
        if provider == "compatible":
            if thinking is not None or preserve_thinking:
                raise ValueError("Select an explicit provider before configuring thinking")
        elif type(thinking) is not bool:
            raise ValueError("An explicit thinking boolean is required for this provider")
        if preserve_thinking and provider != "qwen":
            raise ValueError("preserve_thinking is a Qwen parameter; DeepSeek replay follows its contract")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url 必须是不含凭据、查询参数或片段的 HTTP(S) 地址")
        if (
            not model
            or max_output_tokens < 1
            or not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(temperature)
        ):
            raise ValueError("模型名称、输出限制或请求参数无效")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.supports_tools = supports_tools
        self.transport = transport
        self.provider, self.thinking, self.preserve_thinking = provider, thinking, preserve_thinking

    def prepare(self, messages, tools, max_bytes):
        if tools and not self.supports_tools:
            raise ModelError("Configured provider does not support native tools", kind="capability")
        return PreparedRequest(self.request_bytes(messages, tools), max_bytes)

    def request_bytes(self, messages, tools=None):
        return chat_request_bytes(
            self.model,
            messages,
            tools,
            self.max_output_tokens,
            self.temperature,
            provider=self.provider,
            thinking=self.thinking,
            preserve_thinking=self.preserve_thinking,
            max_completion_tokens=self.max_completion_tokens,
        )

    async def complete(self, request: PreparedRequest) -> ModelResponse:
        request.validate()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            # No SDK/transport retries. The runtime alone owns bounded protocol correction.
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST", self.url, content=request.body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        raise ModelError(
                            "Model HTTP request rejected",
                            kind="capability"
                            if response.status_code in {400, 404, 422}
                            else "http",
                            diagnostics={"http_status": response.status_code},
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 16 * 1024 * 1024:
                            raise ModelError("Response too large", kind="protocol_error")
            data = json.loads(body)
        except httpx.HTTPError:
            raise ModelError("模型网络请求失败或超时，请检查地址与连接") from None
        except (ValueError, UnicodeError):
            raise ModelError(
                "模型服务没有返回有效 JSON",
                kind="protocol_error",
                diagnostics={"field": "$", "actual_type": "bytes", "issue": "invalid_json"},
            ) from None
        parsed = self._parse(data)
        if self.provider == "deepseek" and self.thinking and json.loads(request.body).get("tools"):
            if not isinstance(parsed.provider_metadata.get("reasoning_content"), str):
                raise ModelError("Thinking-tool response lacks required reasoning metadata",
                                 kind="protocol_error", diagnostics={"field": "choices[0].message.reasoning_content",
                                 "issue": "missing_reasoning"}, usage=parsed.usage if parsed.usage_known else None)
        return parsed

    @staticmethod
    def _parse(data) -> ModelResponse:
        usage, usage_problem = _extract_usage(data)
        # 只记录已知的结束状态枚举，避免任意服务端文本进入诊断。
        reason_hint = None
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict) and isinstance(first.get("finish_reason"), str):
                reason_hint = (
                    first["finish_reason"]
                    if first["finish_reason"] in _KNOWN_FINISH_REASONS
                    else "unrecognized"
                )

        def fail(message, field, actual, expected, issue="invalid_type", tool_index=None):
            details = {
                "field": field,
                "actual_type": _actual_type(actual),
                "expected_type": expected,
                "issue": issue,
            }
            if tool_index is not None:
                details["tool_index"] = tool_index
            if reason_hint is not None:
                details["finish_reason"] = reason_hint
            raise ModelError(message, kind="protocol_error", diagnostics=details, usage=usage)

        if usage_problem is not None:
            field, actual, expected, issue = usage_problem
            fail("模型用量字段无效", field, actual, expected, issue)
        if not isinstance(data, dict):
            fail("模型响应必须是对象", "$", data, "dict")
        choices = data.get("choices", _MISSING)
        if not isinstance(choices, list):
            fail("模型响应缺少有效 choices", "choices", choices, "nonempty list")
        if not choices:
            fail("模型响应 choices 不能为空", "choices", choices, "nonempty list", "empty_value")
        choice = data["choices"][0]
        if not isinstance(choice, dict):
            fail("模型 choice 必须是对象", "choices[0]", choice, "dict")
        if not isinstance(choice.get("message"), dict):
            fail(
                "模型响应缺少有效 message",
                "choices[0].message",
                choice.get("message", _MISSING),
                "dict",
            )
        message = choice["message"]
        content, reason = message.get("content"), choice.get("finish_reason")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            fail("Invalid reasoning metadata", "choices[0].message.reasoning_content", reasoning, "str or None")
        if content is not None and not isinstance(content, str):
            fail("本适配器仅支持文本 content", "choices[0].message.content", content, "str or None")
        if not isinstance(reason, str) or not reason:
            fail(
                "模型响应缺少有效 finish_reason",
                "choices[0].finish_reason",
                choice.get("finish_reason", _MISSING),
                "nonempty str",
            )
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            fail("tool_calls 必须是列表", "choices[0].message.tool_calls", calls, "list")
        tool_calls, seen = [], set()
        for index, item in enumerate(calls):
            path = f"choices[0].message.tool_calls[{index}]"
            if not isinstance(item, dict):
                fail("工具调用必须是对象", path, item, "dict", tool_index=index)
            if item.get("type") != "function":
                fail(
                    "工具调用类型必须是 function",
                    f"{path}.type",
                    item.get("type", _MISSING),
                    "function",
                    "unsupported_value",
                    index,
                )
            if not isinstance(item.get("function"), dict):
                fail(
                    "工具调用 function 必须是对象",
                    f"{path}.function",
                    item.get("function", _MISSING),
                    "dict",
                    tool_index=index,
                )
            call_id, function = item.get("id"), item["function"]
            name, arguments = function.get("name"), function.get("arguments")
            if not isinstance(call_id, str) or not call_id.strip():
                fail(
                    "工具调用 id 缺失或为空",
                    f"{path}.id",
                    item.get("id", _MISSING),
                    "nonempty str",
                    "invalid_type" if not isinstance(call_id, str) else "empty_value",
                    index,
                )
            if call_id in seen:
                fail(
                    "工具调用 id 重复",
                    f"{path}.id",
                    call_id,
                    "unique str",
                    "duplicate_value",
                    index,
                )
            if not isinstance(name, str):
                fail(
                    "工具调用名称必须是字符串",
                    f"{path}.function.name",
                    function.get("name", _MISSING),
                    "nonempty str",
                    tool_index=index,
                )
            if not name.strip():
                fail(
                    "工具调用名称不能为空",
                    f"{path}.function.name",
                    name,
                    "nonempty str",
                    "empty_value",
                    index,
                )
            if not isinstance(arguments, str):
                fail(
                    "工具调用参数必须是 JSON 字符串",
                    f"{path}.function.arguments",
                    function.get("arguments", _MISSING),
                    "str",
                    tool_index=index,
                )
            seen.add(call_id)
            # 参数是否是合法 JSON 留给工具层判断，模型随后可以修正。
            tool_calls.append(ToolCall(call_id, name, arguments))
        refusal = message.get("refusal")
        if refusal is not None and not isinstance(refusal, str):
            fail(
                "refusal 必须是字符串或 null", "choices[0].message.refusal", refusal, "str or None"
            )
        if message.get("refusal"):
            reason = "content_filter"
        return ModelResponse(
            content, tool_calls, reason, usage or Usage(), usage_known=usage is not None,
            provider_metadata={"reasoning_content": reasoning} if "reasoning_content" in message else {},
        )


class ScriptedModel:
    """Deterministic fixture model; no network, credentials, or capability claim."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def prepare(self, messages, tools, max_bytes):
        return PreparedRequest(dumps({"messages": messages, "tools": tools}).encode(), max_bytes)

    async def complete(self, request):
        request.validate()
        payload = json.loads(request.body)
        self.requests.append(payload)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(payload)
        return deepcopy(response)
