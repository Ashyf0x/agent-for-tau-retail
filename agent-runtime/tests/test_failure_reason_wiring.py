"""The failure_reason_chars option must be reachable from the real run entry.

Why this file exists
--------------------
Unit tests can pass while a feature is unreachable in production. The project rule
is explicit: if the entry point does not implement a configuration read, report it
rather than assume it took effect. This test pins the whole path so a later edit
that drops one hop fails loudly instead of silently disabling the option.

Path under test:
    --failure-reason-chars  (argparse)
        -> build_config(failure_reason_chars=...)
        -> config["context"]["failure_reason_chars"]
        -> ContextManager(**config["context"]) really projecting the excerpt

These build a real config, so they need the pinned checkout and skip without it,
matching the existing pilot-config tests.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from agentcore.context import ContextManager
from agentcore.core.model import HTTPModelClient
from agentcore.core.types import dumps

ROOT = Path(__file__).parents[1] / ".vendor/tau2-bench"


def client():
    return HTTPModelClient("https://fixture.invalid/v1", "explicit-fixture", provider="qwen",
                           thinking=True, preserve_thinking=True, max_completion_tokens=4096)


def record(source, role, content="", **fields):
    return {"id": source, "message": {"role": role, "content": content, **fields}}


def failed_write_records():
    records = [record("goal", "user", "Exchange both kettles from #W4316152.")]
    records += [record("write", "assistant", tool_calls=[{
        "id": "write", "type": "function",
        "function": {"name": "exchange_delivered_order_items",
                     "arguments": dumps({"order_id": "#W4316152"})}}]),
        record("write-result", "tool", dumps({
            "status": "returned", "ok": False,
            "value": "Error: Insufficient gift card balance to pay for the price difference",
            "error": "Retail environment rejected the operation",
            "category": "business", "decision": "feedback",
            "observation_ref": "ref-write"}), tool_call_id="write")]
    records += [record("now", "user", "Why did that fail?")]
    return records


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "settings.txt"
    path.write_text(
        "AGENT_MODEL=qwen3.7-flash\n"
        "AGENT_BASE_URL=https://configured.invalid/v1\n"
        "RETAIL_USER_MODEL=qwen3.7-flash\n"
        "RETAIL_USER_BASE_URL=https://configured.invalid/v1\n"
        "RETAIL_COMMUNICATE_MODEL=qwen3.7-flash\n"
        "RETAIL_COMMUNICATE_BASE_URL=https://configured.invalid/v1\n"
        "RETAIL_NL_MODEL=qwen3.7-flash\n"
        "RETAIL_NL_BASE_URL=https://configured.invalid/v1\n",
        encoding="utf-8")
    return path


def test_build_config_exposes_the_option(settings):
    from evals.retail_pilot import build_config

    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    config = build_config(ROOT, ["11"], env_file=settings, failure_reason_chars=240)
    assert config["context"]["failure_reason_chars"] == 240


def test_build_config_omits_the_key_when_off(settings):
    """Off must leave the context block byte-identical to the frozen snapshot."""
    from evals.retail_pilot import build_config

    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    config = build_config(ROOT, ["11"], env_file=settings)
    assert config["context"] == {"policy": "capacity", "compaction": "legacy",
                                 "byte_margin": 2048, "handoff_max_bytes": 16384}


def test_build_config_rejects_an_out_of_range_option(settings):
    from evals.retail_pilot import build_config

    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    with pytest.raises(ValueError, match="Failure reason excerpt"):
        build_config(ROOT, ["11"], env_file=settings, failure_reason_chars=1000)
    with pytest.raises(ValueError, match="Failure reason excerpt"):
        build_config(ROOT, ["11"], env_file=settings, failure_reason_chars=-1)


def test_config_context_is_directly_constructible(settings):
    """The dict written into config must build a working manager, not just look right."""
    from evals.retail_pilot import build_config

    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    config = build_config(ROOT, ["11"], env_file=settings, failure_reason_chars=240)
    manager = ContextManager("Policy", **config["context"])
    request, _ = manager.prepare(client(), failed_write_records(), [], [], {}, 131072)

    body = json.loads([m for m in json.loads(request.body)["messages"]
                       if m["role"] == "tool"][-1]["content"])
    assert "Insufficient gift card balance" in body["failure_reason"]["excerpt"]


def test_validate_config_accepts_the_option(settings):
    """A frozen-config check that rejects the new key would block every run with it."""
    from evals.retail_pilot import build_config, validate_config

    if not ROOT.exists():
        pytest.skip("Pinned checkout not installed")
    config = build_config(ROOT, ["11"], env_file=settings, failure_reason_chars=240)
    validate_config(ROOT, config)  # must not raise


def test_cli_carries_the_flag():
    """Argparse must actually define and forward the flag; otherwise it is a no-op."""
    import evals.retail_pilot as pilot

    source = inspect.getsource(pilot.main)
    assert "--failure-reason-chars" in source
    assert "args.failure_reason_chars" in source
