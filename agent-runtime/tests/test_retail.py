import json
from pathlib import Path

import pytest

from agentcore import ModelResponse, ToolCall
from agentcore.core.model import ScriptedModel

ROOT = Path(__file__).parents[1] / ".vendor/tau2-bench"
pytestmark = pytest.mark.retail


@pytest.fixture(scope="module")
def retail():
    if not ROOT.exists():
        pytest.skip("Pinned Retail checkout not installed; see evals/README.md")
    from evals.retail import load_retail

    return load_retail(ROOT)


async def test_official_environment_runs_our_core_without_answer_leakage(store, retail):
    from evals.retail import build_environment, make_agent, scoring_status
    from evals.retail_fixture import USER_EMAIL, fixture_model

    lock, loaded, tasks = retail
    assert lock["release"] == "v2.0.0" and len(tasks) == 50
    environment = build_environment(loaded)
    before = environment.get_db_hash()
    model = fixture_model()
    result = await make_agent(model, store, environment, loaded).run(
        f"My email is {USER_EMAIL}. Find my account."
    )
    assert result.status == "completed" and USER_EMAIL in result.answer
    assert len(model.requests) == 3
    assert environment.get_db_hash() == before
    assert len(environment.get_tools()) == 16
    inputs = json.dumps(model.requests)
    for hidden in (
        "evaluation_criteria",
        "user_scenario",
        "reward_basis",
        "communicate_info",
        "nl_assertions",
    ):
        assert hidden not in inputs
    assert scoring_status()["official_score"] is None


async def test_official_business_error_is_returned_not_internal_failure(store, retail):
    from evals.retail import build_environment, make_agent

    _, loaded, _ = retail
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("fixture", "get_user_details", {"user_id": "nonexistent-fixture-user"})
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse("I could not find that account. Please check your details."),
        ]
    )
    result = await make_agent(model, store, build_environment(loaded), loaded).run(
        "Look up nonexistent-fixture-user"
    )
    body = json.loads(store.messages("default", "default")[2]["message"]["content"])
    assert result.status == "completed"
    assert body["status"] == "returned" and not body["ok"]
    assert "User not found" in body["value"]  # exact official error is retained locally
    assert "User not found" not in str(model.requests[-1])  # compact safe feedback


def test_pinned_tasks_initialize_official_environment_and_are_isolated(retail):
    from evals.retail import build_environment

    _, loaded, tasks = retail
    first = build_environment(loaded, tasks["6"])
    second = build_environment(loaded, tasks["6"])
    assert first.get_db_hash() == second.get_db_hash()
    first.tools.db.users[next(iter(first.tools.db.users))].email = "changed@example.com"
    assert first.get_db_hash() != second.get_db_hash()


def test_lock_rejects_mismatched_data_without_loading_environment(tmp_path, monkeypatch):
    from evals import retail

    manifest = tmp_path / "lock.json"
    manifest.write_text(json.dumps({"commit": "fixture", "files": {"data.json": "incorrect"}}))
    (tmp_path / "data.json").write_text("{}")
    monkeypatch.setattr(retail, "LOCK", manifest)
    monkeypatch.setattr(retail.subprocess, "check_output", lambda *a, **k: "fixture")
    with pytest.raises(ValueError, match="changed"):
        retail.verify_checkout(tmp_path)


def test_git_ownership_exception_is_scoped_to_verified_checkout(retail, monkeypatch):
    from evals import retail as module
    lock, _, _ = retail
    calls = []
    def check(command, *, cwd, text):
        calls.append((command, cwd))
        return lock["commit"]
    monkeypatch.setattr(module.subprocess, "check_output", check)
    module.verify_checkout(ROOT)
    assert calls == [(["git", "-c", f"safe.directory={ROOT.resolve().as_posix()}",
                       "rev-parse", "HEAD"], ROOT.resolve())]
