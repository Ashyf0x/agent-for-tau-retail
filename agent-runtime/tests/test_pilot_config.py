import json
import sys
from pathlib import Path

import pytest

from evals.pilot_config import resolve_roles


def test_all_roles_inherit_actual_file_settings_and_no_credentials_are_parsed(tmp_path, monkeypatch):
    import dotenv
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=user-chosen-alias\nAGENT_BASE_URL=https://configured.invalid/v1\n'
                      'DASHSCOPE_API_KEY=fixture-auth\nRETAIL_NL_MODEL=user-chosen-judge\n', encoding="utf-8")
    original = dotenv.dotenv_values
    def check(*, stream, interpolate):
        assert "fixture-auth" not in stream.getvalue() and interpolate is False
        return original(stream=stream, interpolate=interpolate)
    monkeypatch.setattr(dotenv, "dotenv_values", check)
    roles = resolve_roles(env_file=config)
    assert all(r["base_url"] == "https://configured.invalid/v1" for r in roles.values())
    assert roles["agent"]["model"] == roles["user"]["model"] == "user-chosen-alias"
    assert roles["nl"]["model"] == "user-chosen-judge"
    assert roles["nl"]["model_source"] == "RETAIL_NL_MODEL"
    assert "fixture-auth" not in json.dumps(roles)


def test_no_model_or_url_fallback_and_explicit_override_does_not_open_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "not-a-file-setting")
    with pytest.raises(ValueError, match="Missing"):
        resolve_roles(env_file=tmp_path / "absent")
    def forbidden(*args, **kwargs):
        raise AssertionError("Both explicit overrides must avoid config reads")
    monkeypatch.setattr(Path, "open", forbidden)
    roles = resolve_roles(model="explicit-alias", base_url="https://explicit.invalid/v1")
    assert all(r["model"] == "explicit-alias" for r in roles.values())


def test_agent_only_override_preserves_all_other_role_fields(tmp_path):
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=baseline\nAGENT_BASE_URL=https://configured.invalid/v1\n'
                      'RETAIL_AGENT_MODEL=configured-agent\nRETAIL_USER_MODEL=simulator\n'
                      'RETAIL_NL_MODEL=judge\nRETAIL_NL_BASE_URL=https://judge.invalid/v1\n', encoding="utf-8")
    before = resolve_roles(env_file=config)
    after = resolve_roles(agent_model="experiment-agent", env_file=config)
    assert after["agent"] == {**before["agent"], "model": "experiment-agent",
                              "model_source": "explicit --agent-model"}
    assert all(after[role] == before[role] for role in ("user", "nl", "communicate"))
    with pytest.raises(ValueError, match="Missing agent MODEL"):
        resolve_roles(agent_model=" ", env_file=config)


def test_cli_config_check_needs_no_output_or_auth(tmp_path, monkeypatch, capsys):
    from evals import retail_pilot
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=chosen\nAGENT_BASE_URL=https://chosen.invalid/v1\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["pilot", "--check-config", "--env-file", str(config)])
    retail_pilot.main()
    result = json.loads(capsys.readouterr().out)
    assert result["requests_sent"] == 0 and not result["credentials_loaded"]
    assert result["roles"]["agent"]["model"] == "chosen"


@pytest.mark.parametrize("agent_model", [None, "experiment-agent"])
def test_direct_execution_validates_limits_before_loading_key(tmp_path, monkeypatch, agent_model):
    from agentcore.app import smoke
    from evals import retail_pilot
    root = Path(__file__).parents[1] / ".vendor/tau2-bench"
    if not root.exists():
        pytest.skip("Pinned checkout not installed")
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=chosen\nAGENT_BASE_URL=https://chosen.invalid/v1\n', encoding="utf-8")
    def forbidden(*args):
        raise AssertionError("Invalid limits must be refused before reading authentication")
    monkeypatch.setattr(smoke, "_load_key", forbidden)
    argv = ["pilot", "--checkout", str(root), "--execute", "--task-id", "6",
            "--output", str(tmp_path / "run"), "--env-file", str(config),
            "--agent-max-completion-tokens", "9061"]
    if agent_model is not None:
        argv += ["--agent-model", agent_model]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="Agent output"):
        retail_pilot.main()
    assert not (tmp_path / "run").exists()


def test_comparison_budget_and_agent_override_are_recorded(tmp_path):
    from evals.retail_pilot import build_config, validate_config
    root = Path(__file__).parents[1] / ".vendor/tau2-bench"
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=baseline\nAGENT_BASE_URL=https://configured.invalid/v1\n', encoding="utf-8")
    plan = build_config(root, ["11"], env_file=config, agent_model="experiment-agent",
                  budget="3", input_price="12", output_price="36", max_request_bytes=131072)
    assert plan["configuration_inputs"]["agent_model"] == "experiment-agent"
    assert plan["roles"]["user"]["model"] == "baseline"
    validate_config(root, plan)
    plan["budget_cny"] = "30.01"
    with pytest.raises(ValueError, match="budget"):
        validate_config(root, plan)
    with pytest.raises(ValueError, match="at most 30"):
        build_config(root, ["11"], env_file=config, budget="30.01")


@pytest.mark.parametrize("override", ["--agent-model", "--model"])
def test_direct_cli_resolves_current_roles_without_plan(tmp_path, monkeypatch, capsys, override):
    from agentcore.app import smoke
    from evals import retail_pilot
    config = tmp_path / "settings.txt"
    config.write_text('AGENT_MODEL=baseline\nAGENT_BASE_URL=https://configured.invalid/v1\n'
                      'RETAIL_NL_MODEL=separate-judge\n', encoding="utf-8")
    order = []
    def load_key(path):
        assert path == config
        order.append("auth")
        return "fixture-auth"
    def run(checkout, effective, output, *, key, offline):
        order.append("run")
        assert key == "fixture-auth" and offline is False
        assert effective["roles"]["agent"]["model"] == "experiment-model"
        assert effective["roles"]["user"]["model"] == (
            "experiment-model" if override == "--model" else "baseline")
        assert effective["roles"]["nl"]["model"] == (
            "experiment-model" if override == "--model" else "separate-judge")
        assert all(r["base_url"] == "https://configured.invalid/v1" for r in effective["roles"].values())
        assert effective["limits"]["agent_max_output_tokens"] == 9060
        assert effective["limits"]["max_output_tokens"] == 2048
        assert effective["limits"]["max_request_bytes"] == 131072
        assert effective["budget_cny"] == "2.00"
        return {"test_double": True}
    monkeypatch.setattr(smoke, "_load_key", load_key)
    monkeypatch.setattr(retail_pilot, "run_pilot", run)
    monkeypatch.setattr(sys, "argv", ["pilot", "--execute", "--task-id", "6",
        "--env-file", str(config), override, "experiment-model", "--output", str(tmp_path / "run"),
        "--agent-max-completion-tokens", "9060", "--max-request-bytes", "131072", "--budget-yuan", "2.00"])
    retail_pilot.main()
    assert order == ["auth", "run"]
    assert json.loads(capsys.readouterr().out) == {"test_double": True}


def test_cli_requires_a_mode_and_offline_check_requires_explicit_settings(tmp_path, monkeypatch):
    from evals import retail_pilot
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected CLI must not read settings or run")
    monkeypatch.setattr(retail_pilot, "build_config", forbidden)
    for argv in (["pilot", "--output", str(tmp_path / "unused")],
                 ["pilot", "--offline-check", "--output", str(tmp_path / "unused")],
                 ["pilot", "--execute", "--plan", "old.json", "--output", str(tmp_path / "unused")]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as error:
            retail_pilot.main()
        assert error.value.code == 2


def test_existing_output_is_rejected_before_settings_or_auth(tmp_path, monkeypatch):
    from evals import retail_pilot
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing evidence must be rejected before settings or authentication")
    monkeypatch.setattr(retail_pilot, "build_config", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--execute", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError, match="already exists"):
        retail_pilot.main()


@pytest.mark.parametrize("profile", ["relaxed", "legacy"])
def test_offline_cli_saves_actual_configuration_without_reading_env(tmp_path, monkeypatch, capsys, profile):
    from agentcore.app import smoke
    from evals import retail_pilot
    private = tmp_path / "do-not-read.env"
    private.write_text("not a test configuration", encoding="utf-8")
    original_open = Path.open
    def checked_open(path, *args, **kwargs):
        assert path != private, "Offline fixture must not open any .env file"
        return original_open(path, *args, **kwargs)
    def forbidden(*args):
        raise AssertionError("Offline fixture must not load authentication")
    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(smoke, "_load_key", forbidden)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["pilot", "--offline-check", "--task-id", "6",
        "--env-file", str(private), "--model", "explicit-offline-fixture",
        "--base-url", "https://fixture.invalid/v1", "--output", str(output),
        "--resource-profile", profile,
        "--agent-max-completion-tokens", "9060", "--max-request-bytes", "131072"])
    retail_pilot.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "offline_http_fixture" and summary["calls"] == 3
    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
    from evals.pilot_config import RESOURCE_PROFILES
    assert saved["resources"] == {"profile": profile, **RESOURCE_PROFILES[profile]}
    assert saved["budget_cny"] == RESOURCE_PROFILES[profile]["task_budget_cny"]
    assert all(r["model"] == "explicit-offline-fixture" for r in saved["roles"].values())
    assert not (output / "plan.json").exists()
    rows = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    agent = next(r["request"] for r in rows if r["event"] == "dispatch_intent" and r["role"] == "agent")
    assert agent["max_completion_tokens"] == 9060
    assert agent["enable_thinking"] is agent["preserve_thinking"] is True
    assert (output / "source-snapshot/evals/retail_pilot.py").is_file()
