"""Public Pilot settings only: no authentication parsing or process-env fallback."""

import io
import re
from pathlib import Path

from agentcore.core.model import HTTPModelClient, bailian_protocol

ROLES = ("agent", "user", "communicate", "nl")

# Project execution choices, not official limits or experimentally optimal values.
RESOURCE_PROFILES = {
    "relaxed": {"max_calls_per_task": 80, "core_max_steps": 24, "task_budget_cny": "3.00"},
    "legacy": {"max_calls_per_task": 40, "core_max_steps": 12, "task_budget_cny": "1.00"},
}


def resolve_roles(*, model=None, base_url=None, agent_model=None, role_models=None,
                  role_thinking=None, env_file=Path(".env")):
    """Resolve one model per role. Explicit per-role models win over any shared default.

    `role_thinking` is an optional explicit per-role thinking map. Without it the
    historical default stands (caller thinks, every other role does not). Passing it
    is how a run reproduces the official order where the simulator reasons more than
    the caller; it is frozen into the plan and re-checked at validation.
    """
    from dotenv import dotenv_values

    role_models = dict(role_models or {})
    role_thinking = dict(role_thinking or {})
    unknown = (set(role_models) | set(role_thinking)) - set(ROLES)
    if unknown:
        raise ValueError(f"Unknown role(s) for override: {sorted(unknown)}")
    for role, value in role_thinking.items():
        if type(value) is not bool:
            raise ValueError(f"role_thinking[{role}] must be boolean")
    names = {"AGENT_MODEL", "AGENT_BASE_URL"}
    names |= {f"RETAIL_{role.upper()}_{field}" for role in ROLES for field in ("MODEL", "BASE_URL")}
    values = {}
    path = Path(env_file)
    if (model is None or base_url is None) and path.is_file():
        selected = []
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                match = re.match(r"^\s*(?:export\s+)?([A-Z_]+)\s*=", line)
                if match and match[1] in names:
                    selected.append(line)
        values = dotenv_values(stream=io.StringIO("".join(selected)), interpolate=False)

    def setting(role, field, explicit):
        role_key = f"RETAIL_{role.upper()}_{field}"
        common_key = f"AGENT_{field}"
        if role == "agent" and field == "MODEL" and agent_model is not None:
            value, source = agent_model, "explicit --agent-model"
        elif role in role_models and field == "MODEL":
            value, source = role_models[role], f"explicit --{role}-model"
        elif explicit is not None:
            value, source = explicit, "explicit --" + field.lower().replace("_", "-")
        elif role_key in values:
            value, source = values[role_key], role_key
        else:
            value, source = values.get(common_key), common_key
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {role} {field}: configure {common_key}/{role_key} or explicit CLI")
        return value.strip(), source

    result = {}
    for role in ROLES:
        model_value, model_source = setting(role, "MODEL", model)
        url_value, url_source = setting(role, "BASE_URL", base_url)
        provider = bailian_protocol(model_value)
        thinking = role_thinking.get(role, role == "agent")
        preserve = role == "agent" and provider == "qwen" and thinking
        HTTPModelClient(url_value, model_value, provider=provider, thinking=thinking,
                        preserve_thinking=preserve)
        result[role] = {"model": model_value, "base_url": url_value, "provider": provider,
                        "thinking": thinking, "preserve_thinking": preserve,
                        "temperature": 0, "provider_seed": 42 if role == "user" else None,
                        "model_source": model_source, "base_url_source": url_source}
    return result
