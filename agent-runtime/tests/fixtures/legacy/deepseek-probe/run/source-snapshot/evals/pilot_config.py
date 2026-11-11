"""Public Pilot settings only: no authentication parsing or process-env fallback."""

import io
import re
from pathlib import Path

from research_agent.core.model import HTTPModelClient, bailian_protocol

ROLES = ("agent", "user", "communicate", "nl")

# Project execution choices, not official limits or experimentally optimal values.
RESOURCE_PROFILES = {
    "relaxed": {"max_calls_per_task": 80, "core_max_steps": 24, "task_budget_cny": "3.00"},
    "legacy": {"max_calls_per_task": 40, "core_max_steps": 12, "task_budget_cny": "1.00"},
}


def resolve_roles(*, model=None, base_url=None, agent_model=None, env_file=Path(".env")):
    from dotenv import dotenv_values

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
        preserve = role == "agent" and provider == "qwen"
        HTTPModelClient(url_value, model_value, provider=provider, thinking=role == "agent",
                        preserve_thinking=preserve)
        result[role] = {"model": model_value, "base_url": url_value, "provider": provider,
                        "thinking": role == "agent", "preserve_thinking": preserve,
                        "temperature": 0, "provider_seed": 42 if role == "user" else None,
                        "model_source": model_source, "base_url_source": url_source}
    return result
