"""Minimal pinned tau-Bench 2.0 text-v1/en/retail environment adapter.

Task and scoring data stay on the harness side. This module never starts an upstream
Agent, user simulator or judge. Fixed fixture interactions exercise our own Agent loop.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from agentcore import Agent
from agentcore.tools import ToolRegistry, ToolResult, ToolSpec

LOCK = Path(__file__).with_name("retail.lock.json")


def verify_checkout(root):
    root = Path(root).resolve()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned Retail source/data changed: {name}")
    # The checkout can be created by the desktop sandbox account and later read
    # by the user's account. Trust only this resolved, hash-verified checkout for
    # this read-only command; never change global Git config or trust a wildcard.
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, text=True,
    ).strip()
    if commit != lock["commit"]:
        raise ValueError("Retail checkout commit differs from lock")
    spec = importlib.util.find_spec("tau2")
    if spec is None or Path(spec.origin).resolve() != root / "src/tau2/__init__.py":
        raise ValueError(
            "Install the pinned checkout in editable mode; unrelated tau2 installs are refused"
        )
    return lock


def load_retail(root):
    root = Path(root).resolve()
    lock = verify_checkout(root)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["TAU2_DATA_DIR"] = str(root / "data")
    from tau2.data_model.language import TaskSelection
    from tau2.multilingual.runtime import load_language
    from tau2.utils.utils import DATA_DIR

    if DATA_DIR.resolve() != root / "data":
        raise ValueError("tau2 was already imported with a different data root")
    loaded = load_language(
        "en",
        "retail",
        mode="text",
        selection=TaskSelection(task_set_name="retail", task_split_name="base"),
    )
    subset = json.loads((root / "data/tau2/task_subsets/retail_50.json").read_text())
    tasks = {task.id: task for task in loaded.tasks if task.id in subset["task_ids"]}
    if len(tasks) != 50:
        raise ValueError("Expected the released 50-task English Retail cell")
    return lock, loaded, tasks


def build_environment(loaded, task=None):
    from tau2.domains.retail.environment import get_environment
    from tau2.multilingual.runtime import language_environment_kwargs

    environment = get_environment(**language_environment_kwargs(loaded))
    if task is not None and task.initial_state is not None:
        initial = task.initial_state
        environment.set_state(
            initial.initialization_data,
            initial.initialization_actions,
            initial.message_history or [],
        )
    return environment


def tools_for(environment):
    from tau2.data_model.message import ToolCall as TauToolCall

    registry = ToolRegistry()

    def handler_for(name):
        # Official environment owns return serialization, validation and business semantics.
        def handler(**arguments):
            response = environment.get_response(
                TauToolCall(id=uuid4().hex, name=name, arguments=arguments)
            )
            if response.error:
                return ToolResult(
                    "returned",
                    value=response.content,
                    error="Retail environment rejected the operation",
                    category="business",
                )
            return response.content

        return handler

    for tool in environment.get_tools():
        definition = tool.openai_schema["function"]
        registry.register(
            ToolSpec(
                tool.name,
                definition["description"],
                definition["parameters"],
                handler_for(tool.name),
                read_only=not environment.tools.tool_mutates_state(tool.name),
            )
        )
    return registry


def make_agent(model, store, environment, loaded, *, config=None):
    # No Task, evaluation_criteria, user_scenario or reference actions enter the Agent.
    prompt = environment.get_policy() + "\n" + loaded.pack.agent_language_clause
    return Agent(model, tools_for(environment), store, config=config, system_prompt=prompt)


def scoring_status():
    return {
        "official_score": None,
        "scoring_run": False,
        "unverified": [
            "official user simulator",
            "end-to-end official orchestration",
            "DB reward",
            "communication semantic judge",
            "NL assertion judge",
        ],
        "note": "No scorer was replaced or removed; this is environment wiring evidence only.",
    }
