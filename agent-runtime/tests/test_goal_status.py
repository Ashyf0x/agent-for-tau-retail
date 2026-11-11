"""`goal_status` must report an observation, never a verdict.

The field existed from the start and was never assigned: every run reported
"unverified", so it carried no information. Filling it is only defensible if the value
stays strictly inside what the program can know. The program knows which mutating calls
it dispatched and which of them returned success. It does not know the user's goal.

The distinction is not academic. In experiments/retail-qwen-tier-order-20261108
(35 failures / 115 passes) runs that never dispatched a mutating call failed 62% of the
time and runs with an observed success failed 12% -- but twelve of the passes in the
zero-write group were read-only tasks that were answered correctly. Any label meaning
"the agent did not do the job" would have been wrong for those twelve. These tests hold
the line: report the observation, never the conclusion.
"""

from __future__ import annotations

from agentcore import Agent
from agentcore.core.model import ScriptedModel
from agentcore.core.types import ModelResponse, ToolCall
from agentcore.tools import ToolRegistry, ToolResult, ToolSpec

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def call(name):
    return ModelResponse(
        tool_calls=[ToolCall("provider", name, {})], finish_reason="tool_calls"
    )


def registry(specs):
    return ToolRegistry(specs)


async def test_no_mutating_tool_means_nothing_attempted(store):
    model = ScriptedModel([ModelResponse("I can answer that directly.")])
    result = await Agent(model, ToolRegistry(), store).run("What is 2 + 2?")

    assert result.goal_status == "no_state_change_attempted"


async def test_read_only_work_is_not_counted_as_a_state_change(store):
    async def look():
        return {"value": 1}

    model = ScriptedModel([call("look"), ModelResponse("The value is 1.")])
    result = await Agent(
        model,
        registry([ToolSpec("look", "read", SCHEMA, look, read_only=True)]),
        store,
    ).run("Read it.")

    assert result.goal_status == "no_state_change_attempted"


async def test_successful_mutation_is_observed(store):
    async def write():
        return ToolResult("returned", ok=True, value={"done": True})

    model = ScriptedModel([call("write"), ModelResponse("Done.")])
    result = await Agent(
        model, registry([ToolSpec("write", "mutate", SCHEMA, write)]), store
    ).run("Change it.")

    assert result.goal_status == "state_change_observed"


async def test_failed_mutation_is_unconfirmed_not_observed(store):
    """A rejected write changed nothing; the field must not read as success."""

    async def write():
        return ToolResult("returned", ok=False, error="rejected", category="business")

    model = ScriptedModel([call("write"), ModelResponse("I could not change it.")])
    result = await Agent(
        model, registry([ToolSpec("write", "mutate", SCHEMA, write)]), store
    ).run("Change it.")

    assert result.goal_status == "state_change_unconfirmed"


async def test_a_later_success_counts_even_after_an_earlier_failure(store):
    """The field reports the run's cumulative observation, not its last attempt."""
    calls = {"n": 0}

    async def write():
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolResult("returned", ok=False, error="rejected", category="business")
        return ToolResult("returned", ok=True, value={"done": True})

    model = ScriptedModel([call("write"), call("write"), ModelResponse("Done.")])
    result = await Agent(
        model, registry([ToolSpec("write", "mutate", SCHEMA, write)]), store
    ).run("Change it.")

    assert result.goal_status == "state_change_observed"


async def test_undeclared_attempt_does_not_count_as_confirmed(store):
    """status='unknown' is dispatched but unresolved; it is attempted, never confirmed."""

    async def write():
        from agentcore.core.errors import ToolFailure

        raise ToolFailure("Timed out", "transient", outcome_known=False)

    model = ScriptedModel([call("write"), ModelResponse("I am not sure it went through.")])
    result = await Agent(
        model,
        registry([ToolSpec("write", "mutate", SCHEMA, write, max_attempts=1)]),
        store,
    ).run("Change it.")

    assert result.goal_status == "state_change_unconfirmed"


async def test_status_is_recorded_in_the_persisted_result(store):
    """Observability reads the stored run row, so the field must reach it."""

    async def write():
        return ToolResult("returned", ok=True, value={"done": True})

    model = ScriptedModel([call("write"), ModelResponse("Done.")])
    result = await Agent(
        model, registry([ToolSpec("write", "mutate", SCHEMA, write)]), store
    ).run("Change it.")

    row = store.db.execute(
        "SELECT result FROM runs WHERE id=?", (result.run_id,)
    ).fetchone()
    assert '"goal_status":"state_change_observed"' in row["result"]
