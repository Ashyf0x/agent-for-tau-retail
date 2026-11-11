"""Half-duplex bridge: official orchestration, our unchanged Agent execution loop.

Import only after load_retail has verified and configured the pinned checkout.
No client, credential loader, or real-call CLI is provided here.
"""

import asyncio
import json
from pathlib import Path

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from research_agent import Agent
from research_agent.context import ContextManager
from research_agent.core.agent import AgentConfig
from research_agent.storage import Store
from research_agent.tools import ToolRegistry, ToolResult, ToolSpec


class RetailAgentBridge(HalfDuplexAgent):
    """Drive a suspended Agent coroutine without threads or duplicate tool execution.

Core executes tool batches sequentially; each validated dispatch is exposed as one
official assistant tool message. Official IDs are core IDs (provider mappings stay
in SQLite). Internal ask_user/read_observation remain local runtime operations.
"""

    def __init__(self, model, database, environment, domain_policy, *, config=None,
                 context_options=None):
        if Path(database).exists():
            raise FileExistsError("Retail interaction requires a new database")
        context_manager = ContextManager(domain_policy, **(context_options or {}))
        tools = environment.get_tools()
        super().__init__(tools, domain_policy)
        self.loop = asyncio.new_event_loop()
        self.outputs = asyncio.Queue()
        self.pending = None
        self.task = None
        self.closed = False
        self.stopping = False
        self.used = set()
        self.results = []
        self.store = Store(database)
        registry = ToolRegistry()
        for tool in tools:
            definition = tool.openai_schema["function"]
            registry.register(ToolSpec(
                tool.name, definition["description"], definition["parameters"],
                self._handler(tool.name), read_only=not environment.tools.tool_mutates_state(tool.name),
            ))
        bridge = self

        class Model:
            def prepare(self, *args, **kwargs):
                return model.prepare(*args, **kwargs)

            async def complete(self, request):
                if bridge.stopping:
                    raise asyncio.CancelledError()
                return await model.complete(request)

        self.agent = Agent(Model(), registry, self.store, system_prompt=domain_policy,
                           context=context_manager,
                           config=config or AgentConfig(max_argument_corrections=0,
                                                        max_protocol_corrections=0))

    def _handler(self, name):
        async def relay(**arguments):
            message = next(row["message"] for row in reversed(self.store.messages("retail", "call"))
                           if row["message"].get("tool_calls"))
            call = next(call for call in message["tool_calls"]
                        if call["id"] not in self.used and call["function"]["name"] == name
                        and json.loads(call["function"]["arguments"]) == arguments)
            self.used.add(call["id"])
            self.pending = (call["id"], self.loop.create_future())
            await self.outputs.put(AssistantMessage(role="assistant", tool_calls=[
                ToolCall(id=call["id"], name=name, arguments=arguments)]))
            response = await self.pending[1]
            self.pending = None
            if response.error:
                return ToolResult("returned", value=response.content,
                                  error="Retail environment rejected the operation", category="business")
            return response.content
        return relay

    def get_init_state(self, message_history=None):
        if self.store.messages("retail", "call"):
            raise ValueError("Bridge cannot resume or initialize twice")
        for message in message_history or []:
            if not isinstance(message, AssistantMessage) or message.is_tool_call():
                raise ValueError("Only the official opening greeting is supported as initial history")
            self.store.append("retail", "call", None, {"role": "assistant", "content": message.content})
        return None

    async def _turn(self, content):
        try:
            result = await self.agent.run(content, user_id="retail", session_id="call")
            self.results.append(result)
            if result.status != "completed" and not (
                result.status == "waiting_user" and result.reason == "clarification"
            ):
                raise RuntimeError(f"Agent stopped: {result.status}/{result.reason}")
            await self.outputs.put(AssistantMessage(role="assistant", content=result.answer))
        except Exception as exc:
            await self.outputs.put(exc)

    def generate_next_message(self, message, state):
        if self.closed:
            raise RuntimeError("Bridge is closed")
        if isinstance(message, UserMessage):
            if self.task is not None and not self.task.done():
                raise RuntimeError("User turn arrived during tool dispatch")
            self.task = self.loop.create_task(self._turn(message.content))
        elif isinstance(message, ToolMessage):
            if self.pending is None or message.id != self.pending[0]:
                raise ValueError("Official tool response ID mismatch")
            self.pending[1].set_result(message)
        else:
            raise ValueError("Expected one user message or one official tool response")
        output = self.loop.run_until_complete(self.outputs.get())
        if isinstance(output, Exception):
            raise output
        return output, state

    def stop(self, message=None, state=None):
        if self.closed:
            return
        self.stopping = True
        if self.task is not None and not self.task.done():
            if (isinstance(message, ToolMessage) and self.pending is not None
                    and message.id == self.pending[0] and not self.pending[1].done()):
                # The official step limit may stop after executing the tool but
                # before routing its result back. Persist that known result first;
                # the model gate prevents another provider dispatch during drain.
                self.pending[1].set_result(message)
            else:
                self.task.cancel()
            self.loop.run_until_complete(asyncio.gather(self.task, return_exceptions=True))
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.store.close()
        self.loop.close()
        self.closed = True


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(value.model_dump_json(indent=2) + "\n")


def run_and_score(orchestrator, loaded, output):
    """May call user/model/judges: caller needs fresh authorization for ALL roles.

    Save the official simulation before scoring so judge failure cannot erase it.
    No reduced scoring, fallback scorer, retry, or implied zero on failure.
    """
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.multilingual.runtime import language_environment_kwargs

    context = orchestrator.call_context
    if context is None:
        raise ValueError("A pinned language call context is required")
    from tau2.multilingual.runtime import task_digest
    if task_digest(orchestrator.task) != context.language.task_sha256:
        raise ValueError("Evaluation task differs from pinned language context")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "task.json", orchestrator.task)
    save_new(output / "call-context.json", context)
    simulation = orchestrator.run()
    simulation.policy = orchestrator.environment.get_policy()
    # Official get_cost sums missing per-message costs as zero. The bridge keeps
    # raw Agent usage in SQLite but has no shared provider price accounting yet.
    simulation.agent_cost = None
    simulation.language_context = context.language
    save_new(output / "simulation.unscored.json", simulation)
    simulation.reward_info = evaluate_simulation(
        simulation=simulation, task=orchestrator.task, evaluation_type=EvaluationType.ALL,
        solo_mode=False, domain="retail", env_kwargs=language_environment_kwargs(loaded),
        communicate_scoring=context.communicate_scoring,
        customer_language=context.language.provider_language,
    )
    from tau2.data_model.tasks import RewardType
    reward = simulation.reward_info
    criteria = orchestrator.task.evaluation_criteria
    if reward.reward_basis and RewardType.NL_ASSERTION in reward.reward_basis:
        expected = sorted(criteria.nl_assertions or [])
        actual = sorted(check.nl_assertion for check in reward.nl_assertions or [])
        if actual != expected:
            raise ValueError("Official NL checks do not cover task assertions; no complete score saved")
    save_new(output / "simulation.json", simulation)
    return simulation


def build_interaction(loaded, task, model, database, *, user_model, user_args,
                      communicate_scoring, seed=42, max_steps=100, config=None,
                      user_factory=None, context_options=None):
    """Build without dispatching. Explicit simulator/scorer config; no .env reads.

    user_factory is for deterministic offline fixtures. Production uses the
    official UserSimulator with the official language/persona instructions.
    """
    from tau2.data_model.language import CallContext
    from tau2.multilingual.runtime import resolve_task, agent_language_guidance
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_user, agent_domain_policy
    from .retail import build_environment

    context = resolve_task(loaded, task, seed)
    environment = build_environment(loaded)
    user = (user_factory(environment, task, context) if user_factory else build_user(
        "user_simulator", environment, task, llm=user_model, llm_args=user_args,
        persona_config=context.persona,
    ))
    bridge = RetailAgentBridge(model, database, environment, agent_domain_policy(
        environment, agent_language_guidance(context), language=context.language), config=config,
        context_options=context_options)
    try:
        return Orchestrator(domain="retail", agent=bridge, user=user, environment=environment,
                            task=task, seed=seed, max_steps=max_steps,
                            call_context=CallContext(language=context,
                                                     communicate_scoring=communicate_scoring))
    except BaseException:
        bridge.stop()
        raise
