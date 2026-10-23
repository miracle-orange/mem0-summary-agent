from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.agent_pool import AgentPool  # noqa: E402
from summary_multi_agent.dag import topological_layers  # noqa: E402
from summary_multi_agent.orchestrator import SummaryOrchestrator  # noqa: E402
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    ExecutionPlan,
    PlanSource,
    RunConfig,
    RunStatus,
    SubTask,
    SummaryRequest,
    TaskType,
)
from summary_multi_agent.planner import RuleBasedPlanner  # noqa: E402
from summary_multi_agent.tools import (  # noqa: E402
    CalendarResolveTool,
    EntityQueryTool,
    TemplateAdaptationTool,
    ToolRegistry,
    TypedMemorySearchTool,
)
from summary_multi_agent.working_memory import (  # noqa: E402
    initialize_working_state,
    load_working_state_checkpoint,
)
from summary_multi_agent.workers import (  # noqa: E402
    BaseWorker,
    EntityPersonalizationWorker,
    SemanticMemoryWorker,
    TemplateAdaptationWorker,
    TemporalReasoningWorker,
    WorkerContext,
)


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.create_calls = 0
        self.replan_calls = 0

    async def create_plan(self, planner_input) -> ExecutionPlan:
        del planner_input
        self.create_calls += 1
        return self.plan

    async def replan(self, planner_input) -> ExecutionPlan:
        self.replan_calls += 1
        previous_plan = planner_input.previous_plan
        successful_ids = tuple(
            sorted(
                outcome.task_id
                for outcome in planner_input.task_outcomes
                if outcome.status == AgentStatus.SUCCESS
            )
        )
        return ExecutionPlan(
            source=PlanSource.REPLAN,
            plan_version=previous_plan.plan_version + 1,
            parent_plan_id=previous_plan.plan_id,
            replan_round=planner_input.replan_round,
            replan_reason="; ".join(planner_input.failure_reasons),
            preserved_task_ids=successful_ids,
            tasks=self.plan.tasks,
            rationale="Static test replan.",
        )


class ExecutionTracker:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: dict[str, int] = {}
        self.prior_result_ids: dict[str, tuple[str, ...]] = {}


class IntegrationMemoryBackend:
    def search(self, query, *, filters, top_k):
        del query, top_k
        memory_type = filters.get("memory_type")
        items = [
            {
                "id": "memory_task",
                "memory": "张三负责历史接口任务",
                "score": 0.9,
                "metadata": {
                    "typed_memory_id": "memory_task",
                    "memory_type": "task",
                    "content_text": "张三负责历史接口任务",
                    "entities_text": "张三 | 李四",
                    "source_time": "2026-08-01",
                },
            }
        ]
        if memory_type:
            items = [
                item for item in items if item["metadata"]["memory_type"] == memory_type
            ]
        return {"results": items}


class RecordingWorker(BaseWorker):
    def __init__(
        self,
        task_type: TaskType,
        tracker: ExecutionTracker,
        *,
        delay: float = 0.01,
        fail_first: bool = False,
    ) -> None:
        self.task_type = task_type
        self._tracker = tracker
        self._delay = delay
        self._fail_first = fail_first

    async def execute(self, context: WorkerContext) -> AgentResult:
        self.validate_context(context)
        task_id = context.task.task_id
        prior_result_ids = tuple(
            item.task_id for item in context.prior_results
        )
        self._tracker.prior_result_ids[task_id] = prior_result_ids
        self._tracker.calls[task_id] = self._tracker.calls.get(task_id, 0) + 1
        self._tracker.active += 1
        self._tracker.max_active = max(
            self._tracker.max_active, self._tracker.active
        )
        try:
            await asyncio.sleep(self._delay)
        finally:
            self._tracker.active -= 1
        if self._fail_first and self._tracker.calls[task_id] == 1:
            return AgentResult(
                task_id=task_id,
                task_type=self.task_type,
                status=AgentStatus.FAILED,
                error="planned first-attempt failure",
            )
        return AgentResult(
            task_id=task_id,
            task_type=self.task_type,
            status=AgentStatus.SUCCESS,
            payload={"worker": self.task_type.value},
            consumed_result_ids=prior_result_ids,
        )


def make_request_and_state():
    request = SummaryRequest(
        current_material="张三明天提交测试报告。",
        source_time="2026-08-05",
        summary_request="生成摘要",
    )
    bundle = build_raw_material_bundle(request.current_material, max_segment_chars=100)
    return request, initialize_working_state(request, bundle)


class DagTests(unittest.TestCase):
    def test_topological_layers_are_deterministic(self) -> None:
        plan = ExecutionPlan(
            tasks=(
                SubTask(
                    task_id="b",
                    task_type=TaskType.TEMPLATE_ADAPTATION,
                    description="B",
                ),
                SubTask(
                    task_id="a",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="A",
                ),
                SubTask(
                    task_id="c",
                    task_type=TaskType.TEMPORAL_REASONING,
                    description="C",
                    dependencies=("a", "b"),
                ),
            )
        )
        layers = topological_layers(plan)
        self.assertEqual(("a", "b"), tuple(task.task_id for task in layers[0]))
        self.assertEqual(("c",), tuple(task.task_id for task in layers[1]))


class AgentPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_is_reused(self) -> None:
        tracker = ExecutionTracker()
        pool = AgentPool(
            {
                TaskType.SEMANTIC_MEMORY: lambda: RecordingWorker(
                    TaskType.SEMANTIC_MEMORY, tracker
                )
            },
            max_total=1,
        )
        first = await pool.acquire(TaskType.SEMANTIC_MEMORY)
        await pool.release(first)
        second = await pool.acquire(TaskType.SEMANTIC_MEMORY)
        self.assertIs(first, second)
        await pool.release(second)

    async def test_idle_other_type_is_evicted_at_global_limit(self) -> None:
        tracker = ExecutionTracker()
        pool = AgentPool(
            {
                TaskType.SEMANTIC_MEMORY: lambda: RecordingWorker(
                    TaskType.SEMANTIC_MEMORY, tracker
                ),
                TaskType.TEMPLATE_ADAPTATION: lambda: RecordingWorker(
                    TaskType.TEMPLATE_ADAPTATION, tracker
                ),
            },
            max_total=1,
        )
        memory_worker = await pool.acquire(TaskType.SEMANTIC_MEMORY)
        await pool.release(memory_worker)
        template_worker = await asyncio.wait_for(
            pool.acquire(TaskType.TEMPLATE_ADAPTATION), timeout=0.2
        )
        self.assertEqual(TaskType.TEMPLATE_ADAPTATION, template_worker.task_type)
        await pool.release(template_worker)


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rule_plan_runs_real_workers_with_fake_memory(self) -> None:
        backend = IntegrationMemoryBackend()
        registry = ToolRegistry(
            [
                TypedMemorySearchTool(backend, user_id="user", project="project"),
                EntityQueryTool(backend, user_id="user", project="project"),
                TemplateAdaptationTool(),
                CalendarResolveTool(),
            ]
        )
        pool = AgentPool(
            {
                TaskType.SEMANTIC_MEMORY: SemanticMemoryWorker,
                TaskType.ENTITY_PERSONALIZATION: EntityPersonalizationWorker,
                TaskType.TEMPLATE_ADAPTATION: TemplateAdaptationWorker,
                TaskType.TEMPORAL_REASONING: TemporalReasoningWorker,
            },
            max_total=4,
        )
        request = SummaryRequest(
            current_material="张三明天提交测试报告。",
            source_time="2026-08-05",
            summary_request="生成结构化摘要",
            target_entities=["张三"],
        )
        bundle = build_raw_material_bundle(request.current_material, max_segment_chars=100)
        state = initialize_working_state(request, bundle)
        orchestrator = SummaryOrchestrator(
            planner=RuleBasedPlanner(),
            agent_pool=pool,
            tool_registry=registry,
            config=RunConfig(workflow_mode="multi_agent_v2"),
        )
        state = await orchestrator.run_workers(state)

        self.assertEqual(RunStatus.COLLECTING, state.status)
        self.assertEqual(
            {
                "semantic_memory",
                "entity_personalization",
                "template_adaptation",
                "temporal_reasoning",
            },
            set(state.agent_results),
        )
        self.assertTrue(
            all(result.status == AgentStatus.SUCCESS for result in state.agent_results.values())
        )
        self.assertTrue(state.evidence_index)

    async def test_runs_topological_layers_with_parallel_workers(self) -> None:
        tracker = ExecutionTracker()
        plan = ExecutionPlan(
            tasks=(
                SubTask(
                    task_id="memory",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="Memory",
                ),
                SubTask(
                    task_id="template",
                    task_type=TaskType.TEMPLATE_ADAPTATION,
                    description="Template",
                ),
                SubTask(
                    task_id="temporal",
                    task_type=TaskType.TEMPORAL_REASONING,
                    description="Temporal",
                    dependencies=("memory", "template"),
                ),
            )
        )
        planner = StaticPlanner(plan)
        pool = AgentPool(
            {
                task_type: lambda task_type=task_type: RecordingWorker(
                    task_type, tracker, delay=0.03
                )
                for task_type in (
                    TaskType.SEMANTIC_MEMORY,
                    TaskType.TEMPLATE_ADAPTATION,
                    TaskType.TEMPORAL_REASONING,
                )
            },
            max_total=3,
        )
        request, state = make_request_and_state()
        del request
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = SummaryOrchestrator(
                planner=planner,
                agent_pool=pool,
                tool_registry=ToolRegistry(),
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    max_concurrent_workers=2,
                ),
                output_dir=Path(temp_dir),
            )
            state = await orchestrator.run_workers(state)
            checkpoint_path = Path(temp_dir) / "working_state.json"
            trace_path = Path(temp_dir) / "execution_trace.jsonl"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(trace_path.exists())
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            self.assertEqual(
                ["memory", "template"],
                checkpoint["agent_results"]["temporal"]["consumed_result_ids"],
            )
            restored = load_working_state_checkpoint(checkpoint_path)
            self.assertEqual(
                ("memory", "template"),
                restored.agent_results["temporal"].consumed_result_ids,
            )
            trace = [
                json.loads(line)
                for line in trace_path.read_text("utf-8").splitlines()
            ]
            temporal_started = next(
                item
                for item in trace
                if item["event_type"] == "task.started"
                and item["payload"]["task_id"] == "temporal"
            )
            self.assertEqual(
                ["memory", "template"],
                temporal_started["payload"]["dependency_task_ids"],
            )

        self.assertEqual(RunStatus.COLLECTING, state.status)
        self.assertEqual(2, tracker.max_active)
        self.assertEqual(
            {"memory", "template", "temporal"}, set(state.agent_results)
        )
        self.assertEqual(
            ("memory", "template"), tracker.prior_result_ids["temporal"]
        )
        self.assertTrue(
            all(result.status == AgentStatus.SUCCESS for result in state.agent_results.values())
        )

    async def test_failed_dependency_skips_downstream_without_worker_call(self) -> None:
        tracker = ExecutionTracker()
        plan = ExecutionPlan(
            tasks=(
                SubTask(
                    task_id="memory",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="Memory",
                ),
                SubTask(
                    task_id="template",
                    task_type=TaskType.TEMPLATE_ADAPTATION,
                    description="Template",
                    dependencies=("memory",),
                ),
            )
        )
        pool = AgentPool(
            {
                TaskType.SEMANTIC_MEMORY: lambda: RecordingWorker(
                    TaskType.SEMANTIC_MEMORY, tracker, fail_first=True
                ),
                TaskType.TEMPLATE_ADAPTATION: lambda: RecordingWorker(
                    TaskType.TEMPLATE_ADAPTATION, tracker
                ),
            },
            max_total=2,
        )
        _request, state = make_request_and_state()
        state = await SummaryOrchestrator(
            planner=StaticPlanner(plan),
            agent_pool=pool,
            tool_registry=ToolRegistry(),
            config=RunConfig(workflow_mode="multi_agent_v2"),
        ).run_workers(state)

        self.assertEqual(RunStatus.COLLECTING, state.status)
        self.assertEqual(AgentStatus.FAILED, state.agent_results["memory"].status)
        self.assertEqual(AgentStatus.SKIPPED, state.agent_results["template"].status)
        self.assertNotIn("template", tracker.calls)

    async def test_replan_preserves_success_and_retries_failed_critical_task(self) -> None:
        tracker = ExecutionTracker()
        plan = ExecutionPlan(
            tasks=(
                SubTask(
                    task_id="memory",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="Memory",
                ),
                SubTask(
                    task_id="temporal",
                    task_type=TaskType.TEMPORAL_REASONING,
                    description="Temporal",
                    critical=True,
                ),
            )
        )
        planner = StaticPlanner(plan)
        pool = AgentPool(
            {
                TaskType.SEMANTIC_MEMORY: lambda: RecordingWorker(
                    TaskType.SEMANTIC_MEMORY, tracker
                ),
                TaskType.TEMPORAL_REASONING: lambda: RecordingWorker(
                    TaskType.TEMPORAL_REASONING, tracker, fail_first=True
                ),
            },
            max_total=2,
        )
        _request, state = make_request_and_state()
        orchestrator = SummaryOrchestrator(
            planner=planner,
            agent_pool=pool,
            tool_registry=ToolRegistry(),
            config=RunConfig(
                workflow_mode="multi_agent_v2",
                max_replan_rounds=1,
            ),
        )
        state = await orchestrator.run_workers(state)

        self.assertEqual(RunStatus.COLLECTING, state.status)
        self.assertEqual(1, planner.replan_calls)
        self.assertEqual(1, tracker.calls["memory"])
        self.assertEqual(2, tracker.calls["temporal"])
        self.assertEqual(AgentStatus.SUCCESS, state.agent_results["temporal"].status)
        self.assertEqual(2, len(state.plan_history))
        self.assertEqual(PlanSource.REPLAN, state.plan.source)
        self.assertEqual(("memory",), state.plan.preserved_task_ids)


if __name__ == "__main__":
    unittest.main()
