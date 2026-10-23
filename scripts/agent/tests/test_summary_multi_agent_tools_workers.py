from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

from pydantic import Field


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    AgentStatus,
    FrozenStrictModel,
    SourceKind,
    SubTask,
    SummaryRequest,
    TaskType,
    ToolCall,
    ToolStatus,
)
from summary_multi_agent.temporal_workload import (  # noqa: E402
    REVIEW_TEMPORAL_SCOPE_INPUT,
    find_temporal_expressions,
)
from summary_multi_agent.tools import (  # noqa: E402
    BaseTool,
    CalendarResolveTool,
    EntityQueryTool,
    RetryableToolError,
    TemplateAdaptationTool,
    ToolContext,
    ToolRegistry,
    TypedMemorySearchTool,
)
from summary_multi_agent.workers import (  # noqa: E402
    EntityPersonalizationWorker,
    SemanticMemoryWorker,
    TemplateAdaptationWorker,
    TemporalReasoningWorker,
    WorkerContext,
)


class FakeMemoryBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.items = [
            {
                "id": "mem_task_1",
                "memory": "张三负责周五前提交接口测试报告",
                "score": 0.91,
                "metadata": {
                    "typed_memory_id": "mem_task_1",
                    "memory_type": "task",
                    "content_text": "张三负责周五前提交接口测试报告",
                    "source_time": "2026-08-01",
                    "evidence_span": "张三周五前提交报告",
                    "entities_text": "张三 | 李四",
                },
            },
            {
                "id": "mem_risk_1",
                "memory": "接口测试环境尚未准备",
                "score": 0.82,
                "metadata": {
                    "typed_memory_id": "mem_risk_1",
                    "memory_type": "risk",
                    "content_text": "接口测试环境尚未准备",
                    "source_time": "2026-08-02",
                    "entities_text": "王五",
                },
            },
        ]

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        memory_type = filters.get("memory_type")
        results = self.items
        if memory_type:
            results = [
                item
                for item in results
                if item["metadata"].get("memory_type") == memory_type
            ]
        return {"results": results[:top_k]}


class EchoInput(FrozenStrictModel):
    value: str = Field(min_length=1)


class EchoOutput(FrozenStrictModel):
    value: str


class RetryEchoTool(BaseTool):
    name = "retry_echo"
    input_model = EchoInput
    output_model = EchoOutput
    allowed_task_types = frozenset({TaskType.TEMPLATE_ADAPTATION})

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: EchoInput, context: ToolContext) -> EchoOutput:
        del context
        self.calls += 1
        if self.calls == 1:
            raise RetryableToolError("temporary failure")
        return EchoOutput(value=arguments.value)


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_enforces_permission(self) -> None:
        registry = ToolRegistry([TemplateAdaptationTool()])
        result = await registry.execute(
            ToolCall(
                tool_name="template_adaptation",
                arguments={"summary_request": "结构化摘要"},
            ),
            context=ToolContext(
                run_id="run", task_id="task", task_type=TaskType.TEMPLATE_ADAPTATION
            ),
            allowed_tools=(),
        )
        self.assertEqual(ToolStatus.FAILED, result.status)
        self.assertIn("not allowed", result.error or "")

    async def test_registry_retries_only_retryable_errors(self) -> None:
        tool = RetryEchoTool()
        registry = ToolRegistry([tool])
        result = await registry.execute(
            ToolCall(tool_name=tool.name, arguments={"value": "ok"}),
            context=ToolContext(
                run_id="run", task_id="task", task_type=TaskType.TEMPLATE_ADAPTATION
            ),
            allowed_tools=(tool.name,),
            retry_attempts=2,
        )
        self.assertEqual(ToolStatus.SUCCESS, result.status)
        self.assertEqual(2, result.attempts)
        self.assertEqual("ok", result.payload["value"])


class ToolAndWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.backend = FakeMemoryBackend()
        self.registry = ToolRegistry(
            [
                TypedMemorySearchTool(
                    self.backend, user_id="user", project="project"
                ),
                EntityQueryTool(self.backend, user_id="user", project="project"),
                TemplateAdaptationTool(),
                CalendarResolveTool(),
            ]
        )
        self.request = SummaryRequest(
            current_material=(
                "张三负责整理接口结果，明天中午12点提交。"
                "李四下周三开始复核，王五记录风险。"
            ),
            source_time="2026-08-05",
            summary_request="生成结构化项目摘要",
            target_entities=["张三"],
        )
        self.bundle = build_raw_material_bundle(
            self.request.current_material,
            max_segment_chars=18,
            overlap_chars=3,
        )

    def context(self, task: SubTask) -> WorkerContext:
        return WorkerContext(
            run_id="run_test",
            request=self.request,
            raw_material=self.bundle,
            task=task,
            tool_registry=self.registry,
        )

    async def test_semantic_worker_queries_each_requested_type(self) -> None:
        task = SubTask(
            task_id="memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            description="Retrieve history.",
            allowed_tools=("typed_memory_search",),
            inputs={"queries": ["接口测试"], "memory_types": ["task", "risk"]},
        )
        result = await SemanticMemoryWorker().execute(self.context(task))
        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(2, len(result.payload["records"]))
        queried_types = {call["filters"]["memory_type"] for call in self.backend.calls}
        self.assertEqual({"task", "risk"}, queried_types)

    async def test_entity_worker_exact_matches_and_one_hop(self) -> None:
        task = SubTask(
            task_id="entity",
            task_type=TaskType.ENTITY_PERSONALIZATION,
            description="Query target entity.",
            allowed_tools=("entity_query",),
        )
        result = await EntityPersonalizationWorker().execute(self.context(task))
        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(["李四"], result.payload["related_entities"])
        self.assertTrue(result.payload["current_mentions"])
        current_refs = [
            ref
            for ref in result.evidence
            if ref.source_kind == SourceKind.CURRENT_MATERIAL
        ]
        memory_refs = [
            ref
            for ref in result.evidence
            if ref.source_kind == SourceKind.LONG_TERM_MEMORY
        ]
        self.assertTrue(all(ref.source_id and ref.segment_id for ref in current_refs))
        self.assertTrue(all(ref.memory_id for ref in memory_refs))

    async def test_template_worker_preserves_default_structured_mode(self) -> None:
        task = SubTask(
            task_id="template",
            task_type=TaskType.TEMPLATE_ADAPTATION,
            description="Select template.",
            allowed_tools=("template_adaptation",),
            inputs={"requested_style": "structured"},
        )
        result = await TemplateAdaptationWorker().execute(self.context(task))
        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual("structured", result.payload["format_kind"])

    async def test_temporal_worker_scans_all_segments_and_resolves_dates(self) -> None:
        task = SubTask(
            task_id="temporal",
            task_type=TaskType.TEMPORAL_REASONING,
            description="Resolve time expressions.",
            allowed_tools=("calendar_resolve",),
            max_tool_calls=8,
        )
        result = await TemporalReasoningWorker().execute(self.context(task))
        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(
            {segment.segment_id for segment in self.bundle.segments},
            set(result.payload["processed_segment_ids"]),
        )
        dates = {
            item["resolution"].get("normalized_date")
            for item in result.payload["resolved"]
        }
        self.assertIn("2026-08-06", dates)
        self.assertIn("2026-08-12", dates)

    async def test_production_temporal_worker_ignores_review_scope_input(self) -> None:
        expressions = find_temporal_expressions(self.bundle)
        task = SubTask(
            task_id="temporal_production",
            task_type=TaskType.TEMPORAL_REASONING,
            description="Resolve every production time expression.",
            allowed_tools=("calendar_resolve",),
            inputs={
                REVIEW_TEMPORAL_SCOPE_INPUT: {
                    "issue_id": "not_a_review_task",
                    "selection_reason": "issue_text",
                    "expressions": [expressions[0]],
                }
            },
            max_tool_calls=8,
        )

        result = await TemporalReasoningWorker().execute(self.context(task))

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertGreater(len(expressions), 1)
        self.assertEqual(len(expressions), len(result.payload["resolved"]))


if __name__ == "__main__":
    unittest.main()
