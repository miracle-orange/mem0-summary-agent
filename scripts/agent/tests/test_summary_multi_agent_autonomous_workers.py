from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    EvidenceRef,
    SourceKind,
    SubTask,
    SummaryRequest,
    TaskType,
    WorkerMode,
    WorkerTerminationReason,
)
from summary_multi_agent.temporal_workload import (  # noqa: E402
    REVIEW_TEMPORAL_SCOPE_INPUT,
    find_temporal_expressions,
)
from summary_multi_agent.tools import (  # noqa: E402
    CalendarResolveTool,
    EntityQueryTool,
    TemplateAdaptationTool,
    ToolRegistry,
    TypedMemorySearchTool,
)
from summary_multi_agent.workers import (  # noqa: E402
    WorkerActionLoop,
    WorkerActionLoopConfig,
    WorkerContext,
    build_autonomous_worker,
)


class SequenceModelClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, temperature: float, max_tokens: int):
        del temperature, max_tokens
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected model call")
        response = self._responses.pop(0)
        return response(prompt) if callable(response) else response


class MemoryBackend:
    def search(self, query, *, filters, top_k):
        del top_k
        memory_type = filters.get("memory_type", "task")
        return {
            "results": [
                {
                    "id": f"memory_{memory_type}",
                    "score": 0.9,
                    "metadata": {
                        "typed_memory_id": f"memory_{memory_type}",
                        "memory_type": memory_type,
                        "content_text": f"{query} 的历史任务证据",
                        "entities_text": "张雨晴|陈一诺",
                        "source_time": "2026-03-12",
                        "evidence_span": "张雨晴负责提交任务",
                    },
                }
            ]
        }


class CountingMemoryBackend(MemoryBackend):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, *, filters, top_k):
        self.queries.append(query)
        return super().search(query, filters=filters, top_k=top_k)


class FailingMemoryBackend:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, *, filters, top_k):
        del filters, top_k
        self.queries.append(query)
        raise RuntimeError("memory backend unavailable")


class FixtureMemoryBackend:
    def __init__(self, results):
        self.results = list(results)

    def search(self, query, *, filters, top_k):
        del query, top_k
        memory_type = filters.get("memory_type")
        return {
            "results": [
                item
                for item in self.results
                if item.get("metadata", {}).get("memory_type") == memory_type
            ]
        }


def action(action_type: str, **values) -> str:
    payload = {
        "schema_version": "summary_multi_agent.worker_action.v2",
        "action": action_type,
        "action_rationale": values.pop("rationale", "执行受控动作"),
        "tool_calls": values.pop("tool_calls", []),
        "selected_observation_ids": values.pop(
            "selected_observation_ids", []
        ),
        "referenced_prior_result_ids": values.pop(
            "referenced_prior_result_ids", []
        ),
        "referenced_prior_evidence_ids": values.pop(
            "referenced_prior_evidence_ids", []
        ),
        "result_summary": values.pop("result_summary", None),
        "warnings": values.pop("warnings", []),
    }
    selected_memory_ids = values.pop("selected_memory_ids", "__absent__")
    if selected_memory_ids != "__absent__":
        payload["selected_memory_ids"] = selected_memory_ids
    if values:
        raise AssertionError(f"unknown action fields: {values}")
    return json.dumps(payload, ensure_ascii=False)


def complete_from_observations(prompt: str) -> str:
    match = re.search(
        r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
        prompt,
        re.DOTALL,
    )
    payload = json.loads(match.group(1))
    observation_ids = [
        item["observation_id"] for item in payload["observations"]
    ]
    return action(
        "complete",
        selected_observation_ids=observation_ids,
        result_summary="Observation 已满足子任务目标",
    )


def make_context(
    *,
    task_type: TaskType,
    tool_registry: ToolRegistry,
    allowed_tools: tuple[str, ...],
    inputs=None,
    current_material="张雨晴明天提交报告。",
    source_time="2026-03-19",
    target_entities=None,
    critical=False,
    max_tool_calls=4,
    task_id=None,
    dependencies=(),
    prior_results=(),
) -> WorkerContext:
    request = SummaryRequest(
        current_material=current_material,
        source_time=source_time,
        summary_request="生成结构化摘要",
        target_entities=target_entities or [],
        personalization_preferences={"summary_style": "structured"},
    )
    bundle = build_raw_material_bundle(
        current_material,
        max_segment_chars=100,
        overlap_ratio=0.0,
    )
    task = SubTask(
        task_id=task_id or f"{task_type.value}_autonomous",
        task_type=task_type,
        description="完成受控专业子任务",
        dependencies=dependencies,
        critical=critical,
        allowed_tools=allowed_tools,
        inputs=inputs or {},
        max_model_calls=4,
        max_tool_calls=max_tool_calls,
    )
    return WorkerContext(
        run_id="run_autonomous_worker",
        request=request,
        raw_material=bundle,
        task=task,
        tool_registry=tool_registry,
        tool_retry_attempts=1,
        prior_results=prior_results,
    )


def make_worker(task_type: TaskType, model, *, fallback=False):
    loop = WorkerActionLoop(
        model_client=model,
        config=WorkerActionLoopConfig(
            max_steps=4,
            max_model_calls=4,
            max_tool_calls=16,
            repair_attempts=0,
        ),
    )
    return build_autonomous_worker(
        task_type=task_type,
        action_loop=loop,
        fallback_to_rule=fallback,
    )


class AutonomousWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_worker_assembles_only_selected_observations(self):
        backend = MemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "报告任务",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_from_observations,
            ]
        )
        context = make_context(
            task_type=TaskType.SEMANTIC_MEMORY,
            tool_registry=registry,
            allowed_tools=("typed_memory_search",),
            inputs={
                "queries": ["报告任务"],
                "memory_types": ["task"],
                "top_k_per_type": 5,
            },
            max_tool_calls=1,
        )

        result = await make_worker(TaskType.SEMANTIC_MEMORY, model).execute(
            context
        )

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual("memory_task", result.payload["records"][0]["memory_id"])
        self.assertEqual("memory_task", result.evidence[0].memory_id)
        self.assertEqual(WorkerMode.MODEL, result.worker_execution.mode)
        self.assertEqual(
            WorkerTerminationReason.GOAL_SATISFIED,
            result.worker_execution.termination_reason,
        )

    async def test_semantic_worker_can_select_individual_memory_records(self):
        backend = FixtureMemoryBackend(
            [
                {
                    "id": "relevant",
                    "score": 0.9,
                    "metadata": {
                        "typed_memory_id": "relevant",
                        "memory_type": "task",
                        "content_text": "与当前接口联调直接相关的历史任务",
                        "source_time": "2026-03-12",
                    },
                },
                {
                    "id": "unrelated",
                    "score": 0.8,
                    "metadata": {
                        "typed_memory_id": "unrelated",
                        "memory_type": "task",
                        "content_text": "无关的办公用品采购任务",
                        "source_time": "2026-03-12",
                    },
                },
            ]
        )
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )

        def select_relevant(prompt: str) -> str:
            match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            payload = json.loads(match.group(1))
            self.assertEqual(
                "接口联调", payload["observations"][0]["arguments"]["query"]
            )
            self.assertEqual(
                {"relevant", "unrelated"},
                {
                    item["memory_id"]
                    for item in payload["shared_observation_state"]["records"]
                },
            )
            return action(
                "complete",
                selected_observation_ids=[
                    payload["observations"][0]["observation_id"]
                ],
                selected_memory_ids=["relevant"],
                result_summary="只采用与当前任务相关的记录",
            )

        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "接口联调",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                select_relevant,
            ]
        )
        result = await make_worker(TaskType.SEMANTIC_MEMORY, model).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={"queries": ["接口联调"]},
                max_tool_calls=1,
            )
        )

        self.assertEqual(["relevant"], result.payload["selected_memory_ids"])
        self.assertEqual("record_ids", result.payload["selection_mode"])
        self.assertEqual(
            {"relevant", "unrelated"},
            set(result.payload["available_record_ids"]),
        )
        self.assertEqual(["relevant"], [item.memory_id for item in result.evidence])
        self.assertEqual(2, len(result.tool_results[0].payload["records"]))

    async def test_semantic_worker_explicit_empty_selection_is_legal(self):
        backend = MemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )

        def complete_empty(prompt: str) -> str:
            match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            payload = json.loads(match.group(1))
            return action(
                "complete_empty",
                selected_observation_ids=[
                    payload["observations"][0]["observation_id"]
                ],
                selected_memory_ids=[],
                result_summary="检索有返回，但没有与当前任务相关的记录",
            )

        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "不存在的相关事实",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_empty,
            ]
        )
        result = await make_worker(TaskType.SEMANTIC_MEMORY, model).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={"queries": ["不存在的相关事实"]},
                max_tool_calls=1,
            )
        )

        self.assertTrue(result.is_empty)
        self.assertEqual([], result.payload["records"])
        self.assertEqual("explicit_empty", result.payload["selection_mode"])

    async def test_semantic_worker_rejects_unknown_selected_memory_id(self):
        backend = MemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )

        def select_unknown(prompt: str) -> str:
            match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            payload = json.loads(match.group(1))
            return action(
                "complete",
                selected_observation_ids=[
                    payload["observations"][0]["observation_id"]
                ],
                selected_memory_ids=["unknown"],
            )

        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "报告任务",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                select_unknown,
            ]
        )
        result = await make_worker(TaskType.SEMANTIC_MEMORY, model).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={"queries": ["报告任务"]},
                max_tool_calls=1,
            )
        )

        self.assertEqual(AgentStatus.FAILED, result.status)
        self.assertIn("outside selected successful", result.error)

    async def test_semantic_read_merge_only_exact_same_source_records(self):
        base_metadata = {
            "memory_type": "task",
            "content_text": "李明负责完成接口联调",
            "history_source_id": "meeting-1",
            "source_time": "2026-03-12",
            "memory_status": "pending",
            "structured_fields_json": json.dumps(
                {"owner": "李明", "task": "完成接口联调"},
                ensure_ascii=False,
            ),
        }
        backend = FixtureMemoryBackend(
            [
                {
                    "id": "same-low",
                    "score": 0.7,
                    "metadata": {**base_metadata, "typed_memory_id": "same-low"},
                },
                {
                    "id": "same-high",
                    "score": 0.9,
                    "metadata": {**base_metadata, "typed_memory_id": "same-high"},
                },
                {
                    "id": "different-date",
                    "score": 0.8,
                    "metadata": {
                        **base_metadata,
                        "typed_memory_id": "different-date",
                        "source_time": "2026-03-13",
                    },
                },
            ]
        )
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "接口联调",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_from_observations,
            ]
        )
        result = await make_worker(TaskType.SEMANTIC_MEMORY, model).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={"queries": ["接口联调"]},
                max_tool_calls=1,
            )
        )

        self.assertEqual(
            {"same-high", "different-date"},
            {item["memory_id"] for item in result.payload["records"]},
        )
        self.assertEqual(1, len(result.payload["equivalent_record_merges"]))

    async def test_dependent_worker_preserves_result_and_evidence_lineage(self):
        backend = MemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        prior_evidence = EvidenceRef(
            source_kind=SourceKind.LONG_TERM_MEMORY,
            memory_id="prior_memory",
            evidence_text="历史结果提示需要补查风险",
        )
        prior = AgentResult(
            task_id="history_tasks",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={"records": [{"content": "需要补查风险"}]},
            evidence=(prior_evidence,),
        )

        def complete_dependency(prompt: str) -> str:
            match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            payload = json.loads(match.group(1))
            return action(
                "complete",
                selected_observation_ids=[
                    item["observation_id"]
                    for item in payload["observations"]
                ],
                referenced_prior_result_ids=["history_tasks"],
                referenced_prior_evidence_ids=[prior_evidence.evidence_id],
                result_summary="结合上游任务结果完成风险补查",
            )

        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "依据历史任务补查风险",
                                "memory_types": ["risk"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                    referenced_prior_result_ids=["history_tasks"],
                    referenced_prior_evidence_ids=[prior_evidence.evidence_id],
                ),
                complete_dependency,
            ]
        )
        context = make_context(
            task_type=TaskType.SEMANTIC_MEMORY,
            tool_registry=registry,
            allowed_tools=("typed_memory_search",),
            inputs={
                "queries": ["依据历史任务补查风险"],
                "memory_types": ["risk"],
                "top_k_per_type": 5,
            },
            max_tool_calls=1,
            task_id="risk_followup",
            dependencies=("history_tasks",),
            prior_results=(prior,),
        )

        result = await make_worker(
            TaskType.SEMANTIC_MEMORY, model
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(("history_tasks",), result.consumed_result_ids)
        self.assertIn(prior_evidence.evidence_id, result.consumed_evidence_ids)
        self.assertIn(
            prior_evidence.evidence_id,
            {item.evidence_id for item in result.evidence},
        )
        self.assertEqual(
            ("history_tasks",),
            result.worker_execution.consumed_prior_result_ids,
        )

    async def test_entity_worker_keeps_grounded_current_and_memory_evidence(self):
        backend = MemoryBackend()
        registry = ToolRegistry(
            [EntityQueryTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "entity_query",
                            "arguments": {
                                "entities": ["张雨晴"],
                                "include_one_hop": True,
                                "top_k_per_entity": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_from_observations,
            ]
        )
        context = make_context(
            task_type=TaskType.ENTITY_PERSONALIZATION,
            tool_registry=registry,
            allowed_tools=("entity_query",),
            target_entities=["张雨晴"],
            inputs={"include_one_hop": True, "top_k_per_entity": 5},
            max_tool_calls=1,
        )

        result = await make_worker(
            TaskType.ENTITY_PERSONALIZATION, model
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertTrue(result.payload["current_mentions"])
        self.assertTrue(any(item.memory_id for item in result.evidence))
        self.assertTrue(any(item.start_offset is not None for item in result.evidence))

    async def test_template_worker_uses_observed_template_payload(self):
        registry = ToolRegistry([TemplateAdaptationTool()])
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "template_adaptation",
                            "arguments": {
                                "summary_request": "生成结构化摘要",
                                "requested_style": "structured",
                                "preferences": {"summary_style": "structured"},
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_from_observations,
            ]
        )
        context = make_context(
            task_type=TaskType.TEMPLATE_ADAPTATION,
            tool_registry=registry,
            allowed_tools=("template_adaptation",),
            inputs={"requested_style": "structured"},
            max_tool_calls=1,
        )

        result = await make_worker(
            TaskType.TEMPLATE_ADAPTATION, model
        ).execute(context)

        self.assertEqual("structured_general_v3", result.payload["template_name"])
        self.assertEqual(2, result.worker_execution.final_budget.model_calls_used)

    async def test_temporal_worker_requires_complete_deterministic_coverage(self):
        registry = ToolRegistry([CalendarResolveTool()])
        context = make_context(
            task_type=TaskType.TEMPORAL_REASONING,
            tool_registry=registry,
            allowed_tools=("calendar_resolve",),
            current_material="负责人昨天确认，明天提交报告。",
            critical=True,
            max_tool_calls=2,
        )
        expressions = find_temporal_expressions(context.raw_material)
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "calendar_resolve",
                            "arguments": {
                                "expression": item["expression"],
                                "source_time": context.request.source_time,
                                "direction": item["direction"],
                            },
                            "timeout_seconds": 30,
                        }
                        for item in expressions
                    ],
                ),
                complete_from_observations,
            ]
        )

        result = await make_worker(
            TaskType.TEMPORAL_REASONING, model
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(len(expressions), len(result.payload["resolved"]))
        self.assertEqual(len(expressions), len(result.evidence))

    async def test_review_temporal_worker_only_resolves_trusted_issue_scope(self):
        material = "负责人昨天确认。张雨晴明天提交报告。下周一复盘。"
        bundle = build_raw_material_bundle(
            material,
            max_segment_chars=100,
            overlap_ratio=0.0,
        )
        selected = next(
            item
            for item in find_temporal_expressions(bundle)
            if item["expression"] == "明天"
        )
        context = make_context(
            task_type=TaskType.TEMPORAL_REASONING,
            task_id="review_verify_1_issue_time",
            tool_registry=ToolRegistry([CalendarResolveTool()]),
            allowed_tools=("calendar_resolve",),
            current_material=material,
            critical=False,
            max_tool_calls=1,
            inputs={
                REVIEW_TEMPORAL_SCOPE_INPUT: {
                    "issue_id": "issue_time",
                    "selection_reason": "issue_text",
                    "expressions": [selected],
                }
            },
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "calendar_resolve",
                            "arguments": {
                                "expression": selected["expression"],
                                "source_time": context.request.source_time,
                                "direction": selected["direction"],
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                complete_from_observations,
            ]
        )

        result = await make_worker(
            TaskType.TEMPORAL_REASONING, model
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(["明天"], [item["expression"] for item in result.payload["resolved"]])
        self.assertEqual(1, len(result.evidence))
        self.assertEqual(1, result.worker_execution.final_budget.tool_calls_used)
        self.assertIn('"expression": "明天"', model.calls[0])
        self.assertNotIn('"expression": "昨天"', model.calls[0])
        self.assertNotIn('"expression": "下周一"', model.calls[0])

    async def test_model_failure_uses_explicit_audited_rule_fallback(self):
        registry = ToolRegistry([TemplateAdaptationTool()])
        model = SequenceModelClient(["not-json"])
        context = make_context(
            task_type=TaskType.TEMPLATE_ADAPTATION,
            tool_registry=registry,
            allowed_tools=("template_adaptation",),
            inputs={"requested_style": "structured"},
            max_tool_calls=1,
        )

        result = await make_worker(
            TaskType.TEMPLATE_ADAPTATION,
            model,
            fallback=True,
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(
            WorkerMode.MODEL_WITH_RULE_FALLBACK,
            result.worker_execution.mode,
        )
        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            result.worker_execution.termination_reason,
        )
        self.assertTrue(result.worker_execution.fallback_reason)
        self.assertTrue(any("rule fallback" in item for item in result.warnings))

    async def test_semantic_fallback_reuses_successful_search_and_only_fills_missing_query(self):
        backend = CountingMemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "历史任务",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                "not-json",
            ]
        )
        result = await make_worker(
            TaskType.SEMANTIC_MEMORY,
            model,
            fallback=True,
        ).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={
                    "queries": ["历史任务", "历史风险"],
                    "memory_types": ["task"],
                    "top_k_per_type": 5,
                },
                max_tool_calls=2,
            )
        )

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(["历史任务", "历史风险"], backend.queries)
        self.assertEqual(1, result.payload["reused_search_count"])
        self.assertEqual(1, result.payload["new_search_count"])
        self.assertTrue(result.payload["retrieval_complete"])
        self.assertEqual("rule_recovery", result.payload["execution_origin"])
        self.assertFalse(result.payload["record_selection_completed"])
        self.assertEqual(
            2,
            result.worker_execution.final_budget.tool_calls_used,
        )

    async def test_semantic_fallback_marks_incomplete_when_remaining_budget_is_zero(self):
        backend = CountingMemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "历史任务",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                "not-json",
            ]
        )
        context = make_context(
            task_type=TaskType.SEMANTIC_MEMORY,
            tool_registry=registry,
            allowed_tools=("typed_memory_search",),
            inputs={
                "queries": ["历史任务", "历史风险"],
                "memory_types": ["task"],
                "top_k_per_type": 5,
            },
            max_tool_calls=1,
        )
        result = await make_worker(
            TaskType.SEMANTIC_MEMORY,
            model,
            fallback=True,
        ).execute(context)

        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual(["历史任务"], backend.queries)
        self.assertFalse(result.payload["retrieval_complete"])
        self.assertEqual(["历史风险"], result.payload["incomplete_queries"])
        self.assertTrue(any("did not complete" in item for item in result.warnings))

    async def test_semantic_fallback_without_success_is_not_reported_as_empty(self):
        backend = FailingMemoryBackend()
        registry = ToolRegistry(
            [TypedMemorySearchTool(backend, user_id="u", project="p")]
        )
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "typed_memory_search",
                            "arguments": {
                                "query": "历史任务",
                                "memory_types": ["task"],
                                "top_k_per_type": 5,
                            },
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                "not-json",
            ]
        )
        result = await make_worker(
            TaskType.SEMANTIC_MEMORY,
            model,
            fallback=True,
        ).execute(
            make_context(
                task_type=TaskType.SEMANTIC_MEMORY,
                tool_registry=registry,
                allowed_tools=("typed_memory_search",),
                inputs={
                    "queries": ["历史任务"],
                    "memory_types": ["task"],
                    "top_k_per_type": 5,
                },
                max_tool_calls=1,
            )
        )

        self.assertEqual(AgentStatus.FAILED, result.status)
        self.assertFalse(result.is_empty)
        self.assertEqual(["历史任务"], backend.queries)
        self.assertIn("no budget", result.error)


if __name__ == "__main__":
    unittest.main()
