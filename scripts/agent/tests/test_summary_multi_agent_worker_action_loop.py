from __future__ import annotations

import asyncio
import json
import re
import sys
import unittest
from pathlib import Path

from pydantic import Field, ValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.adapters import ContextOverflowError  # noqa: E402
from summary_multi_agent.input_projection import (  # noqa: E402
    MemoryRecordCatalog,
    project_records_in_payload,
)
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    EvidenceRef,
    FrozenStrictModel,
    SourceKind,
    SubTask,
    SummaryRequest,
    TaskType,
    ToolResult,
    ToolStatus,
    WorkerAction,
    WorkerAgentInput,
    WorkerBudgetSnapshot,
    WorkerObservation,
    WorkerTerminationReason,
)
from summary_multi_agent.tools.registry import (  # noqa: E402
    BaseTool,
    ToolContext,
    ToolRegistry,
)
from summary_multi_agent.workers.action_loop import (  # noqa: E402
    WorkerActionLoop,
    WorkerActionLoopConfig,
    build_worker_action_prompt,
)
from summary_multi_agent.workers.base import WorkerContext  # noqa: E402


class EchoInput(FrozenStrictModel):
    value: str = Field(min_length=1)


class EchoOutput(FrozenStrictModel):
    echoed: str = Field(min_length=1)


class EchoTool(BaseTool):
    """Return a value for action-loop tests."""

    name = "echo"
    input_model = EchoInput
    output_model = EchoOutput
    allowed_task_types = frozenset({TaskType.TEMPLATE_ADAPTATION})

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def execute(
        self, arguments: EchoInput, context: ToolContext
    ) -> EchoOutput:
        del context
        self.calls.append(arguments.value)
        if self.fail:
            raise RuntimeError("echo failed")
        return EchoOutput(echoed=arguments.value)


class SequenceModelClient:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, prompt, *, temperature, max_tokens):
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(prompt)
        return response


class BlockingModelClient:
    async def complete(self, prompt, *, temperature, max_tokens):
        del prompt, temperature, max_tokens
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class ConcurrentModelClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def complete(self, prompt, *, temperature, max_tokens):
        del prompt, temperature, max_tokens
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return action("complete_empty", summary="并发隔离测试")
        finally:
            self.active -= 1


def action(
    action_type: str,
    *,
    tool_calls: list[dict] | None = None,
    selected_observation_ids: list[str] | None = None,
    referenced_prior_result_ids: list[str] | None = None,
    referenced_prior_evidence_ids: list[str] | None = None,
    summary: str | None = None,
) -> str:
    return json.dumps(
        {
            "schema_version": "summary_multi_agent.worker_action.v2",
            "action": action_type,
            "action_rationale": f"选择 {action_type} 以推进受控任务。",
            "tool_calls": tool_calls or [],
            "selected_observation_ids": selected_observation_ids or [],
            "referenced_prior_result_ids": referenced_prior_result_ids or [],
            "referenced_prior_evidence_ids": referenced_prior_evidence_ids or [],
            "result_summary": summary,
            "warnings": [],
        },
        ensure_ascii=False,
    )


def make_context(
    tool: EchoTool,
    *,
    task_id: str = "template_agent",
    max_model_calls: int = 4,
    max_tool_calls: int = 2,
    dependencies: tuple[str, ...] = (),
) -> WorkerContext:
    request = SummaryRequest(
        current_material="完整当前材料第一行。\n完整当前材料第二行。",
        source_time="2026-08-08",
        summary_request="选择合适的摘要结构",
    )
    bundle = build_raw_material_bundle(
        request.current_material, max_segment_chars=100
    )
    task = SubTask(
        task_id=task_id,
        task_type=TaskType.TEMPLATE_ADAPTATION,
        description="根据请求选择输出结构。",
        dependencies=dependencies,
        allowed_tools=("echo",),
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
    )
    return WorkerContext(
        run_id="run_action_loop",
        request=request,
        raw_material=bundle,
        task=task,
        tool_registry=ToolRegistry([tool]),
        tool_retry_attempts=1,
    )


class WorkerActionSchemaTests(unittest.TestCase):
    def test_call_tool_requires_calls(self) -> None:
        with self.assertRaises(ValidationError):
            WorkerAction(
                action="call_tool",
                action_rationale="需要工具",
            )

    def test_terminal_action_rejects_tool_calls(self) -> None:
        with self.assertRaises(ValidationError):
            WorkerAction(
                action="complete",
                action_rationale="已经完成",
                tool_calls=(
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "x"},
                    },
                ),
            )

    def test_repeated_search_records_are_catalogued_once_without_losing_membership(self) -> None:
        tool = EchoTool()
        context = make_context(tool)
        content = "unique-history-business-payload-" * 40
        record = {
            "memory_id": "memory-shared",
            "memory_type": "task",
            "content": content,
            "score": 0.91,
            "metadata": {
                "content_text": content,
                "history_source_id": "history-1",
                "source_time": "2026-08-01",
                "unknown_business_field": {"keep": True},
            },
        }
        observations = tuple(
            WorkerObservation(
                step_index=index + 1,
                tool_result=ToolResult(
                    call_id=f"call-{index}",
                    tool_name="typed_memory_search",
                    status=ToolStatus.SUCCESS,
                    payload={"records": [record]},
                ),
            )
            for index in range(8)
        )
        semantic_task = context.task.model_copy(
            update={
                "task_type": TaskType.SEMANTIC_MEMORY,
                "allowed_tools": ("typed_memory_search",),
                "inputs": {
                    "queries": [f"query-{index}" for index in range(8)],
                    "memory_types": ["task"],
                    "top_k_per_type": 10,
                },
                "max_tool_calls": 8,
            }
        )
        prompt = build_worker_action_prompt(
            WorkerAgentInput(
                task=semantic_task,
                request=context.request,
                raw_material=context.raw_material,
                prior_results=(),
                tool_schemas={},
                observations=observations,
                budget=WorkerBudgetSnapshot(
                    step_limit=6,
                    model_call_limit=4,
                    tool_call_limit=8,
                    steps_used=1,
                    model_calls_used=1,
                    tool_calls_used=8,
                ),
            )
        )
        payload = json.loads(
            re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            ).group(1)
        )

        catalog = payload["shared_observation_state"]
        self.assertEqual(8, catalog["record_occurrence_count"])
        self.assertEqual(1, catalog["unique_memory_id_count"])
        self.assertEqual(1, catalog["unique_record_version_count"])
        self.assertEqual(8, len(payload["observations"]))
        self.assertEqual("memory-shared", catalog["records"][0]["memory_id"])
        self.assertEqual(8, len(catalog["records"][0]["matches"]))
        self.assertEqual(
            set(range(8)),
            {match[0] for match in catalog["records"][0]["matches"]},
        )
        self.assertTrue(all(match[1] == 0.91 for match in catalog["records"][0]["matches"]))
        self.assertTrue(all("record_refs" not in item for item in payload["observations"]))
        self.assertEqual(1, prompt.count(content))
        self.assertTrue(
            catalog["records"][0]["record"]["metadata"][
                "unknown_business_field"
            ]["keep"]
        )
        legacy_observations = [
            {
                "tool_result": {
                    "call_id": f"call-{index}",
                    "tool_name": "typed_memory_search",
                    "status": "success",
                    "payload": {"records": [record]},
                }
            }
            for index in range(8)
        ]
        legacy_text = json.dumps(
            {"observations": legacy_observations},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        projected_text = json.dumps(
            {
                "observations": payload["observations"],
                "shared_observation_state": catalog,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertLess(len(projected_text), len(legacy_text) * 0.5)

    def test_shared_state_retains_conflicting_versions_and_failed_search(self) -> None:
        context = make_context(EchoTool())
        first = {
            "memory_id": "memory-1",
            "memory_type": "task",
            "content": "申请测试资源",
            "score": 0.8,
            "metadata": {
                "typed_memory_id": "memory-1",
                "memory_type": "task",
                "structured_fields_json": '{"owner":"甲"}',
                "unknown_business_field": "keep",
            },
        }
        second = {**first, "content": "申请生产资源", "score": 0.6}
        observations = (
            WorkerObservation(
                observation_id="observation-1",
                step_index=1,
                tool_result=ToolResult(
                    call_id="call-1",
                    tool_name="typed_memory_search",
                    status=ToolStatus.SUCCESS,
                    payload={"records": [first, first]},
                ),
            ),
            WorkerObservation(
                observation_id="observation-2",
                step_index=1,
                tool_result=ToolResult(
                    call_id="call-2",
                    tool_name="typed_memory_search",
                    status=ToolStatus.SUCCESS,
                    payload={"records": [second], "warnings": ["partial index"]},
                ),
            ),
            WorkerObservation(
                observation_id="observation-3",
                step_index=1,
                tool_result=ToolResult(
                    call_id="call-3",
                    tool_name="typed_memory_search",
                    status=ToolStatus.FAILED,
                    error="search failed",
                ),
            ),
        )
        semantic_task = context.task.model_copy(
            update={
                "task_type": TaskType.SEMANTIC_MEMORY,
                "allowed_tools": ("typed_memory_search",),
            }
        )
        agent_input = WorkerAgentInput(
            task=semantic_task,
            request=context.request,
            raw_material=context.raw_material,
            prior_results=(),
            tool_schemas={},
            observations=observations,
            budget=WorkerBudgetSnapshot(
                step_limit=6, model_call_limit=4, tool_call_limit=3,
                steps_used=1, model_calls_used=1, tool_calls_used=3,
            ),
        )
        prompt = build_worker_action_prompt(agent_input)
        block = prompt.split("<worker_agent_input>\n", 1)[1].split("\n</worker_agent_input>", 1)[0]
        payload = json.loads(block)
        shared = payload["shared_observation_state"]
        self.assertEqual(3, shared["record_occurrence_count"])
        self.assertEqual(1, shared["unique_memory_id_count"])
        self.assertEqual(2, shared["unique_record_version_count"])
        self.assertEqual(1, len(shared["record_version_conflicts"]))
        self.assertEqual(2, len(shared["records"]))
        self.assertEqual("甲", shared["records"][0]["record"]["metadata"]["structured_fields_json"]["owner"])
        self.assertEqual("keep", shared["records"][1]["record"]["metadata"]["unknown_business_field"])
        self.assertEqual("partial index", payload["observations"][1]["warnings"][0])
        self.assertEqual("search failed", payload["observations"][2]["error"])
        self.assertNotIn("records", payload["observations"][0])
        self.assertEqual(2, len(observations[0].tool_result.payload["records"]))

    def test_overlapping_searches_use_one_shared_candidate_set(self) -> None:
        context = make_context(EchoTool())
        records = [
            {
                "memory_id": f"memory-{index}",
                "memory_type": "task",
                "content": f"项目事项 {index}：" + "需要核对负责人、进度和约束。" * 5,
                "score": 0.9 - index / 1000,
                "metadata": {
                    "content_text": f"项目事项 {index}：" + "需要核对负责人、进度和约束。" * 5,
                    "source_time": "2026-07-21",
                    "history_source_id": "history-1",
                    "structured_fields_json": json.dumps(
                        {"owner": f"负责人{index}", "status": "pending"},
                        ensure_ascii=False,
                    ),
                },
            }
            for index in range(89)
        ]
        observations = tuple(
            WorkerObservation(
                observation_id=f"observation-{query_index}",
                step_index=1,
                tool_result=ToolResult(
                    call_id=f"call-{query_index}",
                    tool_name="typed_memory_search",
                    status=ToolStatus.SUCCESS,
                    payload={
                        "records": [
                            records[(query_index * 7 + offset) % len(records)]
                            for offset in range(46)
                        ]
                    },
                ),
            )
            for query_index in range(8)
        )
        semantic_task = context.task.model_copy(
            update={
                "task_type": TaskType.SEMANTIC_MEMORY,
                "allowed_tools": ("typed_memory_search",),
                "max_tool_calls": 8,
            }
        )
        agent_input = WorkerAgentInput(
            task=semantic_task,
            request=context.request,
            raw_material=context.raw_material,
            prior_results=(),
            tool_schemas={},
            observations=observations,
            budget=WorkerBudgetSnapshot(
                step_limit=6, model_call_limit=4, tool_call_limit=8,
                steps_used=1, model_calls_used=1, tool_calls_used=8,
            ),
        )
        prompt = build_worker_action_prompt(agent_input)
        payload = json.loads(
            prompt.split("<worker_agent_input>\n", 1)[1].split(
                "\n</worker_agent_input>", 1
            )[0]
        )
        shared = payload["shared_observation_state"]
        self.assertEqual(368, shared["record_occurrence_count"])
        self.assertEqual(89, shared["unique_memory_id_count"])
        self.assertEqual(89, len(shared["records"]))
        self.assertEqual(368, sum(len(item["matches"]) for item in shared["records"]))

        old_catalog = MemoryRecordCatalog()
        old_observations = [
            {
                "observation_id": item.observation_id,
                "tool_result": {
                    "payload": project_records_in_payload(
                        item.tool_result.payload,
                        record_catalog=old_catalog,
                    )
                },
            }
            for item in observations
        ]
        old_view = json.dumps(
            {"observations": old_observations, "memory_record_catalog": old_catalog.payload},
            ensure_ascii=False,
        )
        new_view = json.dumps(
            {"observations": payload["observations"], "shared_observation_state": shared},
            ensure_ascii=False,
        )
        self.assertLess(len(new_view), len(old_view) * 0.7)

    def test_call_tool_rejects_record_selection(self) -> None:
        with self.assertRaises(ValidationError):
            WorkerAction(
                action="call_tool",
                action_rationale="先检索",
                tool_calls=(
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "x"},
                    },
                ),
                selected_memory_ids=("memory-1",),
            )


class WorkerActionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_observation_changes_next_action_and_completes(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "observed"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                action("complete", summary="观察结果满足目标"),
            ]
        )
        outcome = await WorkerActionLoop(model_client=model).run(
            make_context(tool, max_model_calls=2, max_tool_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.GOAL_SATISFIED,
            outcome.termination_reason,
        )
        self.assertEqual(["observed"], tool.calls)
        self.assertEqual(2, len(outcome.steps))
        self.assertEqual(1, len(outcome.observations))
        self.assertIn("observed", model.calls[1]["prompt"])
        self.assertEqual(2, outcome.final_budget.model_calls_used)
        self.assertEqual(1, outcome.final_budget.tool_calls_used)

    async def test_unauthorized_tool_is_repaired_without_execution(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "not_allowed",
                            "arguments": {"value": "x"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                action("complete_empty", summary="无需工具即可合法结束"),
            ]
        )
        loop = WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=1),
        )
        outcome = await loop.run(make_context(tool, max_model_calls=2))

        self.assertEqual(
            WorkerTerminationReason.LEGAL_EMPTY_RESULT,
            outcome.termination_reason,
        )
        self.assertEqual([], tool.calls)
        self.assertEqual("validation_error", outcome.model_attempts[0].failure_kind)
        self.assertEqual("repair", outcome.model_attempts[1].stage)

    async def test_invalid_arguments_fail_closed_after_bounded_repair(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {},
                            "timeout_seconds": 30,
                        }
                    ],
                )
            ]
        )
        loop = WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=0),
        )
        outcome = await loop.run(make_context(tool, max_model_calls=1))

        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            outcome.termination_reason,
        )
        self.assertEqual([], tool.calls)
        self.assertIn("ValidationError", outcome.error)

    async def test_tool_failure_can_request_replan(self) -> None:
        tool = EchoTool(fail=True)
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "will-fail"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                action("request_replan", summary="授权工具失败，需要重规划"),
            ]
        )
        outcome = await WorkerActionLoop(model_client=model).run(
            make_context(tool, max_model_calls=2, max_tool_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.REPLAN_REQUESTED,
            outcome.termination_reason,
        )
        self.assertEqual("failed", outcome.observations[0].tool_result.status.value)

    async def test_tool_budget_violation_can_be_repaired_to_partial(self) -> None:
        tool = EchoTool()

        def partial_from_observation(prompt: str) -> str:
            match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            payload = json.loads(match.group(1))
            observation_id = payload["observations"][0]["observation_id"]
            return action(
                "partial",
                summary="保留已有观察并停止",
                selected_observation_ids=[observation_id],
            )

        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "first"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "over-budget"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                partial_from_observation,
            ]
        )
        outcome = await WorkerActionLoop(model_client=model).run(
            make_context(tool, max_model_calls=3, max_tool_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.PARTIAL_WITH_EVIDENCE,
            outcome.termination_reason,
        )
        self.assertEqual(["first"], tool.calls)
        self.assertEqual(3, outcome.final_budget.model_calls_used)
        self.assertEqual(1, outcome.final_budget.tool_calls_used)

    async def test_tool_budget_exhaustion_has_distinct_termination(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "first"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "over-budget"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
            ]
        )
        outcome = await WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=0),
        ).run(make_context(tool, max_model_calls=2, max_tool_calls=1))

        self.assertEqual(
            WorkerTerminationReason.TOOL_BUDGET_EXHAUSTED,
            outcome.termination_reason,
        )
        self.assertEqual(["first"], tool.calls)

    async def test_step_limit_is_terminal(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "first"},
                            "timeout_seconds": 30,
                        }
                    ],
                )
            ]
        )
        outcome = await WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(max_steps=1),
        ).run(make_context(tool, max_model_calls=2, max_tool_calls=2))

        self.assertEqual(
            WorkerTerminationReason.STEP_LIMIT,
            outcome.termination_reason,
        )
        self.assertEqual(1, outcome.final_budget.steps_used)

    async def test_model_timeout_has_distinct_termination(self) -> None:
        model = SequenceModelClient([asyncio.TimeoutError()])
        outcome = await WorkerActionLoop(model_client=model).run(
            make_context(EchoTool(), max_model_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.TIMEOUT,
            outcome.termination_reason,
        )
        self.assertEqual("timeout", outcome.model_attempts[0].failure_kind)

    async def test_context_overflow_does_not_enter_repair_loop(self) -> None:
        model = SequenceModelClient(
            [ContextOverflowError("context_overflow: configured window exceeded")]
        )
        outcome = await WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=3),
        ).run(make_context(EchoTool(), max_model_calls=4))

        self.assertEqual(
            WorkerTerminationReason.CONTEXT_OVERFLOW,
            outcome.termination_reason,
        )
        self.assertEqual(1, len(model.calls))
        self.assertEqual(1, len(outcome.model_attempts))
        self.assertEqual(
            "context_overflow", outcome.model_attempts[0].failure_kind
        )

    async def test_model_budget_exhaustion_is_terminal(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "only-call"},
                            "timeout_seconds": 30,
                        }
                    ],
                )
            ]
        )
        outcome = await WorkerActionLoop(model_client=model).run(
            make_context(tool, max_model_calls=1, max_tool_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.MODEL_BUDGET_EXHAUSTED,
            outcome.termination_reason,
        )
        self.assertEqual(1, len(outcome.steps))

    async def test_schema_failure_after_observation_preserves_budget_audit(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [
                action(
                    "call_tool",
                    tool_calls=[
                        {
                            "tool_name": "echo",
                            "arguments": {"value": "first"},
                            "timeout_seconds": 30,
                        }
                    ],
                ),
                "not-json",
            ]
        )
        loop = WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=0),
        )
        outcome = await loop.run(
            make_context(tool, max_model_calls=2, max_tool_calls=1)
        )

        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            outcome.termination_reason,
        )
        self.assertEqual(1, len(outcome.steps))
        self.assertEqual(2, outcome.final_budget.model_calls_used)
        self.assertEqual(1, outcome.final_budget.tool_calls_used)

    async def test_prompt_preserves_full_current_material_once(self) -> None:
        tool = EchoTool()
        model = SequenceModelClient(
            [action("complete_empty", summary="测试完整输入")]
        )
        context = make_context(tool, max_model_calls=1)
        await WorkerActionLoop(model_client=model).run(context)

        match = re.search(
            r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
            model.calls[0]["prompt"],
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertNotIn("current_material", payload["request"])
        current_block = model.calls[0]["prompt"].split(
            "<current_material>\n", 1
        )[1].split("\n</current_material>", 1)[0]
        self.assertEqual(context.request.current_material, current_block)
        self.assertEqual(
            1, model.calls[0]["prompt"].count(context.request.current_material)
        )

    async def test_prior_result_outside_dependencies_is_rejected(self) -> None:
        context = make_context(EchoTool())
        prior = AgentResult(
            task_id="undeclared",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
        )
        with self.assertRaisesRegex(ValueError, "exactly match declared dependencies"):
            await WorkerActionLoop(
                model_client=SequenceModelClient(
                    [action("complete_empty")]
                )
            ).run(context, prior_results=(prior,))

    async def test_declared_prior_result_is_exposed_and_audited(self) -> None:
        context = make_context(
            EchoTool(),
            dependencies=("upstream_memory",),
            max_model_calls=2,
            max_tool_calls=1,
        )
        prior = AgentResult(
            task_id="upstream_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={"records": [{"content": "历史任务要求复核风险"}]},
        )

        def call_from_prior(prompt: str) -> str:
            self.assertIn("历史任务要求复核风险", prompt)
            self.assertNotIn('"worker_execution"', prompt)
            return action(
                "call_tool",
                tool_calls=[
                    {
                        "tool_name": "echo",
                        "arguments": {"value": "依据上游结果复核风险"},
                        "timeout_seconds": 30,
                    }
                ],
                referenced_prior_result_ids=["upstream_memory"],
            )

        def complete_with_links(prompt: str) -> str:
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
                referenced_prior_result_ids=["upstream_memory"],
                summary="已根据上游结构化结果完成复核",
            )

        outcome = await WorkerActionLoop(
            model_client=SequenceModelClient(
                [call_from_prior, complete_with_links]
            ),
            config=WorkerActionLoopConfig(repair_attempts=0),
        ).run(context, prior_results=(prior,))

        self.assertEqual(
            WorkerTerminationReason.GOAL_SATISFIED,
            outcome.termination_reason,
        )
        self.assertEqual(
            ("upstream_memory",), outcome.consumed_prior_result_ids
        )

    async def test_dependent_worker_cannot_complete_without_consuming_prior(self) -> None:
        context = make_context(
            EchoTool(),
            dependencies=("upstream_memory",),
            max_model_calls=1,
        )
        prior = AgentResult(
            task_id="upstream_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
        )
        outcome = await WorkerActionLoop(
            model_client=SequenceModelClient(
                [action("complete_empty", summary="错误地忽略上游结果")]
            ),
            config=WorkerActionLoopConfig(repair_attempts=0),
        ).run(context, prior_results=(prior,))

        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            outcome.termination_reason,
        )
        self.assertIn("reference every declared prior result", outcome.error)

    async def test_prior_evidence_requires_its_owning_result_reference(self) -> None:
        context = make_context(
            EchoTool(),
            dependencies=("upstream_memory",),
            max_model_calls=1,
        )
        evidence = EvidenceRef(
            source_kind=SourceKind.LONG_TERM_MEMORY,
            memory_id="memory_1",
        )
        prior = AgentResult(
            task_id="upstream_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            evidence=(evidence,),
        )
        outcome = await WorkerActionLoop(
            model_client=SequenceModelClient(
                [
                    action(
                        "complete_empty",
                        referenced_prior_evidence_ids=[evidence.evidence_id],
                    )
                ]
            ),
            config=WorkerActionLoopConfig(repair_attempts=0),
        ).run(context, prior_results=(prior,))

        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            outcome.termination_reason,
        )
        self.assertIn("owning prior result", outcome.error)

    async def test_unknown_observation_cannot_be_claimed_as_evidence(self) -> None:
        model = SequenceModelClient(
            [
                action(
                    "partial",
                    summary="错误引用不存在的观察",
                    selected_observation_ids=["observation_invented"],
                )
            ]
        )
        outcome = await WorkerActionLoop(
            model_client=model,
            config=WorkerActionLoopConfig(repair_attempts=0),
        ).run(make_context(EchoTool(), max_model_calls=1))

        self.assertEqual(
            WorkerTerminationReason.SCHEMA_FAILURE,
            outcome.termination_reason,
        )
        self.assertIn("unknown observations", outcome.error)

    async def test_cancellation_propagates(self) -> None:
        task = asyncio.create_task(
            WorkerActionLoop(
                model_client=BlockingModelClient(),
                config=WorkerActionLoopConfig(model_timeout_seconds=60),
            ).run(make_context(EchoTool(), max_model_calls=1))
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_shared_model_semaphore_serializes_independent_workers(self) -> None:
        model = ConcurrentModelClient()
        loop = WorkerActionLoop(
            model_client=model,
            model_semaphore=asyncio.Semaphore(1),
        )
        first, second = await asyncio.gather(
            loop.run(
                make_context(
                    EchoTool(), task_id="template_first", max_model_calls=1
                )
            ),
            loop.run(
                make_context(
                    EchoTool(), task_id="template_second", max_model_calls=1
                )
            ),
        )

        self.assertEqual(1, model.max_active)
        self.assertEqual("template_first", first.task_id)
        self.assertEqual("template_second", second.task_id)
        self.assertEqual(1, first.final_budget.model_calls_used)
        self.assertEqual(1, second.final_budget.model_calls_used)


if __name__ == "__main__":
    unittest.main()
