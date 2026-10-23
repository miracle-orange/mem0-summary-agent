from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.evaluation.evaluator import (  # noqa: E402
    _visible_historical_records,
)
from summary_multi_agent.planner import (  # noqa: E402
    _planner_input_payload,
    build_model_planner_prompt,
    build_planner_input,
    build_worker_capabilities,
)
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    CurrentFactCategory,
    CurrentFactItem,
    CurrentSegmentAnalysis,
    EvidenceRef,
    RunConfig,
    SourceKind,
    SubTask,
    SummaryRequest,
    SynthesisContext,
    TaskType,
    ToolCapabilitySpec,
    ToolResult,
    ToolStatus,
    WorkerAgentInput,
    WorkerBudgetSnapshot,
    WorkerMode,
    WorkerObservation,
)
from summary_multi_agent.synthesis import (  # noqa: E402
    build_legacy_summary_synthesis_prompt,
    build_projected_summary_synthesis_prompt,
    build_summary_synthesis_prompt,
    visible_historical_records,
)
from summary_multi_agent.workers.action_loop import (  # noqa: E402
    build_worker_action_prompt,
)


def make_long_context() -> tuple[SynthesisContext, str]:
    current_material = ("项目组逐项核对当前方案、风险和责任人。" "李明确认接口联调继续推进，王芳将在明天复核测试环境。") * 120
    items = []
    evidence_index = {}
    for index in range(70):
        quote = current_material[index * 8 : index * 8 + 180]
        evidence_id = f"current-evidence-{index}"
        items.append(
            CurrentFactItem(
                item_id=f"current-fact-{index}",
                category=CurrentFactCategory.FACT,
                statement=f"当前材料事实 {index}：{quote}",
                evidence_quote=quote,
                evidence_id=evidence_id,
                entities=("李明",) if index % 2 == 0 else ("王芳",),
            )
        )
        evidence_index[evidence_id] = EvidenceRef(
            evidence_id=evidence_id,
            source_kind=SourceKind.CURRENT_MATERIAL,
            source_id="source-1",
            segment_id="segment-1",
            start_offset=index * 8,
            end_offset=index * 8 + len(quote),
            evidence_text=quote,
        )

    canonical_record = {
        "record_ref_id": "canonical-ref-1",
        "memory_id": "memory-canonical-1",
        "memory_type": "task",
        "content": "李明需要完成接口联调。",
        "content_variants": ["李明需要完成接口联调。", "李明需要完成接口联调并复核日志。"],
        "metadata": {
            "source_time": "2026-07-21",
            "memory_status": "pending",
            "structured_fields_json": json.dumps(
                {
                    "owner": "李明",
                    "deadline": "2026-07-29",
                    "constraint": "测试环境可用",
                },
                ensure_ascii=False,
            ),
            "entities_text": "李明",
        },
        "score": 0.93,
        "evidence_ids": ["memory-evidence-1"],
        "record_variants": [
            {
                "memory_type": "task",
                "content": "李明需要完成接口联调并复核日志。",
                "metadata": {
                    "source_time": "2026-07-22",
                    "memory_status": "active",
                    "structured_fields_json": '{"owner":"李明"}',
                },
            }
        ],
    }
    entity_record = {
        "memory_id": "memory-entity-1",
        "memory_type": "constraint",
        "content": "王芳负责确认测试环境资源是否可用。",
        "matched_entities": ["王芳"],
        "related_entities": ["测试环境"],
        "metadata": {
            "source_time": "2026-07-20",
            "memory_status": "confirmed",
        },
        "score": 0.88,
    }
    temporal_expression = "明天"
    temporal_start = current_material.index(temporal_expression)
    context = SynthesisContext(
        request_id="request-1",
        source_time="2026-07-28",
        summary_request="生成项目进展、决定、风险和行动项纪要。",
        target_entities=("李明", "王芳"),
        personalization_preferences={"summary_style": "structured"},
        segment_analyses=(
            CurrentSegmentAnalysis(
                segment_id="segment-1",
                start_offset=0,
                end_offset=len(current_material),
                items=tuple(items),
            ),
        ),
        merged_current_items=tuple(items),
        worker_results={
            "entity-task": {
                "task_id": "entity-task",
                "task_type": "entity_personalization",
                "status": "success",
                "payload": {"records": [entity_record]},
                "evidence_ids": ["memory-evidence-2"],
                "duration_ms": 999,
            },
            "temporal-task": {
                "task_id": "temporal-task",
                "task_type": "temporal_reasoning",
                "status": "success",
                "payload": {
                    "expressions": [
                        {
                            "expression": temporal_expression,
                            "direction": "future",
                            "segment_id": "segment-1",
                            "start_offset": temporal_start,
                            "end_offset": temporal_start + len(temporal_expression),
                        }
                    ],
                    "resolved": [
                        {
                            "expression": temporal_expression,
                            "direction": "future",
                            "segment_id": "segment-1",
                            "start_offset": temporal_start,
                            "end_offset": temporal_start + len(temporal_expression),
                            "resolution": {
                                "normalized_date": "2026-07-29",
                                "rule": "relative_day",
                            },
                        }
                    ],
                },
                "warnings": [
                    "一个模糊时间表达式需谨慎解释",
                    "worker schema_failure after 999 ms",
                ],
            },
        },
        historical_records={"canonical-ref-1": canonical_record},
        evidence_index=evidence_index,
        template={
            "template_name": "structured_v2",
            "format_kind": "structured",
            "sections": ["项目进展", "决定", "风险", "行动项"],
            "instructions": ["行动项包含负责人和截止日期。"],
            "used_default": True,
            "warnings": ["unsupported style used default"],
        },
        auxiliary_context_info={
            "mode": "deterministic",
            "input_chars": 90000,
            "output_chars": 50000,
        },
        warnings=("完整运行 warning 不应整体进入 Summarizer",),
    )
    return context, current_material


class SynthesisProjectionTests(unittest.TestCase):
    def test_projection_preserves_business_information_without_mutation(self) -> None:
        context, current_material = make_long_context()
        before = copy.deepcopy(context.model_dump(mode="json"))

        prompt, report = build_projected_summary_synthesis_prompt(
            context,
            current_material=current_material,
        )

        self.assertEqual(before, context.model_dump(mode="json"))
        self.assertEqual(1, prompt.count(current_material))
        self.assertIn("李明需要完成接口联调", prompt)
        self.assertIn("王芳负责确认测试环境资源是否可用", prompt)
        self.assertIn('"owner":"李明"', prompt)
        self.assertIn("2026-07-29", prompt)
        self.assertIn("同一历史来源存在不同", prompt)
        self.assertIn("项目进展", prompt)
        self.assertIn("一个模糊时间表达式需谨慎解释", prompt)
        self.assertNotIn("worker schema_failure after 999 ms", prompt)
        self.assertNotIn("used_default", prompt)
        self.assertNotIn("unsupported style used default", prompt)

        self.assertNotIn("memory-canonical-1", prompt)
        self.assertNotIn("memory-entity-1", prompt)
        self.assertNotIn("current-evidence-1", prompt)
        self.assertNotIn('"score"', prompt)
        self.assertNotIn("duration_ms", prompt)
        self.assertNotIn("segment_analyses", prompt)
        self.assertNotIn("完整运行 warning 不应整体进入 Summarizer", prompt)

        expected_ids = set(_visible_historical_records(context))
        projected_ids = {item[0] for item in visible_historical_records(context)}
        self.assertEqual(expected_ids, projected_ids)
        self.assertEqual(expected_ids, set(report["history_record_ids"]))
        self.assertEqual(
            hashlib.sha256(current_material.encode("utf-8")).hexdigest(),
            report["current_material_sha256"],
        )

    def test_long_prompt_is_at_least_thirty_percent_smaller(self) -> None:
        context, current_material = make_long_context()
        projected, report = build_projected_summary_synthesis_prompt(
            context,
            current_material=current_material,
        )
        legacy = build_legacy_summary_synthesis_prompt(context)

        self.assertLess(len(projected), len(legacy))
        self.assertGreaterEqual(report["char_reduction_ratio"], 0.30)
        self.assertEqual(len(projected), report["totals"]["projected"]["chars"])
        self.assertEqual(
            len(projected.encode("utf-8")),
            report["totals"]["projected"]["utf8_bytes"],
        )
        self.assertEqual(len(legacy), report["totals"]["legacy_full_context"]["chars"])

    def test_missing_current_material_keeps_legacy_internal_path(self) -> None:
        context, _ = make_long_context()
        prompt = build_summary_synthesis_prompt(context)
        self.assertEqual(build_legacy_summary_synthesis_prompt(context), prompt)
        self.assertIn("<internal_synthesis_input>", prompt)


class PlannerWorkerProjectionTests(unittest.TestCase):
    def test_planner_keeps_full_material_without_hashes_or_versions(self) -> None:
        material = "李明明天完成接口联调，王芳复核测试环境。"
        request = SummaryRequest(
            current_material=material,
            source_time="2026-07-28",
            summary_request="生成项目纪要",
        )
        raw = build_raw_material_bundle(material, max_segment_chars=1000)
        tools = (
            ToolCapabilitySpec(
                tool_name="typed_memory_search",
                description="search",
                allowed_task_types=(TaskType.SEMANTIC_MEMORY,),
            ),
            ToolCapabilitySpec(
                tool_name="template_adaptation",
                description="template",
                allowed_task_types=(TaskType.TEMPLATE_ADAPTATION,),
            ),
            ToolCapabilitySpec(
                tool_name="calendar_resolve",
                description="calendar",
                allowed_task_types=(TaskType.TEMPORAL_REASONING,),
            ),
        )
        planner_input = build_planner_input(
            request=request,
            raw_material=raw,
            worker_capabilities=build_worker_capabilities(
                (
                    TaskType.SEMANTIC_MEMORY,
                    TaskType.TEMPLATE_ADAPTATION,
                    TaskType.TEMPORAL_REASONING,
                ),
                tools,
                max_tasks_per_plan=2,
                execution_mode=WorkerMode.RULE,
            ),
            tool_capabilities=tools,
            config=RunConfig(workflow_mode="multi_agent_v2", planner_mode="rule"),
        )
        payload = _planner_input_payload(planner_input)
        prompt = build_model_planner_prompt(planner_input)
        block = prompt.split("<planner_input>\n", 1)[1].split("\n</planner_input>", 1)[
            0
        ]
        legacy_payload = {
            "request": planner_input.request.model_dump(mode="json"),
            "raw_material_manifest": {
                "schema_version": raw.schema_version,
                "source_id": raw.source_id,
                "content_sha256": raw.content_sha256,
                "total_chars": raw.total_chars,
                "segments": [
                    segment.model_dump(mode="json", exclude={"text"})
                    for segment in raw.segments
                ],
            },
            "worker_capabilities": [
                item.model_dump(mode="json")
                for item in planner_input.worker_capabilities
            ],
            "tool_capabilities": [item.model_dump(mode="json") for item in tools],
            "budget": planner_input.budget.model_dump(mode="json"),
            "remaining_budget": None,
            "workload_profile": planner_input.workload_profile.model_dump(mode="json"),
        }
        legacy_block = json.dumps(
            legacy_payload, ensure_ascii=False, indent=2, sort_keys=True
        )

        self.assertNotIn("current_material", payload["request"])
        self.assertNotIn("schema_version", json.dumps(payload))
        self.assertNotIn("content_sha256", json.dumps(payload))
        self.assertIn(material, prompt)
        self.assertEqual(1, prompt.count(material))
        self.assertLessEqual(len(block), len(legacy_block))

    def test_worker_keeps_reference_ids_and_drops_observation_runtime_stats(
        self,
    ) -> None:
        material = "李明明天完成接口联调。"
        request = SummaryRequest(
            current_material=material,
            source_time="2026-07-28",
        )
        raw = build_raw_material_bundle(material, max_segment_chars=1000)
        prior = AgentResult(
            task_id="upstream",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={"records": [{"memory_id": "memory-1", "content": "历史约束仍有效"}]},
            evidence=(
                EvidenceRef(
                    evidence_id="prior-evidence-1",
                    source_kind=SourceKind.LONG_TERM_MEMORY,
                    memory_id="memory-1",
                    evidence_text="历史约束仍有效",
                ),
            ),
        )
        observation = WorkerObservation(
            observation_id="observation-1",
            step_index=1,
            tool_result=ToolResult(
                call_id="call-1",
                tool_name="typed_memory_search",
                status=ToolStatus.SUCCESS,
                payload={"records": [{"content": "业务检索结果"}]},
                attempts=3,
                duration_ms=9000,
            ),
        )
        agent_input = WorkerAgentInput(
            task=SubTask(
                task_id="semantic-child",
                task_type=TaskType.SEMANTIC_MEMORY,
                description="根据上游证据补充检索",
                dependencies=("upstream",),
                allowed_tools=("typed_memory_search",),
                max_model_calls=2,
                max_tool_calls=1,
            ),
            request=request,
            raw_material=raw,
            prior_results=(prior,),
            tool_schemas={"typed_memory_search": {"type": "object"}},
            observations=(observation,),
            budget=WorkerBudgetSnapshot(
                step_limit=4,
                model_call_limit=2,
                tool_call_limit=1,
                steps_used=1,
                model_calls_used=1,
                tool_calls_used=1,
            ),
        )

        prompt = build_worker_action_prompt(agent_input)
        block = prompt.split("<worker_agent_input>\n", 1)[1].split(
            "\n</worker_agent_input>", 1
        )[0]
        payload = json.loads(block)
        legacy_payload = agent_input.model_dump(mode="json")
        legacy_payload["raw_material"] = {
            "schema_version": raw.schema_version,
            "source_id": raw.source_id,
            "content_sha256": raw.content_sha256,
            "total_chars": raw.total_chars,
            "max_segment_chars": raw.max_segment_chars,
            "overlap_chars": raw.overlap_chars,
            "segments": [
                segment.model_dump(mode="json", exclude={"text"})
                for segment in raw.segments
            ],
            "current_material_location": "request.current_material",
        }
        legacy_payload["prior_results"] = [
            {
                "task_id": prior.task_id,
                "task_type": prior.task_type.value,
                "status": prior.status.value,
                "is_empty": prior.is_empty,
                "payload": prior.payload,
                "evidence": [item.model_dump(mode="json") for item in prior.evidence],
                "confidence": prior.confidence,
                "warnings": list(prior.warnings),
                "consumed_result_ids": list(prior.consumed_result_ids),
                "consumed_evidence_ids": list(prior.consumed_evidence_ids),
            }
        ]
        legacy_block = json.dumps(
            legacy_payload, ensure_ascii=False, indent=2, sort_keys=True
        )

        self.assertNotIn("current_material", payload["request"])
        self.assertEqual(1, prompt.count(material))
        self.assertEqual("observation-1", payload["observations"][0]["observation_id"])
        self.assertEqual("call-1", payload["observations"][0]["call_id"])
        self.assertEqual("upstream", payload["prior_results"][0]["task_id"])
        prior_ref = payload["prior_results"][0]["payload"]["record_refs"][0][
            "record_version_ref"
        ]
        self.assertEqual(
            prior_ref,
            payload["memory_record_catalog"]["records"][0]["record_version_ref"],
        )
        self.assertEqual(
            "prior-evidence-1",
            payload["prior_results"][0]["evidence_ids"][0],
        )
        self.assertEqual(
            "历史约束仍有效",
            payload["dependency_evidence_catalog"]["evidence"][0][
                "evidence_text"
            ],
        )
        self.assertNotIn("attempts", payload["observations"][0])
        self.assertNotIn("duration_ms", payload["observations"][0])
        self.assertLessEqual(len(block), len(legacy_block))
        self.assertEqual(
            "业务检索结果",
            payload["observations"][0]["unkeyed_records"][0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
