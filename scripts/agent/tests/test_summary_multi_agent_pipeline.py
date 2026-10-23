from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.runtime import build_multi_agent_v2_pipeline  # noqa: E402
from summary_multi_agent.context_builder import compress_memory_records  # noqa: E402
from summary_multi_agent.schemas import RunConfig, RunStatus, SummaryRequest  # noqa: E402


class EmptyMemoryBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, filters, top_k):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        return {"results": []}


class FakeSummaryModel:
    def __init__(self, *, fail_marker: str | None = None) -> None:
        self.fail_marker = fail_marker
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if "Summary Multi-Agent V2 的 Model Planner" in prompt:
            planner_match = re.search(
                r"<planner_input>\n(.*?)\n</planner_input>", prompt, re.DOTALL
            )
            if planner_match is None:
                raise AssertionError("planner prompt is missing planner_input")
            planner_payload = json.loads(planner_match.group(1))
            workload = planner_payload["workload_profile"]
            temporal_calls = workload["temporal_required_tool_calls"]
            capability_model_calls = {
                item["capability_id"]: item["min_model_calls"]
                for item in planner_payload["worker_capabilities"]
            }
            tasks = [
                {
                    "task_id": "semantic_memory",
                    "task_type": "semantic_memory",
                    "description": "检索与摘要要求相关的历史 typed memory。",
                    "dependencies": [],
                    "critical": False,
                    "timeout_seconds": 120,
                    "allowed_tools": ["typed_memory_search"],
                    "inputs": {
                        "queries": ["当前材料的历史任务、约束和风险"],
                        "memory_types": [
                            "time",
                            "decision",
                            "constraint",
                            "task",
                            "risk",
                            "preference",
                        ],
                        "top_k_per_type": 10,
                    },
                    "max_model_calls": capability_model_calls["semantic_memory"],
                    "max_tool_calls": 1,
                },
                {
                    "task_id": "template_adaptation",
                    "task_type": "template_adaptation",
                    "description": "选择符合用户要求的摘要结构。",
                    "dependencies": [],
                    "critical": False,
                    "timeout_seconds": 60,
                    "allowed_tools": ["template_adaptation"],
                    "inputs": {"requested_style": "structured"},
                    "max_model_calls": capability_model_calls["template_adaptation"],
                    "max_tool_calls": 1,
                },
            ]
            if workload["temporal_task_required"]:
                tasks.append(
                    {
                        "task_id": "temporal_reasoning",
                        "task_type": "temporal_reasoning",
                        "description": "解析当前材料中的相对时间。",
                        "dependencies": [],
                        "critical": workload["temporal_resolution_required"],
                        "timeout_seconds": 120,
                        "allowed_tools": ["calendar_resolve"],
                        "inputs": {},
                        "max_model_calls": capability_model_calls[
                            "temporal_reasoning"
                        ],
                        "max_tool_calls": temporal_calls or 1,
                    }
                )
            return json.dumps(
                {
                    "schema_version": "summary_multi_agent.model_plan_draft.v2",
                    "tasks": tasks,
                    "rationale": "按请求选择最小充分的专业任务。",
                },
                ensure_ascii=False,
            )
        if "受控专业 Worker Agent" in prompt:
            worker_match = re.search(
                r"<worker_agent_input>\n(.*?)\n</worker_agent_input>",
                prompt,
                re.DOTALL,
            )
            if worker_match is None:
                raise AssertionError("worker prompt is missing worker_agent_input")
            payload = json.loads(worker_match.group(1))
            observations = payload["observations"]
            if observations:
                return json.dumps(
                    {
                        "schema_version": "summary_multi_agent.worker_action.v2",
                        "action": "complete",
                        "action_rationale": "工具 Observation 已满足任务",
                        "tool_calls": [],
                        "selected_observation_ids": [
                            item["observation_id"] for item in observations
                        ],
                        "result_summary": "受控工具结果已收集",
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            task = payload["task"]
            task_type = task["task_type"]
            if task_type == "semantic_memory":
                arguments = {
                    "query": task["inputs"]["queries"][0],
                    "memory_types": task["inputs"]["memory_types"],
                    "top_k_per_type": task["inputs"]["top_k_per_type"],
                }
                tool_name = "typed_memory_search"
            elif task_type == "template_adaptation":
                arguments = {
                    "summary_request": payload["request"]["summary_request"],
                    "requested_style": task["inputs"].get("requested_style"),
                    "preferences": payload["request"][
                        "personalization_preferences"
                    ],
                }
                tool_name = "template_adaptation"
            else:
                raise AssertionError(f"unsupported worker prompt: {task_type}")
            return json.dumps(
                {
                    "schema_version": "summary_multi_agent.worker_action.v2",
                    "action": "call_tool",
                    "action_rationale": "需要读取授权工具 Observation",
                    "tool_calls": [
                        {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "timeout_seconds": 30,
                        }
                    ],
                    "selected_observation_ids": [],
                    "result_summary": None,
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        if "Current Material Mapper" in prompt:
            match = re.search(
                r"<current_segment>\n(.*?)\n</current_segment>", prompt, re.DOTALL
            )
            if match is None:
                return "not-json"
            segment_text = match.group(1)
            if self.fail_marker and self.fail_marker in segment_text:
                return "not-json"
            quote = segment_text[: min(6, len(segment_text))]
            return json.dumps(
                {
                    "items": [
                        {
                            "category": "fact",
                            "statement": f"分段事实：{quote}",
                            "evidence_quote": quote,
                            "entities": [],
                            "time_expression": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Summarizer Agent" in prompt:
            return "这是由全部当前材料分段和 Worker 证据生成的摘要。"
        raise AssertionError("unexpected model prompt")


class RepairingTemporalSummaryModel(FakeSummaryModel):
    def __init__(self, *, always_invalid: bool = False) -> None:
        super().__init__()
        self.always_invalid = always_invalid
        self.summary_calls = 0

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Summarizer Agent" in prompt:
            self.calls.append(
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            self.summary_calls += 1
            if self.always_invalid or self.summary_calls == 1:
                return "报告已于上周五（2026-03-20）提交。"
            return "报告已于上周五（2026-03-13）提交。"
        return super().__call__(
            prompt, temperature=temperature, max_tokens=max_tokens
        )


class MetaPreambleSummaryModel(FakeSummaryModel):
    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Summarizer Agent" in prompt:
            self.calls.append(
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return (
                "好的，以下是根据您提供的 SynthesisContext 生成的最终摘要。\n\n"
                "### 本次进展\n- 项目范围已经确认。"
            )
        return super().__call__(
            prompt, temperature=temperature, max_tokens=max_tokens
        )


class FailingSynthesisModel(FakeSummaryModel):
    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Summarizer Agent" in prompt:
            self.calls.append(
                {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            raise RuntimeError("controlled synthesis failure")
        return super().__call__(
            prompt, temperature=temperature, max_tokens=max_tokens
        )


class MultiAgentPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_synthesis_still_saves_projected_input_audit(self) -> None:
        request = SummaryRequest(
            current_material="项目范围已经确认。",
            source_time="2026-08-10",
            summary_request="生成结构化项目摘要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=FailingSynthesisModel(),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.FAILED, result.status)
            self.assertTrue(
                (output_dir / "multi_agent_summary_prompt.txt").exists()
            )
            self.assertTrue((output_dir / "model_input_report.json").exists())
            self.assertTrue(
                (output_dir / "model_call_input_report.json").exists()
            )
            self.assertTrue(
                (output_dir / "history_selection_report.json").exists()
            )
            self.assertIn("multi_agent_summary_prompt.txt", result.artifact_paths)
            self.assertIn("model_input_report.json", result.artifact_paths)
            self.assertIn("model_call_input_report.json", result.artifact_paths)
            self.assertIn("history_selection_report.json", result.artifact_paths)

    async def test_summary_meta_preamble_is_removed_and_audited(self) -> None:
        request = SummaryRequest(
            current_material="项目范围已经确认。",
            source_time="2026-08-10",
            summary_request="生成结构化项目摘要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=MetaPreambleSummaryModel(),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertIn(
                "summary meta preamble was removed by the output guard",
                result.warnings,
            )
            summary = (output_dir / "summary_multi_agent.md").read_text("utf-8")
            self.assertEqual("### 本次进展\n- 项目范围已经确认。", summary.strip())
            prompt = (output_dir / "multi_agent_summary_prompt.txt").read_text(
                "utf-8"
            )
            self.assertIn("不得添加确认、致谢、生成说明或引导语", prompt)
            trace = json.loads(
                (output_dir / "summary_synthesis_trace.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual(
                ["removed_meta_preamble"],
                trace[0]["normalizations"],
            )

    async def test_model_workers_execute_action_observation_in_runtime(self) -> None:
        model = FakeSummaryModel()
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            source_time="2026-08-10",
            summary_request="生成结构化项目摘要",
            personalization_preferences={"summary_style": "structured"},
        )
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            planner_mode="model",
            worker_mode="model",
            worker_action_repair_attempts=0,
            raw_segment_max_chars=100,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            agent_results = json.loads(
                (output_dir / "agent_results.json").read_text("utf-8")
            )
            self.assertEqual(
                {"semantic_memory", "template_adaptation"},
                set(agent_results),
            )
            for worker_result in agent_results.values():
                execution = worker_result["worker_execution"]
                self.assertEqual("model", execution["mode"])
                self.assertEqual("goal_satisfied", execution["termination_reason"])
                self.assertEqual(2, len(execution["steps"]))
                self.assertIsNone(execution["fallback_reason"])
            trace = (output_dir / "execution_trace.jsonl").read_text("utf-8")
            self.assertIn('"worker_mode": "model"', trace)
            self.assertIn('"worker_termination": "goal_satisfied"', trace)

    async def test_full_pipeline_maps_every_segment_and_saves_artifacts(self) -> None:
        memory = EmptyMemoryBackend()
        model = FakeSummaryModel()
        current_material = (
            "第一段事实。第二段事实。第三段事实。末尾保留。" * 12 + "\n"
        )
        request = SummaryRequest(
            current_material=current_material,
            source_time="2026-08-05",
            summary_request="生成结构化摘要",
            personalization_preferences={"summary_style": "structured"},
        )
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            raw_segment_max_chars=100,
            raw_segment_overlap_ratio=0.25,
            max_concurrent_segment_maps=2,
            schema_repair_attempts=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=memory,
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            self.assertTrue(result.coverage_complete)
            self.assertEqual(
                current_material,
                (output_dir / "current_material.txt").read_text(encoding="utf-8"),
            )
            ledger = json.loads(
                (output_dir / "coverage_ledger.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    set(segment["completed_stages"])
                    == {"current_fact_map", "synthesis_merge"}
                    for segment in ledger["segments"]
                )
            )
            analyses = json.loads(
                (output_dir / "current_segment_analyses.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(ledger["segments"]), len(analyses))
            self.assertEqual(
                len(ledger["segments"]),
                len([call for call in model.calls if "Current Material Mapper" in call["prompt"]]),
            )
            self.assertTrue((output_dir / "summary_multi_agent.md").exists())
            self.assertTrue((output_dir / "multi_agent_summary_prompt.txt").exists())
            self.assertTrue((output_dir / "model_input_report.json").exists())
            model_input_report = json.loads(
                (output_dir / "model_input_report.json").read_text("utf-8")
            )
            self.assertEqual(
                "summary_multi_agent.model_input_report.v1",
                model_input_report["schema_version"],
            )
            self.assertIn("model_input_report.json", result.artifact_paths)
            plan_history = json.loads(
                (output_dir / "plan_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, len(plan_history))
            self.assertEqual("model", plan_history[0]["source"])
            self.assertEqual(1, plan_history[0]["plan_version"])
            planner_report = json.loads(
                (output_dir / "planner_validation_report.json").read_text("utf-8")
            )
            self.assertEqual("model_plan_accepted", planner_report["status"])
            self.assertTrue((output_dir / "planner_prompt.txt").exists())
            self.assertTrue((output_dir / "planner_model_trace.json").exists())
            synthesis_context = json.loads(
                (output_dir / "synthesis_context.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                synthesis_context["auxiliary_context_info"][
                    "current_material_compressed"
                ]
            )
            compression_report = json.loads(
                (output_dir / "context_compression_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("deterministic", compression_report["mode"])
            self.assertTrue(compression_report["invariants_preserved"])
            self.assertEqual(
                compression_report["current_material_sha256_before"],
                compression_report["current_material_sha256_after"],
            )
            self.assertIn(
                '"event_type": "context.compressed"',
                (output_dir / "execution_trace.jsonl").read_text("utf-8"),
            )
            board = json.loads(
                (output_dir / "artifact_board.json").read_text(encoding="utf-8")
            )
            self.assertEqual("summary_artifact_board.v1", board["schema_version"])
            self.assertEqual(0, board["retrieved_memory_count"])

        expected_search_calls = 6
        self.assertEqual(expected_search_calls, len(memory.calls))

    async def test_model_planner_failure_falls_back_without_blocking_summary(self) -> None:
        class PlannerFailingModel(FakeSummaryModel):
            def __call__(self, prompt, *, temperature, max_tokens):
                if "Summary Multi-Agent V2 的 Model Planner" in prompt:
                    self.calls.append(
                        {
                            "prompt": prompt,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                    )
                    return "not-json"
                return super().__call__(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )

        request = SummaryRequest(
            current_material="负责人明天提交项目报告。",
            source_time="2026-08-05",
            summary_request="生成项目摘要",
        )
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            planner_repair_attempts=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=PlannerFailingModel(),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertTrue(any("planner:" in item for item in result.warnings))
            plan = json.loads((output_dir / "plan.json").read_text("utf-8"))
            self.assertEqual("rule_fallback", plan["source"])
            report = json.loads(
                (output_dir / "planner_validation_report.json").read_text("utf-8")
            )
            self.assertTrue(report["fallback_used"])

    async def test_failed_segment_blocks_summary_generation(self) -> None:
        memory = EmptyMemoryBackend()
        model = FakeSummaryModel(fail_marker="FAIL")
        request = SummaryRequest(
            current_material=(
                "正常分段内容。" * 12
                + "FAIL 分段内容不能解析。"
                + "后续正常分段内容。" * 12
            ),
            source_time="2026-08-05",
            summary_request="生成摘要",
            personalization_preferences={"summary_style": "structured"},
        )
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            raw_segment_max_chars=100,
            raw_segment_overlap_ratio=0.0,
            schema_repair_attempts=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=memory,
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.FAILED, result.status)
            self.assertFalse(result.coverage_complete)
            self.assertFalse((output_dir / "summary_multi_agent.md").exists())
            self.assertFalse(
                any("Summarizer Agent" in call["prompt"] for call in model.calls)
            )
            ledger = json.loads(
                (output_dir / "coverage_ledger.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(segment["failed_stages"] for segment in ledger["segments"])
            )

    async def test_summarizer_delegates_temporal_semantics_to_worker(self) -> None:
        model = RepairingTemporalSummaryModel()
        request = SummaryRequest(
            current_material="负责人确认报告已经提交。",
            source_time="2026-03-19",
            summary_request="生成项目摘要",
        )
        config = RunConfig(workflow_mode="multi_agent_v2")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            self.assertEqual(1, model.summary_calls)
            self.assertIn("2026-03-20", result.final_summary or "")
            trace = json.loads(
                (output_dir / "summary_synthesis_trace.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(1, len(trace))
            self.assertEqual([], trace[0]["issues"])
            self.assertEqual("temporal_worker", trace[0]["temporal_authority"])
            self.assertEqual(
                "output_normalization_only",
                trace[0]["validation_mode"],
            )

    async def test_summarizer_does_not_block_on_temporal_pair(self) -> None:
        model = RepairingTemporalSummaryModel(always_invalid=True)
        request = SummaryRequest(
            current_material="负责人确认报告已经提交。",
            source_time="2026-03-19",
            summary_request="生成项目摘要",
        )
        config = RunConfig(workflow_mode="multi_agent_v2")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            self.assertEqual(1, model.summary_calls)
            self.assertTrue((output_dir / "summary_multi_agent.md").exists())
            trace = json.loads(
                (output_dir / "summary_synthesis_trace.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(1, len(trace))
            self.assertEqual([], trace[0]["issues"])

    async def test_worker_warning_is_exposed_in_run_result(self) -> None:
        model = FakeSummaryModel()
        request = SummaryRequest(
            current_material="今天登记风险，明天提交报告。",
            source_time="2026-03-19",
            summary_request="生成项目摘要",
        )
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            planner_mode="rule",
            temporal_max_tool_calls=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=Path(temp_dir),
                config=config,
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertTrue(
                any(
                    warning.startswith("temporal_reasoning:")
                    and "exceeded max_tool_calls" in warning
                    for warning in result.warnings
                )
            )


class AuxiliaryContextCompressionTests(unittest.TestCase):
    def test_only_historical_records_are_ranked_and_bounded(self) -> None:
        records = [
            {
                "memory_id": "low",
                "memory_type": "task",
                "content": "low",
                "score": 0.1,
            },
            {
                "memory_id": "high",
                "memory_type": "task",
                "content": "high",
                "score": 0.9,
            },
            {
                "memory_id": "risk",
                "memory_type": "risk",
                "content": "risk",
                "score": 0.5,
            },
        ]
        retained, info = compress_memory_records(records, top_k_per_type=1)
        self.assertEqual({"high", "risk"}, {item["memory_id"] for item in retained})
        self.assertEqual(3, info["input_count"])
        self.assertEqual(2, info["retained_count"])


if __name__ == "__main__":
    unittest.main()
