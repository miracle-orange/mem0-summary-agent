from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.context_builder import build_synthesis_context
from summary_multi_agent.current_material import CurrentSegmentDraft, build_current_segment_map_prompt
from summary_multi_agent.planner import ModelDrivenPlanner, RuleBasedPlanner
from summary_multi_agent.raw_material import build_raw_material_bundle, mark_segment_processed
from summary_multi_agent.review.blue import build_blue_prompt
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline
from summary_multi_agent.schemas import (
    AgentResult, AgentStatus, CurrentSegmentAnalysis,
    ExecutionPlan, PlanSource, RedReviewReport, RunConfig, RunStatus, SubTask, SummaryRequest, TaskType,
)
from summary_multi_agent.synthesis import build_projected_summary_synthesis_prompt, visible_historical_records
from summary_multi_agent.tools import TemplateAdaptationTool, ToolRegistry
from summary_multi_agent.tools.template import (
    MEETING_SCENES, SCENE_SECTIONS, TemplateAdaptationInput,
    TemplateAdaptationOutput, project_template, resolve_template,
)
from summary_multi_agent.workers import TemplateAdaptationWorker
from summary_multi_agent.working_memory import initialize_working_state, load_working_state_checkpoint, save_working_state_checkpoint
from summary_multi_agent_web.artifact_reader import create_safe_artifact_archive, read_artifact_snapshot
from summary_multi_agent_web.schemas import WebRunSnapshot, WebRunStatus, WebSummarizeRequest, build_summary_request
from summary_multi_agent_web_demo import build_app, build_web_request, build_e2e_request

from test_summary_multi_agent_autonomous_workers import (
    SequenceModelClient, action, complete_from_observations, make_context, make_worker,
)
from test_summary_multi_agent_model_input_projection import make_long_context
from test_summary_multi_agent_planner_contracts import make_initial_input, valid_model_draft
from test_summary_multi_agent_pipeline import FakeSummaryModel
from test_summary_multi_agent_tools_workers import FakeMemoryBackend


def mapped_state(preferences=None):
    request = SummaryRequest(
        current_material="本批短音频已完成，长音频仍在处理；若资源到位，再开展下一批测试。",
        summary_request="生成中文会议纪要",
        personalization_preferences=preferences or {},
    )
    bundle = build_raw_material_bundle(request.current_material, max_segment_chars=100)
    state = initialize_working_state(request, bundle)
    state.current_segment_analyses = [
        CurrentSegmentAnalysis(
            segment_id=s.segment_id, start_offset=s.start_offset, end_offset=s.end_offset,
        )
        for s in bundle.segments
    ]
    for s in bundle.segments:
        state.coverage_ledger = mark_segment_processed(
            state.coverage_ledger, segment_id=s.segment_id, stage="current_fact_map",
        )
    return state


class MeetingTemplateTests(unittest.TestCase):
    def test_all_scene_style_pairs_are_pure_and_distinct(self):
        names = set()
        for scene in MEETING_SCENES:
            for style in ("structured", "natural"):
                with self.subTest(scene=scene, style=style):
                    args = TemplateAdaptationInput(requested_scene=scene, requested_style=style)
                    before = args.model_dump(mode="json")
                    result = resolve_template(args)
                    self.assertEqual(before, args.model_dump(mode="json"))
                    self.assertEqual(scene, result.meeting_scene)
                    self.assertEqual(style, result.format_kind)
                    self.assertFalse(result.used_default)
                    self.assertEqual(
                        SCENE_SECTIONS[scene] if style == "structured" else (),
                        result.sections,
                    )
                    names.add(result.template_name)
        self.assertEqual(8, len(names))

    def test_explicit_scene_overrides_suggestion_and_auto_does_not(self):
        for explicit, expected in (("project_sync", "project_sync"), ("auto", "decision_review")):
            result = resolve_template(TemplateAdaptationInput(
                requested_scene="decision_review", preferences={"meeting_scene": explicit},
            ))
            self.assertEqual(expected, result.meeting_scene)
        self.assertEqual("general", resolve_template(TemplateAdaptationInput()).meeting_scene)
        self.assertEqual("natural", resolve_template(TemplateAdaptationInput(
            summary_request="请用自然段表达",
        )).format_kind)

    def test_unknown_scene_is_not_silently_accepted_as_a_tool_choice(self):
        with self.assertRaises(ValidationError):
            TemplateAdaptationInput(requested_scene="invented")
        result = resolve_template(TemplateAdaptationInput(preferences={"meeting_scene": "old_unknown"}))
        self.assertEqual("general", result.meeting_scene)
        self.assertTrue(result.warnings)

    def test_legacy_template_payload_still_validates(self):
        old = {
            "template_name": "structured_v2", "format_kind": "structured",
            "sections": ["进展"], "instructions": ["只写有据内容"],
        }
        parsed = TemplateAdaptationOutput.model_validate(old)
        self.assertEqual("general", parsed.meeting_scene)
        self.assertEqual(old, project_template(old))

    def test_no_template_task_gets_auditable_default_without_fabricated_result(self):
        state = mapped_state({"meeting_scene": "focused_discussion", "summary_style": "natural"})
        before = copy.deepcopy(state.agent_results)
        context = build_synthesis_context(state)
        self.assertEqual("focused_discussion", context.template["meeting_scene"])
        self.assertEqual("natural", context.template["format_kind"])
        self.assertEqual("default_not_planned", context.auxiliary_context_info["template_selection"]["source"])
        self.assertEqual(before, state.agent_results)
        with tempfile.TemporaryDirectory() as directory:
            path = save_working_state_checkpoint(state, Path(directory))
            restored = load_working_state_checkpoint(path)
            self.assertEqual(context.template, restored.synthesis_context.template)

    def test_failed_or_invalid_template_is_unavailable_not_worker_success(self):
        for status, payload in ((AgentStatus.FAILED, {}), (AgentStatus.SUCCESS, {"instructions": []})):
            with self.subTest(status=status):
                state = mapped_state()
                state.plan = ExecutionPlan(tasks=(SubTask(
                    task_id="template", task_type=TaskType.TEMPLATE_ADAPTATION,
                    description="select a template", allowed_tools=("template_adaptation",),
                ),))
                state.agent_results["template"] = AgentResult(
                    task_id="template", task_type=TaskType.TEMPLATE_ADAPTATION,
                    status=status, payload=payload, error="failed" if status == AgentStatus.FAILED else None,
                )
                before = state.agent_results["template"].model_dump(mode="json")
                context = build_synthesis_context(state)
                self.assertEqual("default_unavailable", context.auxiliary_context_info["template_selection"]["source"])
                self.assertEqual(before, state.agent_results["template"].model_dump(mode="json"))

    def test_successful_selected_template_is_shared_with_blue_and_not_mutated(self):
        state = mapped_state()
        payload = resolve_template(TemplateAdaptationInput(requested_scene="decision_review")).model_dump(mode="json")
        state.agent_results["template"] = AgentResult(
            task_id="template", task_type=TaskType.TEMPLATE_ADAPTATION,
            status=AgentStatus.SUCCESS, payload=payload,
        )
        context = build_synthesis_context(state)
        self.assertEqual(project_template(payload), project_template(context.template))
        self.assertEqual(payload, state.agent_results["template"].payload)
        blue = build_blue_prompt(state, "已有候选", RedReviewReport(round_index=1, candidate_index=0, passed=True), (), 1)
        template_text = re.search(r"<output_template>\n(.*?)\n</output_template>", blue, re.S).group(1)
        self.assertEqual(project_template(payload), json.loads(template_text))
        self.assertEqual(1, blue.count("<output_template>"))

    def test_nonempty_history_and_original_survive_every_template(self):
        context, original = make_long_context()
        for scene in MEETING_SCENES:
            for style in ("structured", "natural"):
                payload = resolve_template(TemplateAdaptationInput(
                    requested_scene=scene, requested_style=style,
                )).model_dump(mode="json")
                selected = context.model_copy(update={"template": payload})
                before = copy.deepcopy(selected.model_dump(mode="json"))
                prompt, report = build_projected_summary_synthesis_prompt(selected, current_material=original)
                self.assertEqual(before, selected.model_dump(mode="json"))
                self.assertEqual(1, prompt.count(original))
                self.assertEqual(
                    [mid for mid, _ in visible_historical_records(context)],
                    report["history_record_ids"],
                )
                self.assertTrue(report["history_record_ids"])
                self.assertGreaterEqual(report["char_reduction_ratio"], 0.3)
                self.assertIn("owner", prompt)
                self.assertIn("2026-07-29", prompt)

    def test_mapper_uses_existing_schema_without_example_fact(self):
        state = mapped_state()
        prompt = build_current_segment_map_prompt(state.request, state.raw_material.segments[0])
        self.assertEqual(1, prompt.count(state.request.current_material))
        self.assertNotIn("张三负责提交测试报告", prompt)
        schema = json.loads(prompt.split("输出 Schema：\n", 1)[1].split("\n\n请求信息：", 1)[0])
        self.assertEqual(CurrentSegmentDraft.model_json_schema(), schema)

    def test_web_scene_is_optional_and_reaches_core_and_e2e(self):
        args = ("中文项目纪要", "user", "project", "", "", "structured",
                "rule", "rule", "deterministic", False, False, 0, 0, 1, 1, 1, 180, 600)
        old_request = build_web_request("当前会议内容", *args)
        self.assertEqual("auto", old_request.meeting_scene)
        self.assertEqual("classic", old_request.minutes_output_mode)
        request = build_web_request("当前会议内容", *args, meeting_scene="project_sync")
        self.assertEqual("project_sync", build_summary_request(request).personalization_preferences["meeting_scene"])
        e2e = build_e2e_request(
            "e2e_run", "历史会议内容", None, "", "当前会议内容", None, "",
            "meeting", "", "", args, meeting_scene="focused_discussion",
        )
        self.assertEqual("focused_discussion", e2e.summary.meeting_scene)
        roundtrip = WebSummarizeRequest.model_validate(request.model_dump(mode="json"))
        self.assertEqual(request, roundtrip)

    def test_web_scene_binding_reaches_callback_as_the_scene_argument(self):
        try:
            import gradio
        except ImportError:
            self.skipTest("Gradio is not installed")
        if not hasattr(gradio, "Blocks"):
            self.skipTest("Gradio Blocks is unavailable")

        class Manager:
            request = None

            def start_summarize(self, request, output_dir):
                self.request = request
                return WebRunSnapshot(
                    status=WebRunStatus.QUEUED, stage="queued", run_id="ui-scene-check",
                    output_dir=output_dir,
                )

        with tempfile.TemporaryDirectory() as directory:
            manager = Manager()
            app = build_app(manager, output_dir=Path(directory))
            functions = app.fns.values() if isinstance(app.fns, dict) else app.fns
            callback = next(f for f in functions if f.fn.__name__ == "start_summarize")
            values = [
                "" if isinstance(item, gradio.Textbox) and item.value is None else item.value
                for item in callback.inputs
            ]
            values[0] = "会议讨论交付范围。"
            values[3] = "生成纪要"
            values[4] = "user"
            values[5] = "project"
            scene_index = next(
                index
                for index, item in enumerate(callback.inputs)
                if item.label == "会议场景"
            )
            self.assertEqual("auto", callback.inputs[scene_index].value)
            values[scene_index] = "decision_review"
            output_mode_index = next(
                index
                for index, item in enumerate(callback.inputs)
                if item.label == "产出模式"
            )
            rich_tokens_index = next(
                index
                for index, item in enumerate(callback.inputs)
                if item.label == "丰富纪要输出上限 rich_summary_max_tokens"
            )
            values[output_mode_index] = "rich"
            values[rich_tokens_index] = 7200
            try:
                result = callback.fn(*values)
                self.assertNotIn("启动失败", result[0])
                self.assertEqual("decision_review", manager.request.meeting_scene)
                self.assertEqual("rich", manager.request.minutes_output_mode)
                self.assertEqual(7200, manager.request.rich_summary_max_tokens)
            finally:
                app.close()


class MeetingTemplateExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_pipeline_template_history_and_download_across_modes(self):
        for mode in ("model", "rule"):
            for compression in ("deterministic", "disabled"):
                for scene in MEETING_SCENES:
                    with self.subTest(mode=mode, compression=compression, scene=scene):
                        model = FakeSummaryModel()
                        config = RunConfig(
                            workflow_mode="multi_agent_v2", planner_mode="model",
                            worker_mode=mode, worker_action_repair_attempts=0,
                            context_compression_mode=compression,
                        )
                        request = SummaryRequest(
                            current_material="负责人确认本批项目范围，后续批次待讨论。",
                            summary_request="生成项目纪要",
                            personalization_preferences={"summary_style": "structured", "meeting_scene": scene},
                        )
                        with tempfile.TemporaryDirectory() as directory:
                            root = Path(directory)
                            pipeline = build_multi_agent_v2_pipeline(
                                memory=FakeMemoryBackend(), llm_call=model, user_id="u",
                                project="p", output_dir=root, config=config,
                            )
                            result = await pipeline.run(request)
                            self.assertEqual(RunStatus.COMPLETED, result.status)
                            context = json.loads((root / "synthesis_context.json").read_text("utf-8"))
                            self.assertEqual(scene, context["template"]["meeting_scene"])
                            report = json.loads((root / "model_input_report.json").read_text("utf-8"))
                            self.assertTrue(report["history_record_ids"])
                            self.assertEqual(scene, report["template_selection"]["meeting_scene"])
                            self.assertEqual("template_worker", report["template_selection"]["source"])
                            prompt = (root / "multi_agent_summary_prompt.txt").read_text("utf-8")
                            self.assertEqual(1, prompt.count(request.current_material))
                            self.assertEqual(result.final_summary, read_artifact_snapshot(root).formal_summary.rstrip())
                            archive = create_safe_artifact_archive(root)
                            with zipfile.ZipFile(archive) as bundle:
                                self.assertIn("model_input_report.json", bundle.namelist())
                                self.assertNotIn("current_material.txt", bundle.namelist())

    async def test_model_worker_can_refine_planner_scene_with_one_tool(self):
        registry = ToolRegistry([TemplateAdaptationTool()])
        context = make_context(
            task_type=TaskType.TEMPLATE_ADAPTATION, tool_registry=registry,
            allowed_tools=("template_adaptation",),
            inputs={"requested_scene": "general"}, max_tool_calls=1,
        )
        model = SequenceModelClient([
            action("call_tool", tool_calls=[{
                "tool_name": "template_adaptation",
                "arguments": {
                    "summary_request": context.request.summary_request,
                    "preferences": context.request.personalization_preferences,
                    "requested_style": "structured", "requested_scene": "decision_review",
                },
            }]),
            complete_from_observations,
        ])
        result = await make_worker(TaskType.TEMPLATE_ADAPTATION, model).execute(context)
        self.assertEqual(AgentStatus.SUCCESS, result.status)
        self.assertEqual("decision_review", result.payload["meeting_scene"])
        self.assertEqual(2, result.worker_execution.final_budget.model_calls_used)
        self.assertEqual(1, result.worker_execution.final_budget.tool_calls_used)

    async def test_rule_and_fallback_respect_explicit_scene_and_plan_suggestion(self):
        for explicit, expected in (("auto", "focused_discussion"), ("project_sync", "project_sync")):
            for fallback in (False, True):
                with self.subTest(explicit=explicit, fallback=fallback):
                    context = make_context(
                        task_type=TaskType.TEMPLATE_ADAPTATION,
                        tool_registry=ToolRegistry([TemplateAdaptationTool()]),
                        allowed_tools=("template_adaptation",),
                        inputs={"requested_scene": "focused_discussion"}, max_tool_calls=1,
                    )
                    request = context.request.model_copy(update={"personalization_preferences": {
                        "summary_style": "natural", "meeting_scene": explicit,
                    }})
                    context = replace(context, request=request)
                    worker = (
                        make_worker(TaskType.TEMPLATE_ADAPTATION, SequenceModelClient(["invalid"]), fallback=True)
                        if fallback else TemplateAdaptationWorker()
                    )
                    result = await worker.execute(context)
                    self.assertEqual(AgentStatus.SUCCESS, result.status)
                    self.assertEqual(expected, result.payload["meeting_scene"])
                    self.assertEqual("natural", result.payload["format_kind"])
                    if fallback:
                        self.assertTrue(result.worker_execution.fallback_reason)

    async def test_planner_accepts_optional_scene_and_rejects_unknown_scene(self):
        for scene, expected_source in (("decision_review", PlanSource.MODEL), ("invented", PlanSource.RULE_FALLBACK)):
            draft = valid_model_draft()
            draft["tasks"].append({
                "task_id": "template", "task_type": "template_adaptation",
                "description": "选择模板", "dependencies": [], "critical": False,
                "timeout_seconds": 60, "allowed_tools": ["template_adaptation"],
                "inputs": {"requested_style": "structured", "requested_scene": scene},
                "max_model_calls": 0, "max_tool_calls": 1,
            })
            model = SequenceModelClient([json.dumps(draft)])
            with tempfile.TemporaryDirectory() as directory:
                planner = ModelDrivenPlanner(
                    model_client=model, fallback_planner=RuleBasedPlanner(),
                    output_dir=Path(directory), repair_attempts=0,
                )
                plan = await planner.create_plan(make_initial_input())
            self.assertEqual(expected_source, plan.source)
            if expected_source == PlanSource.MODEL:
                self.assertEqual(scene, plan.tasks[-1].inputs["requested_scene"])


if __name__ == "__main__":
    unittest.main()
