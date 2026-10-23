from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.runtime import build_multi_agent_v2_pipeline  # noqa: E402
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.review.blue import (  # noqa: E402
    BlueAgent,
    BlueDraftRepair,
    BlueDraftResult,
)
from summary_multi_agent.review.convergence import (  # noqa: E402
    build_issue_signature,
    calculate_text_change_ratio,
)
from summary_multi_agent.review.verify import ReviewVerifier  # noqa: E402
from summary_multi_agent.review.model_io import complete_validated_json  # noqa: E402
from summary_multi_agent.review.red import (  # noqa: E402
    RedDraftIssue,
    validate_red_draft_report,
)
from summary_multi_agent.schemas import (  # noqa: E402
    RedReviewReport,
    ReviewAction,
    ReviewIssue,
    ReviewTermination,
    RunConfig,
    RunStatus,
    SummaryRequest,
)
from summary_multi_agent.working_memory import initialize_working_state  # noqa: E402


SOURCE_TEXT = "项目进展正常。风险登记需要补充负责人。"
ORIGINAL_SUMMARY = "项目进展正常。"
REPAIRED_SUMMARY = "项目进展正常。风险登记需要补充负责人。"


class ReviewMemoryBackend:
    def __init__(self, *, with_record: bool = False) -> None:
        self.with_record = with_record
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, filters, top_k):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        if not self.with_record or filters.get("memory_type") != "task":
            return {"results": []}
        return {
            "results": [
                {
                    "id": "historical_task_1",
                    "memory": "历史负责人是李明",
                    "score": 0.91,
                    "metadata": {
                        "typed_memory_id": "historical_task_1",
                        "memory_type": "task",
                        "content_text": "历史负责人是李明",
                        "source_time": "2026-08-01",
                        "evidence_span": "历史负责人是李明",
                    },
                }
            ]
        }


class ReviewScenarioModel:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []
        self.judge_call_count = 0

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if "Current Material Mapper" in prompt:
            match = re.search(
                r"<current_segment>\n(.*?)\n</current_segment>", prompt, re.DOTALL
            )
            assert match is not None
            segment_text = match.group(1)
            return json.dumps(
                {
                    "items": [
                        {
                            "category": "risk",
                            "statement": segment_text,
                            "evidence_quote": segment_text,
                            "entities": [],
                            "time_expression": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Summarizer Agent" in prompt:
            return ORIGINAL_SUMMARY
        if "Red Issue Adjudicator" in prompt:
            if self.scenario == "adjudication_malformed":
                return "not-json"
            issue_ids = list(
                dict.fromkeys(
                    re.findall(r'"issue_id": "(review_issue_[a-f0-9]+)"', prompt)
                )
            )
            decisions = []
            for index, issue_id in enumerate(issue_ids):
                dismiss = self.scenario in {"false_positive", "protected_dismiss"} and index == 0
                decisions.append(
                    {
                        "issue_id": issue_id,
                        "keep": not dismiss,
                        "reason_code": "already_covered" if dismiss else "confirmed",
                        "reason": "候选摘要已经覆盖" if dismiss else "存在可执行问题",
                    }
                )
            return json.dumps(
                {"decisions": decisions, "warnings": []}, ensure_ascii=False
            )
        if "Red Agent" in prompt:
            if self.scenario == "malformed":
                return "not-json"
            if "<current_segment>" not in prompt and self.scenario == "red_bad_calendar":
                return self._red_issues(
                    [
                        {
                            "issue_type": "time_error",
                            "description": (
                                "摘要中的本周五（2026年8月7日）有误，"
                                "本周五应为8月8日。"
                            ),
                            "evidence_quotes": [],
                        }
                    ]
                )
            if "<current_segment>" not in prompt and self.scenario == "red_valid_calendar":
                return self._red_issues(
                    [
                        {
                            "issue_type": "time_error",
                            "description": "摘要误写本周五（2026年3月21日）。",
                            "evidence_quotes": [],
                        }
                    ]
                )
            if "<current_segment>" not in prompt and self.scenario == "target_leak":
                return self._red_issues(
                    [
                        {
                            "issue_type": "critical_omission",
                            "description": "遗漏张雨晴负责风险登记的任务。",
                            "evidence_quotes": [],
                            "target_entities": ["张雨晴"],
                        }
                    ]
                )
            if (
                "<current_segment>" not in prompt
                and self.scenario
                in {"current_fact_alias", "unknown_current_fact_alias"}
            ):
                fact_match = re.search(
                    r'"item_id": "(current_fact_[a-f0-9]+)"', prompt
                )
                assert fact_match is not None
                evidence_id = (
                    fact_match.group(1)
                    if self.scenario == "current_fact_alias"
                    else "current_fact_unknown0000"
                )
                return self._red_issue(
                    evidence_quotes=[],
                    evidence_ids=[evidence_id],
                    route_to="summary_writer",
                    needs_verification=False,
                )
            if "<current_segment>" in prompt and self.scenario == "false_positive":
                return self._red_issues(
                    [
                        {
                            "issue_type": "other",
                            "description": "遗漏项目进展正常",
                            "evidence_quotes": ["项目进展正常。"],
                        },
                        {
                            "issue_type": "critical_omission",
                            "description": "遗漏负责人风险",
                            "evidence_quotes": ["风险登记需要补充负责人。"],
                        },
                    ]
                )
            if (
                "<current_segment>" in prompt
                and self.scenario
                in {"accepted_tie_with_deferred", "issue_reappearance"}
            ):
                candidate = self._tagged(prompt, "candidate_summary")
                issue_types = ["tie_issue_a", "tie_issue_b"]
                if (
                    self.scenario == "accepted_tie_with_deferred"
                    and "[review-round-1]" in candidate
                ):
                    issue_types = ["tie_issue_c", "tie_issue_d"]
                return self._red_issues(
                    [
                        {
                            "issue_type": issue_types[0],
                            "description": "first traceable review issue",
                            "evidence_quotes": [SOURCE_TEXT],
                        },
                        {
                            "issue_type": issue_types[1],
                            "description": "second traceable review issue",
                            "evidence_quotes": [SOURCE_TEXT],
                        },
                    ]
                )
            if "<current_segment>" in prompt and self.scenario in {
                "repair",
                "adjudication_malformed",
                "blue_malformed",
                "blue_verify",
                "converged",
                "oscillation",
                "segment_delete",
                "judge_bad_date",
                "blue_bad_calendar",
                "text_convergence",
                "text_churn",
            }:
                return self._red_issue(
                    evidence_quotes=["风险登记需要补充负责人。"],
                    route_to="summary_writer",
                    needs_verification=False,
                )
            if "<current_segment>" not in prompt and self.scenario == "verify":
                return self._red_issue(
                    evidence_quotes=[],
                    route_to="semantic_memory",
                    needs_verification=True,
                )
            if "<current_segment>" not in prompt and self.scenario == "protected_dismiss":
                evidence_match = re.search(
                    r'"evidence_id": "(evidence_[a-f0-9]+)"', prompt
                )
                assert evidence_match is not None
                return self._red_issue(
                    evidence_quotes=[],
                    evidence_ids=[evidence_match.group(1)],
                    route_to="summary_writer",
                    needs_verification=False,
                    severity="high",
                )
            if "<current_segment>" not in prompt and self.scenario == "issue_budget":
                return self._red_issues(
                    [
                        {
                            "issue_type": f"budget_issue_{index}",
                            "description": f"预算候选问题 {index}",
                            "evidence_quotes": [],
                        }
                        for index in range(8)
                    ]
                )
            if "<current_segment>" not in prompt and self.scenario == "issue_entities":
                return self._red_issues(
                    [
                        {
                            "issue_type": "owner_error",
                            "description": "任务责任人需要核对",
                            "evidence_quotes": [],
                            "target_entities": [" 周明远 ", "周明远", "赵梓轩"],
                        }
                    ]
                )
            return json.dumps({"issues": [], "warnings": []}, ensure_ascii=False)
        if "Blue Agent" in prompt:
            if self.scenario == "blue_malformed":
                return '{"candidate_summary": "截断输出"'
            issue_id = self._first_issue_id(prompt)
            if self.scenario in {
                "accepted_tie_with_deferred",
                "issue_reappearance",
            }:
                source = self._tagged(prompt, "source_candidate")
                suffix = (
                    " [review-round-1]"
                    if "[review-round-1]" not in source
                    else " [review-round-2]"
                )
                candidate = source + suffix
            elif self.scenario == "text_convergence":
                candidate = self._tagged(prompt, "source_candidate") + "x"
            elif self.scenario == "text_churn":
                source = self._tagged(prompt, "source_candidate")
                candidate = source + (f"\nchange-{self.judge_call_count}-" + "z" * 200)
            elif self.scenario == "blue_bad_calendar":
                candidate = ORIGINAL_SUMMARY + "任务安排在本周五（2026年3月21日）。"
            elif self.scenario == "segment_delete":
                candidate = "项目内容被删除。"
            elif self.scenario == "verify":
                candidate = "项目进展正常。历史负责人是李明。"
            elif self.scenario == "converged":
                candidate = ORIGINAL_SUMMARY
            elif self.scenario == "oscillation" and REPAIRED_SUMMARY in self._tagged(
                prompt, "source_candidate"
            ):
                candidate = ORIGINAL_SUMMARY
            else:
                candidate = REPAIRED_SUMMARY
            return json.dumps(
                {
                    "candidate_summary": candidate,
                    "repairs": [
                        {
                            "issue_id": issue_id,
                            "action": (
                                "delete"
                                if self.scenario == "segment_delete"
                                else
                                "verify"
                                if self.scenario == "blue_verify"
                                else "modify"
                                if self.scenario == "verify"
                                else "add"
                            ),
                            "applied": True,
                            "rationale": "根据已核验的证据进行局部修复",
                            "evidence_ids": [],
                        }
                    ],
                    "unresolved_issue_ids": [],
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        if "Summary Multi-Agent V2 的 Judge" in prompt:
            self.judge_call_count += 1
            if (
                self.scenario == "judge_bad_date"
                and "上一次输出未通过 JSON Schema 校验" not in prompt
            ):
                return json.dumps(
                    {
                        "disposition": "repair",
                        "score": 0.5,
                        "dimension_scores": {"factual_consistency": 0.2},
                        "rationale": "日期仍错误，本周五应为2026年8月8日。",
                        "remaining_issue_ids": [self._first_issue_id(prompt)],
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            disposition = (
                "repair"
                if self.scenario
                in {
                    "converged",
                    "oscillation",
                    "text_convergence",
                    "text_churn",
                }
                else "accept"
            )
            remaining = (
                [self._first_issue_id(prompt)]
                if disposition == "repair"
                else []
            )
            score = 0.6 if disposition == "repair" else 0.95
            if self.scenario in {"text_convergence", "text_churn"}:
                score = [0.6, 0.61, 0.615, 0.616][
                    min(self.judge_call_count - 1, 3)
                ]
            return json.dumps(
                {
                    "disposition": disposition,
                    "score": score,
                    "dimension_scores": {
                        "factual_consistency": 0.95,
                        "coverage": 0.95,
                        "personalization": 0.9,
                        "historical_memory_use": 0.9,
                        "template_adherence": 0.9,
                        "conciseness": 0.9,
                    },
                    "rationale": "候选已解决本轮严重问题",
                    "remaining_issue_ids": remaining,
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected model prompt")

    @staticmethod
    def _first_issue_id(prompt: str) -> str:
        match = re.search(r'"issue_id": "(review_issue_[a-f0-9]+)"', prompt)
        assert match is not None
        return match.group(1)

    @staticmethod
    def _tagged(prompt: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL
        )
        assert match is not None
        return match.group(1)

    @staticmethod
    def _red_issue(
        *,
        evidence_quotes: list[str],
        evidence_ids: list[str] | None = None,
        route_to: str,
        needs_verification: bool,
        severity: str = "medium",
    ) -> str:
        return json.dumps(
            {
                "issues": [
                    {
                        "issue_type": "critical_omission",
                        "severity": severity,
                        "section": "风险",
                        "description": "遗漏了会影响后续执行的负责人风险",
                        "summary_quote": "",
                        "evidence_ids": evidence_ids or [],
                        "evidence_quotes": evidence_quotes,
                        "related_task_ids": [],
                        "recommended_action": "modify" if needs_verification else "add",
                        "repair_instruction": "仅补充已核验的负责人信息",
                        "route_to": route_to,
                        "needs_verification": needs_verification,
                        "verification_query": "历史负责人",
                        "confidence": 0.9,
                    }
                ],
                "warnings": [],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _red_issues(items: list[dict[str, Any]]) -> str:
        issues = []
        for item in items:
            issues.append(
                {
                    "issue_type": item["issue_type"],
                    "severity": "high",
                    "section": "测试章节",
                    "description": item["description"],
                    "summary_quote": "",
                    "evidence_ids": [],
                    "evidence_quotes": item["evidence_quotes"],
                    "target_entities": item.get("target_entities", []),
                    "related_task_ids": [],
                    "recommended_action": "add",
                    "repair_instruction": "局部修复",
                    "route_to": "summary_writer",
                    "needs_verification": False,
                    "verification_query": "",
                    "confidence": 0.9,
                }
            )
        return json.dumps({"issues": issues, "warnings": []}, ensure_ascii=False)


class ReviewPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _run_scenario(
        self,
        scenario: str,
        output_dir: Path,
        *,
        with_memory_record: bool = False,
    ):
        memory = ReviewMemoryBackend(with_record=with_memory_record)
        model = ReviewScenarioModel(scenario)
        config = RunConfig(
            workflow_mode="multi_agent_v2",
            planner_mode="rule",
            review_enabled=True,
            review_shadow_mode=True,
            max_review_rounds=1,
            schema_repair_attempts=1,
        )
        pipeline = build_multi_agent_v2_pipeline(
            memory=memory,
            llm_call=model,
            user_id="user",
            project="project",
            output_dir=output_dir,
            config=config,
        )
        result = await pipeline.run(
            SummaryRequest(
                current_material=SOURCE_TEXT,
                source_time="2026-08-06",
                summary_request="生成项目摘要，保留关键风险和责任信息。",
            )
        )
        return result, memory, model

    async def test_no_issue_review_accepts_original_without_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("no_issue", output_dir)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            self.assertEqual(0, result.selected_candidate_index)
            self.assertEqual(0, result.recommended_candidate_index)
            self.assertEqual(ORIGINAL_SUMMARY, result.final_summary)
            report = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual("accepted", report["termination"])
            self.assertEqual(1, len(report["rounds"]))
            self.assertTrue(report["rounds"][0]["red_report"]["checked_segment_ids"])
            model_trace = json.loads(
                (output_dir / "review_model_trace.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("Red Agent", model_trace["red"][0]["attempts"][0]["prompt"])
            self.assertEqual(0.0, model_trace["judge"][0]["attempts"][0]["temperature"])
            synthesis_prompt = (output_dir / "multi_agent_summary_prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("Memory metadata.source_time", synthesis_prompt)
            self.assertFalse(
                (output_dir / "summary_multi_agent_reviewed_candidate.md").exists()
            )

    async def test_global_red_uses_compact_nonduplicated_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _, _, model = await self._run_scenario(
                "no_issue",
                output_dir,
                with_memory_record=True,
            )

            prompt = next(
                call["prompt"]
                for call in model.calls
                if "第 1 轮全局审核" in call["prompt"]
            )
            self.assertIn("CompactReviewContext", prompt)
            self.assertIn('"current_facts"', prompt)
            self.assertIn('"historical_records"', prompt)
            self.assertIn("历史负责人是李明", prompt)
            self.assertIn("最多 6 个 issue", prompt)
            self.assertNotIn('"segment_analyses"', prompt)

    async def test_repaired_candidate_is_saved_but_does_not_replace_formal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("repair", output_dir)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual(ORIGINAL_SUMMARY, result.final_summary)
            self.assertEqual(0, result.selected_candidate_index)
            self.assertEqual(1, result.recommended_candidate_index)
            self.assertEqual(
                ORIGINAL_SUMMARY,
                (output_dir / "summary_multi_agent.md").read_text(
                    encoding="utf-8"
                ).strip(),
            )
            self.assertEqual(
                REPAIRED_SUMMARY,
                (output_dir / "summary_multi_agent_reviewed_candidate.md").read_text(
                    encoding="utf-8"
                ).strip(),
            )
            red_reports = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )
            evidence_id = red_reports[0]["issues"][0]["evidence_ids"][0]
            evidence = json.loads(
                (output_dir / "evidence_index.json").read_text(encoding="utf-8")
            )[evidence_id]
            self.assertEqual("风险登记需要补充负责人。", evidence["evidence_text"])

    async def test_review_failure_preserves_last_trusted_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, model = await self._run_scenario("malformed", output_dir)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual(ReviewTermination.REVIEW_FAILED, result.review_termination)
            self.assertEqual(ORIGINAL_SUMMARY, result.final_summary)
            self.assertEqual(2, len([call for call in model.calls if "Red Agent" in call["prompt"]]))
            self.assertTrue(any("review failed" in item for item in result.warnings))

    async def test_completed_red_and_verify_artifacts_survive_blue_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("blue_malformed", output_dir)

            self.assertEqual(ReviewTermination.REVIEW_FAILED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )
            verification = json.loads(
                (output_dir / "review_verification_results.json").read_text(
                    encoding="utf-8"
                )
            )
            blue = json.loads(
                (output_dir / "review_blue_results.json").read_text(encoding="utf-8")
            )
            traces = json.loads(
                (output_dir / "review_model_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, len(red))
            self.assertEqual(1, len(verification))
            self.assertEqual([], blue)
            self.assertTrue(
                all(
                    attempt["failure_kind"] == "likely_output_truncation"
                    for attempt in traces["blue"][0]["attempts"]
                )
            )

    async def test_red_adjudication_dismisses_already_covered_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, model = await self._run_scenario("false_positive", output_dir)

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(2, red["candidate_issue_count"])
            self.assertEqual(1, len(red["issues"]))
            self.assertEqual(0, len(red["deferred_issues"]))
            dispositions = {
                item["disposition"] for item in red["issue_dispositions"]
            }
            self.assertEqual({"kept", "dismissed"}, dispositions)
            adjudication_prompt = next(
                call["prompt"]
                for call in model.calls
                if "Red Issue Adjudicator" in call["prompt"]
            )
            self.assertIn("完整 Current Fact Catalog", adjudication_prompt)
            self.assertIn(SOURCE_TEXT, adjudication_prompt)

    async def test_red_adjudication_failure_uses_bounded_fail_closed_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario(
                "adjudication_malformed", output_dir
            )

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            self.assertTrue(
                any(
                    "adjudication failed" in warning
                    for warning in result.warnings
                )
            )
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(1, len(red["issues"]))
            self.assertEqual(
                "adjudicator_missing",
                red["issue_dispositions"][0]["reason_code"],
            )

    async def test_evidence_backed_global_high_issue_can_be_adjudicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("protected_dismiss", output_dir)

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(0, len(red["issues"]))
            self.assertEqual(
                "already_covered",
                red["issue_dispositions"][0]["reason_code"],
            )

    async def test_red_defers_issues_beyond_per_round_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("issue_budget")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=1,
                review_red_max_candidates_per_audit=8,
                review_red_adjudication_batch_size=3,
                review_max_issues_per_round=3,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            self.assertEqual(ReviewTermination.CONVERGED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(8, red["candidate_issue_count"])
            self.assertEqual(3, len(red["issues"]))
            self.assertEqual(5, len(red["deferred_issues"]))
            self.assertEqual(
                5,
                sum(
                    item["disposition"] == "deferred"
                    for item in red["issue_dispositions"]
                ),
            )
            adjudication_calls = [
                call
                for call in model.calls
                if "Red Issue Adjudicator" in call["prompt"]
            ]
            self.assertEqual(3, len(adjudication_calls))
            traces = json.loads(
                (output_dir / "review_model_trace.json").read_text(encoding="utf-8")
            )
            adjudication_traces = [
                item for item in traces["red"] if item["stage"] == "red_adjudication"
            ]
            self.assertEqual([1, 2, 3], [item["batch_index"] for item in adjudication_traces])
            self.assertTrue(all(item["batch_count"] == 3 for item in adjudication_traces))
            judge_prompt = next(
                call["prompt"]
                for call in model.calls
                if "Summary Multi-Agent V2 的 Judge" in call["prompt"]
            )
            self.assertNotIn('"deferred_issues"', judge_prompt)
            current_material_block = judge_prompt.split(
                "<current_material>\n", 1
            )[1].split("\n</current_material>", 1)[0]
            self.assertEqual(SOURCE_TEXT, current_material_block)
            input_report = json.loads(
                (output_dir / "model_call_input_report.json").read_text(
                    encoding="utf-8"
                )
            )
            judge_call = next(
                item
                for item in input_report["calls"]
                if item["stage"] == "review_judge"
            )
            self.assertIn("current_material", judge_call["sections"])
            self.assertIn("candidate_summary", judge_call["sections"])
            self.assertIn(
                "instructions_and_structured_context",
                judge_call["sections"],
            )

    async def test_red_audit_candidate_budget_is_enforced_after_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("issue_budget")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=1,
                review_red_max_candidates_per_audit=4,
                review_max_issues_per_round=4,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(4, red["candidate_issue_count"])
            self.assertTrue(
                any("per-audit budget" in warning for warning in red["warnings"])
            )

    async def test_red_removes_target_entities_without_direct_issue_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            await self._run_scenario("issue_entities", output_dir)

            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual([], red["issues"][0]["target_entities"])
            self.assertTrue(
                any(
                    "removed target entities without direct issue evidence"
                    in warning
                    for warning in red["warnings"]
                )
            )

    async def test_red_discards_ungrounded_target_entity_association(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("target_leak", output_dir)

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual([], red["issues"])
            self.assertTrue(
                any(
                    "discarded issue with ungrounded target-entity association"
                    in warning
                    for warning in red["warnings"]
                )
            )

    async def test_red_ignores_time_issue_without_revalidating_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("red_bad_calendar", output_dir)

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual([], red["issues"])
            self.assertTrue(
                any(
                    "ignored time_error because temporal semantics are owned by Temporal Worker"
                    in warning
                    for warning in red["warnings"]
                )
            )

    async def test_red_ignores_time_issue_even_when_model_marks_it_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            await self._run_scenario("red_valid_calendar", output_dir)

            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual([], red["issues"])
            self.assertTrue(
                any(
                    "ignored time_error because temporal semantics are owned by Temporal Worker"
                    in warning
                    for warning in red["warnings"]
                )
            )

    async def test_red_resolves_current_fact_id_to_grounded_evidence_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            await self._run_scenario("current_fact_alias", output_dir)

            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            evidence_ids = red["issues"][0]["evidence_ids"]
            self.assertEqual(1, len(evidence_ids))
            self.assertTrue(evidence_ids[0].startswith("evidence_"))
            self.assertTrue(
                any(
                    "resolved Current Fact ids" in warning
                    for warning in red["warnings"]
                )
            )
            self.assertFalse(red["issues"][0]["needs_verification"])

    async def test_red_does_not_guess_unknown_current_fact_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            await self._run_scenario("unknown_current_fact_alias", output_dir)

            red = json.loads(
                (output_dir / "review_red_reports.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual([], red["issues"][0]["evidence_ids"])
            self.assertTrue(red["issues"][0]["needs_verification"])
            self.assertTrue(
                any(
                    "discarded unknown evidence ids: current_fact_unknown0000"
                    in warning
                    for warning in red["warnings"]
                )
            )

    async def test_segment_only_issue_cannot_drive_destructive_blue_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("segment_delete", output_dir)

            self.assertEqual(ReviewTermination.CONVERGED, result.review_termination)
            blue = json.loads(
                (output_dir / "review_blue_results.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(ORIGINAL_SUMMARY, blue["candidate_summary"])
            self.assertFalse(blue["repairs"][0]["applied"])
            self.assertTrue(
                any("segment-only" in warning for warning in blue["warnings"])
            )

    async def test_verify_routes_through_read_only_worker_and_supports_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, memory, _ = await self._run_scenario(
                "verify", output_dir, with_memory_record=True
            )

            self.assertEqual(ReviewTermination.ACCEPTED, result.review_termination)
            verifications = json.loads(
                (output_dir / "review_verification_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("verified", verifications[0]["status"])
            self.assertTrue(verifications[0]["evidence_ids"])
            self.assertTrue(memory.calls)
            trace = (output_dir / "execution_trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event_type": "task.ad_hoc_started"', trace)
            self.assertIn('"event_type": "task.ad_hoc_finished"', trace)

    async def test_no_change_and_oscillation_terminate_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            converged, _, _ = await self._run_scenario(
                "converged", Path(temp_dir) / "converged"
            )
            self.assertEqual(ReviewTermination.CONVERGED, converged.review_termination)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            memory = ReviewMemoryBackend()
            model = ReviewScenarioModel("oscillation")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=3,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=memory,
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            oscillation = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )
            self.assertEqual(
                ReviewTermination.OSCILLATION,
                oscillation.review_termination,
            )
            report = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(report["rounds"]))

    async def test_equal_accepted_score_recommends_latest_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("accepted_tie_with_deferred")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=2,
                review_max_issues_per_round=1,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            self.assertEqual(ReviewTermination.MAX_ROUNDS, result.review_termination)
            self.assertEqual(2, result.recommended_candidate_index)
            review = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(
                    "Judge accepted selected issues" in warning
                    for warning in review["warnings"]
                )
            )

    async def test_low_score_and_text_change_converge_without_false_issue_oscillation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("text_convergence")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=4,
                review_min_improvement=0.02,
                review_text_change_convergence_threshold=0.5,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            self.assertEqual(ReviewTermination.CONVERGED, result.review_termination)
            review = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(3, len(review["rounds"]))
            self.assertTrue(
                all(not item["issue_reappearances"] for item in review["rounds"])
            )
            self.assertTrue(
                all(item["text_change_ratio"] > 0 for item in review["rounds"])
            )
            self.assertTrue(
                any(
                    "below both score-improvement and text-change" in warning
                    for warning in review["warnings"]
                )
            )

    async def test_resolved_issue_reappearance_terminates_as_oscillation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("issue_reappearance")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=3,
                review_max_issues_per_round=1,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            self.assertEqual(ReviewTermination.OSCILLATION, result.review_termination)
            review = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(review["rounds"]))
            reappearances = review["rounds"][1]["issue_reappearances"]
            self.assertEqual(1, len(reappearances))
            self.assertEqual(1, reappearances[0]["previous_round_index"])
            self.assertTrue(reappearances[0]["signature"])
            self.assertTrue(reappearances[0]["previous_issue_id"])
            self.assertTrue(reappearances[0]["current_issue_id"])
            self.assertTrue(
                any("resolved issue signature" in item for item in review["warnings"])
            )

    async def test_low_score_improvement_alone_does_not_claim_text_convergence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = ReviewScenarioModel("text_churn")
            config = RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                max_review_rounds=3,
                review_min_improvement=0.02,
                review_text_change_convergence_threshold=0.02,
                schema_repair_attempts=1,
            )
            pipeline = build_multi_agent_v2_pipeline(
                memory=ReviewMemoryBackend(),
                llm_call=model,
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=config,
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=SOURCE_TEXT,
                    source_time="2026-08-06",
                    summary_request="生成项目摘要，保留关键风险和责任信息。",
                )
            )

            self.assertEqual(ReviewTermination.MAX_ROUNDS, result.review_termination)
            review = json.loads(
                (output_dir / "review_run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(3, len(review["rounds"]))
            self.assertTrue(
                all(
                    item["text_change_ratio"]
                    > config.review_text_change_convergence_threshold
                    for item in review["rounds"]
                )
            )

    async def test_blue_cannot_claim_verify_as_a_content_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result, _, _ = await self._run_scenario("blue_verify", output_dir)

            self.assertEqual(ReviewTermination.CONVERGED, result.review_termination)
            self.assertEqual(ORIGINAL_SUMMARY, result.final_summary)
            blue = json.loads(
                (output_dir / "review_blue_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(blue[0]["repairs"][0]["applied"])
            self.assertIn(
                blue[0]["repairs"][0]["issue_id"],
                blue[0]["unresolved_issue_ids"],
            )


class ReviewConvergenceTests(unittest.TestCase):
    def test_text_change_ratio_normalizes_width_case_and_whitespace(self) -> None:
        self.assertEqual(0.0, calculate_text_change_ratio("Ａ  B\nC", "a b c"))
        self.assertGreater(calculate_text_change_ratio("alpha", "omega"), 0.5)

    def test_issue_signature_ignores_generated_id_and_description_wording(self) -> None:
        first = ReviewIssue(
            issue_id="review_issue_first",
            issue_type="owner_error",
            severity="high",
            section="任务安排",
            description="负责人错误",
            route_to="summary_writer",
            evidence_ids=("evidence_1",),
            target_entities=("张雨晴",),
        )
        rewritten = ReviewIssue(
            issue_id="review_issue_second",
            issue_type="owner_error",
            severity="medium",
            section="任务安排",
            description="任务被分配给了错误人员",
            route_to="semantic_memory",
            evidence_ids=("evidence_1",),
            segment_ids=("seg_0001",),
            target_entities=("张雨晴",),
        )
        different_evidence = rewritten.model_copy(
            update={"evidence_ids": ("evidence_2",)}
        )

        self.assertEqual(
            build_issue_signature(first),
            build_issue_signature(rewritten),
        )
        self.assertNotEqual(
            build_issue_signature(first),
            build_issue_signature(different_evidence),
        )


class ReviewConfigTests(unittest.TestCase):
    def test_active_replacement_mode_is_rejected_in_p5(self) -> None:
        with self.assertRaises(ValidationError):
            RunConfig(
                workflow_mode="multi_agent_v2",
                review_enabled=True,
                review_shadow_mode=False,
            )

    def test_red_adjudication_uses_small_default_batches(self) -> None:
        self.assertEqual(3, RunConfig().review_red_adjudication_batch_size)

    def test_review_text_change_uses_conservative_default_threshold(self) -> None:
        self.assertEqual(
            0.02,
            RunConfig().review_text_change_convergence_threshold,
        )

    def test_model_worker_budget_covers_action_completion_and_repairs(self) -> None:
        with self.assertRaises(ValidationError):
            RunConfig(
                worker_mode="model",
                worker_max_model_calls=3,
                worker_action_repair_attempts=1,
            )

    def test_review_verifier_does_not_create_temporal_review_tasks(self) -> None:
        issue = ReviewIssue(
            issue_type="time_error",
            severity="medium",
            description="verify a relative date",
            route_to="temporal_reasoning",
            needs_verification=True,
        )
        model_config = RunConfig(worker_mode="model")
        model_task = ReviewVerifier(
            orchestrator=None,  # type: ignore[arg-type]
            config=model_config,
        )._build_task(issue, 1)
        rule_task = ReviewVerifier(
            orchestrator=None,  # type: ignore[arg-type]
            config=RunConfig(worker_mode="rule"),
        )._build_task(issue, 1)

        self.assertIsNone(model_task)
        self.assertIsNone(rule_task)

    def test_temporal_review_issue_is_not_routed_to_verifier(self) -> None:
        material = "负责人昨天确认。张雨晴明天提交报告。下周一复盘。"
        request = SummaryRequest(
            current_material=material,
            source_time="2026-03-19",
        )
        state = initialize_working_state(
            request,
            build_raw_material_bundle(material, overlap_ratio=0.0),
        )
        issue = ReviewIssue(
            issue_type="time_error",
            severity="high",
            description="摘要中的明天日期需要核验",
            summary_quote="张雨晴明天提交报告",
            route_to="temporal_reasoning",
            needs_verification=True,
        )

        task = ReviewVerifier(
            orchestrator=None,  # type: ignore[arg-type]
            config=RunConfig(worker_mode="model"),
        )._build_task(issue, 1, state=state)

        self.assertIsNone(task)

    def test_temporal_verification_without_issue_expression_does_not_scan_all(self) -> None:
        material = "负责人昨天确认。张雨晴明天提交报告。"
        request = SummaryRequest(
            current_material=material,
            source_time="2026-03-19",
        )
        state = initialize_working_state(
            request,
            build_raw_material_bundle(material, overlap_ratio=0.0),
        )
        issue = ReviewIssue(
            issue_type="state_error",
            severity="high",
            description="历史任务状态需要补查",
            route_to="temporal_reasoning",
            needs_verification=True,
        )

        task = ReviewVerifier(
            orchestrator=None,  # type: ignore[arg-type]
            config=RunConfig(worker_mode="model"),
        )._build_task(issue, 1, state=state)

        self.assertIsNone(task)

    def test_blue_normalizes_applied_issue_out_of_unresolved_state(self) -> None:
        material = "项目状态已确认。"
        state = initialize_working_state(
            SummaryRequest(current_material=material, source_time="2026-03-19"),
            build_raw_material_bundle(material, overlap_ratio=0.0),
        )
        issue = ReviewIssue(
            issue_id="review_issue_state",
            issue_type="state_error",
            severity="high",
            description="删除错误状态",
            recommended_action="delete",
        )
        result = BlueAgent._normalize(
            state,
            "项目状态待确认。",
            RedReviewReport(
                round_index=1,
                candidate_index=0,
                passed=False,
                issues=(issue,),
            ),
            (),
            BlueDraftResult(
                candidate_summary="项目状态已确认。",
                repairs=(
                    BlueDraftRepair(
                        issue_id=issue.issue_id,
                        action=ReviewAction.DELETE,
                        applied=True,
                        rationale="删除错误状态",
                    ),
                ),
                unresolved_issue_ids=(issue.issue_id,),
            ),
            source_candidate_index=0,
            round_index=1,
        )

        self.assertTrue(result.repairs[0].applied)
        self.assertEqual((), result.unresolved_issue_ids)
        self.assertTrue(any("stale unresolved" in item for item in result.warnings))

    def test_blue_rejects_applied_claim_when_candidate_is_unchanged(self) -> None:
        material = "项目状态待确认。"
        state = initialize_working_state(
            SummaryRequest(current_material=material, source_time="2026-03-19"),
            build_raw_material_bundle(material, overlap_ratio=0.0),
        )
        issue = ReviewIssue(
            issue_id="review_issue_state",
            issue_type="state_error",
            severity="high",
            description="删除错误状态",
            recommended_action="delete",
        )
        result = BlueAgent._normalize(
            state,
            material,
            RedReviewReport(
                round_index=1,
                candidate_index=0,
                passed=False,
                issues=(issue,),
            ),
            (),
            BlueDraftResult(
                candidate_summary=material,
                repairs=(
                    BlueDraftRepair(
                        issue_id=issue.issue_id,
                        action=ReviewAction.DELETE,
                        applied=True,
                        rationale="声称已经修改",
                    ),
                ),
            ),
            source_candidate_index=0,
            round_index=1,
        )

        self.assertFalse(result.repairs[0].applied)
        self.assertEqual((issue.issue_id,), result.unresolved_issue_ids)
        self.assertTrue(any("did not change" in item for item in result.warnings))

    def test_red_draft_still_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RedDraftIssue.model_validate(
                {
                    "issue_type": "other",
                    "severity": "low",
                    "description": "测试问题",
                    "recommended_action": "modify",
                    "unexpected_field": "must remain forbidden",
                }
            )

    def test_red_known_output_variants_are_normalized_before_validation(self) -> None:
        report = validate_red_draft_report(
            [
                {
                    "target_section": "风险与阻塞",
                    "description": "遗漏风险负责人",
                    "repair_instruction": "补充负责人",
                    "evidence_quotes": "风险登记需要补充负责人。",
                    "target_entities": "李明",
                }
            ],
            segment_mode=True,
        )

        self.assertEqual(1, len(report.issues))
        issue = report.issues[0]
        self.assertEqual("风险与阻塞", issue.section)
        self.assertEqual("critical_omission", issue.issue_type)
        self.assertEqual("medium", issue.severity.value)
        self.assertEqual("add", issue.recommended_action.value)
        self.assertEqual(("风险登记需要补充负责人。",), issue.evidence_quotes)
        self.assertEqual(("李明",), issue.target_entities)
        self.assertTrue(any("root_array_to_report" in item for item in report.warnings))


class AsyncSequenceModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str:
        del temperature, max_tokens
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class ReviewModelIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_json_retry_requests_compact_complete_output(self) -> None:
        model = AsyncSequenceModel(
            ['{"value": "unfinished', '{"value": "complete"}']
        )
        trace: list[dict[str, Any]] = []

        result = await complete_validated_json(
            model_client=model,
            prompt="return JSON",
            validator=lambda value: value,
            max_tokens=3000,
            timeout_seconds=10,
            repair_attempts=1,
            trace=trace,
            trace_metadata={"stage": "test"},
        )

        self.assertEqual({"value": "complete"}, result)
        self.assertIn("长度被截断", model.prompts[1])
        self.assertEqual(
            "likely_output_truncation",
            trace[0]["attempts"][0]["failure_kind"],
        )
        self.assertNotIn("unfinished", model.prompts[1])

    async def test_repetitive_json_retry_does_not_echo_decoder_loop(self) -> None:
        repeated = "本周五应为2026-03-20，但当前材料日期为周四。"
        model = AsyncSequenceModel(
            [
                '{"issues":[{"description":"' + repeated * 12,
                '{"issues": [], "warnings": []}',
            ]
        )
        trace: list[dict[str, Any]] = []

        result = await complete_validated_json(
            model_client=model,
            prompt="return compact Red JSON",
            validator=lambda value: value,
            max_tokens=2400,
            timeout_seconds=10,
            repair_attempts=1,
            trace=trace,
            trace_metadata={"stage": "red_global"},
            compact_retry_hint="最多输出 6 个 issue。",
        )

        self.assertEqual({"issues": [], "warnings": []}, result)
        self.assertEqual(
            "repetitive_output",
            trace[0]["attempts"][0]["failure_kind"],
        )
        self.assertIn("连续重复", model.prompts[1])
        self.assertIn("最多输出 6 个 issue", model.prompts[1])
        self.assertNotIn(repeated, model.prompts[1])


if __name__ == "__main__":
    unittest.main()
