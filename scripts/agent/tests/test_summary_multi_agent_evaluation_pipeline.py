from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.runtime import build_multi_agent_v2_pipeline  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    EvaluationStatus,
    RunConfig,
    RunStatus,
    SummaryRequest,
)
from scripts.agent.tests.test_summary_multi_agent_pipeline import (  # noqa: E402
    EmptyMemoryBackend,
    FakeSummaryModel,
)


class EvaluationModel(FakeSummaryModel):
    def __init__(
        self,
        *,
        fail_evaluation: bool = False,
        include_ungrounded_current_fact: bool = False,
    ) -> None:
        super().__init__()
        self.fail_evaluation = fail_evaluation
        self.include_ungrounded_current_fact = include_ungrounded_current_fact

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Evaluation " not in prompt:
            return super().__call__(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.fail_evaluation:
            return "not-json"
        if "Evaluation Source Fact Builder" in prompt:
            segment = re.search(
                r"<evaluation_current_segment>\n(.*?)\n</evaluation_current_segment>",
                prompt,
                re.DOTALL,
            ).group(1)
            quote = segment[: min(6, len(segment))]
            facts = [
                {
                    "fact": "负责人确认项目范围",
                    "fact_type": "decision",
                    "state": "confirmed",
                    "salience": "must_include",
                    "entities": [],
                    "evidence_quote": quote,
                    "speaker": None,
                    "turn_id": None,
                    "classification_reason": "摘要请求需要",
                }
            ]
            if self.include_ungrounded_current_fact:
                facts.append(
                    {
                        "fact": "模型虚构的额外事实",
                        "fact_type": "fact",
                        "state": "unknown",
                        "salience": "useful_optional",
                        "entities": [],
                        "evidence_quote": "原文中不存在的证据",
                        "speaker": None,
                        "turn_id": None,
                        "classification_reason": "用于测试 grounding 守卫",
                    }
                )
            return json.dumps({"facts": facts}, ensure_ascii=False)
        if "Evaluation Summary Fact Extractor" in prompt:
            segment = re.search(
                r"<evaluation_summary_segment>\n(.*?)\n</evaluation_summary_segment>",
                prompt,
                re.DOTALL,
            ).group(1)
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "负责人确认项目范围",
                            "evidence_quote": segment,
                            "claimed_scope": "current",
                            "entities": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Evaluation Fact Matcher" in prompt:
            summary_payload = json.loads(
                re.search(
                    r"<evaluation_summary_facts>\n(.*?)\n</evaluation_summary_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            source_payload = json.loads(
                re.search(
                    r"<evaluation_source_facts>\n(.*?)\n</evaluation_source_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            source_id = source_payload[0]["fact_id"]
            return json.dumps(
                {
                    "matches": [
                        {
                            "summary_fact_id": item["summary_fact_id"],
                            "status": "supported_full",
                            "matched_source_fact_ids": [source_id],
                            "primary_source_fact_id": source_id,
                            "stale_history": False,
                            "reason_code": "same_fact_and_state",
                            "reason": "内容和状态一致",
                        }
                        for item in summary_payload
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected evaluation prompt")


class EvaluationPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_sidecar_evaluates_formal_candidate_zero(self) -> None:
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            source_time="2026-08-25",
            summary_request="生成项目纪要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=EvaluationModel(),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                    evaluation_enabled=True,
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED, result.status)
            self.assertEqual(EvaluationStatus.COMPLETED, result.evaluation_status)
            self.assertEqual(0, result.selected_candidate_index)
            self.assertEqual(
                "这是由全部当前材料分段和 Worker 证据生成的摘要。",
                result.final_summary,
            )
            metrics = json.loads(
                (output_dir / "evaluation_metrics.json").read_text("utf-8")
            )
            self.assertEqual(0, metrics["evaluated_candidate_index"])
            self.assertEqual(
                hashlib.sha256(
                    (output_dir / "summary_multi_agent.md").read_bytes()
                ).hexdigest(),
                metrics["evaluated_summary_sha256"],
            )
            self.assertLessEqual(metrics["execution_stats"]["model_call_count"], 48)
            self.assertEqual(
                1, metrics["execution_stats"]["evaluation_current_segment_count"]
            )
            self.assertEqual(
                1, metrics["execution_stats"]["evaluation_summary_segment_count"]
            )
            self.assertEqual(1.0, metrics["core_metrics"]["must_fact_recall"]["value"])
            for name in (
                "evaluation_source_facts.json",
                "evaluation_summary_facts.json",
                "evaluation_fact_matches.json",
                "evaluation_metrics.json",
                "evaluation_report.md",
                "evaluation_model_trace.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    async def test_evaluation_failure_preserves_successful_summary(self) -> None:
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            summary_request="生成项目纪要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=EvaluationModel(fail_evaluation=True),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                    evaluation_enabled=True,
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual(EvaluationStatus.FAILED, result.evaluation_status)
            self.assertTrue(result.final_summary)
            self.assertEqual(
                result.final_summary,
                (output_dir / "summary_multi_agent.md").read_text("utf-8").strip(),
            )
            metrics = json.loads(
                (output_dir / "evaluation_metrics.json").read_text("utf-8")
            )
            self.assertEqual("failed", metrics["status"])

    async def test_partial_ungrounded_fact_is_discarded_with_warning(self) -> None:
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            summary_request="生成项目纪要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=EvaluationModel(include_ungrounded_current_fact=True),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                    evaluation_enabled=True,
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual(EvaluationStatus.COMPLETED, result.evaluation_status)
            self.assertTrue(
                any(
                    "discarded 1 current evaluation facts" in item
                    for item in result.warnings
                )
            )
            facts = json.loads(
                (output_dir / "evaluation_source_facts.json").read_text("utf-8")
            )
            self.assertEqual(1, len(facts))


if __name__ == "__main__":
    unittest.main()
