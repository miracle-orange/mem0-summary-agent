from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.evaluation.personalization import (  # noqa: E402
    calculate_personalization_metrics,
)
from summary_multi_agent.evaluation.schemas import (  # noqa: E402
    EvaluationClaimScope,
    EvaluationEvidence,
    EvaluationFactMatch,
    EvaluationFactState,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
    HistoricalDisposition,
    PersonalizationClaimAnnotation,
    PersonalizationClaimRelevance,
    PersonalizationSourceAnnotation,
    PersonalizationSourceRelevance,
    PersonalizationTargetStatus,
)
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    EvaluationStatus,
    RunConfig,
    RunStatus,
    SummaryRequest,
)
from scripts.agent.tests.test_summary_multi_agent_evaluation_pipeline import (  # noqa: E402
    EvaluationModel,
)
from scripts.agent.tests.test_summary_multi_agent_pipeline import (  # noqa: E402
    EmptyMemoryBackend,
)


def _source(
    fact_id: str,
    fact: str,
    *,
    scope: EvaluationSourceScope = EvaluationSourceScope.CURRENT,
) -> EvaluationSourceFact:
    return EvaluationSourceFact(
        fact_id=fact_id,
        source_scope=scope,
        fact=fact,
        state=EvaluationFactState.CONFIRMED,
        salience=EvaluationSalience.MUST_INCLUDE,
        evidence=(EvaluationEvidence(evidence_text=fact),),
        history_disposition=(
            HistoricalDisposition.ACTIVE
            if scope == EvaluationSourceScope.HISTORICAL
            else None
        ),
    )


def _claim(fact_id: str, fact: str, start: int) -> EvaluationSummaryFact:
    return EvaluationSummaryFact(
        summary_fact_id=fact_id,
        fact=fact,
        evidence_quote=fact,
        start_offset=start,
        end_offset=start + len(fact),
        claimed_scope=EvaluationClaimScope.CURRENT,
    )


class PersonalizationModel(EvaluationModel):
    """在原有整体评测 Fake 上增加严格目标实体分类响应。"""

    def __init__(self, *, fail_personalization: bool = False) -> None:
        super().__init__()
        self.fail_personalization = fail_personalization

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Target Personalization " not in prompt:
            return super().__call__(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.fail_personalization:
            return "not-json"
        target = re.search(r"target_entity: (.+)", prompt).group(1).strip()
        if "Fact Classifier" in prompt:
            payload = json.loads(
                re.search(
                    r"<personalization_source_facts>\n(.*?)\n</personalization_source_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            return json.dumps(
                {
                    "target_entity": target,
                    "annotations": [
                        {
                            "source_fact_id": item["source_fact_id"],
                            "relevance": "target_must_include",
                            "relation_types": ["direct_mention"],
                            "reason": "测试目标事实",
                        }
                        for item in payload
                    ],
                },
                ensure_ascii=False,
            )
        payload = json.loads(
            re.search(
                r"<personalization_summary_facts>\n(.*?)\n</personalization_summary_facts>",
                prompt,
                re.DOTALL,
            ).group(1)
        )
        return json.dumps(
            {
                "target_entity": target,
                "annotations": [
                    {
                        "summary_fact_id": item["summary_fact_id"],
                        "relevance": "target_directed",
                        "relation_types": ["direct_mention"],
                        "reason": "测试目标表达",
                    }
                    for item in payload
                ],
            },
            ensure_ascii=False,
        )


class PersonalizationMetricTests(unittest.TestCase):
    def test_target_metrics_reuse_existing_matches(self) -> None:
        sources = (
            _source("current_task", "张雨晴负责提交说明"),
            _source(
                "history_constraint",
                "张雨晴历史上需要保留三行过程",
                scope=EvaluationSourceScope.HISTORICAL,
            ),
            _source("common", "全班今晚提交作业"),
        )
        claims = (
            _claim("claim_task", "张雨晴负责提交说明", 0),
            _claim("claim_history", "张雨晴需保留三行过程", 20),
            _claim("claim_bad", "张雨晴负责未被证实的任务", 40),
            _claim("claim_common", "全班今晚提交作业", 70),
        )
        matches = (
            EvaluationFactMatch(
                summary_fact_id="claim_task",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("current_task",),
                primary_source_fact_id="current_task",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="same_fact",
            ),
            EvaluationFactMatch(
                summary_fact_id="claim_history",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("history_constraint",),
                primary_source_fact_id="history_constraint",
                primary_source_scope=EvaluationSourceScope.HISTORICAL,
                reason_code="same_fact",
            ),
            EvaluationFactMatch(
                summary_fact_id="claim_bad",
                match_status=EvaluationMatchStatus.UNSUPPORTED,
                reason_code="unsupported",
            ),
            EvaluationFactMatch(
                summary_fact_id="claim_common",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("common",),
                primary_source_fact_id="common",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="same_fact",
            ),
        )
        source_annotations = (
            PersonalizationSourceAnnotation(
                target_entity="张雨晴",
                source_fact_id="current_task",
                relevance=PersonalizationSourceRelevance.TARGET_MUST_INCLUDE,
            ),
            PersonalizationSourceAnnotation(
                target_entity="张雨晴",
                source_fact_id="history_constraint",
                relevance=PersonalizationSourceRelevance.TARGET_USEFUL_OPTIONAL,
            ),
            PersonalizationSourceAnnotation(
                target_entity="张雨晴",
                source_fact_id="common",
                relevance=PersonalizationSourceRelevance.NOT_TARGET_RELEVANT,
            ),
        )
        claim_annotations = tuple(
            PersonalizationClaimAnnotation(
                target_entity="张雨晴",
                summary_fact_id=item.summary_fact_id,
                relevance=(
                    PersonalizationClaimRelevance.NOT_TARGET_DIRECTED
                    if item.summary_fact_id == "claim_common"
                    else PersonalizationClaimRelevance.TARGET_DIRECTED
                ),
            )
            for item in claims
        )
        metrics = calculate_personalization_metrics(
            summary="测试摘要",
            target_entities=("张雨晴",),
            source_facts=sources,
            summary_facts=claims,
            matches=matches,
            source_annotations=source_annotations,
            claim_annotations=claim_annotations,
        )
        target = metrics.per_target[0]
        self.assertEqual(PersonalizationTargetStatus.RESOLVED, target.status)
        self.assertEqual(1.0, target.core_metrics["target_must_fact_recall"].value)
        self.assertEqual(1.0, target.core_metrics["target_useful_recall"].value)
        self.assertAlmostEqual(
            2 / 3, target.core_metrics["target_useful_precision"].value
        )
        self.assertAlmostEqual(
            1 / 3,
            target.safety_metrics["target_unsupported_attribution_rate"].value,
        )
        self.assertEqual(
            0.75, target.descriptive_metrics["target_focus_share"].value
        )

    def test_unresolved_target_keeps_empty_denominators_as_na(self) -> None:
        source = _source("common", "全班提交作业")
        claim = _claim("claim", "全班提交作业", 0)
        match = EvaluationFactMatch(
            summary_fact_id="claim",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("common",),
            primary_source_fact_id="common",
            primary_source_scope=EvaluationSourceScope.CURRENT,
            reason_code="same_fact",
        )
        metrics = calculate_personalization_metrics(
            summary="全班提交作业",
            target_entities=("不存在的人",),
            source_facts=(source,),
            summary_facts=(claim,),
            matches=(match,),
            source_annotations=(
                PersonalizationSourceAnnotation(
                    target_entity="不存在的人",
                    source_fact_id="common",
                    relevance=PersonalizationSourceRelevance.NOT_TARGET_RELEVANT,
                ),
            ),
            claim_annotations=(
                PersonalizationClaimAnnotation(
                    target_entity="不存在的人",
                    summary_fact_id="claim",
                    relevance=PersonalizationClaimRelevance.NOT_TARGET_DIRECTED,
                ),
            ),
        )
        target = metrics.per_target[0]
        self.assertEqual(PersonalizationTargetStatus.UNRESOLVED, target.status)
        self.assertIsNone(target.core_metrics["target_must_fact_recall"].value)
        self.assertIsNone(target.core_metrics["target_useful_precision"].value)

    def test_multiple_targets_publish_macro_and_micro_without_combined_score(self) -> None:
        source = _source("shared_task", "张雨晴和陈一诺共同完成任务")
        claim = _claim("shared_claim", "张雨晴和陈一诺共同完成任务", 0)
        match = EvaluationFactMatch(
            summary_fact_id="shared_claim",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("shared_task",),
            primary_source_fact_id="shared_task",
            primary_source_scope=EvaluationSourceScope.CURRENT,
            reason_code="same_fact",
        )
        targets = ("张雨晴", "陈一诺")
        metrics = calculate_personalization_metrics(
            summary="张雨晴和陈一诺共同完成任务",
            target_entities=targets,
            source_facts=(source,),
            summary_facts=(claim,),
            matches=(match,),
            source_annotations=tuple(
                PersonalizationSourceAnnotation(
                    target_entity=target,
                    source_fact_id="shared_task",
                    relevance=PersonalizationSourceRelevance.TARGET_MUST_INCLUDE,
                )
                for target in targets
            ),
            claim_annotations=tuple(
                PersonalizationClaimAnnotation(
                    target_entity=target,
                    summary_fact_id="shared_claim",
                    relevance=PersonalizationClaimRelevance.TARGET_DIRECTED,
                )
                for target in targets
            ),
        )
        self.assertEqual(2, len(metrics.per_target))
        self.assertEqual(1.0, metrics.macro_metrics["target_must_fact_recall"])
        self.assertEqual(
            1.0, metrics.micro_metrics["target_useful_precision"].value
        )
        self.assertNotIn("combined_score", metrics.macro_metrics)


class PersonalizationPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        target_entities: list[str],
        model: PersonalizationModel,
    ):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
        pipeline = build_multi_agent_v2_pipeline(
            memory=EmptyMemoryBackend(),
            llm_call=model,
            user_id="user",
            project="project",
            output_dir=output_dir,
            config=RunConfig(
                workflow_mode="multi_agent_v2",
                planner_mode="rule",
                evaluation_enabled=True,
                evaluation_repair_attempts=0,
            ),
        )
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            summary_request="生成提及负责人的项目纪要",
            target_entities=target_entities,
        )
        result = await pipeline.run(request)
        return temp_dir, output_dir, result, model

    async def test_explicit_target_triggers_personalization_branch(self) -> None:
        temp_dir, output_dir, result, model = await self._run(
            target_entities=["负责人"], model=PersonalizationModel()
        )
        with temp_dir:
            self.assertEqual(EvaluationStatus.COMPLETED, result.evaluation_status)
            self.assertEqual(
                EvaluationStatus.COMPLETED,
                result.personalization_evaluation_status,
            )
            metrics = json.loads(
                (output_dir / "evaluation_personalization_metrics.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual(["负责人"], metrics["target_entities"])
            self.assertTrue(metrics["per_target"])
            self.assertIn("entity_worker_result_count", metrics["diagnostics"])
            for name in (
                "evaluation_personalization_source_annotations.json",
                "evaluation_personalization_claim_annotations.json",
                "evaluation_personalization_metrics.json",
                "evaluation_personalization_report.md",
                "evaluation_personalization_model_trace.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            self.assertTrue(
                any(
                    "Target Personalization" in item["prompt"]
                    for item in model.calls
                )
            )

    async def test_request_text_alone_does_not_trigger_branch(self) -> None:
        temp_dir, output_dir, result, model = await self._run(
            target_entities=[], model=PersonalizationModel()
        )
        with temp_dir:
            self.assertEqual(
                EvaluationStatus.DISABLED,
                result.personalization_evaluation_status,
            )
            metrics = json.loads(
                (output_dir / "evaluation_personalization_metrics.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual("disabled", metrics["status"])
            self.assertFalse(
                any(
                    "Target Personalization" in item["prompt"]
                    for item in model.calls
                )
            )

    async def test_personalization_failure_isolated_from_overall_evaluation(self) -> None:
        temp_dir, output_dir, result, _ = await self._run(
            target_entities=["负责人"],
            model=PersonalizationModel(fail_personalization=True),
        )
        with temp_dir:
            self.assertEqual(RunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual(EvaluationStatus.COMPLETED, result.evaluation_status)
            self.assertEqual(
                EvaluationStatus.FAILED,
                result.personalization_evaluation_status,
            )
            overall = json.loads(
                (output_dir / "evaluation_metrics.json").read_text("utf-8")
            )
            personal = json.loads(
                (output_dir / "evaluation_personalization_metrics.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual("completed", overall["status"])
            self.assertEqual("failed", personal["status"])
            self.assertTrue((output_dir / "summary_multi_agent.md").is_file())


if __name__ == "__main__":
    unittest.main()
