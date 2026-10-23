from __future__ import annotations

import asyncio
import json
import re
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.adapters.model import SyncCallableModelClient  # noqa: E402
from summary_multi_agent.evaluation.comparison import (  # noqa: E402
    EvaluationComparisonRunner,
)
from summary_multi_agent.evaluation.evaluator import SummaryEvaluator  # noqa: E402
from summary_multi_agent.evaluation.metrics import (  # noqa: E402
    calculate_evaluation_metrics,
)
from summary_multi_agent.evaluation.personalization import (  # noqa: E402
    calculate_personalization_metrics,
)
from summary_multi_agent.evaluation.prompts import (  # noqa: E402
    build_current_fact_prompt,
)
from summary_multi_agent.evaluation.schemas import (  # noqa: E402
    EvaluationClaimScope,
    EvaluationCompareConfig,
    EvaluationCompareRequest,
    EvaluationEvidence,
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationFactRelationType,
    EvaluationFactState,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceDocument,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryDocument,
    EvaluationSummaryFact,
    EvaluationTransitionAssessment,
    HistoricalDisposition,
    PersonalizationClaimAnnotation,
    PersonalizationClaimRelevance,
    PersonalizationSourceAnnotation,
    PersonalizationSourceRelevance,
)
from summary_multi_agent.evaluation.snapshot import (  # noqa: E402
    EvaluationSourceSnapshotBuilder,
)
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import RunConfig, SummaryRequest  # noqa: E402


def _source(
    fact_id: str,
    fact: str,
    *,
    scope: EvaluationSourceScope,
    state: EvaluationFactState,
) -> EvaluationSourceFact:
    return EvaluationSourceFact(
        fact_id=fact_id,
        source_scope=scope,
        fact=fact,
        state=state,
        salience=EvaluationSalience.MUST_INCLUDE,
        evidence=(EvaluationEvidence(evidence_text=fact),),
        history_disposition=(
            HistoricalDisposition.UPDATED
            if scope == EvaluationSourceScope.HISTORICAL
            else None
        ),
    )


def _summary_fact(fact_id: str, fact: str) -> EvaluationSummaryFact:
    return EvaluationSummaryFact(
        summary_fact_id=fact_id,
        fact=fact,
        evidence_quote=fact,
        start_offset=0,
        end_offset=len(fact),
        claimed_scope=EvaluationClaimScope.CURRENT,
    )


class P73MetricSemanticsTests(unittest.TestCase):
    def test_online_snapshot_and_scorer_share_hard_budgets(self) -> None:
        evaluator = SummaryEvaluator(
            model_client=SyncCallableModelClient(lambda *args, **kwargs: "{}"),
            config=RunConfig(
                evaluation_max_model_calls=7,
                evaluation_personalization_max_model_calls=3,
            ),
        )
        self.assertIs(
            evaluator._snapshot_builder._model_io._model_client,
            evaluator._scorer._model_io._model_client,
        )
        self.assertIs(
            evaluator._snapshot_builder._personalization._model_io._model_client,
            evaluator._scorer._personalization._model_io._model_client,
        )
        self.assertEqual(7, evaluator._shared_overall_client._max_calls)
        self.assertEqual(3, evaluator._shared_personalization_client._max_calls)

    def test_current_source_prompt_is_independent_of_targets(self) -> None:
        material = "周明远决定本周不讲含参导数。"
        bundle = build_raw_material_bundle(material, source_id="current")
        first = SummaryRequest(
            current_material=material,
            source_time="2026-03-19",
            summary_request="生成课堂纪要",
            target_entities=["周明远"],
        )
        second = first.model_copy(update={"target_entities": ["张雨晴"]})
        self.assertEqual(
            build_current_fact_prompt(first, bundle.segments[0]),
            build_current_fact_prompt(second, bundle.segments[0]),
        )
        self.assertNotIn("target_entities", build_current_fact_prompt(first, bundle.segments[0]))

    def test_target_content_and_directed_metrics_are_separate(self) -> None:
        source = _source(
            "c1",
            "周明远决定本周不讲含参导数",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
        )
        claim = _summary_fact("s1", "本周不讲含参导数")
        match = EvaluationFactMatch(
            summary_fact_id="s1",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("c1",),
            primary_source_fact_id="c1",
            primary_source_scope=EvaluationSourceScope.CURRENT,
            reason_code="same_fact",
        )
        metrics = calculate_personalization_metrics(
            summary=claim.fact,
            target_entities=("周明远",),
            source_facts=(source,),
            summary_facts=(claim,),
            matches=(match,),
            source_annotations=(
                PersonalizationSourceAnnotation(
                    target_entity="周明远",
                    source_fact_id="c1",
                    relevance=PersonalizationSourceRelevance.TARGET_MUST_INCLUDE,
                ),
            ),
            claim_annotations=(
                PersonalizationClaimAnnotation(
                    target_entity="周明远",
                    summary_fact_id="s1",
                    relevance=PersonalizationClaimRelevance.NOT_TARGET_DIRECTED,
                ),
            ),
        )
        target = metrics.per_target[0]
        self.assertEqual(
            1.0, target.core_metrics["target_content_must_recall"].value
        )
        self.assertEqual(
            0.0, target.core_metrics["target_directed_must_recall"].value
        )
        self.assertEqual(1, target.counts["gate_excluded_covered_fact_count"])
        self.assertEqual(1, target.counts["implicit_target_content_claim_count"])

    def test_transition_and_history_family_metrics_do_not_double_count(self) -> None:
        current = _source(
            "c1",
            "张雨晴已经完成错题提醒",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.COMPLETED,
        )
        historical = _source(
            "h1",
            "张雨晴需要整理错题提醒",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.PENDING,
        )
        relation = EvaluationFactRelation(
            relation_id="r1",
            fact_family_id="family1",
            historical_fact_id="h1",
            current_fact_ids=("c1",),
            relation_type=EvaluationFactRelationType.COMPLETED,
            historical_state=EvaluationFactState.PENDING,
            current_state=EvaluationFactState.COMPLETED,
        )
        claim = _summary_fact("s1", "张雨晴原定整理错题提醒，目前已经完成")
        match = EvaluationFactMatch(
            summary_fact_id="s1",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("c1",),
            primary_source_fact_id="c1",
            primary_source_scope=EvaluationSourceScope.CURRENT,
            matched_relation_ids=("r1",),
            transition_assessment=EvaluationTransitionAssessment.TRANSITION_FULL,
            reason_code="transition_full",
        )
        metrics = calculate_evaluation_metrics(
            summary=claim.fact,
            current_material=current.fact,
            visible_history_text=historical.fact,
            source_facts=(current, historical),
            summary_facts=(claim,),
            matches=(match,),
            fact_relations=(relation,),
        )
        self.assertEqual(2, metrics.core_metrics["must_fact_recall"].denominator)
        self.assertEqual(1.0, metrics.core_metrics["must_fact_recall"].value)
        self.assertEqual(
            1, metrics.history_metrics["historical_semantic_recall"].denominator
        )
        self.assertEqual(
            1.0, metrics.history_metrics["historical_semantic_recall"].value
        )
        self.assertEqual(
            1.0, metrics.transition_metrics["state_transition_f1"].value
        )

    def test_current_confirmed_history_counts_by_family(self) -> None:
        current = _source(
            "c1",
            "本周不讲含参导数",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
        )
        historical = _source(
            "h1",
            "本周不讲含参导数",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.CONFIRMED,
        ).model_copy(
            update={"history_disposition": HistoricalDisposition.ACTIVE}
        )
        relation = EvaluationFactRelation(
            relation_id="r1",
            fact_family_id="family1",
            historical_fact_id="h1",
            current_fact_ids=("c1",),
            relation_type=EvaluationFactRelationType.CONFIRMED,
            historical_state=EvaluationFactState.CONFIRMED,
            current_state=EvaluationFactState.CONFIRMED,
        )
        claim = _summary_fact("s1", "本周不讲含参导数")
        match = EvaluationFactMatch(
            summary_fact_id="s1",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("c1",),
            primary_source_fact_id="c1",
            primary_source_scope=EvaluationSourceScope.CURRENT,
            transition_assessment=(
                EvaluationTransitionAssessment.CURRENT_STATE_ONLY
            ),
            reason_code="current_confirmed",
        )
        metrics = calculate_evaluation_metrics(
            summary=claim.fact,
            current_material=current.fact,
            visible_history_text=historical.fact,
            source_facts=(current, historical),
            summary_facts=(claim,),
            matches=(match,),
            fact_relations=(relation,),
        )
        self.assertEqual(1, metrics.core_metrics["must_fact_recall"].denominator)
        self.assertEqual(
            1.0, metrics.history_metrics["historical_semantic_recall"].value
        )
        self.assertEqual(
            1.0,
            metrics.history_metrics["current_confirmed_history_recall"].value,
        )

    def test_stale_old_state_is_not_correct_v2_coverage(self) -> None:
        current = _source(
            "c1",
            "张雨晴已经完成错题提醒",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.COMPLETED,
        )
        historical = _source(
            "h1",
            "张雨晴需要整理错题提醒",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.PENDING,
        )
        relation = EvaluationFactRelation(
            relation_id="r1",
            fact_family_id="family1",
            historical_fact_id="h1",
            current_fact_ids=("c1",),
            relation_type=EvaluationFactRelationType.COMPLETED,
            historical_state=EvaluationFactState.PENDING,
            current_state=EvaluationFactState.COMPLETED,
        )
        claim = _summary_fact("s1", "张雨晴目前仍需整理错题提醒")
        match = EvaluationFactMatch(
            summary_fact_id="s1",
            match_status=EvaluationMatchStatus.SUPPORTED_FULL,
            matched_source_fact_ids=("h1",),
            primary_source_fact_id="h1",
            primary_source_scope=EvaluationSourceScope.HISTORICAL,
            matched_relation_ids=("r1",),
            transition_assessment=(
                EvaluationTransitionAssessment.STALE_AS_CURRENT
            ),
            stale_history=True,
            reason_code="stale_as_current",
        )
        metrics = calculate_evaluation_metrics(
            summary=claim.fact,
            current_material=current.fact,
            visible_history_text=historical.fact,
            source_facts=(current, historical),
            summary_facts=(claim,),
            matches=(match,),
            fact_relations=(relation,),
        )
        self.assertEqual(0.0, metrics.core_metrics["must_fact_recall"].value)
        self.assertEqual(
            0.0, metrics.transition_metrics["state_transition_recall"].value
        )
        self.assertEqual(
            0.0, metrics.transition_metrics["current_state_accuracy"].value
        )
        self.assertEqual(
            1.0, metrics.transition_metrics["stale_state_error_rate"].value
        )
        self.assertEqual(
            0.5,
            metrics.legacy_metrics["legacy_must_fact_recall"].value,
        )

    def test_unsupported_match_cannot_claim_a_transition(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationFactMatch(
                summary_fact_id="s1",
                match_status=EvaluationMatchStatus.UNSUPPORTED,
                matched_relation_ids=("r1",),
                transition_assessment=(
                    EvaluationTransitionAssessment.TRANSITION_FULL
                ),
                reason_code="invalid",
            )

    def test_unsupported_target_directed_claim_reduces_directed_precision(self) -> None:
        source = _source(
            "c1",
            "周明远决定本周不讲含参导数",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
        )
        claim = _summary_fact("s1", "周明远决定下月取消考试")
        match = EvaluationFactMatch(
            summary_fact_id="s1",
            match_status=EvaluationMatchStatus.UNSUPPORTED,
            reason_code="unsupported",
        )
        metrics = calculate_personalization_metrics(
            summary=claim.fact,
            target_entities=("周明远",),
            source_facts=(source,),
            summary_facts=(claim,),
            matches=(match,),
            source_annotations=(
                PersonalizationSourceAnnotation(
                    target_entity="周明远",
                    source_fact_id="c1",
                    relevance=(
                        PersonalizationSourceRelevance.TARGET_MUST_INCLUDE
                    ),
                ),
            ),
            claim_annotations=(
                PersonalizationClaimAnnotation(
                    target_entity="周明远",
                    summary_fact_id="s1",
                    relevance=(
                        PersonalizationClaimRelevance.TARGET_DIRECTED
                    ),
                ),
            ),
        ).per_target[0]
        self.assertEqual(
            0.0,
            metrics.core_metrics["target_directed_useful_precision"].value,
        )
        self.assertEqual(
            1, metrics.counts["unsupported_target_directed_claim_count"]
        )


class ComparisonModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        self.calls.append(prompt)
        if "Evaluation Source Fact Builder" in prompt:
            text = re.search(
                r"<evaluation_current_segment>\n(.*?)\n</evaluation_current_segment>",
                prompt,
                re.DOTALL,
            ).group(1)
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "项目范围已确认",
                            "fact_type": "decision",
                            "state": "confirmed",
                            "salience": "must_include",
                            "entities": [],
                            "evidence_quote": text,
                            "speaker": None,
                            "turn_id": None,
                            "classification_reason": "必要决定",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Evaluation Summary Fact Extractor" in prompt:
            text = re.search(
                r"<evaluation_summary_segment>\n(.*?)\n</evaluation_summary_segment>",
                prompt,
                re.DOTALL,
            ).group(1)
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "项目范围已确认",
                            "evidence_quote": text,
                            "claimed_scope": "current",
                            "entities": [],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Evaluation Fact Matcher" in prompt:
            claims = json.loads(
                re.search(
                    r"<evaluation_summary_facts>\n(.*?)\n</evaluation_summary_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            sources = json.loads(
                re.search(
                    r"<evaluation_source_facts>\n(.*?)\n</evaluation_source_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            return json.dumps(
                {
                    "matches": [
                        {
                            "summary_fact_id": item["summary_fact_id"],
                            "status": "supported_full",
                            "matched_source_fact_ids": [sources[0]["fact_id"]],
                            "primary_source_fact_id": sources[0]["fact_id"],
                            "reason_code": "same_fact",
                            "reason": "一致",
                        }
                        for item in claims
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected prompt")


class RawHistorySnapshotModel:
    """返回可逐字 grounding 的当前事实和历史状态变化。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        if "Evaluation Source Fact Builder" in prompt:
            quote = "张雨晴已经完成错题提醒。"
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "张雨晴已经完成错题提醒",
                            "fact_type": "task",
                            "state": "completed",
                            "salience": "must_include",
                            "entities": ["张雨晴"],
                            "evidence_quote": quote,
                            "speaker": None,
                            "turn_id": None,
                            "classification_reason": "当前完成状态",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Evaluation Historical Source Builder" in prompt:
            current_facts = json.loads(
                re.search(
                    r"<evaluation_current_facts>\n(.*?)\n</evaluation_current_facts>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "张雨晴需要整理错题提醒",
                            "fact_type": "task",
                            "state": "pending",
                            "salience": "must_include",
                            "disposition": "updated",
                            "entities": ["张雨晴"],
                            "evidence_quote": "张雨晴需要整理错题提醒。",
                            "speaker": None,
                            "turn_id": 7,
                            "related_current_fact_ids": [
                                current_facts[0]["fact_id"]
                            ],
                            "relation_type": "completed",
                            "relation_reason": "当前确认完成",
                            "classification_reason": "历史待办",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected prompt")


class MiddleSummaryFailureModel(ComparisonModel):
    """让第二份纪要抽取失败，用于验证第三份仍会继续。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Evaluation Summary Fact Extractor" in prompt and "BROKEN" in prompt:
            self.calls.append(prompt)
            raise RuntimeError("synthetic summary failure")
        return super().__call__(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class SummaryWarningModel(ComparisonModel):
    """返回一条无法 grounding 的 Claim，用于验证 warning 上浮。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Evaluation Summary Fact Extractor" in prompt:
            self.calls.append(prompt)
            text = re.search(
                r"<evaluation_summary_segment>\n(.*?)\n</evaluation_summary_segment>",
                prompt,
                re.DOTALL,
            ).group(1)
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "项目范围已确认",
                            "evidence_quote": text,
                            "claimed_scope": "current",
                            "entities": [],
                        },
                        {
                            "fact": "不存在的事实",
                            "evidence_quote": "摘要中不存在的引文",
                            "claimed_scope": "unknown",
                            "entities": [],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        return super().__call__(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class TargetAwareComparisonModel(ComparisonModel):
    """额外返回目标源事实标注，用于验证 target 不污染整体事实。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Target Personalization Fact Classifier" in prompt:
            self.calls.append(prompt)
            target = re.search(r"target_entity: (.*?)\n", prompt).group(1)
            facts = json.loads(
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
                            "relevance": "target_useful_optional",
                            "relation_types": ["affected_party"],
                            "reason": "目标相关",
                        }
                        for item in facts
                    ],
                },
                ensure_ascii=False,
            )
        return super().__call__(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class SlowComparisonModelClient:
    """阻塞 Source Snapshot 构建，用于验证核心横评整轮超时。"""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        del prompt, temperature, max_tokens
        await asyncio.sleep(2)
        return "{}"


class AdaptiveSplitModelClient:
    """较大源分段返回截断 JSON，较小子段返回可 grounding 事实。"""

    def __init__(self) -> None:
        self.segment_lengths: list[int] = []

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        del temperature, max_tokens
        text = re.search(
            r"<evaluation_current_segment>\n(.*?)\n</evaluation_current_segment>",
            prompt,
            re.DOTALL,
        ).group(1)
        self.segment_lengths.append(len(text))
        if len(text) > 600:
            return '{"facts": ['
        quote = text[: min(8, len(text))]
        return json.dumps(
            {
                "facts": [
                    {
                        "fact": quote,
                        "salience": "must_include",
                        "evidence_quote": quote,
                    }
                ]
            },
            ensure_ascii=False,
        )


class AlwaysMalformedSourceModelClient:
    """所有源事实调用均返回截断 JSON，用于验证失败审计。"""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        del prompt, temperature, max_tokens
        return '{"facts": ['


class P73ComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_protocol_failure_adaptively_splits_one_segment(self) -> None:
        material = "甲乙丙丁戊己庚辛壬癸" * 100
        model = AdaptiveSplitModelClient()
        builder = EvaluationSourceSnapshotBuilder(
            model_client=model,
            config=RunConfig(
                evaluation_segment_max_chars=1000,
                evaluation_source_fact_max_tokens=3000,
                evaluation_repair_attempts=0,
                evaluation_adaptive_split_enabled=True,
                evaluation_adaptive_split_min_chars=500,
                evaluation_adaptive_split_max_depth=1,
            ),
        )
        result = await builder.build_from_documents(
            summary_request="生成课堂纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前材料",
                source_time="2026-09-04",
                text=material,
            ),
        )

        self.assertEqual(1, result.snapshot.execution_stats["evaluation_adaptive_split_count"])
        self.assertEqual(2, result.snapshot.execution_stats["evaluation_adaptive_split_child_count"])
        self.assertEqual(3, len(model.segment_lengths))
        self.assertEqual(1000, model.segment_lengths[0])
        self.assertTrue(all(length < 600 for length in model.segment_lengths[1:]))
        self.assertTrue(result.snapshot.source_facts)
        for fact in result.snapshot.source_facts:
            evidence = fact.evidence[0]
            self.assertEqual(
                evidence.evidence_text,
                material[evidence.start_offset : evidence.end_offset],
            )
        self.assertTrue(
            any("adaptive split" in warning for warning in result.snapshot.warnings)
        )

    async def test_snapshot_failure_exposes_source_trace_and_stage(self) -> None:
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成课堂纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前材料",
                source_time="2026-09-04",
                text="甲" * 1000,
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="纪要 A"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="纪要 B"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                segment_max_chars=1000,
                repair_attempts=0,
                adaptive_split_enabled=True,
                adaptive_split_min_chars=500,
                adaptive_split_max_depth=1,
            ),
        )
        runner = EvaluationComparisonRunner(
            model_client=AlwaysMalformedSourceModelClient()
        )
        with self.assertRaises(ValueError):
            await runner.run(request)

        self.assertEqual("building_source_snapshot", runner.last_stage)
        self.assertEqual(2, len(runner.last_source_model_trace))
        self.assertEqual(
            "likely_output_truncation",
            runner.last_source_model_trace[-1]["attempts"][-1]["failure_kind"],
        )
        self.assertEqual(
            1,
            runner.last_source_execution_stats["evaluation_adaptive_split_count"],
        )

    async def test_comparison_runner_enforces_whole_run_timeout(self) -> None:
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="项目范围已确认。"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="项目范围已确认。"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_timeout_seconds=1,
                comparison_model_timeout_seconds=10,
                repair_attempts=0,
            ),
        )
        with self.assertRaises(asyncio.TimeoutError):
            await EvaluationComparisonRunner(
                model_client=SlowComparisonModelClient()
            ).run(request)

    async def test_target_changes_annotations_but_not_overall_source_facts(self) -> None:
        document = EvaluationSourceDocument(
            source_id="current",
            label="当前会议",
            source_time="2026-09-03",
            text="项目范围已确认。",
        )
        config = RunConfig(evaluation_repair_attempts=0)
        without_target = await EvaluationSourceSnapshotBuilder(
            model_client=SyncCallableModelClient(TargetAwareComparisonModel()),
            config=config,
        ).build_from_documents(
            summary_request="生成项目纪要",
            current_source=document,
        )
        with_target = await EvaluationSourceSnapshotBuilder(
            model_client=SyncCallableModelClient(TargetAwareComparisonModel()),
            config=config,
        ).build_from_documents(
            summary_request="生成项目纪要",
            current_source=document,
            target_entities=("周明远",),
        )
        self.assertEqual(
            without_target.snapshot.source_facts,
            with_target.snapshot.source_facts,
        )
        self.assertEqual(1, len(with_target.snapshot.target_source_annotations))

    async def test_raw_history_is_grounded_and_related_in_snapshot(self) -> None:
        builder = EvaluationSourceSnapshotBuilder(
            model_client=SyncCallableModelClient(RawHistorySnapshotModel()),
            config=RunConfig(
                evaluation_segment_max_chars=1000,
                evaluation_repair_attempts=0,
            ),
        )
        history_text = "历史记录：张雨晴需要整理错题提醒。"
        result = await builder.build_from_documents(
            summary_request="生成课堂纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前课堂",
                source_time="2026-03-19",
                text="张雨晴已经完成错题提醒。",
            ),
            historical_sources=(
                EvaluationSourceDocument(
                    source_id="history_1",
                    label="历史课堂",
                    source_time="2026-03-12",
                    text=history_text,
                ),
            ),
        )
        historical = next(
            item
            for item in result.snapshot.source_facts
            if item.source_scope == EvaluationSourceScope.HISTORICAL
        )
        evidence = historical.evidence[0]
        self.assertEqual("history_1", evidence.source_id)
        self.assertEqual("2026-03-12", evidence.source_time)
        self.assertEqual(
            "张雨晴需要整理错题提醒。",
            history_text[evidence.start_offset : evidence.end_offset],
        )
        self.assertEqual(1, len(result.snapshot.fact_relations))
        self.assertEqual(
            EvaluationFactRelationType.COMPLETED,
            result.snapshot.fact_relations[0].relation_type,
        )

    async def test_two_summaries_share_one_snapshot_and_denominator(self) -> None:
        model = ComparisonModel()
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="项目范围已确认。"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="项目范围已确认。"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_max_total_model_calls=20,
                comparison_source_max_model_calls=5,
                comparison_per_summary_max_model_calls=5,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(model)
        ).run(request)
        self.assertEqual("completed", result.status)
        self.assertEqual(2, len(result.summaries))
        self.assertEqual(
            1,
            sum("Evaluation Source Fact Builder" in item for item in model.calls),
        )
        denominators = {
            item.artifacts.metrics.core_metrics["must_fact_recall"].denominator
            for item in result.summaries
            if item.artifacts is not None
        }
        self.assertEqual({1}, denominators)
        snapshot_ids = {
            row["snapshot_id"]
            for row in result.comparison_metrics["rows"]
        }
        self.assertEqual({result.snapshot_bundle.snapshot.snapshot_id}, snapshot_ids)

    async def test_one_summary_failure_does_not_stop_later_summaries(self) -> None:
        model = MiddleSummaryFailureModel()
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="项目范围已确认。"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="BROKEN"
                ),
                EvaluationSummaryDocument(
                    summary_id="c", display_name="C", text="项目范围已确认。"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_max_total_model_calls=20,
                comparison_source_max_model_calls=5,
                comparison_per_summary_max_model_calls=5,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(model)
        ).run(request)
        self.assertEqual("completed_with_warnings", result.status)
        self.assertEqual(
            ["completed", "failed", "completed"],
            [item.status for item in result.summaries],
        )
        self.assertEqual(2, result.comparison_metrics["successful_summary_count"])
        self.assertIsNotNone(result.summaries[2].artifacts)

    async def test_summary_warnings_are_promoted_to_comparison_status(self) -> None:
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="项目范围已确认。"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="项目范围已确认。"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_max_total_model_calls=20,
                comparison_source_max_model_calls=5,
                comparison_per_summary_max_model_calls=5,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(SummaryWarningModel())
        ).run(request)
        self.assertEqual("completed_with_warnings", result.status)
        self.assertEqual(
            ["completed_with_warnings", "completed_with_warnings"],
            [item.status for item in result.summaries],
        )
        self.assertTrue(
            any("discarded 1 evaluation facts" in item for item in result.warnings)
        )

    async def test_shared_total_model_budget_is_a_hard_limit(self) -> None:
        model = ComparisonModel()
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="项目范围已确认。"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="项目范围已确认。"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_max_total_model_calls=3,
                comparison_source_max_model_calls=5,
                comparison_per_summary_max_model_calls=5,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(model)
        ).run(request)
        # 双范围整体指标也计入同一硬预算；三个调用不足以完成任一纪要的
        # 新范围，因此按新契约判定所有范围失败，而不是沿用旧横评成功状态。
        self.assertEqual("failed", result.status)
        self.assertEqual("failed", result.summaries[0].status)
        self.assertEqual("failed", result.summaries[1].status)
        self.assertEqual(3, result.comparison_metrics["total_model_call_count"])


if __name__ == "__main__":
    unittest.main()
