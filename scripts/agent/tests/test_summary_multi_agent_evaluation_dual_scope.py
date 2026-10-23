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

from summary_multi_agent.adapters.model import SyncCallableModelClient  # noqa: E402
from summary_multi_agent.evaluation.comparison import (  # noqa: E402
    EvaluationComparisonRunner,
)
from summary_multi_agent.evaluation.comparison_artifacts import (  # noqa: E402
    write_comparison_artifacts,
)
from summary_multi_agent.evaluation.comparison_scope import (  # noqa: E402
    EvaluationComparisonScopeEvaluator,
    build_comparison_basis,
    calculate_comparison_scope_metrics,
)
from summary_multi_agent.evaluation.prompts import (  # noqa: E402
    build_comparison_scope_match_prompt,
    build_match_prompt,
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
    HistoricalDisposition,
)
from summary_multi_agent.schemas import RunConfig  # noqa: E402
from summary_multi_agent_web.artifact_reader import SAFE_ARCHIVE_NAMES  # noqa: E402


def _source_fact(
    fact_id: str,
    text: str,
    *,
    scope: EvaluationSourceScope,
    state: EvaluationFactState,
    source_id: str,
    source_time: str,
    salience: EvaluationSalience = EvaluationSalience.MUST_INCLUDE,
    entities: tuple[str, ...] = (),
) -> EvaluationSourceFact:
    """构造带来源身份和时间的最小可审计源事实。"""
    return EvaluationSourceFact(
        fact_id=fact_id,
        source_scope=scope,
        fact=text,
        fact_type="task",
        state=state,
        salience=salience,
        entities=entities,
        evidence=(
            EvaluationEvidence(
                evidence_text=text,
                source_id=source_id,
                source_time=source_time,
            ),
        ),
        history_disposition=(
            HistoricalDisposition.ACTIVE
            if scope == EvaluationSourceScope.HISTORICAL
            else None
        ),
    )


def _claim(fact_id: str, text: str) -> EvaluationSummaryFact:
    return EvaluationSummaryFact(
        summary_fact_id=fact_id,
        fact=text,
        evidence_quote=text,
        start_offset=0,
        end_offset=len(text),
        claimed_scope=EvaluationClaimScope.MIXED,
    )


class ScopeMatcherModel:
    """让历史 Claim 仅在 all_sources 范围内得到完整支持。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        self.prompts.append(prompt)
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
        historical = next(
            (item for item in sources if item["fact_id"] == "EH1"),
            None,
        )
        matches = []
        for claim in claims:
            if historical is None:
                matches.append(
                    {
                        "summary_fact_id": claim["summary_fact_id"],
                        "status": "unsupported",
                        "matched_source_fact_ids": [],
                        "primary_source_fact_id": None,
                        "reason_code": "scope_has_no_support",
                        "reason": "当前范围无支持",
                    }
                )
            else:
                matches.append(
                    {
                        "summary_fact_id": claim["summary_fact_id"],
                        "status": "supported_full",
                        "matched_source_fact_ids": [historical["fact_id"]],
                        "primary_source_fact_id": historical["fact_id"],
                        "reason_code": "history_supports_claim",
                        "reason": "历史材料完整支持",
                    }
                )
        return json.dumps({"matches": matches}, ensure_ascii=False)


class ComparisonRunnerModel:
    """覆盖无历史横评的 Source、Claim 和 Matcher 三类调用。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
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
                            "reason": "事实一致",
                        }
                        for item in claims
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected evaluation prompt")


class ComparisonDifferenceModel:
    """为横评生成一条新增覆盖、一条减少覆盖和一条新增问题。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        if "Evaluation Source Fact Builder" in prompt:
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "项目范围已确认",
                            "fact_type": "decision",
                            "state": "confirmed",
                            "salience": "must_include",
                            "entities": [],
                            "evidence_quote": "项目范围已确认。",
                            "speaker": None,
                            "turn_id": None,
                            "classification_reason": "必要决定",
                        },
                        {
                            "fact": "风险登记已完成",
                            "fact_type": "task",
                            "state": "completed",
                            "salience": "must_include",
                            "entities": [],
                            "evidence_quote": "风险登记已完成。",
                            "speaker": None,
                            "turn_id": None,
                            "classification_reason": "必要任务",
                        },
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
            facts = []
            for value in ("项目范围已确认", "风险登记已完成", "额外推断"):
                if value in text:
                    facts.append(
                        {
                            "fact": value,
                            "evidence_quote": value,
                            "claimed_scope": "current",
                            "entities": [],
                        }
                    )
            return json.dumps({"facts": facts}, ensure_ascii=False)
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
            matches = []
            for claim in claims:
                source = next(
                    (
                        item
                        for item in sources
                        if item["fact"] == claim["fact"]
                    ),
                    None,
                )
                if source is None:
                    matches.append(
                        {
                            "summary_fact_id": claim["summary_fact_id"],
                            "status": "unsupported",
                            "matched_source_fact_ids": [],
                            "primary_source_fact_id": None,
                            "reason_code": "no_source_support",
                            "reason": "源材料没有对应事实",
                        }
                    )
                else:
                    matches.append(
                        {
                            "summary_fact_id": claim["summary_fact_id"],
                            "status": "supported_full",
                            "matched_source_fact_ids": [source["fact_id"]],
                            "primary_source_fact_id": source["fact_id"],
                            "reason_code": "same_fact",
                            "reason": "事实一致",
                        }
                    )
            return json.dumps({"matches": matches}, ensure_ascii=False)
        raise AssertionError("unexpected evaluation prompt")


class PartialScopeFailureModel(ScopeMatcherModel):
    """模拟全材料范围失败，同时允许当前范围继续完成。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "source_scope: all_sources" in prompt:
            self.prompts.append(prompt)
            raise RuntimeError("all-sources unavailable")
        return super().__call__(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class JointSupportModel(ScopeMatcherModel):
    """在全材料范围用一条当前事实和一条历史事实共同支持 Claim。"""

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        self.prompts.append(prompt)
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
        if len(sources) == 1:
            return json.dumps(
                {
                    "matches": [
                        {
                            "summary_fact_id": claims[0]["summary_fact_id"],
                            "status": "supported_partial",
                            "matched_source_fact_ids": [sources[0]["fact_id"]],
                            "primary_source_fact_id": sources[0]["fact_id"],
                            "reason_code": "needs_history",
                            "reason": "当前材料只支持一部分",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "matches": [
                    {
                        "summary_fact_id": claims[0]["summary_fact_id"],
                        "status": "supported_full",
                        "matched_source_fact_ids": [item["fact_id"] for item in sources],
                        "primary_source_fact_id": "EC1",
                        "reason_code": "joint_cross_source_support",
                        "reason": "当前与历史共同支持",
                    }
                ]
            },
            ensure_ascii=False,
        )


class ComparisonBasisTests(unittest.TestCase):
    def test_confirmed_and_duplicate_history_merge_but_state_change_does_not(self) -> None:
        current_confirmed = _source_fact(
            "EC1",
            "本周不讲含参导数",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
            source_id="current",
            source_time="2026-03-19",
        )
        history_confirmed = _source_fact(
            "EH1",
            "本周不讲含参导数",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.CONFIRMED,
            source_id="history-1",
            source_time="2026-03-12",
        )
        current_completed = _source_fact(
            "EC2",
            "张雨晴已完成错题提醒",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.COMPLETED,
            source_id="current",
            source_time="2026-03-19",
            entities=("张雨晴",),
        )
        history_pending = _source_fact(
            "EH2",
            "张雨晴需整理错题提醒",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.PENDING,
            source_id="history-1",
            source_time="2026-03-12",
            entities=("张雨晴",),
        )
        duplicate_1 = _source_fact(
            "EH3",
            "复习资料沿用五栏结构",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.ACTIVE,
            source_id="history-1",
            source_time="2026-03-12",
        )
        duplicate_2 = _source_fact(
            "EH4",
            "复习资料沿用五栏结构",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.ACTIVE,
            source_id="history-copy",
            source_time="2026-03-12",
        )
        relations = (
            EvaluationFactRelation(
                relation_id="R1",
                fact_family_id="F1",
                historical_fact_id="EH1",
                current_fact_ids=("EC1",),
                relation_type=EvaluationFactRelationType.CONFIRMED,
                historical_state=EvaluationFactState.CONFIRMED,
                current_state=EvaluationFactState.CONFIRMED,
            ),
            EvaluationFactRelation(
                relation_id="R2",
                fact_family_id="F2",
                historical_fact_id="EH2",
                current_fact_ids=("EC2",),
                relation_type=EvaluationFactRelationType.COMPLETED,
                historical_state=EvaluationFactState.PENDING,
                current_state=EvaluationFactState.COMPLETED,
            ),
        )
        facts = (
            current_confirmed,
            history_confirmed,
            current_completed,
            history_pending,
            duplicate_1,
            duplicate_2,
        )

        basis = build_comparison_basis(
            snapshot_id="snapshot-1",
            source_facts=facts,
            fact_relations=relations,
        )
        current_again = build_comparison_basis(
            snapshot_id="snapshot-1",
            source_facts=(current_confirmed, current_completed),
            fact_relations=(),
        )

        self.assertEqual(4, len(basis.all_sources.units))
        self.assertEqual(2, len(basis.current_only.units))
        self.assertEqual(
            basis.current_only.model_dump(mode="json"),
            current_again.current_only.model_dump(mode="json"),
        )
        units = [set(item.source_fact_ids) for item in basis.all_sources.units]
        self.assertIn({"EC1", "EH1"}, units)
        self.assertIn({"EH3", "EH4"}, units)
        self.assertNotIn({"EC2", "EH2"}, units)
        self.assertEqual(4, len(basis.all_sources.required_unit_ids))

    def test_scope_metrics_keep_full_claim_denominator_and_strict_partial(self) -> None:
        current = _source_fact(
            "EC1",
            "当前必要事实",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
            source_id="current",
            source_time="2026-03-19",
        )
        basis = build_comparison_basis(
            snapshot_id="snapshot-2",
            source_facts=(current,),
            fact_relations=(),
        ).current_only
        claims = (_claim("S1", "当前必要事实"), _claim("S2", "内容不完整"))
        matches = (
            EvaluationFactMatch(
                summary_fact_id="S1",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("EC1",),
                primary_source_fact_id="EC1",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="same_fact",
            ),
            EvaluationFactMatch(
                summary_fact_id="S2",
                match_status=EvaluationMatchStatus.SUPPORTED_PARTIAL,
                matched_source_fact_ids=("EC1",),
                primary_source_fact_id="EC1",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="partial_fact",
            ),
        )

        metrics = calculate_comparison_scope_metrics(
            scope_basis=basis,
            summary="当前必要事实\n内容不完整",
            summary_facts=claims,
            matches=matches,
        )

        self.assertEqual(1.0, metrics.core_metrics["must_fact_recall"].value)
        self.assertEqual(0.5, metrics.core_metrics["useful_fact_precision"].value)
        self.assertEqual(0.5, metrics.core_metrics["fact_support_precision"].value)
        self.assertEqual(0.5, metrics.safety_metrics["partial_rate"].value)


class ComparisonScopeEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_supported_claim_is_separated_from_current_scope(self) -> None:
        current = _source_fact(
            "EC1",
            "当前课堂完成练习",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.COMPLETED,
            source_id="current",
            source_time="2026-03-19",
        )
        history = _source_fact(
            "EH1",
            "历史约束仍然有效",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.ACTIVE,
            source_id="history",
            source_time="2026-03-12",
        )
        facts = (current, history)
        basis = build_comparison_basis(
            snapshot_id="snapshot-3",
            source_facts=facts,
            fact_relations=(),
        )
        model = ScopeMatcherModel()
        evaluator = EvaluationComparisonScopeEvaluator(
            model_client=SyncCallableModelClient(model),
            config=RunConfig(evaluation_repair_attempts=0),
            max_model_calls=8,
        )

        result = await evaluator.evaluate(
            basis=basis,
            source_facts=facts,
            fact_relations=(),
            summary="历史约束仍然有效",
            summary_facts=(_claim("S1", "历史约束仍然有效"),),
        )

        self.assertEqual(0.5, result.all_sources.core_metrics["must_fact_recall"].value)
        self.assertEqual(1.0, result.all_sources.core_metrics["fact_support_precision"].value)
        self.assertEqual(0.0, result.current_only.core_metrics["must_fact_recall"].value)
        self.assertEqual(1.0, result.current_only.safety_metrics["unsupported_rate"].value)
        self.assertEqual(
            "supported_by_all_sources_not_current",
            result.claim_diagnostics[0]["interpretation"],
        )
        self.assertEqual("completed", result.scope_statuses["all_sources"].value)
        self.assertEqual("completed", result.scope_statuses["current_only"].value)
        self.assertEqual(2, len(model.prompts))

    async def test_no_history_reuses_current_match_for_identical_metrics(self) -> None:
        current = _source_fact(
            "EC1",
            "项目范围已确认",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
            source_id="current",
            source_time="2026-03-19",
        )
        basis = build_comparison_basis(
            snapshot_id="snapshot-4",
            source_facts=(current,),
            fact_relations=(),
        )
        model = ScopeMatcherModel()
        evaluator = EvaluationComparisonScopeEvaluator(
            model_client=SyncCallableModelClient(model),
            config=RunConfig(evaluation_repair_attempts=0),
            max_model_calls=8,
        )

        result = await evaluator.evaluate(
            basis=basis,
            source_facts=(current,),
            fact_relations=(),
            summary="项目范围已确认",
            summary_facts=(_claim("S1", "项目范围已确认"),),
        )

        self.assertEqual(
            result.all_sources.core_metrics,
            result.current_only.core_metrics,
        )
        self.assertEqual(
            result.all_sources.safety_metrics,
            result.current_only.safety_metrics,
        )
        self.assertEqual(1, len(model.prompts))
        self.assertEqual(1, result.execution_stats["all_sources_reused_current_only"])

    async def test_one_scope_failure_preserves_the_other_scope(self) -> None:
        current = _source_fact(
            "EC1",
            "当前课堂完成练习",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.COMPLETED,
            source_id="current",
            source_time="2026-03-19",
        )
        history = _source_fact(
            "EH1",
            "历史约束仍然有效",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.ACTIVE,
            source_id="history",
            source_time="2026-03-12",
        )
        facts = (current, history)
        basis = build_comparison_basis(
            snapshot_id="snapshot-5",
            source_facts=facts,
            fact_relations=(),
        )
        evaluator = EvaluationComparisonScopeEvaluator(
            model_client=SyncCallableModelClient(PartialScopeFailureModel()),
            config=RunConfig(evaluation_repair_attempts=0),
            max_model_calls=8,
        )

        result = await evaluator.evaluate(
            basis=basis,
            source_facts=facts,
            fact_relations=(),
            summary="历史约束仍然有效",
            summary_facts=(_claim("S1", "历史约束仍然有效"),),
        )

        self.assertIsNone(result.all_sources)
        self.assertIsNotNone(result.current_only)
        self.assertEqual("failed", result.scope_statuses["all_sources"].value)
        self.assertEqual("completed", result.scope_statuses["current_only"].value)
        self.assertIn("all-sources unavailable", result.scope_errors["all_sources"])

    async def test_all_sources_accepts_joint_cross_material_support(self) -> None:
        current = _source_fact(
            "EC1",
            "任务由张雨晴负责",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
            source_id="current",
            source_time="2026-03-19",
            entities=("张雨晴",),
        )
        history = _source_fact(
            "EH1",
            "任务截止时间为周五",
            scope=EvaluationSourceScope.HISTORICAL,
            state=EvaluationFactState.ACTIVE,
            source_id="history",
            source_time="2026-03-12",
            entities=("张雨晴",),
        )
        facts = (current, history)
        basis = build_comparison_basis(
            snapshot_id="snapshot-6",
            source_facts=facts,
            fact_relations=(),
        )
        evaluator = EvaluationComparisonScopeEvaluator(
            model_client=SyncCallableModelClient(JointSupportModel()),
            config=RunConfig(evaluation_repair_attempts=0),
            max_model_calls=8,
        )

        result = await evaluator.evaluate(
            basis=basis,
            source_facts=facts,
            fact_relations=(),
            summary="张雨晴负责该任务，截止时间为周五",
            summary_facts=(_claim("S1", "张雨晴负责该任务，截止时间为周五"),),
        )

        self.assertEqual(
            EvaluationMatchStatus.SUPPORTED_FULL,
            result.all_sources_matches[0].match_status,
        )
        self.assertEqual(
            {"EC1", "EH1"},
            set(result.all_sources_matches[0].matched_source_fact_ids),
        )
        self.assertEqual(
            EvaluationMatchStatus.SUPPORTED_PARTIAL,
            result.current_only_matches[0].match_status,
        )


class ComparisonArtifactsTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_differences_explain_gain_loss_and_new_problem(self) -> None:
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前会议",
                source_time="2026-09-03",
                text="项目范围已确认。风险登记已完成。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a",
                    display_name="A",
                    text="项目范围已确认。",
                ),
                EvaluationSummaryDocument(
                    summary_id="b",
                    display_name="B",
                    text="风险登记已完成。额外推断。",
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                comparison_max_total_model_calls=30,
                comparison_source_max_model_calls=8,
                comparison_per_summary_max_model_calls=12,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(ComparisonDifferenceModel())
        ).run(request)

        for scope in ("all_sources", "current_only"):
            difference = result.comparison_metrics[
                "scope_content_differences"
            ][scope]["b"]
            self.assertEqual("completed", difference["status"])
            self.assertEqual(1, difference["added_covered_count"])
            self.assertEqual(1, difference["reduced_covered_count"])
            self.assertEqual(1, difference["added_problem_count"])
            self.assertEqual(
                "风险登记已完成",
                difference["added_covered_units"][0]["fact"],
            )
            self.assertEqual(
                "项目范围已确认",
                difference["reduced_covered_units"][0]["fact"],
            )
        self.assertIn(
            "事实覆盖差异（二次核验，仅供核查）", result.report_markdown
        )
        self.assertNotIn("新增覆盖 1 条，减少覆盖 1 条", result.report_markdown)
        self.assertIn("LLM 直接定性对比", result.report_markdown)
        self.assertIn(
            "<summary>问题诊断（模型判定，仅供核查）</summary>",
            result.report_markdown,
        )
        self.assertGreater(
            result.report_markdown.index("问题诊断（模型判定，仅供核查）"),
            result.report_markdown.index("LLM 直接定性对比"),
        )

    async def test_runner_writes_dual_scope_artifacts_and_keeps_trace_private(self) -> None:
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
                comparison_max_total_model_calls=30,
                comparison_source_max_model_calls=8,
                comparison_per_summary_max_model_calls=8,
                repair_attempts=0,
            ),
        )
        result = await EvaluationComparisonRunner(
            model_client=SyncCallableModelClient(ComparisonRunnerModel())
        ).run(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_comparison_artifacts(root, request, result)

            self.assertEqual("completed", result.status)
            self.assertTrue((root / "evaluation_comparison_basis.json").is_file())
            for summary_id in ("a", "b"):
                summary_dir = root / "summaries" / summary_id
                self.assertTrue((summary_dir / "evaluation_scope_metrics.json").is_file())
                self.assertTrue((summary_dir / "fact_matches_all_sources.json").is_file())
                self.assertTrue((summary_dir / "fact_matches_current_only.json").is_file())
                self.assertTrue((summary_dir / "evaluation_scope_model_trace.json").is_file())
            self.assertIn("evaluation_comparison_basis.json", SAFE_ARCHIVE_NAMES)
            self.assertIn("evaluation_scope_metrics.json", SAFE_ARCHIVE_NAMES)
            self.assertNotIn("evaluation_scope_model_trace.json", SAFE_ARCHIVE_NAMES)
            self.assertTrue(
                (
                    root
                    / "evaluation_comparison_interpretation_model_trace.json"
                ).is_file()
            )
            self.assertNotIn(
                "evaluation_comparison_interpretation_model_trace.json",
                SAFE_ARCHIVE_NAMES,
            )
            self.assertEqual(
                "summary_multi_agent.evaluation_comparison_metrics.v2",
                result.comparison_metrics["schema_version"],
            )
            self.assertIn("全材料整体指标", result.report_markdown)
            self.assertIn("当前材料整体指标", result.report_markdown)

    def test_compare_default_per_summary_budget_is_96(self) -> None:
        self.assertEqual(
            96,
            EvaluationCompareConfig().comparison_per_summary_max_model_calls,
        )


class ComparisonPromptIsolationTests(unittest.TestCase):
    def test_comparison_rule_does_not_change_online_match_prompt(self) -> None:
        fact = _source_fact(
            "EC1",
            "项目范围已确认",
            scope=EvaluationSourceScope.CURRENT,
            state=EvaluationFactState.CONFIRMED,
            source_id="current",
            source_time="2026-03-19",
        )
        claim = _claim("S1", "项目范围已确认")

        online = build_match_prompt(
            (claim,),
            (fact,),
            source_scope="current",
            temporal_support={},
        )
        comparison = build_comparison_scope_match_prompt(
            (claim,),
            (fact,),
            source_scope="current_only",
            temporal_support={},
        )

        self.assertNotIn("evaluation_comparison_scope_rule", online)
        self.assertIn("evaluation_comparison_scope_rule", comparison)
        self.assertIn("只允许使用当前材料事实", comparison)


if __name__ == "__main__":
    unittest.main()
