from __future__ import annotations

import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.evaluation.metrics import calculate_evaluation_metrics  # noqa: E402
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
)


def source_fact(
    fact_id: str,
    *,
    scope: EvaluationSourceScope,
    salience: EvaluationSalience,
    disposition: HistoricalDisposition | None = None,
) -> EvaluationSourceFact:
    return EvaluationSourceFact(
        fact_id=fact_id,
        source_scope=scope,
        fact=f"事实 {fact_id}",
        state=EvaluationFactState.CONFIRMED,
        salience=salience,
        evidence=(EvaluationEvidence(evidence_text=f"证据 {fact_id}"),),
        history_disposition=disposition,
    )


def summary_fact(fact_id: str, offset: int) -> EvaluationSummaryFact:
    return EvaluationSummaryFact(
        summary_fact_id=fact_id,
        fact=f"摘要事实 {fact_id}",
        evidence_quote=f"摘要事实 {fact_id}",
        start_offset=offset,
        end_offset=offset + 5,
        claimed_scope=EvaluationClaimScope.UNKNOWN,
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_core_safety_and_conciseness_metrics_are_deterministic(self) -> None:
        sources = (
            source_fact(
                "c1",
                scope=EvaluationSourceScope.CURRENT,
                salience=EvaluationSalience.MUST_INCLUDE,
            ),
            source_fact(
                "c2",
                scope=EvaluationSourceScope.CURRENT,
                salience=EvaluationSalience.USEFUL_OPTIONAL,
            ),
            source_fact(
                "h1",
                scope=EvaluationSourceScope.HISTORICAL,
                salience=EvaluationSalience.MUST_INCLUDE,
                disposition=HistoricalDisposition.ACTIVE,
            ),
            source_fact(
                "h2",
                scope=EvaluationSourceScope.HISTORICAL,
                salience=EvaluationSalience.EXCLUDE,
                disposition=HistoricalDisposition.SUPERSEDED,
            ),
        )
        claims = tuple(summary_fact(f"q{index}", index * 10) for index in range(1, 7))
        matches = (
            EvaluationFactMatch(
                summary_fact_id="q1",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("c1",),
                primary_source_fact_id="c1",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="supported",
            ),
            EvaluationFactMatch(
                summary_fact_id="q2",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("h1",),
                primary_source_fact_id="h1",
                primary_source_scope=EvaluationSourceScope.HISTORICAL,
                reason_code="supported",
            ),
            EvaluationFactMatch(
                summary_fact_id="q3",
                match_status=EvaluationMatchStatus.CONTRADICTED,
                matched_source_fact_ids=("h2",),
                primary_source_fact_id="h2",
                primary_source_scope=EvaluationSourceScope.HISTORICAL,
                stale_history=True,
                reason_code="stale",
            ),
            EvaluationFactMatch(
                summary_fact_id="q4",
                match_status=EvaluationMatchStatus.UNSUPPORTED,
                reason_code="unsupported",
            ),
            EvaluationFactMatch(
                summary_fact_id="q5",
                match_status=EvaluationMatchStatus.SUPPORTED_FULL,
                matched_source_fact_ids=("c1",),
                primary_source_fact_id="c1",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                duplicate_of_summary_fact_id="q1",
                reason_code="duplicate",
            ),
            EvaluationFactMatch(
                summary_fact_id="q6",
                match_status=EvaluationMatchStatus.SUPPORTED_PARTIAL,
                matched_source_fact_ids=("c2",),
                primary_source_fact_id="c2",
                primary_source_scope=EvaluationSourceScope.CURRENT,
                reason_code="partial",
            ),
        )
        metrics = calculate_evaluation_metrics(
            summary="摘要事实一二三四五六",
            current_material="当前会议原文内容",
            visible_history_text="历史事实内容",
            source_facts=sources,
            summary_facts=claims,
            matches=matches,
        )

        self.assertEqual(1.0, metrics.core_metrics["must_fact_recall"].value)
        self.assertEqual(
            2 / 6, metrics.core_metrics["useful_fact_precision"].value
        )
        self.assertEqual(
            3 / 6, metrics.core_metrics["fact_support_precision"].value
        )
        self.assertEqual(1 / 6, metrics.safety_metrics["unsupported_rate"].value)
        self.assertEqual(1 / 6, metrics.safety_metrics["contradiction_rate"].value)
        self.assertEqual(1 / 6, metrics.safety_metrics["partial_rate"].value)
        self.assertEqual(1 / 6, metrics.safety_metrics["stale_history_rate"].value)
        self.assertEqual(1 / 6, metrics.conciseness_metrics["redundancy_rate"].value)

    def test_zero_denominators_are_na(self) -> None:
        metrics = calculate_evaluation_metrics(
            summary="",
            current_material="当前材料",
            visible_history_text="",
            source_facts=(),
            summary_facts=(),
            matches=(),
        )
        self.assertIsNone(metrics.core_metrics["must_fact_recall"].value)
        self.assertIsNone(metrics.core_metrics["useful_fact_precision"].value)
        self.assertIsNone(metrics.core_metrics["minutes_fact_f1"].value)
        self.assertIsNone(metrics.safety_metrics["unsupported_rate"].value)


if __name__ == "__main__":
    unittest.main()
