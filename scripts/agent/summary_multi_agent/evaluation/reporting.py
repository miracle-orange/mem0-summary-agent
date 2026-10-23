"""将 V2 事实、历史、状态和个性化指标渲染为可读报告。"""

from __future__ import annotations

from typing import Any

from .metrics import VALID_HISTORY_DISPOSITIONS
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationFactRelationType,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceFact,
    EvaluationSourceScope,
)


def _metric_text(section: dict[str, Any], name: str) -> str:
    """将 MetricValue 渲染为带分子/分母的文本。"""
    metric = section[name]
    if metric.value is None:
        return f"N/A（{metric.numerator}/{metric.denominator}）"
    return f"{metric.value:.4f}（{metric.numerator}/{metric.denominator}）"


def build_evaluation_report(
    source_facts: tuple[EvaluationSourceFact, ...],
    summary_facts: tuple[Any, ...],
    matches: tuple[EvaluationFactMatch, ...],
    metrics: Any,
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
) -> str:
    """生成整体、历史、状态、安全性和缺失事实报告。"""
    source_by_id = {item.fact_id: item for item in source_facts}
    covered = {
        source_id
        for match in matches
        if match.match_status == EvaluationMatchStatus.SUPPORTED_FULL
        and match.duplicate_of_summary_fact_id is None
        for source_id in match.matched_source_fact_ids
    }
    transition_types = {
        EvaluationFactRelationType.COMPLETED,
        EvaluationFactRelationType.UPDATED,
        EvaluationFactRelationType.REPLACED,
        EvaluationFactRelationType.CANCELLED,
    }
    transition_covered = {
        relation_id
        for match in matches
        if match.transition_assessment.value == "transition_full"
        for relation_id in match.matched_relation_ids
    }
    history_relation_by_id = {
        item.historical_fact_id: item for item in fact_relations
    }
    required = []
    for item in source_facts:
        if item.salience != EvaluationSalience.MUST_INCLUDE:
            continue
        if item.source_scope == EvaluationSourceScope.CURRENT:
            required.append(item)
            continue
        relation = history_relation_by_id.get(item.fact_id)
        if relation is None or relation.relation_type == EvaluationFactRelationType.HISTORY_ONLY:
            if item.history_disposition in VALID_HISTORY_DISPOSITIONS:
                required.append(item)
    missing = [item for item in required if item.fact_id not in covered]
    missing_transitions = [
        item
        for item in fact_relations
        if item.relation_type in transition_types
        and source_by_id.get(item.historical_fact_id) is not None
        and source_by_id[item.historical_fact_id].salience
        == EvaluationSalience.MUST_INCLUDE
        and item.relation_id not in transition_covered
    ]
    problematic = [
        item
        for item in matches
        if item.match_status != EvaluationMatchStatus.SUPPORTED_FULL
        or item.stale_history
        or item.duplicate_of_summary_fact_id is not None
    ]
    lines = [
        "# 纪要事实级评测 V2",
        "",
        "## 整体指标",
        "",
    ]
    for label, key in (
        ("Must Fact Recall", "must_fact_recall"),
        ("Current Must Fact Recall", "current_must_fact_recall"),
        ("Useful Fact Precision", "useful_fact_precision"),
        ("Fact Support Precision", "fact_support_precision"),
        ("Minutes Fact F1", "minutes_fact_f1"),
        ("Weighted Must Recall", "weighted_must_recall"),
    ):
        lines.append(f"- {label}：{_metric_text(metrics.core_metrics, key)}")
    lines.extend(["", "## 历史语义指标", ""])
    for label, key in (
        ("Historical Semantic Recall", "historical_semantic_recall"),
        ("Historical Useful Precision", "historical_useful_precision"),
        ("Historical Fact F1", "historical_fact_f1"),
        ("Historical Unique Recall", "historical_unique_recall"),
        ("Historical Unique Contribution Rate", "historical_unique_contribution_rate"),
        ("Current-Confirmed History Recall", "current_confirmed_history_recall"),
    ):
        lines.append(f"- {label}：{_metric_text(metrics.history_metrics, key)}")
    lines.extend(["", "## 状态变化指标", ""])
    for label, key in (
        ("State Transition Recall", "state_transition_recall"),
        ("State Transition Precision", "state_transition_precision"),
        ("State Transition F1", "state_transition_f1"),
        ("Current State Accuracy", "current_state_accuracy"),
        ("Stale State Error Rate", "stale_state_error_rate"),
    ):
        lines.append(f"- {label}：{_metric_text(metrics.transition_metrics, key)}")
    lines.extend(["", "## 安全性与简洁性", ""])
    for label, key, section in (
        ("Unsupported Rate", "unsupported_rate", metrics.safety_metrics),
        ("Contradiction Rate", "contradiction_rate", metrics.safety_metrics),
        ("Partial Rate", "partial_rate", metrics.safety_metrics),
        ("Stale History Rate", "stale_history_rate", metrics.safety_metrics),
        ("Redundancy Rate", "redundancy_rate", metrics.conciseness_metrics),
        ("Information Density", "information_density", metrics.conciseness_metrics),
        ("Compression Ratio", "compression_ratio", metrics.conciseness_metrics),
    ):
        lines.append(f"- {label}：{_metric_text(section, key)}")
    lines.extend(["", "## 未覆盖的必要事实与状态变化", ""])
    lines.extend(
        [f"- [{item.source_scope.value}] {item.fact}" for item in missing]
        + [
            f"- [transition:{item.relation_type.value}] {item.reason or item.relation_id}"
            for item in missing_transitions
        ]
        or ["- 无"]
    )
    lines.extend(["", "## 问题事实", ""])
    lines.extend(
        [
            (
                f"- {item.summary_fact_id}: {item.match_status.value}; "
                f"transition={item.transition_assessment.value}; "
                f"stale={item.stale_history}; duplicate_of="
                f"{item.duplicate_of_summary_fact_id or '-'}; {item.reason}"
            )
            for item in problematic
        ]
        or ["- 无"]
    )
    lines.extend(
        [
            "",
            "## 规模",
            "",
            f"- 源事实：{len(source_facts)}",
            f"- 事实关系：{len(fact_relations)}",
            f"- 摘要事实：{len(summary_facts)}",
            f"- 匹配记录：{len(matches)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
