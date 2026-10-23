"""不调用模型的确定性纪要事实指标。"""

from __future__ import annotations

import hashlib
import os
import re

from ..schemas import EvaluationStatus
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationFactRelationType,
    EvaluationMatchStatus,
    EvaluationMetrics,
    EvaluationSalience,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
    EvaluationTransitionAssessment,
    HistoricalDisposition,
    MetricValue,
)


VALID_HISTORY_DISPOSITIONS = {
    HistoricalDisposition.ACTIVE,
    HistoricalDisposition.COMPLETED,
    HistoricalDisposition.UPDATED,
}


def _ratio(numerator: int, denominator: int, *, unit: str = "ratio") -> MetricValue:
    """生成可表达空分母的指标值，空分母返回 ``None`` 而不是伪造数值。"""
    return MetricValue(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
        unit=unit,
    )


def _non_whitespace_chars(value: str) -> int:
    """统计摘要或上下文中的非空白字符数。"""
    return len(re.sub(r"\s+", "", value))


def persisted_summary_sha256(summary: str) -> str:
    """严格复现 Pipeline 文本产物的换行规范化后计算摘要 hash。"""

    persisted = summary.rstrip() + os.linesep
    return hashlib.sha256(persisted.encode("utf-8")).hexdigest()


def _harmonic(left: MetricValue, right: MetricValue) -> MetricValue:
    """计算两个比例的调和平均，保留空分母语义。"""
    if left.value is None or right.value is None:
        return MetricValue(numerator=0, denominator=0, value=None, unit="harmonic_mean")
    denominator = left.value + right.value
    return MetricValue(
        numerator=left.numerator + right.numerator,
        denominator=left.denominator + right.denominator,
        value=(2 * left.value * right.value / denominator if denominator else 0.0),
        unit="harmonic_mean",
    )


def calculate_evaluation_metrics(
    *,
    summary: str,
    current_material: str,
    visible_history_text: str,
    source_facts: tuple[EvaluationSourceFact, ...],
    summary_facts: tuple[EvaluationSummaryFact, ...],
    matches: tuple[EvaluationFactMatch, ...],
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
    config: dict[str, object] | None = None,
    execution_stats: dict[str, int] | None = None,
    warnings: tuple[str, ...] = (),
) -> EvaluationMetrics:
    """根据固定源事实、关系与匹配结果计算 V2 和兼容旧口径指标。"""
    current_required = {
        item.fact_id
        for item in source_facts
        if item.source_scope == EvaluationSourceScope.CURRENT
        and item.salience == EvaluationSalience.MUST_INCLUDE
    }
    legacy_history_required = {
        item.fact_id
        for item in source_facts
        if item.source_scope == EvaluationSourceScope.HISTORICAL
        and item.salience == EvaluationSalience.MUST_INCLUDE
        and item.history_disposition in VALID_HISTORY_DISPOSITIONS
    }
    legacy_useful_source_ids = {
        item.fact_id
        for item in source_facts
        if item.salience
        in {EvaluationSalience.MUST_INCLUDE, EvaluationSalience.USEFUL_OPTIONAL}
        and (
            item.source_scope == EvaluationSourceScope.CURRENT
            or item.history_disposition in VALID_HISTORY_DISPOSITIONS
        )
    }
    legacy_full_matches = tuple(
        item
        for item in matches
        if item.match_status == EvaluationMatchStatus.SUPPORTED_FULL
    )
    strict_matches = tuple(
        item
        for item in legacy_full_matches
        if not item.stale_history
        and item.transition_assessment
        not in {
            EvaluationTransitionAssessment.STALE_AS_CURRENT,
            EvaluationTransitionAssessment.TRANSITION_CONTRADICTED,
        }
    )
    covered_source_ids = {
        source_id
        for item in strict_matches
        if item.duplicate_of_summary_fact_id is None
        for source_id in item.matched_source_fact_ids
    }
    legacy_covered_source_ids = {
        source_id
        for item in legacy_full_matches
        if item.duplicate_of_summary_fact_id is None
        for source_id in item.matched_source_fact_ids
    }
    current_covered = len(current_required & covered_source_ids)
    legacy_history_covered = len(
        legacy_history_required & legacy_covered_source_ids
    )
    legacy_required = current_required | legacy_history_required
    legacy_required_covered = len(
        legacy_required & legacy_covered_source_ids
    )

    source_by_id = {item.fact_id: item for item in source_facts}
    summary_by_id = {item.summary_fact_id: item for item in summary_facts}
    relation_by_id = {item.relation_id: item for item in fact_relations}
    relation_by_history = {
        item.historical_fact_id: item for item in fact_relations
    }
    relations_by_current: dict[str, list[EvaluationFactRelation]] = {}
    for relation in fact_relations:
        for current_id in relation.current_fact_ids:
            relations_by_current.setdefault(current_id, []).append(relation)

    transition_types = {
        EvaluationFactRelationType.COMPLETED,
        EvaluationFactRelationType.UPDATED,
        EvaluationFactRelationType.REPLACED,
        EvaluationFactRelationType.CANCELLED,
    }
    valid_history_ids = {
        item.fact_id
        for item in source_facts
        if item.source_scope == EvaluationSourceScope.HISTORICAL
        and item.salience
        in {EvaluationSalience.MUST_INCLUDE, EvaluationSalience.USEFUL_OPTIONAL}
        and (
            item.history_disposition in VALID_HISTORY_DISPOSITIONS
            or (
                item.fact_id in relation_by_history
                and relation_by_history[item.fact_id].relation_type
                in ({EvaluationFactRelationType.CONFIRMED} | transition_types)
            )
        )
    }
    useful_source_ids = {
        item.fact_id
        for item in source_facts
        if item.salience
        in {EvaluationSalience.MUST_INCLUDE, EvaluationSalience.USEFUL_OPTIONAL}
        and (
            item.source_scope == EvaluationSourceScope.CURRENT
            or item.fact_id in valid_history_ids
        )
    }
    # 旧产物没有 relation；此时有效历史事实按 history-only 处理。
    history_only_ids = {
        item_id
        for item_id in valid_history_ids
        if item_id not in relation_by_history
        or relation_by_history[item_id].relation_type
        == EvaluationFactRelationType.HISTORY_ONLY
    }
    confirmed_history_ids = {
        item_id
        for item_id in valid_history_ids
        if item_id in relation_by_history
        and relation_by_history[item_id].relation_type
        == EvaluationFactRelationType.CONFIRMED
    }
    useful_transition_ids = {
        item.relation_id
        for item in fact_relations
        if item.relation_type in transition_types
        and item.historical_fact_id in valid_history_ids
    }

    transition_full_matches = tuple(
        item
        for item in strict_matches
        if item.duplicate_of_summary_fact_id is None
        and item.transition_assessment
        == EvaluationTransitionAssessment.TRANSITION_FULL
    )
    transition_covered_ids = {
        relation_id
        for item in transition_full_matches
        for relation_id in item.matched_relation_ids
        if relation_id in useful_transition_ids
    }
    confirmed_covered_ids = {
        history_id
        for history_id in confirmed_history_ids
        if history_id in covered_source_ids
        or bool(
            set(relation_by_history[history_id].current_fact_ids)
            & covered_source_ids
        )
    }
    history_family_by_id = {
        history_id: (
            relation_by_history[history_id].fact_family_id
            if history_id in relation_by_history
            else f"history_family:{history_id}"
        )
        for history_id in valid_history_ids
    }
    valid_history_families = set(history_family_by_id.values())
    confirmed_history_families = {
        history_family_by_id[item_id]
        for item_id in confirmed_history_ids
        if item_id in history_family_by_id
    }
    confirmed_covered_families = {
        history_family_by_id[item_id]
        for item_id in confirmed_covered_ids
        if item_id in history_family_by_id
    }
    history_semantic_covered_ids = (
        (valid_history_ids & covered_source_ids)
        | confirmed_covered_ids
        | {
            relation.historical_fact_id
            for current_id in covered_source_ids
            for relation in relations_by_current.get(current_id, [])
            if relation.historical_fact_id in valid_history_ids
            and relation.relation_type
            not in {
                EvaluationFactRelationType.HISTORY_ONLY,
                EvaluationFactRelationType.UNRESOLVED_CONFLICT,
            }
        }
        | {
            relation_by_id[relation_id].historical_fact_id
            for relation_id in transition_covered_ids
        }
    )
    history_semantic_covered_families = {
        history_family_by_id[item_id]
        for item_id in history_semantic_covered_ids
        if item_id in history_family_by_id
    }

    history_only_required = {
        item_id
        for item_id in history_only_ids
        if source_by_id[item_id].salience == EvaluationSalience.MUST_INCLUDE
    }
    required_transition_ids = {
        relation_id
        for relation_id in useful_transition_ids
        if source_by_id[relation_by_id[relation_id].historical_fact_id].salience
        == EvaluationSalience.MUST_INCLUDE
    }
    required_units = (
        {f"source:{item}" for item in current_required}
        | {f"history:{item}" for item in history_only_required}
        | {f"transition:{item}" for item in required_transition_ids}
    )
    covered_required_units = (
        {f"source:{item}" for item in current_required & covered_source_ids}
        | {f"history:{item}" for item in history_only_required & covered_source_ids}
        | {
            f"transition:{item}"
            for item in required_transition_ids & transition_covered_ids
        }
    )
    required_covered = len(covered_required_units)
    useful_claim_count = sum(
        1
        for item in strict_matches
        if item.duplicate_of_summary_fact_id is None
        and bool(useful_source_ids & set(item.matched_source_fact_ids))
    )
    claim_count = len(summary_facts)
    support_count = len(strict_matches)
    unsupported_count = sum(
        item.match_status == EvaluationMatchStatus.UNSUPPORTED for item in matches
    )
    contradiction_count = sum(
        item.match_status == EvaluationMatchStatus.CONTRADICTED for item in matches
    )
    partial_count = sum(
        item.match_status == EvaluationMatchStatus.SUPPORTED_PARTIAL for item in matches
    )
    stale_count = sum(item.stale_history for item in matches)
    duplicate_count = sum(
        item.duplicate_of_summary_fact_id is not None for item in matches
    )

    must_recall = _ratio(required_covered, len(required_units))
    useful_precision = _ratio(useful_claim_count, claim_count)
    minutes_f1 = _harmonic(must_recall, useful_precision)

    legacy_must_recall = _ratio(
        legacy_required_covered, len(legacy_required)
    )
    legacy_useful_precision = _ratio(
        sum(
            1
            for item in legacy_full_matches
            if item.duplicate_of_summary_fact_id is None
            and bool(
                legacy_useful_source_ids
                & set(item.matched_source_fact_ids)
            )
        ),
        claim_count,
    )
    legacy_minutes_f1 = _harmonic(
        legacy_must_recall, legacy_useful_precision
    )

    relation_ids_for_claim: dict[str, set[str]] = {}
    for match in matches:
        values = set(match.matched_relation_ids)
        for source_id in match.matched_source_fact_ids:
            if source_id in relation_by_history:
                values.add(relation_by_history[source_id].relation_id)
            for relation in relations_by_current.get(source_id, []):
                if relation.relation_type == EvaluationFactRelationType.CONFIRMED:
                    values.add(relation.relation_id)
        relation_ids_for_claim[match.summary_fact_id] = values

    history_related_claim_ids = {
        item.summary_fact_id
        for item in summary_facts
        if item.claimed_scope.value in {"historical", "mixed"}
    } | {
        item.summary_fact_id
        for item in matches
        if item.primary_source_scope == EvaluationSourceScope.HISTORICAL
        or bool(relation_ids_for_claim.get(item.summary_fact_id))
    }
    correct_history_claim_ids = {
        item.summary_fact_id
        for item in strict_matches
        if item.duplicate_of_summary_fact_id is None
        and (
            bool(valid_history_ids & set(item.matched_source_fact_ids))
            or any(
                relation_by_id[relation_id].historical_fact_id
                in valid_history_ids
                for relation_id in relation_ids_for_claim.get(
                    item.summary_fact_id, set()
                )
                if relation_id in relation_by_id
            )
        )
        and item.transition_assessment
        not in {
            EvaluationTransitionAssessment.STALE_AS_CURRENT,
            EvaluationTransitionAssessment.TRANSITION_CONTRADICTED,
        }
    }
    historical_semantic_recall = _ratio(
        len(history_semantic_covered_families),
        len(valid_history_families),
    )
    historical_useful_precision = _ratio(
        len(correct_history_claim_ids), len(history_related_claim_ids)
    )
    historical_fact_f1 = _harmonic(
        historical_semantic_recall, historical_useful_precision
    )

    asserted_transition_claims = {
        item.summary_fact_id
        for item in matches
        if item.transition_assessment
        in {
            EvaluationTransitionAssessment.TRANSITION_FULL,
            EvaluationTransitionAssessment.TRANSITION_PARTIAL,
            EvaluationTransitionAssessment.STALE_AS_CURRENT,
            EvaluationTransitionAssessment.TRANSITION_CONTRADICTED,
        }
    }
    correct_transition_claims = {
        item.summary_fact_id for item in transition_full_matches
    }
    state_related_claims = {
        item.summary_fact_id
        for item in matches
        if item.transition_assessment
        != EvaluationTransitionAssessment.NOT_APPLICABLE
    }
    correct_current_state_claims = {
        item.summary_fact_id
        for item in strict_matches
        if item.transition_assessment
        in {
            EvaluationTransitionAssessment.CURRENT_STATE_ONLY,
            EvaluationTransitionAssessment.TRANSITION_FULL,
        }
    }
    stale_state_claims = {
        item.summary_fact_id
        for item in matches
        if item.transition_assessment
        == EvaluationTransitionAssessment.STALE_AS_CURRENT
    }
    state_transition_recall = _ratio(
        len(transition_covered_ids), len(useful_transition_ids)
    )
    state_transition_precision = _ratio(
        len(correct_transition_claims), len(asserted_transition_claims)
    )

    summary_chars = _non_whitespace_chars(summary)
    current_chars = _non_whitespace_chars(current_material)
    history_chars = _non_whitespace_chars(visible_history_text)
    context_chars = current_chars + history_chars
    useful_unique_covered = len(useful_source_ids & covered_source_ids)
    current_family_by_id: dict[str, str] = {}
    for source_id in useful_source_ids:
        if source_by_id[source_id].source_scope != EvaluationSourceScope.CURRENT:
            continue
        related = [
            relation
            for relation in relations_by_current.get(source_id, [])
            if relation.historical_fact_id in valid_history_ids
            and relation.relation_type
            != EvaluationFactRelationType.UNRESOLVED_CONFLICT
        ]
        current_family_by_id[source_id] = (
            sorted(related, key=lambda item: item.relation_id)[0].fact_family_id
            if related
            else f"current_family:{source_id}"
        )
    useful_covered_families = {
        current_family_by_id[item]
        for item in useful_source_ids & covered_source_ids
        if item in current_family_by_id
    } | {
        history_family_by_id[item]
        for item in valid_history_ids & covered_source_ids
        if item in history_family_by_id
    } | {
        relation_by_id[item].fact_family_id
        for item in transition_covered_ids
        if item in relation_by_id
    }
    history_only_families = {
        history_family_by_id[item]
        for item in history_only_ids
        if item in history_family_by_id
    }
    history_only_covered_families = {
        history_family_by_id[item]
        for item in history_only_ids & covered_source_ids
        if item in history_family_by_id
    }

    partial_required_units: set[str] = set()
    for item in matches:
        if item.match_status != EvaluationMatchStatus.SUPPORTED_PARTIAL:
            continue
        partial_required_units.update(
            f"source:{source_id}"
            for source_id in item.matched_source_fact_ids
            if f"source:{source_id}" in required_units
        )
        if (
            item.transition_assessment
            == EvaluationTransitionAssessment.TRANSITION_PARTIAL
        ):
            partial_required_units.update(
                f"transition:{relation_id}"
                for relation_id in item.matched_relation_ids
                if f"transition:{relation_id}" in required_units
            )
    partial_required_units -= covered_required_units
    weighted_denominator = len(required_units) * 2
    weighted_numerator = required_covered * 2 + len(partial_required_units)
    information_density = MetricValue(
        numerator=useful_unique_covered,
        denominator=summary_chars,
        value=(
            useful_unique_covered / summary_chars * 100
            if summary_chars
            else None
        ),
        unit="useful_facts_per_100_non_whitespace_chars",
    )

    return EvaluationMetrics(
        status=EvaluationStatus.COMPLETED,
        evaluated_summary_sha256=persisted_summary_sha256(summary),
        counts={
            "source_fact_count": len(source_facts),
            "current_source_fact_count": sum(
                item.source_scope == EvaluationSourceScope.CURRENT
                for item in source_facts
            ),
            "historical_source_fact_count": sum(
                item.source_scope == EvaluationSourceScope.HISTORICAL
                for item in source_facts
            ),
            "summary_fact_count": claim_count,
            "supported_full_count": support_count,
            "supported_partial_count": partial_count,
            "contradicted_count": contradiction_count,
            "unsupported_count": unsupported_count,
            "stale_history_count": stale_count,
            "duplicate_count": duplicate_count,
            "current_required_count": len(current_required),
            "historical_required_count": (
                len(history_only_required) + len(required_transition_ids)
            ),
            "historical_semantic_fact_count": len(valid_history_ids),
            "historical_semantic_family_count": len(valid_history_families),
            "historical_unique_fact_count": len(history_only_ids),
            "confirmed_history_fact_count": len(confirmed_history_ids),
            "state_transition_count": len(useful_transition_ids),
            "useful_source_fact_count": len(useful_source_ids),
            "useful_unique_covered_count": useful_unique_covered,
        },
        core_metrics={
            "must_fact_recall": must_recall,
            "current_must_fact_recall": _ratio(
                current_covered, len(current_required)
            ),
            "historical_useful_recall": _ratio(
                legacy_history_covered, len(legacy_history_required)
            ),
            "useful_fact_precision": useful_precision,
            "fact_support_precision": _ratio(support_count, claim_count),
            "minutes_fact_f1": minutes_f1,
            "weighted_must_recall": MetricValue(
                numerator=weighted_numerator,
                denominator=weighted_denominator,
                value=(
                    weighted_numerator / weighted_denominator
                    if weighted_denominator
                    else None
                ),
                unit="half_credit_ratio",
            ),
        },
        history_metrics={
            "historical_semantic_recall": historical_semantic_recall,
            "historical_useful_precision": historical_useful_precision,
            "historical_fact_f1": historical_fact_f1,
            "historical_unique_recall": _ratio(
                len(history_only_covered_families), len(history_only_families)
            ),
            "historical_unique_contribution_rate": _ratio(
                len(history_only_covered_families),
                len(useful_covered_families),
            ),
            "current_confirmed_history_recall": _ratio(
                len(confirmed_covered_families), len(confirmed_history_families)
            ),
        },
        transition_metrics={
            "state_transition_recall": state_transition_recall,
            "state_transition_precision": state_transition_precision,
            "state_transition_f1": _harmonic(
                state_transition_recall, state_transition_precision
            ),
            "current_state_accuracy": _ratio(
                len(correct_current_state_claims), len(state_related_claims)
            ),
            "stale_state_error_rate": _ratio(
                len(stale_state_claims), len(state_related_claims)
            ),
        },
        safety_metrics={
            "unsupported_rate": _ratio(unsupported_count, claim_count),
            "contradiction_rate": _ratio(contradiction_count, claim_count),
            "partial_rate": _ratio(partial_count, claim_count),
            "stale_history_rate": _ratio(stale_count, claim_count),
        },
        conciseness_metrics={
            "redundancy_rate": _ratio(duplicate_count, claim_count),
            "information_density": information_density,
            "compression_ratio": _ratio(summary_chars, context_chars),
        },
        lengths={
            "summary_non_whitespace_chars": summary_chars,
            "current_material_non_whitespace_chars": current_chars,
            "visible_history_non_whitespace_chars": history_chars,
            "evaluation_context_non_whitespace_chars": context_chars,
        },
        execution_stats=dict(execution_stats or {}),
        config=dict(config or {}),
        legacy_metrics={
            "legacy_must_fact_recall": legacy_must_recall,
            "legacy_historical_useful_recall": _ratio(
                legacy_history_covered, len(legacy_history_required)
            ),
            "legacy_minutes_fact_f1": legacy_minutes_f1,
        },
        warnings=warnings,
    )
