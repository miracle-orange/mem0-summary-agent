"""同一 Source Snapshot 下串行评测多份纪要并聚合横向差异。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..adapters.model import ModelClient
from ..schemas import EvaluationStatus, RunConfig
from .metrics import VALID_HISTORY_DISPOSITIONS
from .model_io import SharedEvaluationModelClient
from .comparison_scope import (
    EvaluationComparisonScopeEvaluator,
    build_comparison_basis,
)
from .comparison_interpretation import ComparisonInterpretationRunner
from .schemas import (
    EvaluationArtifacts,
    EvaluationComparisonBasis,
    EvaluationComparisonScopeResult,
    EvaluationCompareRequest,
    EvaluationFactMatch,
    EvaluationFactRelationType,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceScope,
    EvaluationTransitionAssessment,
)
from .scorer import EvaluationSummaryScorer
from .snapshot import EvaluationSourceSnapshotBuilder, EvaluationSourceSnapshotBundle


@dataclass(frozen=True)
class EvaluationComparisonSummaryResult:
    """一份纪要的成功或失败结果，失败不影响其他纪要。"""

    summary_id: str
    display_name: str
    status: str
    artifacts: Optional[EvaluationArtifacts] = None
    scope_result: Optional[EvaluationComparisonScopeResult] = None
    error: Optional[str] = None
    model_trace: tuple[dict[str, Any], ...] = ()
    scope_model_trace: tuple[dict[str, Any], ...] = ()
    personalization_model_trace: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvaluationComparisonResult:
    """独立横评运行的固定 Snapshot、逐份结果和聚合报告。"""

    status: str
    snapshot_bundle: EvaluationSourceSnapshotBundle
    summaries: tuple[EvaluationComparisonSummaryResult, ...]
    comparison_metrics: dict[str, Any]
    report_markdown: str
    comparison_basis: Optional[EvaluationComparisonBasis] = None
    source_model_trace: tuple[dict[str, Any], ...] = ()
    source_personalization_model_trace: tuple[dict[str, Any], ...] = ()
    comparison_interpretation_model_trace: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()


def comparison_run_config(request: EvaluationCompareRequest) -> RunConfig:
    """把横评配置映射到既有评测组件使用的 RunConfig。"""
    value = request.evaluation_config
    return RunConfig(
        evaluation_enabled=True,
        raw_segment_max_chars=value.segment_max_chars,
        evaluation_segment_max_chars=value.segment_max_chars,
        evaluation_source_fact_max_tokens=value.source_fact_max_tokens,
        evaluation_summary_fact_max_tokens=value.summary_fact_max_tokens,
        evaluation_match_max_tokens=value.match_max_tokens,
        evaluation_repair_attempts=value.repair_attempts,
        evaluation_timeout_seconds=value.comparison_per_summary_timeout_seconds,
        evaluation_model_timeout_seconds=value.comparison_model_timeout_seconds,
        evaluation_max_model_calls=value.comparison_per_summary_max_model_calls,
        evaluation_max_concurrent_segment_maps=1,
        evaluation_match_batch_size=value.match_batch_size,
        evaluation_history_batch_size=value.history_batch_size,
        evaluation_match_full_scan_threshold=value.match_full_scan_threshold,
        evaluation_match_candidate_limit_per_fact=(
            value.match_candidate_limit_per_fact
        ),
        evaluation_match_candidate_union_limit=value.match_candidate_union_limit,
        evaluation_personalization_batch_size=value.personalization_batch_size,
        evaluation_personalization_max_tokens=value.personalization_max_tokens,
        evaluation_personalization_max_model_calls=(
            value.comparison_personalization_per_summary_max_model_calls
        ),
        evaluation_personalization_timeout_seconds=(
            value.comparison_per_summary_timeout_seconds
        ),
        evaluation_adaptive_split_enabled=value.adaptive_split_enabled,
        evaluation_adaptive_split_min_chars=value.adaptive_split_min_chars,
        evaluation_adaptive_split_max_depth=value.adaptive_split_max_depth,
    )


def _metric_value(artifacts: EvaluationArtifacts, name: str) -> Optional[float]:
    """从整体、历史或状态 section 查找一个指标值。"""
    for section in (
        artifacts.metrics.core_metrics,
        artifacts.metrics.history_metrics,
        artifacts.metrics.transition_metrics,
        artifacts.metrics.safety_metrics,
        artifacts.metrics.conciseness_metrics,
    ):
        if name in section:
            return section[name].value
    personalization = artifacts.personalization
    if personalization is not None and name in personalization.metrics.macro_metrics:
        return personalization.metrics.macro_metrics[name]
    return None


def _covered_fact_ids(matches: tuple[EvaluationFactMatch, ...]) -> set[str]:
    """提取严格且非重复 Claim 覆盖的源事实 ID。"""
    return {
        source_id
        for match in matches
        if match.match_status == EvaluationMatchStatus.SUPPORTED_FULL
        and match.duplicate_of_summary_fact_id is None
        and not match.stale_history
        and match.transition_assessment
        not in {
            EvaluationTransitionAssessment.STALE_AS_CURRENT,
            EvaluationTransitionAssessment.TRANSITION_CONTRADICTED,
        }
        for source_id in match.matched_source_fact_ids
    }


def aggregate_comparison(
    request: EvaluationCompareRequest,
    snapshot_bundle: EvaluationSourceSnapshotBundle,
    results: tuple[EvaluationComparisonSummaryResult, ...],
    *,
    comparison_basis: Optional[EvaluationComparisonBasis] = None,
    total_model_calls: int,
) -> dict[str, Any]:
    """确定性生成指标表、相对首份差值及内容覆盖差异。"""
    comparison_basis = comparison_basis or build_comparison_basis(
        snapshot_id=snapshot_bundle.snapshot.snapshot_id,
        source_facts=snapshot_bundle.snapshot.source_facts,
        fact_relations=snapshot_bundle.snapshot.fact_relations,
    )
    successful = [
        item
        for item in results
        if item.artifacts is not None
        and (
            item.scope_result is None
            or item.scope_result.all_sources is not None
            or item.scope_result.current_only is not None
        )
    ]
    metric_names = (
        "must_fact_recall",
        "current_must_fact_recall",
        "useful_fact_precision",
        "fact_support_precision",
        "minutes_fact_f1",
        "historical_semantic_recall",
        "historical_useful_precision",
        "historical_fact_f1",
        "historical_unique_recall",
        "historical_unique_contribution_rate",
        "current_confirmed_history_recall",
        "state_transition_recall",
        "state_transition_precision",
        "state_transition_f1",
        "current_state_accuracy",
        "stale_state_error_rate",
        "target_content_must_recall",
        "target_content_useful_precision",
        "target_content_f1",
        "target_directed_must_recall",
        "target_directed_useful_precision",
        "target_directed_f1",
    )
    scope_metric_names = (
        "must_fact_recall",
        "useful_fact_precision",
        "fact_support_precision",
        "minutes_fact_f1",
        "unsupported_rate",
        "contradiction_rate",
        "partial_rate",
        "redundancy_rate",
    )

    def scope_values(
        result: EvaluationComparisonSummaryResult,
        scope: str,
    ) -> Optional[dict[str, Optional[float]]]:
        """提取一份纪要指定范围的核心与安全指标。"""
        if result.scope_result is None:
            return None
        metrics = getattr(result.scope_result, scope)
        if metrics is None:
            return None
        sections = {**metrics.core_metrics, **metrics.safety_metrics}
        return {
            name: sections[name].value if name in sections else None
            for name in scope_metric_names
        }

    def scope_details(
        result: EvaluationComparisonSummaryResult,
        scope: str,
    ) -> Optional[dict[str, dict[str, Any]]]:
        """保留横评范围指标的分子、分母和值，供报告和 Web 审计。"""
        if result.scope_result is None:
            return None
        metrics = getattr(result.scope_result, scope)
        if metrics is None:
            return None
        sections = {**metrics.core_metrics, **metrics.safety_metrics}
        return {
            name: sections[name].model_dump(mode="json")
            for name in scope_metric_names
            if name in sections
        }

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "summary_id": result.summary_id,
            "display_name": result.display_name,
            "status": result.status,
            "error": result.error,
            "snapshot_id": snapshot_bundle.snapshot.snapshot_id,
        }
        if result.artifacts is not None:
            row["metrics"] = {
                name: _metric_value(result.artifacts, name)
                for name in metric_names
            }
            row["overall_evaluation_status"] = result.artifacts.metrics.status.value
            row["personalization_status"] = (
                result.artifacts.personalization.metrics.status.value
                if result.artifacts.personalization is not None
                else EvaluationStatus.DISABLED.value
            )
        row["scope_metrics"] = {
            scope: values
            for scope in ("all_sources", "current_only")
            if (values := scope_values(result, scope)) is not None
        }
        row["scope_metric_details"] = {
            scope: details
            for scope in ("all_sources", "current_only")
            if (details := scope_details(result, scope)) is not None
        }
        row["scope_statuses"] = (
            {
                scope: status.value
                for scope, status in result.scope_result.scope_statuses.items()
            }
            if result.scope_result is not None
            else {}
        )
        row["scope_errors"] = (
            dict(result.scope_result.scope_errors)
            if result.scope_result is not None
            else {}
        )
        rows.append(row)

    baseline_result = results[0] if results else None
    baseline = (
        baseline_result
        if baseline_result is not None and baseline_result.artifacts is not None
        else None
    )
    baseline_metrics = (
        {
            name: _metric_value(baseline.artifacts, name)
            for name in metric_names
        }
        if baseline is not None and baseline.artifacts is not None
        else {}
    )
    deltas: dict[str, dict[str, Optional[float]]] = {}
    for result in successful:
        assert result.artifacts is not None
        deltas[result.summary_id] = {}
        for name in metric_names:
            value = _metric_value(result.artifacts, name)
            base = baseline_metrics.get(name)
            deltas[result.summary_id][name] = (
                value - base if value is not None and base is not None else None
            )

    baseline_scope_metrics = {
        scope: scope_values(baseline_result, scope) if baseline_result else None
        for scope in ("all_sources", "current_only")
    }
    scope_deltas: dict[str, dict[str, dict[str, Optional[float]]]] = {}
    for result in results:
        scope_deltas[result.summary_id] = {}
        for scope in ("all_sources", "current_only"):
            values = scope_values(result, scope)
            baseline_values = baseline_scope_metrics.get(scope)
            if values is None:
                continue
            scope_deltas[result.summary_id][scope] = {
                name: (
                    values[name] - baseline_values[name]
                    if (
                        baseline_values is not None
                        and values.get(name) is not None
                        and baseline_values.get(name) is not None
                    )
                    else None
                )
                for name in scope_metric_names
            }

    relation_by_history = {
        item.historical_fact_id: item
        for item in snapshot_bundle.snapshot.fact_relations
    }
    relation_valid_types = {
        EvaluationFactRelationType.CONFIRMED,
        EvaluationFactRelationType.COMPLETED,
        EvaluationFactRelationType.UPDATED,
        EvaluationFactRelationType.REPLACED,
        EvaluationFactRelationType.CANCELLED,
    }
    useful_ids = {
        item.fact_id
        for item in snapshot_bundle.snapshot.source_facts
        if item.salience
        in {EvaluationSalience.MUST_INCLUDE, EvaluationSalience.USEFUL_OPTIONAL}
        and (
            item.source_scope == EvaluationSourceScope.CURRENT
            or item.history_disposition in VALID_HISTORY_DISPOSITIONS
            or (
                item.fact_id in relation_by_history
                and relation_by_history[item.fact_id].relation_type
                in relation_valid_types
            )
        )
    }
    source_by_id = {
        item.fact_id: item for item in snapshot_bundle.snapshot.source_facts
    }

    def fact_rows(fact_ids: set[str] | list[str]) -> list[dict[str, str]]:
        """将比较集合投影为可读但不包含原文段落的原子事实。"""
        return [
            {
                "fact_id": fact_id,
                "source_scope": source_by_id[fact_id].source_scope.value,
                "fact": source_by_id[fact_id].fact,
            }
            for fact_id in sorted(fact_ids)
            if fact_id in source_by_id
        ]
    covered_by_summary = {
        item.summary_id: _covered_fact_ids(item.artifacts.matches) & useful_ids
        for item in successful
        if item.artifacts is not None
    }
    covered_sets = list(covered_by_summary.values())
    common_covered = set.intersection(*covered_sets) if covered_sets else set()
    union_covered = set.union(*covered_sets) if covered_sets else set()
    common_missed = useful_ids - union_covered
    unique_covered = {
        summary_id: sorted(
            covered
            - set().union(
                *(
                    other
                    for other_id, other in covered_by_summary.items()
                    if other_id != summary_id
                )
            )
        )
        for summary_id, covered in covered_by_summary.items()
    }
    problem_claims: dict[str, list[dict[str, Any]]] = {}
    for result in successful:
        assert result.artifacts is not None
        summary_fact_by_id = {
            item.summary_fact_id: item for item in result.artifacts.summary_facts
        }
        problem_claims[result.summary_id] = [
            {
                "summary_fact_id": item.summary_fact_id,
                "claim": (
                    summary_fact_by_id[item.summary_fact_id].fact
                    if item.summary_fact_id in summary_fact_by_id
                    else ""
                ),
                "status": item.match_status.value,
                "transition_assessment": item.transition_assessment.value,
                "stale_history": item.stale_history,
                "reason_code": item.reason_code,
                "reason": item.reason,
            }
            for item in result.artifacts.matches
            if item.match_status != EvaluationMatchStatus.SUPPORTED_FULL
            or item.stale_history
        ]

    scope_problem_claims: dict[str, dict[str, list[dict[str, Any]]]] = {
        "all_sources": {},
        "current_only": {},
    }
    for result in results:
        if result.artifacts is None or result.scope_result is None:
            continue
        summary_fact_by_id = {
            item.summary_fact_id: item for item in result.artifacts.summary_facts
        }
        diagnostic_by_id = {
            str(item.get("summary_fact_id")): item
            for item in result.scope_result.claim_diagnostics
        }
        for scope in ("all_sources", "current_only"):
            matches = getattr(result.scope_result, f"{scope}_matches")
            scope_problem_claims[scope][result.summary_id] = [
                {
                    "summary_fact_id": item.summary_fact_id,
                    "claim": (
                        summary_fact_by_id[item.summary_fact_id].fact
                        if item.summary_fact_id in summary_fact_by_id
                        else ""
                    ),
                    "status": item.match_status.value,
                    "duplicate_of_summary_fact_id": (
                        item.duplicate_of_summary_fact_id
                    ),
                    "reason_code": item.reason_code,
                    "reason": item.reason,
                    "scope_interpretation": diagnostic_by_id.get(
                        item.summary_fact_id, {}
                    ).get("interpretation"),
                }
                for item in matches
                if (
                    item.match_status != EvaluationMatchStatus.SUPPORTED_FULL
                    or item.duplicate_of_summary_fact_id is not None
                )
            ]

    def scope_unit_rows(
        scope: str,
        unit_ids: set[str] | list[str],
    ) -> list[dict[str, Any]]:
        """把范围事实单位投影为简洁、可审计的内容差异记录。"""
        basis = getattr(comparison_basis, scope)
        unit_by_id = {item.unit_id: item for item in basis.units}
        return [
            {
                "unit_id": unit_id,
                "fact": unit_by_id[unit_id].fact,
                "fact_type": unit_by_id[unit_id].fact_type,
                "salience": unit_by_id[unit_id].salience.value,
                "source_fact_ids": list(unit_by_id[unit_id].source_fact_ids),
                "source_scopes": [
                    value.value for value in unit_by_id[unit_id].source_scopes
                ],
            }
            for unit_id in sorted(unit_ids)
            if unit_id in unit_by_id
        ]

    def scope_coverage(scope: str) -> dict[str, Any]:
        """汇总一个新范围中共同、遗漏和各纪要独有的事实单位。"""
        basis = getattr(comparison_basis, scope)
        source_to_unit = {
            source_id: unit.unit_id
            for unit in basis.units
            for source_id in unit.source_fact_ids
        }
        useful_units = set(basis.useful_unit_ids)
        covered_by_result: dict[str, set[str]] = {}
        for item in results:
            if (
                item.scope_result is None
                or getattr(item.scope_result, scope) is None
            ):
                continue
            matches = getattr(item.scope_result, f"{scope}_matches")
            covered_by_result[item.summary_id] = {
                source_to_unit[source_id]
                for source_id in _covered_fact_ids(matches)
                if source_id in source_to_unit
            } & useful_units
        sets = list(covered_by_result.values())
        common = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        unique = {
            summary_id: sorted(
                covered
                - set().union(
                    *(
                        other
                        for other_id, other in covered_by_result.items()
                        if other_id != summary_id
                    )
                )
            )
            for summary_id, covered in covered_by_result.items()
        }
        return {
            "covered_unit_ids_by_summary": {
                summary_id: sorted(ids)
                for summary_id, ids in covered_by_result.items()
            },
            "common_covered_unit_ids": sorted(common),
            "common_missed_unit_ids": sorted(useful_units - union),
            "unique_covered_unit_ids": unique,
            "common_covered_units": scope_unit_rows(scope, common),
            "common_missed_units": scope_unit_rows(scope, useful_units - union),
            "unique_covered_units": {
                summary_id: scope_unit_rows(scope, ids)
                for summary_id, ids in unique.items()
            },
        }

    scope_coverages = {
        scope: scope_coverage(scope)
        for scope in ("all_sources", "current_only")
    }

    def problem_signature(item: dict[str, Any]) -> tuple[str, str]:
        """用规范化 Claim 文本和状态识别相对基线新增的问题。"""
        claim = " ".join(str(item.get("claim") or "").split()).casefold()
        return claim, str(item.get("status") or "")

    def scope_content_differences(scope: str) -> dict[str, dict[str, Any]]:
        """按首份纪要生成新增覆盖、减少覆盖和新增问题。"""
        if baseline_result is None:
            return {}
        baseline_id = baseline_result.summary_id
        covered_by_summary = scope_coverages[scope].get(
            "covered_unit_ids_by_summary", {}
        )
        baseline_covered = set(covered_by_summary.get(baseline_id, []))
        baseline_available = baseline_id in covered_by_summary
        baseline_problems = scope_problem_claims.get(scope, {}).get(
            baseline_id, []
        )
        baseline_problem_signatures = {
            problem_signature(item) for item in baseline_problems
        }
        differences: dict[str, dict[str, Any]] = {}
        for result in results[1:]:
            candidate_available = result.summary_id in covered_by_summary
            if not baseline_available or not candidate_available:
                differences[result.summary_id] = {
                    "status": "unavailable",
                    "baseline_summary_id": baseline_id,
                    "reason": "基线或当前纪要的该范围评测未完成",
                    "added_covered_count": 0,
                    "reduced_covered_count": 0,
                    "added_problem_count": 0,
                    "added_covered_units": [],
                    "reduced_covered_units": [],
                    "added_problem_claims": [],
                }
                continue
            candidate_covered = set(
                covered_by_summary.get(result.summary_id, [])
            )
            added_ids = candidate_covered - baseline_covered
            reduced_ids = baseline_covered - candidate_covered
            candidate_problems = scope_problem_claims.get(scope, {}).get(
                result.summary_id, []
            )
            added_problems = [
                item
                for item in candidate_problems
                if problem_signature(item) not in baseline_problem_signatures
            ]
            differences[result.summary_id] = {
                "status": "completed",
                "baseline_summary_id": baseline_id,
                "reason": None,
                "added_covered_count": len(added_ids),
                "reduced_covered_count": len(reduced_ids),
                "added_problem_count": len(added_problems),
                "added_covered_units": scope_unit_rows(scope, added_ids),
                "reduced_covered_units": scope_unit_rows(scope, reduced_ids),
                "added_problem_claims": added_problems,
            }
        return differences

    scope_content_difference_rows = {
        scope: scope_content_differences(scope)
        for scope in ("all_sources", "current_only")
    }

    return {
        "schema_version": "summary_multi_agent.evaluation_comparison_metrics.v2",
        "snapshot_id": snapshot_bundle.snapshot.snapshot_id,
        "baseline_summary_id": (
            baseline_result.summary_id if baseline_result is not None else None
        ),
        "baseline_status": (
            baseline_result.status if baseline_result is not None else None
        ),
        "rows": rows,
        "deltas_from_first_summary": deltas,
        "scope_deltas_from_first_summary": scope_deltas,
        "scope_metric_names": list(scope_metric_names),
        "scope_coverage": scope_coverages,
        "scope_content_differences": scope_content_difference_rows,
        "scope_problem_claims": scope_problem_claims,
        "common_covered_fact_ids": sorted(common_covered),
        "common_missed_fact_ids": sorted(common_missed),
        "unique_covered_fact_ids": unique_covered,
        "common_covered_facts": fact_rows(common_covered),
        "common_missed_facts": fact_rows(common_missed),
        "unique_covered_facts": {
            summary_id: fact_rows(fact_ids)
            for summary_id, fact_ids in unique_covered.items()
        },
        "problem_claims": problem_claims,
        "source_fact_count": len(snapshot_bundle.snapshot.source_facts),
        "fact_relation_count": len(snapshot_bundle.snapshot.fact_relations),
        "successful_summary_count": len(successful),
        "failed_summary_count": len(results) - len(successful),
        "total_model_call_count": total_model_calls,
        "max_total_model_calls": (
            request.evaluation_config.comparison_max_total_model_calls
        ),
    }


def build_comparison_report(
    request: EvaluationCompareRequest,
    metrics: dict[str, Any],
) -> str:
    """渲染不宣布赢家的横向指标与差值报告。"""
    scope_metric_names = (
        "must_fact_recall",
        "useful_fact_precision",
        "fact_support_precision",
        "minutes_fact_f1",
        "unsupported_rate",
        "contradiction_rate",
        "partial_rate",
        "redundancy_rate",
    )
    metric_groups = (
        (
            "历史事实指标",
            (
                "historical_semantic_recall",
                "historical_useful_precision",
                "historical_fact_f1",
                "historical_unique_recall",
                "historical_unique_contribution_rate",
                "current_confirmed_history_recall",
            ),
        ),
        (
            "状态变化指标",
            (
                "state_transition_recall",
                "state_transition_precision",
                "state_transition_f1",
                "current_state_accuracy",
                "stale_state_error_rate",
            ),
        ),
        (
            "个性化指标",
            (
                "target_content_must_recall",
                "target_content_useful_precision",
                "target_content_f1",
                "target_directed_must_recall",
                "target_directed_useful_precision",
                "target_directed_f1",
            ),
        ),
    )

    def fmt(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    def fmt_scope_metric(row: dict[str, Any], scope: str, name: str) -> str:
        """把范围指标渲染为值与分子/分母，空分母保持 N/A。"""
        details = row.get("scope_metric_details", {}).get(scope, {}).get(name)
        if not isinstance(details, dict):
            return "N/A"
        value = fmt(details.get("value"))
        return f"{value} ({details.get('numerator', 0)}/{details.get('denominator', 0)})"

    def fmt_content_unit(item: dict[str, Any]) -> str:
        """用来源和重要性标记一条差异事实。"""
        scopes = "/".join(item.get("source_scopes", [])) or "unknown"
        salience = {
            "must_include": "must",
            "useful_optional": "useful",
        }.get(str(item.get("salience") or ""), str(item.get("salience") or "N/A"))
        return f"[{scopes}][{salience}] {item.get('fact') or item.get('unit_id')}"

    lines = [
        "# 多纪要事实级横向评测",
        "",
        f"- Snapshot：`{metrics['snapshot_id']}`",
        f"- 当前材料：`{request.current_source.label}`",
        f"- 纪要数量：{len(request.summaries)}",
        f"- 成功/失败：{metrics['successful_summary_count']}/{metrics['failed_summary_count']}",
        "- 第一份纪要仅作为差值基线；若其失败则差值为 N/A。本报告不自动推荐最佳纪要。",
        "",
    ]
    scope_titles = (
        ("all_sources", "全材料整体指标"),
        ("current_only", "当前材料整体指标"),
    )
    for scope, title in scope_titles:
        scope_note = (
            "当前与历史材料统一形成事实单位；等价重复只计一次，状态关系不额外增加整体分母。"
            if scope == "all_sources"
            else "只允许当前材料提供事实支持；仅被历史材料支持的 Claim 在本范围显示为无支持。"
        )
        lines.extend(
            [
                f"## {title}",
                "",
                scope_note,
                "",
                "| 纪要 | 状态 | " + " | ".join(scope_metric_names) + " |",
                "|---|---|" + "---|" * len(scope_metric_names),
            ]
        )
        for row in metrics["rows"]:
            scope_metrics = row.get("scope_metrics", {}).get(scope, {})
            scope_status = row.get("scope_statuses", {}).get(
                scope,
                "completed" if scope_metrics else "failed",
            )
            lines.append(
                "| "
                + str(row["display_name"])
                + " | "
                + str(scope_status)
                + " | "
                + " | ".join(
                    fmt_scope_metric(row, scope, name)
                    for name in scope_metric_names
                )
                + " |"
            )
        lines.append("")
    for title, names in metric_groups:
        if title == "个性化指标" and not request.target_entities:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| 纪要 | 状态 | " + " | ".join(names) + " |",
                "|---|---|" + "---|" * len(names),
            ]
        )
        for row in metrics["rows"]:
            row_metrics = row.get("metrics", {})
            lines.append(
                "| "
                + str(row["display_name"])
                + " | "
                + str(row["status"])
                + " | "
                + " | ".join(fmt(row_metrics.get(name)) for name in names)
                + " |"
            )
        lines.append("")

    lines.extend(["## LLM 直接定性对比", ""])
    lines.append(
        "以下内容由独立解释层直接阅读原始材料和各份纪要后生成，"
        "不参与数值指标计算，也不自动推荐赢家。"
    )
    lines.append("")
    qualitative = metrics.get("qualitative_comparison", {})
    for scope, title in scope_titles:
        lines.append(f"### {title}")
        result = qualitative.get(scope, {}) if isinstance(qualitative, dict) else {}
        if not isinstance(result, dict) or result.get("status") not in {
            "completed",
            "completed_with_warnings",
        }:
            lines.append(
                f"- 未完成：{result.get('error') or '本次运行未启用该解释层'}"
                if isinstance(result, dict)
                else "- 本次运行未启用该解释层"
            )
            lines.append("")
            continue
        if result.get("status") == "completed_with_warnings":
            lines.append(
                "- 注意：部分定性观察因证据 Span 无效而被隔离；"
                f"未核验 {result.get('unverified_note_count', 0)} 条。"
            )
        lines.append(f"- 总览：{result.get('overview') or 'N/A'}")
        for item in result.get("summaries", []):
            summary_id = str(item.get("summary_id") or "")
            display_name = next(
                (
                    row.get("display_name")
                    for row in metrics["rows"]
                    if str(row.get("summary_id")) == summary_id
                ),
                summary_id,
            )
            lines.append(f"- {display_name}：{item.get('relative_to_baseline') or 'N/A'}")
            for label, notes in (
                ("优势", item.get("strengths", [])),
                ("局限", item.get("limitations", [])),
            ):
                for note in notes:
                    lines.append(
                        f"  - {label}：{note.get('description')} "
                        f"[来源：{note.get('source_id')}；原文：{note.get('source_quote')}]"
                    )
        lines.append("")
    lines.extend(["", "## 相对第一份纪要的单项差值", ""])
    for summary_id, values in metrics["deltas_from_first_summary"].items():
        lines.append(f"### {summary_id}")
        scope_deltas = metrics.get("scope_deltas_from_first_summary", {}).get(
            summary_id, {}
        )
        for scope, title in scope_titles:
            lines.append(f"- {title}：")
            values_for_scope = scope_deltas.get(scope, {})
            lines.extend(
                f"  - {name}: {fmt(values_for_scope.get(name))}"
                for name in scope_metric_names
            )
        for title, names in metric_groups:
            if title == "个性化指标" and not request.target_entities:
                continue
            lines.append(f"- {title}：")
            lines.extend(
                f"  - {name}: {fmt(values.get(name))}" for name in names
            )
        lines.append("")

    lines.extend(
        [
            "<details>",
            "<summary>事实覆盖差异（二次核验，仅供核查）</summary>",
            "",
            "原始增减候选来自结构化 Claim 匹配；下列内容经过模型重新阅读两份纪要并对称判断。"
            "它不回写或修改数值指标。",
            "",
        ]
    )
    verified = metrics.get("verified_content_differences", {})
    row_by_id = {
        str(item.get("summary_id")): item for item in metrics["rows"]
    }
    baseline_id = str(metrics.get("baseline_summary_id") or "")
    baseline_name = row_by_id.get(baseline_id, {}).get(
        "display_name", baseline_id or "第一份纪要"
    )
    for scope, title in scope_titles:
        lines.append(f"### {title}")
        scope_results = verified.get(scope, {}) if isinstance(verified, dict) else {}
        if not scope_results:
            lines.append("- 本次运行没有可用的二次核验结果。")
        for summary_id, difference in scope_results.items():
            display_name = row_by_id.get(summary_id, {}).get(
                "display_name", summary_id
            )
            if difference.get("status") not in {
                "completed", "completed_with_warnings"
            }:
                lines.append(
                    f"- {display_name} 相对 {baseline_name}：未完成；"
                    f"{difference.get('error') or difference.get('reason') or 'N/A'}"
                )
                continue
            lines.append(
                f"- {display_name} 相对 {baseline_name}：确认新增 "
                f"{difference.get('confirmed_added_count', 0)} 条，确认减少 "
                f"{difference.get('confirmed_reduced_count', 0)} 条；"
                f"未确认 {difference.get('not_confirmed_count', 0)} 条，"
                f"不确定 {difference.get('uncertain_count', 0)} 条，"
                f"未核验 {difference.get('unverified_count', 0)} 条。"
            )
            confirmed_items = [
                item
                for item in difference.get("items", [])
                if item.get("verification_result")
                in {"confirmed_added", "confirmed_reduced"}
            ]
            for item in confirmed_items:
                kind = (
                    "确认新增"
                    if item.get("verification_result") == "confirmed_added"
                    else "确认减少"
                )
                lines.append(
                    f"  - {kind}：{fmt_content_unit(item)}；"
                    f"基线证据={item.get('baseline_quote') or '无'}；"
                    f"候选证据={item.get('candidate_quote') or '无'}"
                )
        lines.append("")
    lines.append("</details>")
    lines.extend(
        [
            "",
            "<details>",
            "<summary>问题诊断（模型判定，仅供核查）</summary>",
            "",
            "以下问题由模型评测器生成，可能存在误判，不作为必须接受的结论。",
            "",
            "## 相对首份纪要新增问题",
            "",
        ]
    )
    row_by_id = {
        str(item.get("summary_id")): item for item in metrics["rows"]
    }
    for scope, title in scope_titles:
        lines.append(f"### {title}")
        differences = metrics.get("scope_content_differences", {}).get(scope, {})
        found_added_problem = False
        for summary_id, difference in differences.items():
            display_name = row_by_id.get(summary_id, {}).get(
                "display_name", summary_id
            )
            for item in difference.get("added_problem_claims", []):
                found_added_problem = True
                lines.append(
                    f"- {display_name}：[{item.get('status') or 'unknown'}] "
                    f"{item.get('claim') or item.get('summary_fact_id')}"
                )
        if not found_added_problem:
            lines.append("- 无")
        lines.append("")
    lines.extend(["## 全部范围问题 Claim", ""])
    for scope, title in scope_titles:
        lines.append(f"### {title}")
        found_scope_problem = False
        for summary in metrics["rows"]:
            summary_id = str(summary["summary_id"])
            claims = metrics.get("scope_problem_claims", {}).get(scope, {}).get(
                summary_id, []
            )
            for item in claims:
                found_scope_problem = True
                interpretation = item.get("scope_interpretation")
                interpretation_text = (
                    "；当前材料未提供支持，全材料已支持"
                    if interpretation == "supported_by_all_sources_not_current"
                    else ""
                )
                lines.append(
                    f"- {summary['display_name']}："
                    f"{item['summary_fact_id']}（{item['claim']}）："
                    f"{item['status']}；{item['reason']}{interpretation_text}"
                )
        if not found_scope_problem:
            lines.append("- 无")
        lines.append("")
    lines.extend(["## 历史与状态专项问题 Claim", ""])
    for summary_id, claims in metrics["problem_claims"].items():
        lines.append(f"### {summary_id}")
        lines.extend(
            (
                f"- {item['summary_fact_id']}（{item['claim']}）: "
                f"{item['status']}; "
                f"transition={item['transition_assessment']}; {item['reason']}"
            )
            for item in claims
        )
        if not claims:
            lines.append("- 无")
        lines.append("")
    lines.append("</details>")
    return "\n".join(lines).rstrip() + "\n"


class EvaluationComparisonRunner:
    """构建一次 Snapshot，并串行、隔离地评测 2 至 5 份纪要。"""

    def __init__(self, *, model_client: ModelClient) -> None:
        """保存底层模型客户端；预算在每次 run 中初始化。"""
        self._model_client = model_client
        self.last_stage = "not_started"
        self.last_source_model_trace: tuple[dict[str, Any], ...] = ()
        self.last_source_personalization_model_trace: tuple[
            dict[str, Any], ...
        ] = ()
        self.last_source_execution_stats: dict[str, int] = {}

    async def run(
        self,
        request: EvaluationCompareRequest,
        *,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationComparisonResult:
        """在核心层执行整轮超时约束，供 CLI 和 Web 复用。"""
        self.last_stage = "validating"
        self.last_source_model_trace = ()
        self.last_source_personalization_model_trace = ()
        self.last_source_execution_stats = {}
        return await asyncio.wait_for(
            self._run_impl(request, status_callback=status_callback),
            timeout=request.evaluation_config.comparison_timeout_seconds,
        )

    async def _run_impl(
        self,
        request: EvaluationCompareRequest,
        *,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationComparisonResult:
        """执行完整横评，Source Snapshot 失败会使整个运行失败。"""
        def update_stage(stage: str) -> None:
            """同步当前阶段，供失败产物和 Web 状态共同使用。"""
            self.last_stage = stage
            if status_callback is not None:
                status_callback(stage)

        config = comparison_run_config(request)
        shared_client = SharedEvaluationModelClient(
            self._model_client,
            max_calls=request.evaluation_config.comparison_max_total_model_calls,
        )
        snapshot_builder = EvaluationSourceSnapshotBuilder(
            model_client=shared_client,
            config=config,
            max_model_calls=(
                request.evaluation_config.comparison_source_max_model_calls
            ),
            personalization_max_model_calls=(
                request.evaluation_config.
                comparison_personalization_source_max_model_calls
            ),
        )
        update_stage("building_source_snapshot")
        try:
            snapshot_bundle = await snapshot_builder.build_from_documents(
                summary_request=request.summary_request,
                current_source=request.current_source,
                historical_sources=request.historical_sources,
                target_entities=request.target_entities,
                status_callback=update_stage,
            )
        finally:
            self.last_source_model_trace = snapshot_builder.model_trace
            self.last_source_personalization_model_trace = (
                snapshot_builder.personalization_model_trace
            )
            self.last_source_execution_stats = snapshot_builder.execution_stats
        comparison_basis = build_comparison_basis(
            snapshot_id=snapshot_bundle.snapshot.snapshot_id,
            source_facts=snapshot_bundle.snapshot.source_facts,
            fact_relations=snapshot_bundle.snapshot.fact_relations,
        )
        results: list[EvaluationComparisonSummaryResult] = []
        warnings: list[str] = list(snapshot_bundle.snapshot.warnings)
        for summary_index, summary in enumerate(request.summaries, start=1):
            update_stage(f"scoring_summary_{summary_index}")
            per_summary_client = SharedEvaluationModelClient(
                shared_client,
                max_calls=(
                    request.evaluation_config.comparison_per_summary_max_model_calls
                ),
            )
            personalization_client = SharedEvaluationModelClient(
                shared_client,
                max_calls=(
                    request.evaluation_config.
                    comparison_personalization_per_summary_max_model_calls
                ),
            )
            scorer = EvaluationSummaryScorer(
                model_client=per_summary_client,
                personalization_model_client=personalization_client,
                config=config,
                max_model_calls=(
                    request.evaluation_config.comparison_per_summary_max_model_calls
                ),
                personalization_max_model_calls=(
                    request.evaluation_config.
                    comparison_personalization_per_summary_max_model_calls
                ),
            )
            scope_evaluator = EvaluationComparisonScopeEvaluator(
                model_client=per_summary_client,
                config=config,
                max_model_calls=(
                    request.evaluation_config.comparison_per_summary_max_model_calls
                ),
            )
            try:
                summary_started = time.monotonic()
                per_summary_timeout = (
                    request.evaluation_config.comparison_per_summary_timeout_seconds
                )
                artifacts = await asyncio.wait_for(
                    scorer.score(
                        snapshot_bundle=snapshot_bundle,
                        summary=summary.text,
                    ),
                    timeout=per_summary_timeout,
                )
                remaining_timeout = max(
                    0.0,
                    per_summary_timeout - (time.monotonic() - summary_started),
                )
                scope_result = await scope_evaluator.evaluate(
                    basis=comparison_basis,
                    source_facts=snapshot_bundle.snapshot.source_facts,
                    fact_relations=snapshot_bundle.snapshot.fact_relations,
                    summary=summary.text,
                    summary_facts=artifacts.summary_facts,
                    timeout_seconds=remaining_timeout,
                )
                personalization_status = (
                    artifacts.personalization.metrics.status
                    if artifacts.personalization is not None
                    else EvaluationStatus.DISABLED
                )
                personalization_failed = (
                    personalization_status == EvaluationStatus.FAILED
                )
                metric_warnings = tuple(artifacts.metrics.warnings)
                scope_warnings = tuple(scope_result.warnings)
                personalization_warnings = tuple(
                    artifacts.personalization.metrics.warnings
                    if (
                        artifacts.personalization is not None
                        and personalization_status != EvaluationStatus.DISABLED
                    )
                    else ()
                )
                scope_all_failed = (
                    scope_result.all_sources is None
                    and scope_result.current_only is None
                )
                scope_partially_failed = (
                    (scope_result.all_sources is None)
                    != (scope_result.current_only is None)
                )
                status = (
                    "failed"
                    if scope_all_failed
                    else (
                        "completed_with_warnings"
                        if (
                            scope_partially_failed
                            or personalization_failed
                            or metric_warnings
                            or scope_warnings
                            or personalization_warnings
                        )
                        else "completed"
                    )
                )
                if personalization_failed:
                    warnings.append(
                        f"summary {summary.summary_id}: personalization failed"
                    )
                warnings.extend(
                    f"summary {summary.summary_id}: {warning}"
                    for warning in (
                        *metric_warnings,
                        *scope_warnings,
                        *personalization_warnings,
                    )
                )
                results.append(
                    EvaluationComparisonSummaryResult(
                        summary_id=summary.summary_id,
                        display_name=summary.display_name,
                        status=status,
                        artifacts=artifacts,
                        scope_result=scope_result,
                        model_trace=scorer.model_trace,
                        scope_model_trace=scope_evaluator.model_trace,
                        personalization_model_trace=(
                            scorer.personalization_model_trace
                        ),
                    )
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"summary {summary.summary_id} failed: {error}")
                results.append(
                    EvaluationComparisonSummaryResult(
                        summary_id=summary.summary_id,
                        display_name=summary.display_name,
                        status="failed",
                        error=error,
                        model_trace=scorer.model_trace,
                        scope_model_trace=scope_evaluator.model_trace,
                        personalization_model_trace=(
                            scorer.personalization_model_trace
                        ),
                    )
                )
        result_tuple = tuple(results)
        update_stage("aggregating")
        metrics = aggregate_comparison(
            request,
            snapshot_bundle,
            result_tuple,
            comparison_basis=comparison_basis,
            total_model_calls=shared_client.model_call_count,
        )
        interpretation_trace: tuple[dict[str, Any], ...] = ()
        if (
            request.evaluation_config.comparison_interpretation_enabled
            and metrics["successful_summary_count"] > 0
        ):
            update_stage("explaining_comparison")
            interpretation_client = SharedEvaluationModelClient(
                shared_client,
                max_calls=(
                    request.evaluation_config.
                    comparison_interpretation_max_model_calls
                ),
            )
            interpreter = ComparisonInterpretationRunner(
                model_client=interpretation_client,
                config=request.evaluation_config,
            )
            try:
                interpretation = await interpreter.run(
                    request=request,
                    metrics=metrics,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                interpretation = {
                    "schema_version": (
                        "summary_multi_agent."
                        "evaluation_comparison_interpretation.v1"
                    ),
                    "status": "failed",
                    "qualitative_by_scope": {},
                    "verified_content_differences": {},
                    "execution_stats": interpreter.execution_stats,
                    "warnings": [f"comparison interpretation failed: {error}"],
                }
            finally:
                interpretation_trace = interpreter.model_trace
            metrics = {
                **metrics,
                "comparison_interpretation_status": interpretation["status"],
                "qualitative_comparison": interpretation.get(
                    "qualitative_by_scope", {}
                ),
                "verified_content_differences": interpretation.get(
                    "verified_content_differences", {}
                ),
                "comparison_interpretation_execution_stats": interpretation.get(
                    "execution_stats", {}
                ),
                "total_model_call_count": shared_client.model_call_count,
            }
            warnings.extend(interpretation.get("warnings", []))
        else:
            metrics = {
                **metrics,
                "comparison_interpretation_status": "disabled",
                "qualitative_comparison": {},
                "verified_content_differences": {},
                "comparison_interpretation_execution_stats": {},
            }
        success_count = metrics["successful_summary_count"]
        status = (
            "failed"
            if success_count == 0
            else (
                "completed_with_warnings"
                if (
                    success_count < len(request.summaries)
                    or warnings
                    or any(item.status != "completed" for item in result_tuple)
                )
                else "completed"
            )
        )
        return EvaluationComparisonResult(
            status=status,
            snapshot_bundle=snapshot_bundle,
            comparison_basis=comparison_basis,
            summaries=result_tuple,
            comparison_metrics=metrics,
            report_markdown=build_comparison_report(request, metrics),
            source_model_trace=snapshot_builder.model_trace,
            source_personalization_model_trace=(
                snapshot_builder.personalization_model_trace
            ),
            comparison_interpretation_model_trace=interpretation_trace,
            warnings=tuple(dict.fromkeys(warnings)),
        )
