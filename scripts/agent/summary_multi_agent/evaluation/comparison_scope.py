"""独立横评的全材料/当前材料双范围事实基准与指标。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from typing import Any, Optional

from ..adapters.model import ModelClient
from ..schemas import EvaluationStatus, RunConfig
from .matcher import EvaluationFactMatcher
from .metrics import VALID_HISTORY_DISPOSITIONS, persisted_summary_sha256
from .model_io import EvaluationModelIO
from .schemas import (
    EvaluationComparisonBasis,
    EvaluationComparisonFactUnit,
    EvaluationComparisonScopeBasis,
    EvaluationComparisonScopeMetrics,
    EvaluationComparisonScopeResult,
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationFactRelationType,
    EvaluationFactState,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
    EvaluationTransitionAssessment,
    MetricValue,
)


def _stable_id(prefix: str, *parts: object) -> str:
    """按稳定 JSON 内容生成横评计数单位 ID。"""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize(value: str) -> str:
    """移除不承载语义的符号，供保守的确定性重复归并使用。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _ratio(numerator: int, denominator: int, *, unit: str = "ratio") -> MetricValue:
    """保留空分母的 N/A 语义。"""
    return MetricValue(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
        unit=unit,
    )


def _harmonic(left: MetricValue, right: MetricValue) -> MetricValue:
    """计算两项比例的调和平均。"""
    if left.value is None or right.value is None:
        return MetricValue(numerator=0, denominator=0, value=None, unit="harmonic_mean")
    denominator = left.value + right.value
    return MetricValue(
        numerator=left.numerator + right.numerator,
        denominator=left.denominator + right.denominator,
        value=(2 * left.value * right.value / denominator if denominator else 0.0),
        unit="harmonic_mean",
    )


def _evidence_source_ids(fact: EvaluationSourceFact) -> tuple[str, ...]:
    """提取原文 source ID，Memory 记录则回退到 source_record_ids。"""
    values = [item.source_id for item in fact.evidence if item.source_id]
    values.extend(fact.source_record_ids)
    return tuple(dict.fromkeys(str(item) for item in values if item))


def _evidence_source_times(fact: EvaluationSourceFact) -> tuple[str, ...]:
    """提取事实 Evidence 的时间锚点。"""
    return tuple(
        dict.fromkeys(
            str(item.source_time) for item in fact.evidence if item.source_time
        )
    )


def _historical_fact_is_useful(
    fact: EvaluationSourceFact,
    relation: Optional[EvaluationFactRelation],
) -> bool:
    """复用现有历史有效性口径，避免改变历史专项指标的含义。"""
    if fact.history_disposition in VALID_HISTORY_DISPOSITIONS:
        return True
    return bool(
        relation
        and relation.relation_type
        in {
            EvaluationFactRelationType.CONFIRMED,
            EvaluationFactRelationType.COMPLETED,
            EvaluationFactRelationType.UPDATED,
            EvaluationFactRelationType.REPLACED,
            EvaluationFactRelationType.CANCELLED,
        }
    )


class _DisjointSet:
    """为确定性事实单位归并维护小规模并查集。"""

    def __init__(self, ids: tuple[str, ...]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        retained, merged = sorted((left_root, right_root))
        self.parent[merged] = retained


def build_comparison_basis(
    *,
    snapshot_id: str,
    source_facts: tuple[EvaluationSourceFact, ...],
    fact_relations: tuple[EvaluationFactRelation, ...],
) -> EvaluationComparisonBasis:
    """从固定 Snapshot 构建 current-only 与 all-sources 计数单位。"""
    source_by_id = {item.fact_id: item for item in source_facts}
    relation_by_history = {
        item.historical_fact_id: item for item in fact_relations
    }
    current_facts = tuple(
        item for item in source_facts
        if item.source_scope == EvaluationSourceScope.CURRENT
    )

    def make_unit(
        members: tuple[EvaluationSourceFact, ...],
        *,
        scope: str,
        merge_reason: str,
    ) -> EvaluationComparisonFactUnit:
        """把一组等价 Source Facts 投影为一个可审计计数单位。"""
        ordered = tuple(
            sorted(
                members,
                key=lambda item: (
                    item.source_scope != EvaluationSourceScope.CURRENT,
                    item.fact_id,
                ),
            )
        )
        canonical = ordered[0]
        salience = (
            EvaluationSalience.MUST_INCLUDE
            if any(item.salience == EvaluationSalience.MUST_INCLUDE for item in ordered)
            else (
                EvaluationSalience.USEFUL_OPTIONAL
                if any(
                    item.salience == EvaluationSalience.USEFUL_OPTIONAL
                    for item in ordered
                )
                else EvaluationSalience.EXCLUDE
            )
        )
        semantic_key = (
            scope,
            _normalize(canonical.fact),
            canonical.fact_type.casefold(),
            canonical.state.value,
            tuple(sorted(entity.casefold() for entity in canonical.entities)),
            tuple(sorted(_evidence_source_times(canonical))),
            tuple(sorted(item.fact_id for item in ordered if item.source_scope == EvaluationSourceScope.CURRENT)),
        )
        return EvaluationComparisonFactUnit(
            unit_id=_stable_id("comparison_unit", *semantic_key),
            canonical_fact_id=canonical.fact_id,
            source_fact_ids=tuple(item.fact_id for item in ordered),
            source_scopes=tuple(
                dict.fromkeys(item.source_scope for item in ordered)
            ),
            source_ids=tuple(
                dict.fromkeys(
                    source_id
                    for item in ordered
                    for source_id in _evidence_source_ids(item)
                )
            ),
            source_times=tuple(
                dict.fromkeys(
                    source_time
                    for item in ordered
                    for source_time in _evidence_source_times(item)
                )
            ),
            fact=canonical.fact,
            fact_type=canonical.fact_type,
            state=canonical.state,
            salience=salience,
            entities=tuple(
                dict.fromkeys(entity for item in ordered for entity in item.entities)
            ),
            merge_reason=merge_reason,
        )

    current_units = tuple(
        make_unit((item,), scope="current_only", merge_reason="current_fact")
        for item in current_facts
    )

    ids = tuple(item.fact_id for item in source_facts)
    groups = _DisjointSet(ids)
    confirmed_history_ids: set[str] = set()
    for relation in fact_relations:
        if relation.relation_type != EvaluationFactRelationType.CONFIRMED:
            continue
        confirmed_history_ids.add(relation.historical_fact_id)
        for current_id in relation.current_fact_ids:
            if relation.historical_fact_id in source_by_id and current_id in source_by_id:
                groups.union(relation.historical_fact_id, current_id)

    # 仅合并文本、状态、实体和 Evidence 时间都相同的历史重复项。
    exact_history: dict[tuple[object, ...], str] = {}
    for fact in source_facts:
        if fact.source_scope != EvaluationSourceScope.HISTORICAL:
            continue
        key = (
            _normalize(fact.fact),
            fact.fact_type.casefold(),
            fact.state.value,
            tuple(sorted(entity.casefold() for entity in fact.entities)),
            tuple(sorted(_evidence_source_times(fact))),
        )
        previous = exact_history.get(key)
        if previous is None:
            exact_history[key] = fact.fact_id
        else:
            groups.union(previous, fact.fact_id)

    grouped: dict[str, list[EvaluationSourceFact]] = defaultdict(list)
    for fact in source_facts:
        grouped[groups.find(fact.fact_id)].append(fact)
    all_units: list[EvaluationComparisonFactUnit] = []
    for members in grouped.values():
        member_tuple = tuple(members)
        member_ids = {item.fact_id for item in member_tuple}
        if member_ids & confirmed_history_ids and any(
            item.source_scope == EvaluationSourceScope.CURRENT for item in member_tuple
        ):
            reason = "confirmed_current_history_family"
        elif len(member_tuple) > 1:
            reason = "exact_historical_duplicate"
        else:
            reason = "independent_fact"
        all_units.append(
            make_unit(member_tuple, scope="all_sources", merge_reason=reason)
        )
    all_units_tuple = tuple(sorted(all_units, key=lambda item: item.unit_id))

    def scope_basis(
        scope: str,
        units: tuple[EvaluationComparisonFactUnit, ...],
    ) -> EvaluationComparisonScopeBasis:
        """建立一个范围的必要/有用集合和事实白名单。"""
        useful: list[str] = []
        required: list[str] = []
        allowed_source_ids: list[str] = []
        for unit in units:
            unit_facts = [source_by_id[item] for item in unit.source_fact_ids]
            allowed_source_ids.extend(unit.source_fact_ids)
            useful_member = any(
                item.salience
                in {EvaluationSalience.MUST_INCLUDE, EvaluationSalience.USEFUL_OPTIONAL}
                and (
                    item.source_scope == EvaluationSourceScope.CURRENT
                    or _historical_fact_is_useful(
                        item, relation_by_history.get(item.fact_id)
                    )
                )
                for item in unit_facts
            )
            required_member = any(
                item.salience == EvaluationSalience.MUST_INCLUDE
                and (
                    item.source_scope == EvaluationSourceScope.CURRENT
                    or _historical_fact_is_useful(
                        item, relation_by_history.get(item.fact_id)
                    )
                )
                for item in unit_facts
            )
            if useful_member:
                useful.append(unit.unit_id)
            if required_member:
                required.append(unit.unit_id)
        return EvaluationComparisonScopeBasis(
            scope=scope,
            units=units,
            required_unit_ids=tuple(required),
            useful_unit_ids=tuple(useful),
            source_fact_ids=tuple(dict.fromkeys(allowed_source_ids)),
        )

    current_basis = scope_basis("current_only", current_units)
    all_basis = scope_basis("all_sources", all_units_tuple)
    return EvaluationComparisonBasis(
        snapshot_id=snapshot_id,
        all_sources=all_basis,
        current_only=current_basis,
        diagnostics={
            "source_fact_count": len(source_facts),
            "current_source_fact_count": len(current_facts),
            "all_sources_unit_count": len(all_basis.units),
            "current_only_unit_count": len(current_basis.units),
            "merged_source_fact_count": len(source_facts) - len(all_basis.units),
            "confirmed_merge_count": sum(
                item.merge_reason == "confirmed_current_history_family"
                for item in all_basis.units
            ),
            "exact_history_duplicate_merge_count": sum(
                item.merge_reason == "exact_historical_duplicate"
                for item in all_basis.units
            ),
        },
    )


def calculate_comparison_scope_metrics(
    *,
    scope_basis: EvaluationComparisonScopeBasis,
    summary: str,
    summary_facts: tuple[EvaluationSummaryFact, ...],
    matches: tuple[EvaluationFactMatch, ...],
    execution_stats: Optional[dict[str, int]] = None,
    warnings: tuple[str, ...] = (),
) -> EvaluationComparisonScopeMetrics:
    """根据固定事实单位和范围匹配确定性计算横评整体指标。"""
    source_to_unit = {
        source_id: unit.unit_id
        for unit in scope_basis.units
        for source_id in unit.source_fact_ids
    }
    strict_matches = tuple(
        item
        for item in matches
        if item.match_status == EvaluationMatchStatus.SUPPORTED_FULL
        and not item.stale_history
        and item.transition_assessment
        not in {
            EvaluationTransitionAssessment.STALE_AS_CURRENT,
            EvaluationTransitionAssessment.TRANSITION_CONTRADICTED,
        }
    )
    covered_unit_ids = {
        source_to_unit[source_id]
        for item in strict_matches
        if item.duplicate_of_summary_fact_id is None
        for source_id in item.matched_source_fact_ids
        if source_id in source_to_unit
    }
    required = set(scope_basis.required_unit_ids)
    useful = set(scope_basis.useful_unit_ids)
    useful_claim_count = sum(
        item.duplicate_of_summary_fact_id is None
        and bool(
            useful
            & {
                source_to_unit[source_id]
                for source_id in item.matched_source_fact_ids
                if source_id in source_to_unit
            }
        )
        for item in strict_matches
    )
    claim_count = len(summary_facts)
    unsupported_count = sum(
        item.match_status == EvaluationMatchStatus.UNSUPPORTED for item in matches
    )
    contradiction_count = sum(
        item.match_status == EvaluationMatchStatus.CONTRADICTED for item in matches
    )
    partial_count = sum(
        item.match_status == EvaluationMatchStatus.SUPPORTED_PARTIAL for item in matches
    )
    duplicate_count = sum(
        item.duplicate_of_summary_fact_id is not None for item in matches
    )
    must_recall = _ratio(len(required & covered_unit_ids), len(required))
    useful_precision = _ratio(useful_claim_count, claim_count)
    return EvaluationComparisonScopeMetrics(
        scope=scope_basis.scope,
        status=EvaluationStatus.COMPLETED,
        evaluated_summary_sha256=persisted_summary_sha256(summary),
        counts={
            "source_fact_count": len(scope_basis.source_fact_ids),
            "fact_unit_count": len(scope_basis.units),
            "required_unit_count": len(required),
            "useful_unit_count": len(useful),
            "covered_required_unit_count": len(required & covered_unit_ids),
            "covered_useful_unit_count": len(useful & covered_unit_ids),
            "summary_fact_count": claim_count,
            "supported_full_count": len(strict_matches),
            "supported_partial_count": partial_count,
            "contradicted_count": contradiction_count,
            "unsupported_count": unsupported_count,
            "duplicate_count": duplicate_count,
        },
        core_metrics={
            "must_fact_recall": must_recall,
            "useful_fact_precision": useful_precision,
            "fact_support_precision": _ratio(len(strict_matches), claim_count),
            "minutes_fact_f1": _harmonic(must_recall, useful_precision),
        },
        safety_metrics={
            "unsupported_rate": _ratio(unsupported_count, claim_count),
            "contradiction_rate": _ratio(contradiction_count, claim_count),
            "partial_rate": _ratio(partial_count, claim_count),
            "redundancy_rate": _ratio(duplicate_count, claim_count),
        },
        execution_stats=dict(execution_stats or {}),
        warnings=warnings,
    )


class EvaluationComparisonScopeEvaluator:
    """复用已抽取 Claim，为横评执行全材料与当前材料独立匹配。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        max_model_calls: int,
    ) -> None:
        self._config = config
        self._model_io = EvaluationModelIO(
            model_client,
            max_model_calls=max_model_calls,
        )

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回双范围 Matcher trace。"""
        return tuple(self._model_io.traces)

    async def evaluate(
        self,
        *,
        basis: EvaluationComparisonBasis,
        source_facts: tuple[EvaluationSourceFact, ...],
        fact_relations: tuple[EvaluationFactRelation, ...],
        summary: str,
        summary_facts: tuple[EvaluationSummaryFact, ...],
        timeout_seconds: Optional[float] = None,
    ) -> EvaluationComparisonScopeResult:
        """逐范围匹配并隔离失败；无历史时复用 current-only 结果。"""
        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        source_by_id = {item.fact_id: item for item in source_facts}
        scope_metrics: dict[str, EvaluationComparisonScopeMetrics] = {}
        scope_matches: dict[str, tuple[EvaluationFactMatch, ...]] = {}
        warnings: list[str] = []
        errors: dict[str, str] = {}
        stats: dict[str, int] = {}

        async def run_scope(
            scope_basis: EvaluationComparisonScopeBasis,
            *,
            relations: tuple[EvaluationFactRelation, ...],
        ) -> None:
            started = time.monotonic()
            calls_before = self._model_io.model_call_count
            repairs_before = self._model_io.repair_call_count
            matcher = EvaluationFactMatcher(
                model_io=self._model_io,
                config=self._config,
            )
            selected = tuple(
                source_by_id[item]
                for item in scope_basis.source_fact_ids
                if item in source_by_id
            )
            try:
                matched = await matcher.match_against_scope(
                    summary_facts,
                    selected,
                    stage_scope=scope_basis.scope,
                    fact_relations=relations,
                )
                scope_stats = {
                    **matcher.execution_stats,
                    "model_call_count": (
                        self._model_io.model_call_count - calls_before
                    ),
                    "repair_call_count": (
                        self._model_io.repair_call_count - repairs_before
                    ),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
                metrics = calculate_comparison_scope_metrics(
                    scope_basis=scope_basis,
                    summary=summary,
                    summary_facts=summary_facts,
                    matches=matched,
                    execution_stats=scope_stats,
                    warnings=tuple(matcher.warnings),
                )
                scope_matches[scope_basis.scope] = matched
                scope_metrics[scope_basis.scope] = metrics
                for key, value in scope_stats.items():
                    stats[f"{scope_basis.scope}_{key}"] = value
                warnings.extend(
                    f"{scope_basis.scope}: {item}" for item in matcher.warnings
                )
            except Exception as exc:
                errors[scope_basis.scope] = f"{type(exc).__name__}: {exc}"
                failed_stats = {
                    **matcher.execution_stats,
                    "model_call_count": (
                        self._model_io.model_call_count - calls_before
                    ),
                    "repair_call_count": (
                        self._model_io.repair_call_count - repairs_before
                    ),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
                for key, value in failed_stats.items():
                    stats[f"{scope_basis.scope}_{key}"] = value
                warnings.append(
                    f"{scope_basis.scope} evaluation failed: {errors[scope_basis.scope]}"
                )

        async def run_bounded_scope(
            scope_basis: EvaluationComparisonScopeBasis,
            *,
            relations: tuple[EvaluationFactRelation, ...],
        ) -> None:
            """让两个范围共享剩余的单纪要超时，并保留先完成的范围。"""
            bounded_started = time.monotonic()
            calls_before = self._model_io.model_call_count
            repairs_before = self._model_io.repair_call_count
            if deadline is None:
                await run_scope(scope_basis, relations=relations)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors[scope_basis.scope] = (
                    "TimeoutError: comparison per-summary timeout exhausted"
                )
                stats[f"{scope_basis.scope}_model_call_count"] = 0
                stats[f"{scope_basis.scope}_repair_call_count"] = 0
                stats[f"{scope_basis.scope}_duration_ms"] = 0
                warnings.append(
                    f"{scope_basis.scope} evaluation was not executed: "
                    "per-summary timeout exhausted"
                )
                return
            try:
                await asyncio.wait_for(
                    run_scope(scope_basis, relations=relations),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                errors[scope_basis.scope] = (
                    "TimeoutError: comparison per-summary timeout exhausted"
                )
                stats.setdefault(
                    f"{scope_basis.scope}_model_call_count",
                    self._model_io.model_call_count - calls_before,
                )
                stats.setdefault(
                    f"{scope_basis.scope}_repair_call_count",
                    self._model_io.repair_call_count - repairs_before,
                )
                stats.setdefault(
                    f"{scope_basis.scope}_duration_ms",
                    int((time.monotonic() - bounded_started) * 1000),
                )
                warnings.append(
                    f"{scope_basis.scope} evaluation failed: "
                    "per-summary timeout exhausted"
                )

        has_history = any(
            item.source_scope == EvaluationSourceScope.HISTORICAL
            for item in source_facts
        )
        if has_history:
            await run_bounded_scope(basis.all_sources, relations=fact_relations)
            await run_bounded_scope(basis.current_only, relations=())
        else:
            await run_bounded_scope(basis.current_only, relations=())
            if "current_only" in scope_metrics:
                current_metrics = scope_metrics["current_only"]
                current_matches = scope_matches["current_only"]
                scope_metrics["all_sources"] = current_metrics.model_copy(
                    update={"scope": "all_sources"}
                )
                scope_matches["all_sources"] = current_matches
                stats["all_sources_reused_current_only"] = 1
                for key, value in current_metrics.execution_stats.items():
                    stats[f"all_sources_{key}"] = value
            else:
                errors["all_sources"] = (
                    "NotExecuted: current_only failed and no historical source exists"
                )
                warnings.append(
                    "all_sources evaluation was not executed because the shared "
                    "current-only match failed"
                )

        diagnostics: list[dict[str, Any]] = []
        all_by_id = {
            item.summary_fact_id: item
            for item in scope_matches.get("all_sources", ())
        }
        current_by_id = {
            item.summary_fact_id: item
            for item in scope_matches.get("current_only", ())
        }
        summary_by_id = {item.summary_fact_id: item for item in summary_facts}
        for fact_id in summary_by_id:
            all_match = all_by_id.get(fact_id)
            current_match = current_by_id.get(fact_id)
            interpretation = "same_scope_outcome"
            if (
                all_match is not None
                and current_match is not None
                and all_match.match_status == EvaluationMatchStatus.SUPPORTED_FULL
                and current_match.match_status == EvaluationMatchStatus.UNSUPPORTED
            ):
                interpretation = "supported_by_all_sources_not_current"
            elif all_match is None or current_match is None:
                interpretation = "scope_result_unavailable"
            diagnostics.append(
                {
                    "summary_fact_id": fact_id,
                    "all_sources_status": (
                        all_match.match_status.value if all_match else None
                    ),
                    "current_only_status": (
                        current_match.match_status.value if current_match else None
                    ),
                    "interpretation": interpretation,
                }
            )

        stats.update(
            {
                "model_call_count": self._model_io.model_call_count,
                "repair_call_count": self._model_io.repair_call_count,
            }
        )
        return EvaluationComparisonScopeResult(
            evaluated_summary_sha256=persisted_summary_sha256(summary),
            all_sources=scope_metrics.get("all_sources"),
            current_only=scope_metrics.get("current_only"),
            all_sources_matches=scope_matches.get("all_sources", ()),
            current_only_matches=scope_matches.get("current_only", ()),
            scope_statuses={
                scope: (
                    EvaluationStatus.COMPLETED
                    if scope in scope_metrics
                    else (
                        EvaluationStatus.FAILED
                        if scope in errors
                        else EvaluationStatus.PENDING
                    )
                )
                for scope in ("all_sources", "current_only")
            },
            scope_errors=errors,
            claim_diagnostics=tuple(diagnostics),
            execution_stats=stats,
            warnings=tuple(dict.fromkeys(warnings)),
        )
