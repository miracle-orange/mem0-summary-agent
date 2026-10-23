"""基于既有事实与匹配结果的目标实体个性化评测分支。"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Optional

from pydantic import Field

from ..adapters.model import ModelClient
from ..schemas import EvaluationStatus, FrozenStrictModel, RunConfig, TaskType
from .metrics import VALID_HISTORY_DISPOSITIONS, persisted_summary_sha256
from .model_io import EvaluationModelIO
from .prompts import (
    build_personalization_claim_prompt,
    build_personalization_source_prompt,
)
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationMatchStatus,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
    MetricValue,
    PersonalizationClaimAnnotation,
    PersonalizationClaimRelevance,
    PersonalizationEvaluationArtifacts,
    PersonalizationEvaluationMetrics,
    PersonalizationRelationType,
    PersonalizationSourceAnnotation,
    PersonalizationSourceRelevance,
    PersonalizationTargetStatus,
    TargetPersonalizationMetrics,
)


class _SourceAnnotationDraft(FrozenStrictModel):
    """模型返回的单条目标实体源事实分类。"""

    source_fact_id: str = Field(min_length=1)
    relevance: PersonalizationSourceRelevance
    relation_types: tuple[PersonalizationRelationType, ...] = ()
    reason: str = ""


class _SourceAnnotationResponse(FrozenStrictModel):
    """一个目标实体在一个源事实批次上的完整响应。"""

    target_entity: str = Field(min_length=1)
    annotations: tuple[_SourceAnnotationDraft, ...] = ()


class _ClaimAnnotationDraft(FrozenStrictModel):
    """模型返回的单条目标实体摘要 claim 分类。"""

    summary_fact_id: str = Field(min_length=1)
    relevance: PersonalizationClaimRelevance
    relation_types: tuple[PersonalizationRelationType, ...] = ()
    reason: str = ""


class _ClaimAnnotationResponse(FrozenStrictModel):
    """一个目标实体在一个摘要事实批次上的完整响应。"""

    target_entity: str = Field(min_length=1)
    annotations: tuple[_ClaimAnnotationDraft, ...] = ()


def _ratio(numerator: int, denominator: int, *, unit: str = "ratio") -> MetricValue:
    """生成可表达空分母的确定性指标值。"""

    return MetricValue(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
        unit=unit,
    )


def _harmonic(left: MetricValue, right: MetricValue) -> MetricValue:
    """计算两个比例的调和平均，任一分母为空时返回 N/A。"""

    if left.value is None or right.value is None:
        return MetricValue(numerator=0, denominator=0, value=None, unit="harmonic_mean")
    denominator = left.value + right.value
    return MetricValue(
        numerator=left.numerator + right.numerator,
        denominator=left.denominator + right.denominator,
        value=(2 * left.value * right.value / denominator if denominator else 0.0),
        unit="harmonic_mean",
    )


def _valid_source_fact(item: EvaluationSourceFact) -> bool:
    """判断目标事实是否可作为本轮有效当前/历史事实。"""

    return (
        item.source_scope == EvaluationSourceScope.CURRENT
        or item.history_disposition in VALID_HISTORY_DISPOSITIONS
    )


def _target_claim_ids(
    annotations: Iterable[PersonalizationClaimAnnotation],
) -> set[str]:
    """返回直接或上下文面向目标实体的摘要事实 ID。"""

    return {
        item.summary_fact_id
        for item in annotations
        if item.relevance
        in {
            PersonalizationClaimRelevance.TARGET_DIRECTED,
            PersonalizationClaimRelevance.TARGET_CONTEXTUAL,
        }
    }


def calculate_personalization_metrics(
    *,
    summary: str,
    target_entities: tuple[str, ...],
    source_facts: tuple[EvaluationSourceFact, ...],
    summary_facts: tuple[EvaluationSummaryFact, ...],
    matches: tuple[EvaluationFactMatch, ...],
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
    source_annotations: tuple[PersonalizationSourceAnnotation, ...],
    claim_annotations: tuple[PersonalizationClaimAnnotation, ...],
    execution_stats: Optional[dict[str, int]] = None,
    config: Optional[dict[str, Any]] = None,
    diagnostics: Optional[dict[str, Any]] = None,
    warnings: tuple[str, ...] = (),
) -> PersonalizationEvaluationMetrics:
    """根据目标标注和整体 Matcher 结果确定性计算个性化指标。"""

    source_by_id = {item.fact_id: item for item in source_facts}
    match_by_id = {item.summary_fact_id: item for item in matches}
    relation_by_id = {item.relation_id: item for item in fact_relations}
    relations_by_source: dict[str, set[str]] = defaultdict(set)
    for relation in fact_relations:
        relations_by_source[relation.historical_fact_id].add(relation.relation_id)
        for current_id in relation.current_fact_ids:
            relations_by_source[current_id].add(relation.relation_id)
    source_by_target: dict[str, list[PersonalizationSourceAnnotation]] = defaultdict(list)
    claim_by_target: dict[str, list[PersonalizationClaimAnnotation]] = defaultdict(list)
    for item in source_annotations:
        source_by_target[item.target_entity].append(item)
    for item in claim_annotations:
        claim_by_target[item.target_entity].append(item)

    per_target: list[TargetPersonalizationMetrics] = []
    all_warnings = list(warnings)
    for target in target_entities:
        target_sources = source_by_target.get(target, [])
        target_claim_annotations = claim_by_target.get(target, [])
        must_ids = {
            item.source_fact_id
            for item in target_sources
            if item.relevance == PersonalizationSourceRelevance.TARGET_MUST_INCLUDE
            and item.source_fact_id in source_by_id
            and _valid_source_fact(source_by_id[item.source_fact_id])
        }
        useful_ids = must_ids | {
            item.source_fact_id
            for item in target_sources
            if item.relevance
            == PersonalizationSourceRelevance.TARGET_USEFUL_OPTIONAL
            and item.source_fact_id in source_by_id
            and _valid_source_fact(source_by_id[item.source_fact_id])
        }
        current_useful_ids = {
            item_id
            for item_id in useful_ids
            if source_by_id[item_id].source_scope == EvaluationSourceScope.CURRENT
        }
        historical_useful_ids = useful_ids - current_useful_ids
        directed_claim_ids = _target_claim_ids(target_claim_annotations)
        relation_ids_for_target = {
            relation_id
            for source_id in useful_ids
            for relation_id in relations_by_source.get(source_id, set())
        }
        matched_target_claim_ids: set[str] = set()
        for item in matches:
            direct = bool(useful_ids & set(item.matched_source_fact_ids))
            relation_ids = set(item.matched_relation_ids)
            for source_id in item.matched_source_fact_ids:
                relation_ids.update(relations_by_source.get(source_id, set()))
            if direct or bool(relation_ids_for_target & relation_ids):
                matched_target_claim_ids.add(item.summary_fact_id)
        content_claim_ids = directed_claim_ids | matched_target_claim_ids
        content_matches = [
            match_by_id[item_id]
            for item_id in content_claim_ids
            if item_id in match_by_id
        ]
        directed_matches = [
            match_by_id[item_id]
            for item_id in directed_claim_ids
            if item_id in match_by_id
        ]
        content_strict_matches = [
            item
            for item in content_matches
            if item.match_status == EvaluationMatchStatus.SUPPORTED_FULL
        ]
        directed_strict_matches = [
            item
            for item in directed_matches
            if item.match_status == EvaluationMatchStatus.SUPPORTED_FULL
        ]

        def covered_ids(items: Iterable[EvaluationFactMatch]) -> set[str]:
            """把严格匹配及其事实关系投影为目标源事实覆盖。"""
            covered: set[str] = set()
            for match in items:
                if match.duplicate_of_summary_fact_id is not None:
                    continue
                covered.update(useful_ids & set(match.matched_source_fact_ids))
                relation_ids = set(match.matched_relation_ids)
                for source_id in match.matched_source_fact_ids:
                    relation_ids.update(relations_by_source.get(source_id, set()))
                for relation_id in relation_ids & relation_ids_for_target:
                    relation = relation_by_id.get(relation_id)
                    if relation is None:
                        continue
                    covered.update(
                        useful_ids
                        & {
                            relation.historical_fact_id,
                            *relation.current_fact_ids,
                        }
                    )
            return covered

        content_covered_source_ids = covered_ids(content_strict_matches)
        directed_covered_source_ids = covered_ids(directed_strict_matches)
        legacy_covered_source_ids = {
            source_id
            for item in directed_strict_matches
            if item.duplicate_of_summary_fact_id is None
            for source_id in item.matched_source_fact_ids
        }
        content_useful_claim_count = sum(
            (
                item.duplicate_of_summary_fact_id is None
                and bool(
                    (useful_ids & set(item.matched_source_fact_ids))
                    or (
                        relation_ids_for_target
                        & (
                            set(item.matched_relation_ids)
                            | {
                                relation_id
                                for source_id in item.matched_source_fact_ids
                                for relation_id in relations_by_source.get(
                                    source_id, set()
                                )
                            }
                        )
                    )
                )
            )
            for item in content_strict_matches
        )
        directed_useful_claim_count = sum(
            item.summary_fact_id in content_claim_ids
            and item.duplicate_of_summary_fact_id is None
            and item.match_status == EvaluationMatchStatus.SUPPORTED_FULL
            and item.summary_fact_id in matched_target_claim_ids
            for item in directed_matches
        )
        unsupported_count = sum(
            item.match_status == EvaluationMatchStatus.UNSUPPORTED
            for item in directed_matches
        )
        contradiction_count = sum(
            item.match_status == EvaluationMatchStatus.CONTRADICTED
            for item in directed_matches
        )
        partial_count = sum(
            item.match_status == EvaluationMatchStatus.SUPPORTED_PARTIAL
            for item in directed_matches
        )
        stale_count = sum(item.stale_history for item in directed_matches)
        duplicate_count = sum(
            item.duplicate_of_summary_fact_id is not None
            for item in directed_matches
        )
        content_must_recall = _ratio(
            len(must_ids & content_covered_source_ids), len(must_ids)
        )
        content_useful_recall = _ratio(
            len(useful_ids & content_covered_source_ids), len(useful_ids)
        )
        content_useful_precision = _ratio(
            content_useful_claim_count, len(content_claim_ids)
        )
        directed_must_recall = _ratio(
            len(must_ids & directed_covered_source_ids), len(must_ids)
        )
        directed_useful_recall = _ratio(
            len(useful_ids & directed_covered_source_ids), len(useful_ids)
        )
        directed_useful_precision = _ratio(
            directed_useful_claim_count, len(directed_claim_ids)
        )
        legacy_must_recall = _ratio(
            len(must_ids & legacy_covered_source_ids), len(must_ids)
        )
        legacy_useful_precision = _ratio(
            sum(
                item.duplicate_of_summary_fact_id is None
                and bool(useful_ids & set(item.matched_source_fact_ids))
                for item in directed_strict_matches
            ),
            len(directed_claim_ids),
        )
        target_warnings: list[str] = []
        status = PersonalizationTargetStatus.RESOLVED
        if not useful_ids:
            status = PersonalizationTargetStatus.UNRESOLVED
            target_warnings.append(
                f"target entity {target!r} has no grounded relevant source facts"
            )
            all_warnings.extend(target_warnings)
        per_target.append(
            TargetPersonalizationMetrics(
                target_entity=target,
                status=status,
                counts={
                    "target_source_annotation_count": len(target_sources),
                    "target_claim_annotation_count": len(target_claim_annotations),
                    "target_must_fact_count": len(must_ids),
                    "target_useful_fact_count": len(useful_ids),
                    "target_current_useful_fact_count": len(current_useful_ids),
                    "target_historical_useful_fact_count": len(historical_useful_ids),
                    "target_content_claim_count": len(content_claim_ids),
                    "target_directed_claim_count": len(directed_claim_ids),
                    "target_supported_full_count": len(content_strict_matches),
                    "target_useful_claim_count": content_useful_claim_count,
                    "gate_excluded_covered_fact_count": len(
                        (useful_ids & content_covered_source_ids)
                        - directed_covered_source_ids
                    ),
                    "implicit_target_content_claim_count": len(
                        content_claim_ids - directed_claim_ids
                    ),
                    "unsupported_target_directed_claim_count": unsupported_count,
                    "target_unsupported_count": unsupported_count,
                    "target_contradicted_count": contradiction_count,
                    "target_partial_count": partial_count,
                    "target_stale_history_count": stale_count,
                    "target_duplicate_count": duplicate_count,
                },
                core_metrics={
                    "target_content_must_recall": content_must_recall,
                    "target_content_useful_recall": content_useful_recall,
                    "target_content_useful_precision": content_useful_precision,
                    "target_content_f1": _harmonic(
                        content_must_recall, content_useful_precision
                    ),
                    "target_directed_must_recall": directed_must_recall,
                    "target_directed_useful_recall": directed_useful_recall,
                    "target_directed_useful_precision": directed_useful_precision,
                    "target_directed_f1": _harmonic(
                        directed_must_recall, directed_useful_precision
                    ),
                    "target_support_precision": _ratio(
                        len(content_strict_matches), len(content_claim_ids)
                    ),
                    "target_current_recall": _ratio(
                        len(current_useful_ids & content_covered_source_ids),
                        len(current_useful_ids),
                    ),
                    "target_historical_recall": _ratio(
                        len(historical_useful_ids & content_covered_source_ids),
                        len(historical_useful_ids),
                    ),
                    # P7.2 兼容别名；正式旧口径同时保存在 legacy_metrics。
                    "target_must_fact_recall": legacy_must_recall,
                    "target_useful_recall": _ratio(
                        len(useful_ids & legacy_covered_source_ids),
                        len(useful_ids),
                    ),
                    "target_useful_precision": legacy_useful_precision,
                    "target_fact_f1": _harmonic(
                        legacy_must_recall, legacy_useful_precision
                    ),
                },
                safety_metrics={
                    "target_unsupported_attribution_rate": _ratio(
                        unsupported_count, len(directed_claim_ids)
                    ),
                    "target_contradiction_rate": _ratio(
                        contradiction_count, len(directed_claim_ids)
                    ),
                    "target_partial_rate": _ratio(
                        partial_count, len(directed_claim_ids)
                    ),
                    "target_stale_history_rate": _ratio(
                        stale_count, len(directed_claim_ids)
                    ),
                },
                descriptive_metrics={
                    "target_redundancy_rate": _ratio(
                        duplicate_count, len(directed_claim_ids)
                    ),
                    "target_focus_share": _ratio(
                        len(directed_claim_ids), len(summary_facts)
                    ),
                },
                legacy_metrics={
                    "legacy_target_must_fact_recall": legacy_must_recall,
                    "legacy_target_useful_precision": legacy_useful_precision,
                    "legacy_target_fact_f1": _harmonic(
                        legacy_must_recall, legacy_useful_precision
                    ),
                },
                warnings=tuple(target_warnings),
            )
        )

    metric_groups: dict[str, list[MetricValue]] = defaultdict(list)
    for target_metrics in per_target:
        for section in (
            target_metrics.core_metrics,
            target_metrics.safety_metrics,
            target_metrics.descriptive_metrics,
        ):
            for name, metric in section.items():
                metric_groups[name].append(metric)
    macro_metrics = {
        name: (
            fmean(values)
            if (
                values := [item.value for item in metrics if item.value is not None]
            )
            else None
        )
        for name, metrics in metric_groups.items()
    }
    micro_metrics: dict[str, MetricValue] = {}
    for name, metrics in metric_groups.items():
        if name in {
            "target_content_f1",
            "target_directed_f1",
            "target_fact_f1",
        }:
            continue
        micro_metrics[name] = _ratio(
            sum(item.numerator for item in metrics),
            sum(item.denominator for item in metrics),
            unit=metrics[0].unit if metrics else "ratio",
        )
    for prefix in ("target_content", "target_directed"):
        recall_name = f"{prefix}_must_recall"
        precision_name = f"{prefix}_useful_precision"
        if recall_name in micro_metrics and precision_name in micro_metrics:
            micro_metrics[f"{prefix}_f1"] = _harmonic(
                micro_metrics[recall_name], micro_metrics[precision_name]
            )
    if "target_must_fact_recall" in micro_metrics and "target_useful_precision" in micro_metrics:
        micro_metrics["target_fact_f1"] = _harmonic(
            micro_metrics["target_must_fact_recall"],
            micro_metrics["target_useful_precision"],
        )
    legacy_macro_groups: dict[str, list[float]] = defaultdict(list)
    legacy_micro_groups: dict[str, list[MetricValue]] = defaultdict(list)
    for target_metrics in per_target:
        for name, metric in target_metrics.legacy_metrics.items():
            legacy_micro_groups[name].append(metric)
            if metric.value is not None:
                legacy_macro_groups[name].append(metric.value)
    legacy_micro = {
        name: _ratio(
            sum(item.numerator for item in values),
            sum(item.denominator for item in values),
            unit=values[0].unit if values else "ratio",
        )
        for name, values in legacy_micro_groups.items()
        if name != "legacy_target_fact_f1"
    }
    if (
        "legacy_target_must_fact_recall" in legacy_micro
        and "legacy_target_useful_precision" in legacy_micro
    ):
        legacy_micro["legacy_target_fact_f1"] = _harmonic(
            legacy_micro["legacy_target_must_fact_recall"],
            legacy_micro["legacy_target_useful_precision"],
        )
    return PersonalizationEvaluationMetrics(
        status=EvaluationStatus.COMPLETED,
        evaluated_summary_sha256=persisted_summary_sha256(summary),
        target_entities=target_entities,
        per_target=tuple(per_target),
        macro_metrics=macro_metrics,
        micro_metrics=micro_metrics,
        execution_stats=dict(execution_stats or {}),
        config=dict(config or {}),
        diagnostics=dict(diagnostics or {}),
        legacy_metrics={
            "macro": {
                name: (fmean(values) if values else None)
                for name, values in legacy_macro_groups.items()
            },
            "micro": {
                name: value.model_dump(mode="json")
                for name, value in legacy_micro.items()
            },
        },
        warnings=tuple(dict.fromkeys(all_warnings)),
    )


def _validate_exact_ids(
    *,
    target_entity: str,
    returned_target: str,
    returned_ids: list[str],
    expected_ids: set[str],
    id_kind: str,
) -> None:
    """强制模型对当前批次每个已知 ID 恰好分类一次。"""

    if returned_target != target_entity:
        raise ValueError(
            f"personalization classifier changed target entity: {returned_target!r}"
        )
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError(f"personalization classifier returned duplicate {id_kind} ids")
    unknown = set(returned_ids) - expected_ids
    missing = expected_ids - set(returned_ids)
    if unknown:
        raise ValueError(
            f"personalization classifier returned unknown {id_kind} ids: {sorted(unknown)}"
        )
    if missing:
        raise ValueError(
            f"personalization classifier omitted {id_kind} ids: {sorted(missing)}"
        )


def _worker_diagnostics(worker_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """汇总 Entity Worker 是否被计划和执行，不参与指标分数。"""

    entity_results = [
        item
        for item in worker_results.values()
        if isinstance(item, dict)
        and item.get("task_type") == TaskType.ENTITY_PERSONALIZATION.value
    ]
    return {
        "entity_worker_result_count": len(entity_results),
        "entity_worker_success_count": sum(
            item.get("status") == "success" for item in entity_results
        ),
        "entity_worker_failed_count": sum(
            item.get("status") in {"failed", "timeout"} for item in entity_results
        ),
        "entity_worker_evidence_count": sum(
            len(item.get("evidence_ids", [])) for item in entity_results
        ),
    }


class TargetPersonalizationEvaluator:
    """在独立预算内标注目标相关性并计算确定性个性化指标。"""

    def __init__(self, *, model_client: ModelClient, config: RunConfig) -> None:
        """创建与整体 Evaluation 隔离的模型 I/O 和调用预算。"""

        self._config = config
        self._model_io = EvaluationModelIO(
            model_client,
            max_model_calls=config.evaluation_personalization_max_model_calls,
        )
        self._source_batch_count = 0
        self._claim_batch_count = 0

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回个性化分类调用的独立审计 trace。"""

        return tuple(self._model_io.traces)

    @property
    def execution_stats(self) -> dict[str, int]:
        """返回个性化分支的调用、repair 和批次数。"""

        return {
            "model_call_count": self._model_io.model_call_count,
            "repair_call_count": self._model_io.repair_call_count,
            "source_annotation_batch_count": self._source_batch_count,
            "claim_annotation_batch_count": self._claim_batch_count,
        }

    async def evaluate(
        self,
        *,
        summary: str,
        target_entities: tuple[str, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
        summary_facts: tuple[EvaluationSummaryFact, ...],
        matches: tuple[EvaluationFactMatch, ...],
        worker_results: dict[str, dict[str, Any]],
        fact_relations: tuple[EvaluationFactRelation, ...] = (),
        source_annotations: Optional[
            tuple[PersonalizationSourceAnnotation, ...]
        ] = None,
    ) -> PersonalizationEvaluationArtifacts:
        """按目标和批次标注事实，再复用整体匹配结果计算指标。"""
        resolved_source_annotations = source_annotations
        if resolved_source_annotations is None:
            resolved_source_annotations = await self.annotate_sources(
                target_entities=target_entities,
                source_facts=source_facts,
            )
        claim_annotations = await self.annotate_claims(
            target_entities=target_entities,
            summary_facts=summary_facts,
            matches=matches,
            source_facts=source_facts,
            source_annotations=resolved_source_annotations,
            fact_relations=fact_relations,
        )
        metrics = calculate_personalization_metrics(
            summary=summary,
            target_entities=target_entities,
            source_facts=source_facts,
            summary_facts=summary_facts,
            matches=matches,
            fact_relations=fact_relations,
            source_annotations=resolved_source_annotations,
            claim_annotations=claim_annotations,
            execution_stats=self.execution_stats,
            config={
                "trigger": "explicit_target_entities",
                "batch_size": self._config.evaluation_personalization_batch_size,
                "max_tokens": self._config.evaluation_personalization_max_tokens,
                "max_model_calls": (
                    self._config.evaluation_personalization_max_model_calls
                ),
                "timeout_seconds": (
                    self._config.evaluation_personalization_timeout_seconds
                ),
                "model_timeout_seconds": self._config.evaluation_model_timeout_seconds,
            },
            diagnostics=_worker_diagnostics(worker_results),
        )
        return PersonalizationEvaluationArtifacts(
            source_annotations=resolved_source_annotations,
            claim_annotations=claim_annotations,
            metrics=metrics,
            report_markdown=build_personalization_report(metrics),
            model_trace=self.model_trace,
        )

    async def annotate_sources(
        self,
        *,
        target_entities: tuple[str, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
    ) -> tuple[PersonalizationSourceAnnotation, ...]:
        """为固定 Snapshot 只标注一次目标源事实集合。"""
        source_annotations: list[PersonalizationSourceAnnotation] = []
        batch_size = self._config.evaluation_personalization_batch_size
        for target_index, target in enumerate(target_entities):
            for start in range(0, len(source_facts), batch_size):
                batch = source_facts[start : start + batch_size]
                expected = {item.fact_id for item in batch}

                def validate_source(response: _SourceAnnotationResponse) -> None:
                    _validate_exact_ids(
                        target_entity=target,
                        returned_target=response.target_entity,
                        returned_ids=[item.source_fact_id for item in response.annotations],
                        expected_ids=expected,
                        id_kind="source fact",
                    )

                response = await self._model_io.complete_json(
                    stage=f"evaluation.personalization.source.t{target_index + 1}.b{start // batch_size + 1}",
                    prompt=build_personalization_source_prompt(target, batch),
                    response_model=_SourceAnnotationResponse,
                    max_tokens=self._config.evaluation_personalization_max_tokens,
                    repair_attempts=self._config.evaluation_repair_attempts,
                    timeout_seconds=self._config.evaluation_model_timeout_seconds,
                    validator=validate_source,
                    trace_metadata={
                        "target_entity": target,
                        "source_fact_count": len(batch),
                    },
                )
                self._source_batch_count += 1
                source_annotations.extend(
                    PersonalizationSourceAnnotation(
                        target_entity=target,
                        source_fact_id=item.source_fact_id,
                        relevance=item.relevance,
                        relation_types=item.relation_types,
                        reason=item.reason,
                    )
                    for item in response.annotations
                )
        return tuple(source_annotations)

    async def annotate_claims(
        self,
        *,
        target_entities: tuple[str, ...],
        summary_facts: tuple[EvaluationSummaryFact, ...],
        matches: tuple[EvaluationFactMatch, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
        source_annotations: tuple[PersonalizationSourceAnnotation, ...],
        fact_relations: tuple[EvaluationFactRelation, ...] = (),
    ) -> tuple[PersonalizationClaimAnnotation, ...]:
        """只标注一份纪要的目标指向性，不重新判断事实真伪。"""
        claim_annotations: list[PersonalizationClaimAnnotation] = []
        batch_size = self._config.evaluation_personalization_batch_size
        match_by_id = {item.summary_fact_id: item for item in matches}
        source_by_id = {item.fact_id: item for item in source_facts}
        annotation_by_key = {
            (item.target_entity, item.source_fact_id): item
            for item in source_annotations
        }
        relation_by_id = {item.relation_id: item for item in fact_relations}
        for target_index, target in enumerate(target_entities):
            for start in range(0, len(summary_facts), batch_size):
                batch = summary_facts[start : start + batch_size]
                expected = {item.summary_fact_id for item in batch}
                batch_matches = tuple(
                    match_by_id[item.summary_fact_id]
                    for item in batch
                    if item.summary_fact_id in match_by_id
                )
                batch_source_ids = {
                    source_id
                    for item in batch_matches
                    for source_id in item.matched_source_fact_ids
                }
                batch_relation_ids = {
                    relation_id
                    for item in batch_matches
                    for relation_id in item.matched_relation_ids
                }
                batch_sources = tuple(
                    source_by_id[source_id]
                    for source_id in sorted(batch_source_ids)
                    if source_id in source_by_id
                )
                batch_source_annotations = tuple(
                    annotation_by_key[(target, item.fact_id)]
                    for item in batch_sources
                    if (target, item.fact_id) in annotation_by_key
                )
                batch_relations = tuple(
                    relation_by_id[relation_id]
                    for relation_id in sorted(batch_relation_ids)
                    if relation_id in relation_by_id
                )

                def validate_claim(response: _ClaimAnnotationResponse) -> None:
                    _validate_exact_ids(
                        target_entity=target,
                        returned_target=response.target_entity,
                        returned_ids=[item.summary_fact_id for item in response.annotations],
                        expected_ids=expected,
                        id_kind="summary fact",
                    )

                response = await self._model_io.complete_json(
                    stage=f"evaluation.personalization.claim.t{target_index + 1}.b{start // batch_size + 1}",
                    prompt=build_personalization_claim_prompt(
                        target,
                        batch,
                        matches=batch_matches,
                        source_facts=batch_sources,
                        source_annotations=batch_source_annotations,
                        fact_relations=batch_relations,
                    ),
                    response_model=_ClaimAnnotationResponse,
                    max_tokens=self._config.evaluation_personalization_max_tokens,
                    repair_attempts=self._config.evaluation_repair_attempts,
                    timeout_seconds=self._config.evaluation_model_timeout_seconds,
                    validator=validate_claim,
                    trace_metadata={
                        "target_entity": target,
                        "summary_fact_count": len(batch),
                    },
                )
                self._claim_batch_count += 1
                claim_annotations.extend(
                    PersonalizationClaimAnnotation(
                        target_entity=target,
                        summary_fact_id=item.summary_fact_id,
                        relevance=item.relevance,
                        relation_types=item.relation_types,
                        reason=item.reason,
                    )
                    for item in response.annotations
                )
        return tuple(claim_annotations)


def disabled_personalization_artifacts(summary: str) -> PersonalizationEvaluationArtifacts:
    """构造未显式提供目标实体时的零调用 disabled 产物。"""

    metrics = PersonalizationEvaluationMetrics(
        status=EvaluationStatus.DISABLED,
        evaluated_summary_sha256=persisted_summary_sha256(summary),
        warnings=("personalization evaluation disabled: no explicit target_entities",),
    )
    return PersonalizationEvaluationArtifacts(
        metrics=metrics,
        report_markdown=(
            "# 目标实体个性化评测\n\n未显式提供 `target_entities`，本分支未启动。\n"
        ),
    )


def failed_personalization_artifacts(
    summary: str,
    target_entities: tuple[str, ...],
    error: str,
    *,
    execution_stats: Optional[dict[str, int]] = None,
    model_trace: tuple[dict[str, Any], ...] = (),
) -> PersonalizationEvaluationArtifacts:
    """构造失败隔离产物，不影响整体评测和正式摘要。"""

    metrics = PersonalizationEvaluationMetrics(
        status=EvaluationStatus.FAILED,
        evaluated_summary_sha256=persisted_summary_sha256(summary),
        target_entities=target_entities,
        execution_stats=dict(execution_stats or {}),
        warnings=(error,),
    )
    return PersonalizationEvaluationArtifacts(
        metrics=metrics,
        report_markdown=(
            "# 目标实体个性化评测\n\n个性化评测失败，但整体评测和正式摘要保持不变。\n\n"
            f"错误：{error}\n"
        ),
        model_trace=model_trace,
    )


def _metric_text(metric: MetricValue) -> str:
    """将个性化 MetricValue 渲染为报告文本。"""

    if metric.value is None:
        return f"N/A（{metric.numerator}/{metric.denominator}）"
    return f"{metric.value:.4f}（{metric.numerator}/{metric.denominator}）"


def build_personalization_report(metrics: PersonalizationEvaluationMetrics) -> str:
    """生成整体指标之外的目标实体个性化评测中文报告。"""

    lines = [
        "# 目标实体个性化评测",
        "",
        "本报告与整体事实级指标并列展示，只评测正式 candidate 0。",
    ]
    for item in metrics.per_target:
        lines.extend(["", f"## {item.target_entity}", "", f"- 状态：{item.status.value}"])
        for name, metric in item.core_metrics.items():
            lines.append(f"- {name}: {_metric_text(metric)}")
        lines.extend(["", "### 安全与聚焦", ""])
        for name, metric in item.safety_metrics.items():
            lines.append(f"- {name}: {_metric_text(metric)}")
        for name, metric in item.descriptive_metrics.items():
            lines.append(f"- {name}: {_metric_text(metric)}")
        lines.extend(f"- 警告：{warning}" for warning in item.warnings)
    if metrics.macro_metrics:
        lines.extend(["", "## 多目标宏平均", ""])
        for name, value in metrics.macro_metrics.items():
            lines.append(f"- {name}: {'N/A' if value is None else f'{value:.4f}'}")
    return "\n".join(lines).rstrip() + "\n"
