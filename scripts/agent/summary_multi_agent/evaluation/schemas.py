"""事实级评测的严格数据契约。"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ..schemas import (
    EvaluationMode,
    EvaluationStatus,
    FrozenStrictModel,
    LiteEvaluationConfig,
)


# 这些枚举和模型会写入评测产物；普通注释用于说明语义，不改变 JSON Schema。
# 源事实来自当前材料或本轮实际可见的历史上下文。
class EvaluationSourceScope(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


# 摘要 claim 声明的事实来源范围。
class EvaluationClaimScope(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# 评测源事实对当前摘要用途的重要程度。
class EvaluationSalience(str, Enum):
    MUST_INCLUDE = "must_include"
    USEFUL_OPTIONAL = "useful_optional"
    EXCLUDE = "exclude"


# 源事实在当前上下文中的状态。
class EvaluationFactState(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


# 历史事实相对于当前请求和当前事实的有效性分类。
class HistoricalDisposition(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    UPDATED = "updated"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    BACKGROUND = "background"
    IRRELEVANT = "irrelevant"
    UNCERTAIN = "uncertain"


# 历史事实与当前事实之间的可审计语义/状态关系。
class EvaluationFactRelationType(str, Enum):
    HISTORY_ONLY = "history_only"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    UPDATED = "updated"
    REPLACED = "replaced"
    CANCELLED = "cancelled"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


# 摘要 claim 对状态变化的表达完整性。
class EvaluationTransitionAssessment(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    CURRENT_STATE_ONLY = "current_state_only"
    TRANSITION_FULL = "transition_full"
    TRANSITION_PARTIAL = "transition_partial"
    STALE_AS_CURRENT = "stale_as_current"
    TRANSITION_CONTRADICTED = "transition_contradicted"


# 摘要原子事实与源事实的支持关系。
class EvaluationMatchStatus(str, Enum):
    SUPPORTED_FULL = "supported_full"
    SUPPORTED_PARTIAL = "supported_partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


# 目标实体评测只标注事实选择价值，不改变整体事实的 salience。
class PersonalizationSourceRelevance(str, Enum):
    TARGET_MUST_INCLUDE = "target_must_include"
    TARGET_USEFUL_OPTIONAL = "target_useful_optional"
    NOT_TARGET_RELEVANT = "not_target_relevant"
    UNCERTAIN = "uncertain"


# 摘要 claim 是否明确服务于指定目标实体。
class PersonalizationClaimRelevance(str, Enum):
    TARGET_DIRECTED = "target_directed"
    TARGET_CONTEXTUAL = "target_contextual"
    NOT_TARGET_DIRECTED = "not_target_directed"
    UNCERTAIN = "uncertain"


# 目标实体与事实之间允许被审计的关系类型。
class PersonalizationRelationType(str, Enum):
    ACTOR = "actor"
    OWNER_ASSIGNEE = "owner_assignee"
    DECISION_AUTHORITY = "decision_authority"
    AFFECTED_PARTY = "affected_party"
    SPEAKER = "speaker"
    PREFERENCE = "preference"
    COMMITMENT = "commitment"
    RISK_OWNER = "risk_owner"
    DIRECT_MENTION = "direct_mention"
    ROLE_CONTEXT = "role_context"


class PersonalizationTargetStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


# 事实级 Evidence 可带当前 offset、历史 Memory ID 或 speaker/turn 定位。
class EvaluationEvidence(FrozenStrictModel):
    evidence_id: Optional[str] = None
    evidence_text: str = ""
    source_id: Optional[str] = None
    segment_id: Optional[str] = None
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    memory_id: Optional[str] = None
    source_time: Optional[str] = None
    speaker: Optional[str] = None
    turn_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvaluationEvidence":
        """确保 offset 要么成对出现，要么都缺省且不会形成空区间。"""
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("evidence offsets must be both present or both absent")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("evidence end_offset must be greater than start_offset")
        return self


# Evaluation 用于计算分母的当前/历史源事实。
class EvaluationSourceFact(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_source_fact.v1"] = (
        "summary_multi_agent.evaluation_source_fact.v1"
    )
    fact_id: str = Field(min_length=1)
    source_scope: EvaluationSourceScope
    fact: str = Field(min_length=1)
    fact_type: str = Field(default="fact", min_length=1)
    state: EvaluationFactState = EvaluationFactState.UNKNOWN
    salience: EvaluationSalience
    entities: tuple[str, ...] = ()
    evidence: tuple[EvaluationEvidence, ...] = Field(min_length=1)
    source_record_ids: tuple[str, ...] = ()
    related_evidence_ids: tuple[str, ...] = ()
    history_disposition: Optional[HistoricalDisposition] = None
    classification_reason: str = ""

    @model_validator(mode="after")
    def validate_scope(self) -> "EvaluationSourceFact":
        """确保历史事实有 disposition，当前事实不携带历史状态。"""
        if (
            self.source_scope == EvaluationSourceScope.HISTORICAL
            and self.history_disposition is None
        ):
            raise ValueError("historical source fact requires history_disposition")
        if (
            self.source_scope == EvaluationSourceScope.CURRENT
            and self.history_disposition is not None
        ):
            raise ValueError("current source fact cannot have history_disposition")
        return self


# 同一事项在历史与当前材料中的关系，不改变当前事实的真值优先级。
class EvaluationFactRelation(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_fact_relation.v1"] = (
        "summary_multi_agent.evaluation_fact_relation.v1"
    )
    relation_id: str = Field(min_length=1)
    fact_family_id: str = Field(min_length=1)
    historical_fact_id: str = Field(min_length=1)
    current_fact_ids: tuple[str, ...] = ()
    relation_type: EvaluationFactRelationType
    historical_state: EvaluationFactState = EvaluationFactState.UNKNOWN
    current_state: EvaluationFactState = EvaluationFactState.UNKNOWN
    reason: str = ""

    @model_validator(mode="after")
    def validate_relation_shape(self) -> "EvaluationFactRelation":
        """确保 history-only 无当前引用，其余关系至少关联一个当前事实。"""
        unique = tuple(dict.fromkeys(self.current_fact_ids))
        if unique != self.current_fact_ids:
            raise ValueError("relation current fact ids must be unique")
        if self.relation_type == EvaluationFactRelationType.HISTORY_ONLY:
            if self.current_fact_ids:
                raise ValueError("history-only relation cannot reference current facts")
        elif not self.current_fact_ids:
            raise ValueError("non-history-only relation requires current facts")
        return self


# 从正式 candidate 0 抽取的带摘要 offset 原子事实。
class EvaluationSummaryFact(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_summary_fact.v1"] = (
        "summary_multi_agent.evaluation_summary_fact.v1"
    )
    summary_fact_id: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    claimed_scope: EvaluationClaimScope = EvaluationClaimScope.UNKNOWN
    entities: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_span(self) -> "EvaluationSummaryFact":
        """确保摘要原子事实的 offset 构成正向区间。"""
        if self.end_offset <= self.start_offset:
            raise ValueError("summary fact end_offset must be greater than start_offset")
        return self


# 单条摘要事实的匹配状态、主来源和 stale/duplicate 标记。
class EvaluationFactMatch(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_fact_match.v2"] = (
        "summary_multi_agent.evaluation_fact_match.v2"
    )
    summary_fact_id: str = Field(min_length=1)
    match_status: EvaluationMatchStatus
    matched_source_fact_ids: tuple[str, ...] = ()
    primary_source_fact_id: Optional[str] = None
    primary_source_scope: Optional[EvaluationSourceScope] = None
    matched_relation_ids: tuple[str, ...] = ()
    transition_assessment: EvaluationTransitionAssessment = (
        EvaluationTransitionAssessment.NOT_APPLICABLE
    )
    stale_history: bool = False
    duplicate_of_summary_fact_id: Optional[str] = None
    reason_code: str = Field(min_length=1)
    reason: str = ""

    @model_validator(mode="after")
    def validate_match(self) -> "EvaluationFactMatch":
        """确保 unsupported 不引用来源，其他状态有合法主来源。"""
        matched = tuple(dict.fromkeys(self.matched_source_fact_ids))
        if matched != self.matched_source_fact_ids:
            raise ValueError("matched source fact ids must be unique")
        relation_ids = tuple(dict.fromkeys(self.matched_relation_ids))
        if relation_ids != self.matched_relation_ids:
            raise ValueError("matched relation ids must be unique")
        if self.match_status == EvaluationMatchStatus.UNSUPPORTED:
            if (
                matched
                or self.primary_source_fact_id
                or self.primary_source_scope
                or relation_ids
                or self.transition_assessment
                != EvaluationTransitionAssessment.NOT_APPLICABLE
                or self.stale_history
            ):
                raise ValueError(
                    "unsupported match cannot reference source facts, relations, "
                    "or transition state"
                )
        elif not self.primary_source_fact_id or not self.primary_source_scope:
            raise ValueError("non-unsupported match requires a primary source fact")
        if self.primary_source_fact_id and self.primary_source_fact_id not in matched:
            raise ValueError("primary source fact must be included in matched ids")
        return self


# 独立横评和在线 fresh 评测共享的固定源事实快照。
class EvaluationSourceSnapshot(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_source_snapshot.v1"] = (
        "summary_multi_agent.evaluation_source_snapshot.v1"
    )
    snapshot_id: str = Field(min_length=1)
    current_source_id: str = Field(min_length=1)
    current_source_time: str = Field(min_length=1)
    current_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_source_sha256: dict[str, str] = Field(default_factory=dict)
    historical_source_times: dict[str, str] = Field(default_factory=dict)
    summary_request: str = Field(min_length=1)
    source_facts: tuple[EvaluationSourceFact, ...] = ()
    fact_relations: tuple[EvaluationFactRelation, ...] = ()
    target_entities: tuple[str, ...] = ()
    target_source_annotations: tuple["PersonalizationSourceAnnotation", ...] = ()
    config: dict[str, Any] = Field(default_factory=dict)
    execution_stats: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_snapshot_references(self) -> "EvaluationSourceSnapshot":
        """保证事实、关系和目标标注共享同一组唯一且可解析的 ID。"""
        source_ids = [item.fact_id for item in self.source_facts]
        relation_ids = [item.relation_id for item in self.fact_relations]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("snapshot source fact ids must be unique")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("snapshot relation ids must be unique")
        known_sources = set(source_ids)
        scope_by_source = {
            item.fact_id: item.source_scope for item in self.source_facts
        }
        for relation in self.fact_relations:
            if relation.historical_fact_id not in known_sources:
                raise ValueError("relation references an unknown historical fact")
            if (
                scope_by_source[relation.historical_fact_id]
                != EvaluationSourceScope.HISTORICAL
            ):
                raise ValueError("relation historical id must reference history")
            if set(relation.current_fact_ids) - known_sources:
                raise ValueError("relation references unknown current facts")
            if any(
                scope_by_source[item] != EvaluationSourceScope.CURRENT
                for item in relation.current_fact_ids
            ):
                raise ValueError("relation current ids must reference current facts")
        known_targets = set(self.target_entities)
        annotation_keys: set[tuple[str, str]] = set()
        for annotation in self.target_source_annotations:
            if annotation.target_entity not in known_targets:
                raise ValueError("source annotation references an unknown target")
            if annotation.source_fact_id not in known_sources:
                raise ValueError("source annotation references an unknown fact")
            key = (annotation.target_entity, annotation.source_fact_id)
            if key in annotation_keys:
                raise ValueError("source annotations must be unique per target/fact")
            annotation_keys.add(key)
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in self.historical_source_sha256.values()
        ):
            raise ValueError("historical source hashes must be SHA-256 hex")
        return self


class EvaluationSourceDocument(FrozenStrictModel):
    """独立横评的一份原始当前或历史材料。"""

    source_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    )
    label: str = Field(min_length=1)
    source_time: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_document(self) -> "EvaluationSourceDocument":
        """验证日期可解析且原文非空，但不修改原文字节。"""
        if not self.text.strip():
            raise ValueError("source document text must contain non-whitespace text")
        try:
            datetime.fromisoformat(self.source_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("source_time must be an ISO date or datetime") from exc
        return self


class EvaluationSummaryDocument(FrozenStrictModel):
    """独立横评的一份待评测纪要。"""

    summary_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    )
    display_name: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary_document(self) -> "EvaluationSummaryDocument":
        """只验证非空，不 strip 或改写待评测纪要。"""
        if not self.text.strip():
            raise ValueError("summary text must contain non-whitespace text")
        return self


class EvaluationCompareConfig(FrozenStrictModel):
    """多纪要横评的源阶段、单纪要阶段和共享总预算。"""

    comparison_max_summaries: int = Field(default=5, ge=2, le=5)
    evaluation_mode: EvaluationMode = EvaluationMode.STRICT
    lite_evaluation: LiteEvaluationConfig = Field(
        default_factory=LiteEvaluationConfig
    )
    comparison_source_max_model_calls: int = Field(default=48, ge=1)
    comparison_per_summary_max_model_calls: int = Field(default=96, ge=1)
    comparison_personalization_source_max_model_calls: int = Field(
        default=24, ge=1
    )
    comparison_personalization_per_summary_max_model_calls: int = Field(
        default=24, ge=1
    )
    comparison_interpretation_enabled: bool = False
    comparison_interpretation_max_model_calls: int = Field(default=48, ge=1)
    comparison_interpretation_max_tokens: int = Field(
        default=2400, ge=128, le=3000
    )
    comparison_difference_verification_max_units_per_pair_scope: int = Field(
        default=48, ge=1, le=96
    )
    comparison_max_total_model_calls: int = Field(default=320, ge=1)
    comparison_model_timeout_seconds: int = Field(default=180, ge=1)
    comparison_per_summary_timeout_seconds: int = Field(default=1200, ge=1)
    comparison_timeout_seconds: int = Field(default=7200, ge=1)
    comparison_max_concurrent_model_calls: int = Field(default=1, ge=1, le=1)
    segment_max_chars: int = Field(default=500, ge=100)
    source_fact_max_tokens: int = Field(default=3000, ge=128, le=3000)
    summary_fact_max_tokens: int = Field(default=3000, ge=128, le=3000)
    match_max_tokens: int = Field(default=3000, ge=128, le=3000)
    repair_attempts: int = Field(default=2, ge=0, le=5)
    match_batch_size: int = Field(default=6, ge=1, le=50)
    history_batch_size: int = Field(default=12, ge=1, le=50)
    match_full_scan_threshold: int = Field(default=48, ge=1)
    match_candidate_limit_per_fact: int = Field(default=12, ge=1)
    match_candidate_union_limit: int = Field(default=48, ge=1)
    personalization_batch_size: int = Field(default=24, ge=1, le=50)
    personalization_max_tokens: int = Field(default=2400, ge=128, le=3000)
    adaptive_split_enabled: bool = True
    adaptive_split_min_chars: int = Field(default=500, ge=100)
    adaptive_split_max_depth: int = Field(default=1, ge=0, le=3)


class EvaluationCompareRequest(FrozenStrictModel):
    """同一源事实快照下 2 至 5 份纪要的横向评测请求。"""

    schema_version: Literal["summary_multi_agent.evaluation_compare_request.v1"] = (
        "summary_multi_agent.evaluation_compare_request.v1"
    )
    project: str = Field(min_length=1)
    summary_request: str = Field(min_length=1)
    current_source: EvaluationSourceDocument
    historical_sources: tuple[EvaluationSourceDocument, ...] = Field(
        default=(), max_length=5
    )
    summaries: tuple[EvaluationSummaryDocument, ...] = Field(
        min_length=2, max_length=5
    )
    target_entities: tuple[str, ...] = ()
    evaluation_config: EvaluationCompareConfig = Field(
        default_factory=EvaluationCompareConfig
    )

    @model_validator(mode="after")
    def validate_compare_request(self) -> "EvaluationCompareRequest":
        """确保 ID 唯一、目标规范且纪要数量服从显式配置。"""
        source_ids = [
            self.current_source.source_id,
            *(item.source_id for item in self.historical_sources),
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        summary_ids = [item.summary_id for item in self.summaries]
        if len(summary_ids) != len(set(summary_ids)):
            raise ValueError("summary_id values must be unique")
        if len(self.summaries) > self.evaluation_config.comparison_max_summaries:
            raise ValueError("summary count exceeds comparison_max_summaries")
        normalized_targets = tuple(
            dict.fromkeys(item.strip() for item in self.target_entities if item.strip())
        )
        if normalized_targets != self.target_entities:
            raise ValueError("target_entities must be stripped and unique")
        return self


# 保留分子、分母和可为空的确定性指标值。
class MetricValue(FrozenStrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: Optional[float] = Field(default=None, ge=0.0)
    unit: str = "ratio"


# 独立横评把源事实投影为稳定计数单位；不改变原始 Source Snapshot。
class EvaluationComparisonFactUnit(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.evaluation_comparison_fact_unit.v1"
    ] = "summary_multi_agent.evaluation_comparison_fact_unit.v1"
    unit_id: str = Field(min_length=1)
    canonical_fact_id: str = Field(min_length=1)
    source_fact_ids: tuple[str, ...] = Field(min_length=1)
    source_scopes: tuple[EvaluationSourceScope, ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = ()
    source_times: tuple[str, ...] = ()
    fact: str = Field(min_length=1)
    fact_type: str = Field(min_length=1)
    state: EvaluationFactState
    salience: EvaluationSalience
    entities: tuple[str, ...] = ()
    merge_reason: str = "independent_fact"

    @model_validator(mode="after")
    def validate_fact_unit(self) -> "EvaluationComparisonFactUnit":
        """保证计数单位引用唯一事实且 canonical fact 属于该单位。"""
        if tuple(dict.fromkeys(self.source_fact_ids)) != self.source_fact_ids:
            raise ValueError("comparison unit source fact ids must be unique")
        if self.canonical_fact_id not in self.source_fact_ids:
            raise ValueError("comparison unit canonical fact must be referenced")
        if tuple(dict.fromkeys(self.source_scopes)) != self.source_scopes:
            raise ValueError("comparison unit source scopes must be unique")
        return self


class EvaluationComparisonScopeBasis(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.evaluation_comparison_scope_basis.v1"
    ] = "summary_multi_agent.evaluation_comparison_scope_basis.v1"
    scope: Literal["all_sources", "current_only"]
    units: tuple[EvaluationComparisonFactUnit, ...] = ()
    required_unit_ids: tuple[str, ...] = ()
    useful_unit_ids: tuple[str, ...] = ()
    source_fact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_scope_basis(self) -> "EvaluationComparisonScopeBasis":
        """保证分母和有用集合仅引用本范围的唯一计数单位。"""
        unit_ids = [item.unit_id for item in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("comparison scope unit ids must be unique")
        known = set(unit_ids)
        if set(self.required_unit_ids) - known:
            raise ValueError("comparison required units reference unknown ids")
        if set(self.useful_unit_ids) - known:
            raise ValueError("comparison useful units reference unknown ids")
        if set(self.required_unit_ids) - set(self.useful_unit_ids):
            raise ValueError("comparison required units must also be useful")
        return self


class EvaluationComparisonBasis(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.evaluation_comparison_basis.v1"
    ] = "summary_multi_agent.evaluation_comparison_basis.v1"
    snapshot_id: str = Field(min_length=1)
    all_sources: EvaluationComparisonScopeBasis
    current_only: EvaluationComparisonScopeBasis
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class EvaluationComparisonScopeMetrics(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.evaluation_comparison_scope_metrics.v1"
    ] = "summary_multi_agent.evaluation_comparison_scope_metrics.v1"
    scope: Literal["all_sources", "current_only"]
    status: EvaluationStatus
    evaluated_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: dict[str, int] = Field(default_factory=dict)
    core_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    safety_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    execution_stats: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: Optional[str] = None


class EvaluationComparisonScopeResult(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.evaluation_comparison_scope_result.v1"
    ] = "summary_multi_agent.evaluation_comparison_scope_result.v1"
    evaluated_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    all_sources: Optional[EvaluationComparisonScopeMetrics] = None
    current_only: Optional[EvaluationComparisonScopeMetrics] = None
    all_sources_matches: tuple[EvaluationFactMatch, ...] = ()
    current_only_matches: tuple[EvaluationFactMatch, ...] = ()
    scope_statuses: dict[str, EvaluationStatus] = Field(default_factory=dict)
    scope_errors: dict[str, str] = Field(default_factory=dict)
    claim_diagnostics: tuple[dict[str, Any], ...] = ()
    execution_stats: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# 单条源事实对一个显式目标实体的价值标注。
class PersonalizationSourceAnnotation(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.personalization_source_annotation.v1"
    ] = "summary_multi_agent.personalization_source_annotation.v1"
    target_entity: str = Field(min_length=1)
    source_fact_id: str = Field(min_length=1)
    relevance: PersonalizationSourceRelevance
    relation_types: tuple[PersonalizationRelationType, ...] = ()
    reason: str = ""


# 单条正式摘要 claim 对一个显式目标实体的指向性标注。
class PersonalizationClaimAnnotation(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.personalization_claim_annotation.v1"
    ] = "summary_multi_agent.personalization_claim_annotation.v1"
    target_entity: str = Field(min_length=1)
    summary_fact_id: str = Field(min_length=1)
    relevance: PersonalizationClaimRelevance
    relation_types: tuple[PersonalizationRelationType, ...] = ()
    reason: str = ""


# 一个目标实体的确定性个性化指标；模型只提供上面的分类标签。
class TargetPersonalizationMetrics(FrozenStrictModel):
    target_entity: str = Field(min_length=1)
    status: PersonalizationTargetStatus
    counts: dict[str, int] = Field(default_factory=dict)
    core_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    safety_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    descriptive_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    legacy_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# 显式 target_entities 触发的独立评测分支结果。
class PersonalizationEvaluationMetrics(FrozenStrictModel):
    schema_version: Literal[
        "summary_multi_agent.personalization_evaluation_metrics.v2"
    ] = "summary_multi_agent.personalization_evaluation_metrics.v2"
    status: EvaluationStatus
    trigger: Literal["explicit_target_entities"] = "explicit_target_entities"
    evaluated_candidate_index: Literal[0] = 0
    evaluated_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_entities: tuple[str, ...] = ()
    per_target: tuple[TargetPersonalizationMetrics, ...] = ()
    macro_metrics: dict[str, Optional[float]] = Field(default_factory=dict)
    micro_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    execution_stats: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    legacy_metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# 个性化分支的标注、指标、报告和独立模型 trace。
class PersonalizationEvaluationArtifacts(FrozenStrictModel):
    source_annotations: tuple[PersonalizationSourceAnnotation, ...] = ()
    claim_annotations: tuple[PersonalizationClaimAnnotation, ...] = ()
    metrics: PersonalizationEvaluationMetrics
    report_markdown: str = Field(min_length=1)
    model_trace: tuple[dict[str, Any], ...] = ()


# 一次 Evaluation Sidecar 的指标、计数、配置和执行统计。
class EvaluationMetrics(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_metrics.v2"] = (
        "summary_multi_agent.evaluation_metrics.v2"
    )
    status: EvaluationStatus
    evaluated_candidate_index: Literal[0] = 0
    evaluated_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: dict[str, int] = Field(default_factory=dict)
    core_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    history_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    transition_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    safety_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    conciseness_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    lengths: dict[str, int] = Field(default_factory=dict)
    execution_stats: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    legacy_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# Evaluation 各中间事实、匹配、指标、报告和模型 trace 的聚合结果。
class EvaluationArtifacts(FrozenStrictModel):
    source_facts: tuple[EvaluationSourceFact, ...] = ()
    summary_facts: tuple[EvaluationSummaryFact, ...] = ()
    matches: tuple[EvaluationFactMatch, ...] = ()
    metrics: EvaluationMetrics
    report_markdown: str = Field(min_length=1)
    model_trace: tuple[dict[str, Any], ...] = ()
    personalization: Optional[PersonalizationEvaluationArtifacts] = None
