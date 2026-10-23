"""当前事实优先的分批语义匹配与确定性重复检测。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from pydantic import Field

from ..schemas import FrozenStrictModel, RunConfig
from .model_io import EvaluationModelIO
from .prompts import build_comparison_scope_match_prompt, build_match_prompt
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationMatchStatus,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
    EvaluationTransitionAssessment,
)


# Matcher draft 模型通过显式字段校验，使用普通注释避免改变评测协议描述。
# 模型返回的一条摘要事实匹配草案。
class MatchDraft(FrozenStrictModel):
    summary_fact_id: str = Field(min_length=1)
    status: EvaluationMatchStatus
    matched_source_fact_ids: tuple[str, ...] = ()
    primary_source_fact_id: Optional[str] = None
    matched_relation_ids: tuple[str, ...] = ()
    transition_assessment: EvaluationTransitionAssessment = (
        EvaluationTransitionAssessment.NOT_APPLICABLE
    )
    stale_history: bool = False
    reason_code: str = Field(min_length=1)
    reason: str = ""


# 模型返回的一批摘要事实匹配草案。
class MatchDraftList(FrozenStrictModel):
    matches: tuple[MatchDraft, ...] = ()


_STATUS_RANK = {
    EvaluationMatchStatus.UNSUPPORTED: 0,
    EvaluationMatchStatus.SUPPORTED_PARTIAL: 1,
    EvaluationMatchStatus.CONTRADICTED: 2,
    EvaluationMatchStatus.SUPPORTED_FULL: 3,
}


def _normalize(value: str) -> str:
    """统一事实文本的大小写和非语义符号，供候选排序与重复检测使用。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _entities(value: EvaluationSummaryFact | EvaluationSourceFact) -> set[str]:
    """提取规范化实体集合，用于确定性候选筛选。"""
    return {_normalize(item) for item in value.entities if _normalize(item)}


def _number_tokens(value: str) -> set[str]:
    """提取数字/日期片段，避免候选筛选忽略关键数值差异。"""
    return set(re.findall(r"\d+(?:[./:-]\d+)*", value))


def _char_bigrams(value: str) -> set[str]:
    """生成字符 bigram 集合，作为大规模候选池的稳定相似度信号。"""
    normalized = _normalize(value)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _candidate_score(
    summary_fact: EvaluationSummaryFact,
    source_fact: EvaluationSourceFact,
) -> tuple[float, ...]:
    """计算摘要事实与源事实的确定性候选排序特征。"""
    summary_text = _normalize(summary_fact.fact)
    source_text = _normalize(source_fact.fact)
    summary_entities = _entities(summary_fact)
    source_entities = _entities(source_fact)
    summary_numbers = _number_tokens(summary_fact.fact)
    source_numbers = _number_tokens(source_fact.fact)
    summary_bigrams = _char_bigrams(summary_fact.fact)
    source_bigrams = _char_bigrams(source_fact.fact)
    union = summary_bigrams | source_bigrams
    bigram_overlap = (
        len(summary_bigrams & source_bigrams) / len(union) if union else 0.0
    )
    text_similarity = SequenceMatcher(None, summary_text, source_text).ratio()
    return (
        float(bool(summary_text and (summary_text in source_text or source_text in summary_text))),
        float(len(summary_entities & source_entities)),
        float(len(summary_numbers & source_numbers)),
        bigram_overlap,
        text_similarity,
    )


class EvaluationFactMatcher:
    """执行当前事实优先、历史 fallback 的分批语义匹配和重复检测。"""

    def __init__(
        self,
        *,
        model_io: EvaluationModelIO,
        config: RunConfig,
    ) -> None:
        """绑定评测模型 I/O、候选阈值配置和本轮执行统计。"""
        self._model_io = model_io
        self._config = config
        self.warnings: list[str] = []
        self.execution_stats: dict[str, int] = {
            "matcher_call_count": 0,
            "matcher_full_scan_batch_count": 0,
            "matcher_candidate_batch_count": 0,
            "matcher_omitted_summary_fact_count": 0,
        }

    async def match(
        self,
        summary_facts: tuple[EvaluationSummaryFact, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
        *,
        temporal_support: dict[str, object],
        fact_relations: tuple[EvaluationFactRelation, ...] = (),
    ) -> tuple[EvaluationFactMatch, ...]:
        """先匹配当前事实，再仅为 current unsupported claim 匹配历史事实。"""
        if not summary_facts:
            return ()
        self.warnings.clear()
        for key in self.execution_stats:
            self.execution_stats[key] = 0
        current = tuple(
            item
            for item in source_facts
            if item.source_scope == EvaluationSourceScope.CURRENT
        )
        historical = tuple(
            item
            for item in source_facts
            if item.source_scope == EvaluationSourceScope.HISTORICAL
        )
        # 当前材料是本轮第一事实来源；只有它无法支持的 claim 才允许使用历史匹配。
        current_matches = await self._match_scope(
            summary_facts,
            current,
            scope=EvaluationSourceScope.CURRENT,
            temporal_support=temporal_support,
            fact_relations=fact_relations,
        )
        unresolved = tuple(
            fact
            for fact in summary_facts
            if current_matches[fact.summary_fact_id].match_status
            == EvaluationMatchStatus.UNSUPPORTED
        )
        historical_matches = await self._match_scope(
            unresolved,
            historical,
            scope=EvaluationSourceScope.HISTORICAL,
            temporal_support=temporal_support,
            fact_relations=fact_relations,
        )
        combined = [
            (
                historical_matches[fact.summary_fact_id]
                if current_matches[fact.summary_fact_id].match_status
                == EvaluationMatchStatus.UNSUPPORTED
                and historical
                and fact.summary_fact_id in historical_matches
                else current_matches[fact.summary_fact_id]
            )
            for fact in summary_facts
        ]
        return self._mark_duplicates(summary_facts, combined)

    async def match_against_scope(
        self,
        summary_facts: tuple[EvaluationSummaryFact, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
        *,
        stage_scope: str,
        temporal_support: Optional[dict[str, object]] = None,
        fact_relations: tuple[EvaluationFactRelation, ...] = (),
    ) -> tuple[EvaluationFactMatch, ...]:
        """在一个显式横评范围内直接匹配，不应用 current-first fallback。"""
        if not summary_facts:
            return ()
        self.warnings.clear()
        for key in self.execution_stats:
            self.execution_stats[key] = 0
        matches = await self._match_scope(
            summary_facts,
            source_facts,
            scope=None,
            stage_scope=stage_scope,
            temporal_support=dict(temporal_support or {}),
            fact_relations=fact_relations,
        )
        ordered = [matches[item.summary_fact_id] for item in summary_facts]
        return self._mark_duplicates(summary_facts, ordered)

    async def _match_scope(
        self,
        summary_facts: tuple[EvaluationSummaryFact, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
        *,
        scope: Optional[EvaluationSourceScope],
        stage_scope: Optional[str] = None,
        temporal_support: dict[str, object],
        fact_relations: tuple[EvaluationFactRelation, ...],
    ) -> dict[str, EvaluationFactMatch]:
        """在一个 source scope 内按批次调用模型并保守补齐遗漏 claim。"""
        scope_label = stage_scope or (scope.value if scope is not None else "scope")
        if not summary_facts:
            return {}
        if not source_facts:
            return {
                item.summary_fact_id: self._unsupported(item.summary_fact_id)
                for item in summary_facts
            }
        source_by_id = {item.fact_id: item for item in source_facts}
        candidates: dict[str, list[EvaluationFactMatch]] = {
            item.summary_fact_id: [] for item in summary_facts
        }
        size = self._config.evaluation_match_batch_size
        for summary_start in range(0, len(summary_facts), size):
            summary_batch = summary_facts[summary_start : summary_start + size]
            source_batch, strategy, filtered_count = self._candidate_union(
                summary_batch, source_facts
            )
            source_ids_for_relations = {item.fact_id for item in source_batch}
            relation_batch = tuple(
                item
                for item in fact_relations
                if (
                    item.historical_fact_id in source_ids_for_relations
                    or bool(set(item.current_fact_ids) & source_ids_for_relations)
                )
            )
            if strategy == "full_scan":
                self.execution_stats["matcher_full_scan_batch_count"] += 1
            else:
                self.execution_stats["matcher_candidate_batch_count"] += 1
            allowed_source_ids = {item.fact_id for item in source_batch}
            allowed_relation_ids = {item.relation_id for item in relation_batch}
            expected = {item.summary_fact_id for item in summary_batch}

            def validate_result(result: MatchDraftList) -> None:
                """验证模型只能返回本批摘要和候选源事实 ID。"""
                returned = [item.summary_fact_id for item in result.matches]
                if not returned:
                    raise ValueError(
                        "fact matcher omitted all summary fact ids in the batch"
                    )
                if len(returned) != len(set(returned)):
                    raise ValueError("fact matcher returned duplicate summary fact ids")
                unknown_summary = set(returned) - expected
                if unknown_summary:
                    raise ValueError(
                        "fact matcher referenced unknown summary ids: "
                        f"{sorted(unknown_summary)}"
                    )
                for draft in result.matches:
                    unknown_relations = (
                        set(draft.matched_relation_ids) - allowed_relation_ids
                    )
                    if unknown_relations:
                        raise ValueError(
                            "fact matcher referenced unknown relation ids: "
                            f"{sorted(unknown_relations)}"
                        )
                    unknown_source = (
                        set(draft.matched_source_fact_ids) - allowed_source_ids
                    )
                    if draft.primary_source_fact_id and (
                        draft.primary_source_fact_id not in allowed_source_ids
                    ):
                        unknown_source.add(draft.primary_source_fact_id)
                    if unknown_source:
                        raise ValueError(
                            "fact matcher referenced unknown source ids: "
                            f"{sorted(unknown_source)}"
                        )
                    if draft.status == EvaluationMatchStatus.UNSUPPORTED:
                        if draft.matched_source_fact_ids or draft.primary_source_fact_id:
                            raise ValueError(
                                "unsupported matcher output cannot reference a source id"
                            )
                    elif not draft.primary_source_fact_id:
                        raise ValueError(
                            "non-unsupported matcher output requires a valid primary id"
                        )

            prompt_builder = (
                build_comparison_scope_match_prompt
                if stage_scope is not None
                else build_match_prompt
            )
            result = await self._model_io.complete_json(
                stage=(
                    f"evaluation.match.{scope_label}."
                    f"s{summary_start // size + 1}"
                ),
                prompt=prompt_builder(
                    summary_batch,
                    source_batch,
                    source_scope=scope_label,
                    temporal_support=temporal_support,
                    fact_relations=relation_batch,
                ),
                response_model=MatchDraftList,
                max_tokens=self._config.evaluation_match_max_tokens,
                repair_attempts=self._config.evaluation_repair_attempts,
                timeout_seconds=self._config.evaluation_model_timeout_seconds,
                validator=validate_result,
                trace_metadata={
                    "source_scope": scope_label,
                    "summary_fact_count": len(summary_batch),
                    "source_fact_count": len(source_facts),
                    "candidate_source_fact_count": len(source_batch),
                    "candidate_source_fact_filtered_count": filtered_count,
                    "candidate_strategy": strategy,
                    "candidate_limit_per_fact": (
                        self._config.evaluation_match_candidate_limit_per_fact
                    ),
                    "candidate_union_limit": (
                        self._config.evaluation_match_candidate_union_limit
                    ),
                    "candidate_relation_count": len(relation_batch),
                },
            )
            self.execution_stats["matcher_call_count"] += 1
            returned = {item.summary_fact_id for item in result.matches}
            missing = expected - returned
            if missing:
                self.execution_stats["matcher_omitted_summary_fact_count"] += len(
                    missing
                )
                self.warnings.append(
                    f"{scope_label} matcher omitted {len(missing)} summary fact ids; "
                    "conservatively marked unsupported"
                )
                for fact_id in sorted(missing):
                    candidates[fact_id].append(
                        self._unsupported(
                            fact_id,
                            reason_code="matcher_omitted_summary_fact",
                            reason="模型未返回该摘要事实，按保守策略标记为 unsupported。",
                        )
                    )
            for draft in result.matches:
                if draft.status == EvaluationMatchStatus.UNSUPPORTED:
                    match = self._unsupported(draft.summary_fact_id)
                else:
                    primary = draft.primary_source_fact_id
                    matched_ids = tuple(
                        dict.fromkeys((*draft.matched_source_fact_ids, primary))
                    )
                    stale = draft.stale_history
                    if (
                        draft.transition_assessment
                        == EvaluationTransitionAssessment.STALE_AS_CURRENT
                    ):
                        stale = True
                    primary_scope = source_by_id[primary].source_scope
                    if primary_scope == EvaluationSourceScope.HISTORICAL:
                        disposition = source_by_id[primary].history_disposition
                        if disposition is not None and disposition.value in {
                            "superseded",
                            "contradicted",
                        }:
                            stale = True
                    match = EvaluationFactMatch(
                        summary_fact_id=draft.summary_fact_id,
                        match_status=draft.status,
                        matched_source_fact_ids=matched_ids,
                        primary_source_fact_id=primary,
                        primary_source_scope=primary_scope,
                        matched_relation_ids=tuple(
                            dict.fromkeys(draft.matched_relation_ids)
                        ),
                        transition_assessment=draft.transition_assessment,
                        stale_history=stale,
                        reason_code=draft.reason_code,
                        reason=draft.reason,
                    )
                candidates[draft.summary_fact_id].append(match)

        selected: dict[str, EvaluationFactMatch] = {}
        for fact_id, options in candidates.items():
            selected[fact_id] = max(
                options or [self._unsupported(fact_id)],
                key=lambda item: (
                    _STATUS_RANK[item.match_status],
                    not item.stale_history,
                    item.primary_source_fact_id or "",
                ),
            )
        return selected

    @staticmethod
    def _unsupported(
        summary_fact_id: str,
        *,
        reason_code: str = "no_supported_source_fact",
        reason: str = "本批候选源事实中没有可确认的对应事实。",
    ) -> EvaluationFactMatch:
        """生成不引用任何源事实的保守 unsupported 匹配。"""
        return EvaluationFactMatch(
            summary_fact_id=summary_fact_id,
            match_status=EvaluationMatchStatus.UNSUPPORTED,
            reason_code=reason_code,
            reason=reason,
        )

    def _candidate_union(
        self,
        summary_facts: tuple[EvaluationSummaryFact, ...],
        source_facts: tuple[EvaluationSourceFact, ...],
    ) -> tuple[tuple[EvaluationSourceFact, ...], str, int]:
        """按源事实规模选择全量扫描或稳定的 top-k 候选并集。"""
        if len(source_facts) <= self._config.evaluation_match_full_scan_threshold:
            return source_facts, "full_scan", 0
        limit = self._config.evaluation_match_candidate_limit_per_fact
        ranked: dict[str, list[EvaluationSourceFact]] = {}
        for summary_fact in summary_facts:
            ranked[summary_fact.summary_fact_id] = sorted(
                source_facts,
                key=lambda source: (
                    tuple(-value for value in _candidate_score(summary_fact, source)),
                    source.fact_id,
                ),
            )[:limit]
        union_limit = self._config.evaluation_match_candidate_union_limit
        selected: list[EvaluationSourceFact] = []
        selected_ids: set[str] = set()
        for rank in range(limit):
            for summary_fact in summary_facts:
                options = ranked[summary_fact.summary_fact_id]
                if rank >= len(options):
                    continue
                candidate = options[rank]
                if candidate.fact_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.fact_id)
                if len(selected) >= union_limit:
                    return tuple(selected), "candidate_top_k", len(source_facts) - len(selected)
        return tuple(selected), "candidate_top_k", len(source_facts) - len(selected)

    @staticmethod
    def _mark_duplicates(
        summary_facts: tuple[EvaluationSummaryFact, ...],
        matches: list[EvaluationFactMatch],
    ) -> tuple[EvaluationFactMatch, ...]:
        """按来源一致性和文本相似度标记重复摘要事实。"""
        fact_by_id = {item.summary_fact_id: item for item in summary_facts}
        retained: list[EvaluationFactMatch] = []
        prior: list[EvaluationFactMatch] = []
        for match in matches:
            duplicate_of = None
            current_text = _normalize(fact_by_id[match.summary_fact_id].fact)
            for previous in prior:
                previous_text = _normalize(
                    fact_by_id[previous.summary_fact_id].fact
                )
                same_source = (
                    match.primary_source_fact_id is not None
                    and match.primary_source_fact_id
                    == previous.primary_source_fact_id
                )
                both_unsupported = (
                    match.match_status == EvaluationMatchStatus.UNSUPPORTED
                    and previous.match_status == EvaluationMatchStatus.UNSUPPORTED
                )
                similarity = SequenceMatcher(
                    None, current_text, previous_text
                ).ratio()
                if (same_source and similarity >= 0.85) or (
                    both_unsupported and similarity >= 0.95
                ):
                    duplicate_of = previous.summary_fact_id
                    break
            if duplicate_of:
                match = match.model_copy(
                    update={"duplicate_of_summary_fact_id": duplicate_of}
                )
            retained.append(match)
            prior.append(match)
        return tuple(retained)
