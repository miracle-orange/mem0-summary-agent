"""独立构建当前事实，并分类本轮实际可见的历史事实。"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Optional

from pydantic import Field, model_validator

from ..grounding import match_evidence_quote
from ..raw_material import build_raw_material_bundle
from ..schemas import (
    FrozenStrictModel,
    RawMaterialBundle,
    RawSegment,
    RunConfig,
    SummaryRequest,
)
from .model_io import EvaluationModelIO
from .prompts import (
    build_current_fact_prompt,
    build_history_fact_prompt,
    build_raw_history_fact_prompt,
)
from .schemas import (
    EvaluationEvidence,
    EvaluationFactRelation,
    EvaluationFactRelationType,
    EvaluationFactState,
    EvaluationSalience,
    EvaluationSourceDocument,
    EvaluationSourceFact,
    EvaluationSourceScope,
    HistoricalDisposition,
)


# 评测 draft 模型的 Schema 由显式字段定义；使用普通注释避免改变协议描述。
# 当前事实抽取模型的一条原子事实草案。
class CurrentFactDraft(FrozenStrictModel):
    fact: str = Field(min_length=1)
    fact_type: str = Field(default="fact", min_length=1)
    state: EvaluationFactState = EvaluationFactState.UNKNOWN
    salience: EvaluationSalience
    entities: tuple[str, ...] = ()
    evidence_quote: str = Field(min_length=1)
    speaker: Optional[str] = None
    turn_id: Optional[str] = None
    classification_reason: str = ""


# 当前事实抽取模型的批量响应。
class CurrentFactDraftList(FrozenStrictModel):
    facts: tuple[CurrentFactDraft, ...] = ()


# 历史记录分类模型的一条原子事实草案。
class HistoricalFactDraft(FrozenStrictModel):
    record_id: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    fact_type: str = Field(default="fact", min_length=1)
    state: EvaluationFactState = EvaluationFactState.UNKNOWN
    salience: EvaluationSalience
    disposition: HistoricalDisposition
    entities: tuple[str, ...] = ()
    related_current_fact_ids: tuple[str, ...] = ()
    relation_type: EvaluationFactRelationType = (
        EvaluationFactRelationType.HISTORY_ONLY
    )
    relation_reason: str = ""
    classification_reason: str = ""

    @model_validator(mode="after")
    def validate_relation(self) -> "HistoricalFactDraft":
        """确保模型不会返回缺少当前事实引用的非 history-only 关系。"""
        if len(self.related_current_fact_ids) != len(
            set(self.related_current_fact_ids)
        ):
            raise ValueError("related current fact ids must be unique")
        if self.relation_type == EvaluationFactRelationType.HISTORY_ONLY:
            if self.related_current_fact_ids:
                raise ValueError("history-only draft cannot reference current facts")
        elif not self.related_current_fact_ids:
            raise ValueError("historical relation requires current fact ids")
        return self


# 历史事实分类模型的批量响应。
class HistoricalFactDraftList(FrozenStrictModel):
    facts: tuple[HistoricalFactDraft, ...] = ()


class RawHistoricalFactDraft(FrozenStrictModel):
    """独立横评从历史原文抽取的一条可 grounding 事实。"""

    fact: str = Field(min_length=1)
    fact_type: str = Field(default="fact", min_length=1)
    state: EvaluationFactState = EvaluationFactState.UNKNOWN
    salience: EvaluationSalience
    disposition: HistoricalDisposition
    entities: tuple[str, ...] = ()
    evidence_quote: str = Field(min_length=1)
    speaker: Optional[str] = None
    turn_id: Optional[str] = None
    related_current_fact_ids: tuple[str, ...] = ()
    relation_type: EvaluationFactRelationType = (
        EvaluationFactRelationType.HISTORY_ONLY
    )
    relation_reason: str = ""
    classification_reason: str = ""

    @model_validator(mode="after")
    def validate_relation(self) -> "RawHistoricalFactDraft":
        """复用历史关系形状约束。"""
        if len(self.related_current_fact_ids) != len(
            set(self.related_current_fact_ids)
        ):
            raise ValueError("related current fact ids must be unique")
        if self.relation_type == EvaluationFactRelationType.HISTORY_ONLY:
            if self.related_current_fact_ids:
                raise ValueError("history-only draft cannot reference current facts")
        elif not self.related_current_fact_ids:
            raise ValueError("historical relation requires current fact ids")
        return self


class RawHistoricalFactDraftList(FrozenStrictModel):
    """一段历史原文的原子事实集合。"""

    facts: tuple[RawHistoricalFactDraft, ...] = ()


def _stable_id(prefix: str, *parts: object) -> str:
    """根据稳定输入生成可重复的事实或 Evidence ID。"""
    payload = "\x1f".join(str(part) for part in parts)
    return prefix + "_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _normalize_fact(value: str) -> str:
    """去除展示标点并统一大小写，用于确定性事实去重。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _record_content(record: dict[str, object]) -> str:
    """合并历史记录正文及内容变体，保留可评测的全部版本。"""
    values: list[str] = []
    for key in ("content", "content_variants"):
        value = record.get(key)
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            values.append(str(value).strip())
    if not values:
        values.append(str(record))
    return " / ".join(dict.fromkeys(values))


class EvaluationSourceFactBuilder:
    """独立构建当前源事实，并分类本轮 Synthesis 实际可见的历史事实。"""

    def __init__(
        self,
        *,
        model_io: EvaluationModelIO,
        config: RunConfig,
    ) -> None:
        """绑定评测模型 I/O，并初始化分段、批次和 grounding warning 统计。"""
        self._model_io = model_io
        self._config = config
        self.warnings: list[str] = []
        self.current_segment_count = 0
        self.historical_batch_count = 0
        self.adaptive_split_count = 0
        self.adaptive_split_child_count = 0
        self.relations: tuple[EvaluationFactRelation, ...] = ()

    def _adaptive_children(
        self,
        segment: RawSegment,
        *,
        depth: int,
    ) -> tuple[RawSegment, ...]:
        """把结构化输出失败的源分段稳定拆成两个带少量重叠的子段。"""
        if not self._config.evaluation_adaptive_split_enabled:
            return ()
        if depth >= self._config.evaluation_adaptive_split_max_depth:
            return ()
        minimum = self._config.evaluation_adaptive_split_min_chars
        if len(segment.text) <= minimum:
            return ()
        midpoint = len(segment.text) // 2
        half_overlap = max(
            1,
            int(minimum * self._config.raw_segment_overlap_ratio / 2),
        )
        boundaries = (
            (0, min(len(segment.text), midpoint + half_overlap)),
            (max(0, midpoint - half_overlap), len(segment.text)),
        )
        children: list[RawSegment] = []
        for child_index, (local_start, local_end) in enumerate(boundaries):
            child_text = segment.text[local_start:local_end]
            if not child_text or child_text == segment.text:
                return ()
            digest = hashlib.sha256(child_text.encode("utf-8")).hexdigest()
            children.append(
                RawSegment(
                    segment_id=(
                        f"{segment.segment_id}_split{depth + 1}_"
                        f"{child_index}_{digest[:10]}"
                    ),
                    source_id=segment.source_id,
                    ordinal=child_index,
                    start_offset=segment.start_offset + local_start,
                    end_offset=segment.start_offset + local_end,
                    text=child_text,
                    content_sha256=digest,
                )
            )
        self.adaptive_split_count += 1
        self.adaptive_split_child_count += len(children)
        return tuple(children)

    def _split_after_protocol_failure(
        self,
        *,
        stage: str,
        segment: RawSegment,
        depth: int,
    ) -> tuple[RawSegment, ...]:
        """仅对已审计的截断或协议校验失败启用自适应拆分。"""
        failure_kind = self._model_io.last_failure_kind(stage)
        if failure_kind not in {"likely_output_truncation", "validation_error"}:
            return ()
        children = self._adaptive_children(segment, depth=depth)
        if children:
            self.warnings.append(
                f"{segment.segment_id}: adaptive split after {failure_kind}; "
                f"created {len(children)} child segments"
            )
        return children

    async def build_current_facts(
        self,
        request: SummaryRequest,
        raw_material: RawMaterialBundle,
    ) -> tuple[EvaluationSourceFact, ...]:
        """用独立分段抽取当前源事实，再进行 quote grounding 和确定性去重。

        评测分段独立于生产 chunk，避免生产分段大小改变被测源事实数量；每段
        可受限并发，单段无法 grounding 的情况会记录 warning 或抛出失败。
        """
        # 评测使用独立分段，避免生产 chunk 大小改变被测源证据数量。
        evaluation_material = build_raw_material_bundle(
            request.current_material,
            source_id=raw_material.source_id,
            max_segment_chars=self._config.evaluation_segment_max_chars,
            overlap_ratio=self._config.raw_segment_overlap_ratio,
        )
        self.current_segment_count = len(evaluation_material.segments)
        semaphore = asyncio.Semaphore(
            self._config.evaluation_max_concurrent_segment_maps
        )

        async def map_segment(
            segment: RawSegment,
            *,
            depth: int = 0,
            parent_segment_id: str | None = None,
        ):
            """抽取并 grounding 一个评测分段的当前事实。"""
            stage = f"evaluation.current_source.{segment.segment_id}"
            try:
                async with semaphore:
                    draft = await self._model_io.complete_json(
                        stage=stage,
                        prompt=build_current_fact_prompt(request, segment),
                        response_model=CurrentFactDraftList,
                        max_tokens=self._config.evaluation_source_fact_max_tokens,
                        repair_attempts=self._config.evaluation_repair_attempts,
                        timeout_seconds=self._config.evaluation_model_timeout_seconds,
                        trace_metadata={
                            "source_id": segment.source_id,
                            "segment_id": segment.segment_id,
                            "source_scope": "current",
                            "start_offset": segment.start_offset,
                            "end_offset": segment.end_offset,
                            "adaptive_split_depth": depth,
                            "parent_segment_id": parent_segment_id,
                        },
                    )
            except Exception:
                children = self._split_after_protocol_failure(
                    stage=stage,
                    segment=segment,
                    depth=depth,
                )
                if not children:
                    raise
                groups = []
                for child in children:
                    groups.extend(
                        await map_segment(
                            child,
                            depth=depth + 1,
                            parent_segment_id=segment.segment_id,
                        )
                    )
                return tuple(groups)
            facts: list[EvaluationSourceFact] = []
            rejected: list[str] = []
            for item in draft.facts:
                quote_match = match_evidence_quote(segment.text, item.evidence_quote)
                if quote_match.grounded_quote is None:
                    rejected.append(item.fact)
                    continue
                grounded = quote_match.grounded_quote
                start = segment.start_offset + grounded.start
                end = segment.start_offset + grounded.end
                fact_id = _stable_id(
                    "eval_current",
                    raw_material.source_id,
                    start,
                    end,
                    item.fact,
                    item.state.value,
                )
                facts.append(
                    EvaluationSourceFact(
                        fact_id=fact_id,
                        source_scope=EvaluationSourceScope.CURRENT,
                        fact=item.fact.strip(),
                        fact_type=item.fact_type.strip(),
                        state=item.state,
                        salience=item.salience,
                        entities=tuple(dict.fromkeys(item.entities)),
                        evidence=(
                            EvaluationEvidence(
                                evidence_id=_stable_id(
                                    "eval_evidence",
                                    raw_material.source_id,
                                    start,
                                    end,
                                ),
                                evidence_text=grounded.evidence_text,
                                source_id=raw_material.source_id,
                                segment_id=segment.segment_id,
                                start_offset=start,
                                end_offset=end,
                                source_time=request.source_time,
                                speaker=item.speaker,
                                turn_id=item.turn_id,
                            ),
                        ),
                        classification_reason=item.classification_reason,
                    )
                )
            if rejected:
                self.warnings.append(
                    f"{segment.segment_id}: discarded {len(rejected)} current "
                    "evaluation facts whose quotes could not be grounded"
                )
                if draft.facts and not facts:
                    raise ValueError(
                        f"{segment.segment_id} returned only ungrounded "
                        "current evaluation facts"
                    )
            return tuple(facts)

        mapped = await asyncio.gather(
            *(map_segment(segment) for segment in evaluation_material.segments)
        )
        flattened = [fact for group in mapped for fact in group]
        return self._deduplicate_current(flattened)

    @staticmethod
    def _deduplicate_current(
        facts: list[EvaluationSourceFact],
    ) -> tuple[EvaluationSourceFact, ...]:
        """按事实、类型、状态和实体合并重复当前事实及其 Evidence。"""
        retained: dict[tuple[object, ...], EvaluationSourceFact] = {}
        for fact in sorted(
            facts,
            key=lambda item: (
                item.evidence[0].start_offset or 0,
                item.evidence[0].end_offset or 0,
                item.fact_id,
            ),
        ):
            key = (
                _normalize_fact(fact.fact),
                fact.fact_type.casefold(),
                fact.state.value,
                tuple(sorted(entity.casefold() for entity in fact.entities)),
            )
            existing = retained.get(key)
            if existing is None:
                retained[key] = fact
                continue
            evidence = tuple(
                {
                    (
                        item.source_id,
                        item.start_offset,
                        item.end_offset,
                        item.evidence_text,
                    ): item
                    for item in (*existing.evidence, *fact.evidence)
                }.values()
            )
            salience = (
                EvaluationSalience.MUST_INCLUDE
                if EvaluationSalience.MUST_INCLUDE
                in {existing.salience, fact.salience}
                else existing.salience
            )
            retained[key] = existing.model_copy(
                update={"evidence": evidence, "salience": salience}
            )
        return tuple(retained.values())

    async def build_raw_historical_facts(
        self,
        *,
        summary_request: str,
        current_facts: tuple[EvaluationSourceFact, ...],
        historical_sources: tuple[EvaluationSourceDocument, ...],
    ) -> tuple[EvaluationSourceFact, ...]:
        """从显式历史原文逐段抽取、grounding，并同时建立事实关系。"""
        self.relations = ()
        self.historical_batch_count = 0
        if not historical_sources:
            return ()
        current_by_id = {item.fact_id: item for item in current_facts}
        facts: list[EvaluationSourceFact] = []
        relations: list[EvaluationFactRelation] = []
        seen_fact_ids: set[str] = set()
        for source in historical_sources:
            bundle = build_raw_material_bundle(
                source.text,
                source_id=source.source_id,
                max_segment_chars=self._config.evaluation_segment_max_chars,
                overlap_ratio=self._config.raw_segment_overlap_ratio,
            )
            self.historical_batch_count += len(bundle.segments)
            allowed_current_ids = set(current_by_id)

            def validate_raw_history(
                response: RawHistoricalFactDraftList,
            ) -> None:
                """让未知当前事实 ID 在模型 repair 循环内修复。"""
                unknown = {
                    current_id
                    for item in response.facts
                    for current_id in item.related_current_fact_ids
                } - allowed_current_ids
                if unknown:
                    raise ValueError(
                        "historical source builder referenced unknown current "
                        f"fact ids: {sorted(unknown)}"
                    )

            async def map_historical_segment(
                segment: RawSegment,
                *,
                depth: int = 0,
                parent_segment_id: str | None = None,
            ) -> None:
                """抽取一个历史分段，协议失败时仅拆分该分段后重试。"""
                stage = (
                    "evaluation.raw_history_source."
                    f"{source.source_id}.{segment.segment_id}"
                )
                try:
                    result = await self._model_io.complete_json(
                        stage=stage,
                        prompt=build_raw_history_fact_prompt(
                            summary_request=summary_request,
                            source_id=source.source_id,
                            source_time=source.source_time,
                            segment=segment,
                            current_facts=current_facts,
                        ),
                        response_model=RawHistoricalFactDraftList,
                        max_tokens=self._config.evaluation_source_fact_max_tokens,
                        repair_attempts=self._config.evaluation_repair_attempts,
                        timeout_seconds=self._config.evaluation_model_timeout_seconds,
                        trace_metadata={
                            "source_id": source.source_id,
                            "segment_id": segment.segment_id,
                            "source_scope": "historical",
                            "start_offset": segment.start_offset,
                            "end_offset": segment.end_offset,
                            "adaptive_split_depth": depth,
                            "parent_segment_id": parent_segment_id,
                        },
                        validator=validate_raw_history,
                    )
                except Exception:
                    children = self._split_after_protocol_failure(
                        stage=stage,
                        segment=segment,
                        depth=depth,
                    )
                    if not children:
                        raise
                    for child in children:
                        await map_historical_segment(
                            child,
                            depth=depth + 1,
                            parent_segment_id=segment.segment_id,
                        )
                    return
                accepted = 0
                for item in result.facts:
                    quote_match = match_evidence_quote(
                        segment.text, item.evidence_quote
                    )
                    if quote_match.grounded_quote is None:
                        self.warnings.append(
                            f"historical segment {segment.segment_id}: discarded "
                            "an evaluation fact whose quote could not be grounded"
                        )
                        continue
                    grounded = quote_match.grounded_quote
                    start = segment.start_offset + grounded.start
                    end = segment.start_offset + grounded.end
                    fact_id = _stable_id(
                        "eval_history",
                        source.source_id,
                        start,
                        end,
                        item.fact,
                        item.state.value,
                    )
                    fact = EvaluationSourceFact(
                        fact_id=fact_id,
                        source_scope=EvaluationSourceScope.HISTORICAL,
                        fact=item.fact.strip(),
                        fact_type=item.fact_type.strip(),
                        state=item.state,
                        salience=item.salience,
                        entities=tuple(dict.fromkeys(item.entities)),
                        evidence=(
                            EvaluationEvidence(
                                evidence_id=_stable_id(
                                    "eval_evidence",
                                    source.source_id,
                                    start,
                                    end,
                                ),
                                evidence_text=grounded.evidence_text,
                                source_id=source.source_id,
                                segment_id=segment.segment_id,
                                start_offset=start,
                                end_offset=end,
                                source_time=source.source_time,
                                speaker=item.speaker,
                                turn_id=item.turn_id,
                            ),
                        ),
                        source_record_ids=(source.source_id,),
                        history_disposition=item.disposition,
                        classification_reason=item.classification_reason,
                    )
                    if fact_id in seen_fact_ids:
                        accepted += 1
                        self.warnings.append(
                            f"historical fact {fact_id} repeated across overlap; "
                            "retained the first grounded occurrence"
                        )
                        continue
                    seen_fact_ids.add(fact_id)
                    facts.append(fact)
                    current_states = {
                        current_by_id[current_id].state
                        for current_id in item.related_current_fact_ids
                    }
                    current_state = (
                        next(iter(current_states))
                        if len(current_states) == 1
                        else EvaluationFactState.UNKNOWN
                    )
                    relation = EvaluationFactRelation(
                        relation_id=_stable_id(
                            "eval_relation",
                            fact_id,
                            item.relation_type.value,
                            *sorted(item.related_current_fact_ids),
                        ),
                        fact_family_id=_stable_id(
                            "eval_family",
                            *(
                                sorted(item.related_current_fact_ids)
                                or [fact_id]
                            ),
                        ),
                        historical_fact_id=fact_id,
                        current_fact_ids=tuple(
                            dict.fromkeys(item.related_current_fact_ids)
                        ),
                        relation_type=item.relation_type,
                        historical_state=item.state,
                        current_state=current_state,
                        reason=item.relation_reason or item.classification_reason,
                    )
                    relations.append(relation)
                    accepted += 1
                if result.facts and not accepted:
                    raise ValueError(
                        f"historical segment {segment.segment_id} returned only "
                        "ungrounded evaluation facts"
                    )

            for segment in bundle.segments:
                await map_historical_segment(segment)
        self.relations = tuple(relations)
        return tuple(facts)

    async def build_historical_facts(
        self,
        request: SummaryRequest,
        current_facts: tuple[EvaluationSourceFact, ...],
        historical_records: dict[str, dict[str, object]],
        evidence_index: dict[str, object],
    ) -> tuple[EvaluationSourceFact, ...]:
        """只分类 SynthesisContext 可见的历史记录，并标注其当前有效性。"""
        self.relations = ()
        if not historical_records:
            return ()
        items = list(sorted(historical_records.items()))
        batch_size = self._config.evaluation_history_batch_size
        self.historical_batch_count = (len(items) + batch_size - 1) // batch_size
        drafts: list[HistoricalFactDraft] = []
        for start in range(0, len(items), batch_size):
            batch = dict(items[start : start + batch_size])
            current_ids = {item.fact_id for item in current_facts}

            def validate_history(response: HistoricalFactDraftList) -> None:
                """在 repair 循环内拒绝未知记录或当前事实 ID。"""
                unknown_records = {
                    item.record_id for item in response.facts
                } - set(batch)
                if unknown_records:
                    raise ValueError(
                        "historical classifier referenced unknown record ids: "
                        f"{sorted(unknown_records)}"
                    )
                unknown_current = {
                    current_id
                    for item in response.facts
                    for current_id in item.related_current_fact_ids
                } - current_ids
                if unknown_current:
                    raise ValueError(
                        "historical classifier referenced unknown current fact ids: "
                        f"{sorted(unknown_current)}"
                    )

            result = await self._model_io.complete_json(
                stage=f"evaluation.history_source.batch_{start // batch_size + 1}",
                prompt=build_history_fact_prompt(request, current_facts, batch),
                response_model=HistoricalFactDraftList,
                max_tokens=self._config.evaluation_source_fact_max_tokens,
                repair_attempts=self._config.evaluation_repair_attempts,
                timeout_seconds=self._config.evaluation_model_timeout_seconds,
                validator=validate_history,
            )
            drafts.extend(result.facts)

        by_record: dict[str, list[HistoricalFactDraft]] = {}
        for draft in drafts:
            by_record.setdefault(draft.record_id, []).append(draft)

        facts: list[EvaluationSourceFact] = []
        relations: list[EvaluationFactRelation] = []
        current_by_id = {item.fact_id: item for item in current_facts}
        for record_id, record in items:
            record_drafts = by_record.get(record_id, [])
            if not record_drafts:
                self.warnings.append(
                    f"historical record {record_id} was not classified; "
                    "retained as uncertain/exclude"
                )
                record_drafts = [
                    HistoricalFactDraft(
                        record_id=record_id,
                        fact=_record_content(record),
                        fact_type=str(record.get("memory_type") or "fact"),
                        state=EvaluationFactState.UNKNOWN,
                        salience=EvaluationSalience.EXCLUDE,
                        disposition=HistoricalDisposition.UNCERTAIN,
                        classification_reason="model did not classify this visible record",
                    )
                ]
            evidence_ids = tuple(
                str(item)
                for item in record.get("evidence_ids", [])
                if str(item).strip()
            )
            evidences: list[EvaluationEvidence] = []
            for evidence_id in evidence_ids:
                raw_evidence = evidence_index.get(evidence_id)
                if raw_evidence is None:
                    continue
                if hasattr(raw_evidence, "model_dump"):
                    payload = raw_evidence.model_dump(mode="json")
                elif isinstance(raw_evidence, dict):
                    payload = raw_evidence
                else:
                    continue
                evidences.append(
                    EvaluationEvidence(
                        evidence_id=evidence_id,
                        evidence_text=str(payload.get("evidence_text") or ""),
                        source_id=payload.get("source_id"),
                        segment_id=payload.get("segment_id"),
                        start_offset=payload.get("start_offset"),
                        end_offset=payload.get("end_offset"),
                        memory_id=payload.get("memory_id") or record.get("memory_id"),
                        source_time=payload.get("source_time"),
                    )
                )
            if not evidences:
                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                evidences.append(
                    EvaluationEvidence(
                        evidence_text=_record_content(record),
                        memory_id=str(record.get("memory_id") or record_id),
                        source_time=str(metadata.get("source_time") or "") or None,
                    )
                )
            for index, draft in enumerate(record_drafts):
                fact_id = _stable_id(
                    "eval_history", record_id, index, draft.fact
                )
                facts.append(
                    EvaluationSourceFact(
                        fact_id=fact_id,
                        source_scope=EvaluationSourceScope.HISTORICAL,
                        fact=draft.fact.strip(),
                        fact_type=draft.fact_type.strip(),
                        state=draft.state,
                        salience=draft.salience,
                        entities=tuple(dict.fromkeys(draft.entities)),
                        evidence=tuple(evidences),
                        source_record_ids=(record_id,),
                        related_evidence_ids=evidence_ids,
                        history_disposition=draft.disposition,
                        classification_reason=draft.classification_reason,
                    )
                )
                current_states = {
                    current_by_id[item].state
                    for item in draft.related_current_fact_ids
                    if item in current_by_id
                }
                current_state = (
                    next(iter(current_states))
                    if len(current_states) == 1
                    else EvaluationFactState.UNKNOWN
                )
                family_parts = tuple(
                    sorted(draft.related_current_fact_ids) or [fact_id]
                )
                relations.append(
                    EvaluationFactRelation(
                        relation_id=_stable_id(
                            "eval_relation",
                            fact_id,
                            draft.relation_type.value,
                            *sorted(draft.related_current_fact_ids),
                        ),
                        fact_family_id=_stable_id("eval_family", *family_parts),
                        historical_fact_id=fact_id,
                        current_fact_ids=tuple(
                            dict.fromkeys(draft.related_current_fact_ids)
                        ),
                        relation_type=draft.relation_type,
                        historical_state=draft.state,
                        current_state=current_state,
                        reason=draft.relation_reason or draft.classification_reason,
                    )
                )
        self.relations = tuple(relations)
        return tuple(facts)
