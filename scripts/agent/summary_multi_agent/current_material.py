"""将完整当前原文映射为带连续 Evidence 的事实条目，并维护 Coverage。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import Field

from .adapters.model import ContextOverflowError, ModelClient
from .grounding import match_evidence_quote, normalize_presentation_text
from .raw_material import mark_segment_failed, mark_segment_processed
from .schemas import (
    CurrentFactCategory,
    CurrentFactItem,
    CurrentSegmentAnalysis,
    EvidenceRef,
    FrozenStrictModel,
    RawSegment,
    RunConfig,
    SourceKind,
    SummaryRequest,
    SummaryWorkingState,
)


# 模型返回的单条当前事实草案；其 evidence_quote 必须经系统 grounding。
class CurrentFactDraft(FrozenStrictModel):
    category: CurrentFactCategory
    statement: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    entities: tuple[str, ...] = ()
    time_expression: Optional[str] = None


# 模型返回的单个当前分段事实草案集合。
class CurrentSegmentDraft(FrozenStrictModel):
    items: tuple[CurrentFactDraft, ...] = ()


@dataclass(frozen=True)
class SegmentMapOutcome:
    """保存单个当前分段的事实映射、Evidence、追踪信息和可恢复错误。"""

    segment_id: str
    analysis: Optional[CurrentSegmentAnalysis]
    evidence: tuple[EvidenceRef, ...]
    trace: dict[str, Any]
    error: Optional[str] = None


@dataclass(frozen=True)
class DraftGroundingOutcome:
    """保存 grounding 后可接受的事实、Evidence 和被拒绝的模型条目。"""

    analysis: CurrentSegmentAnalysis
    evidence: tuple[EvidenceRef, ...]
    rejected_items: tuple[CurrentFactDraft, ...]
    audit: tuple[dict[str, Any], ...]


TEMPORAL_ASSERTION_PATTERN = re.compile(
    r"(?:今天|今日|明天|明日|后天|昨天|昨日|前天)"
    r"|(?:(?:上|本|这|下)?(?:周|星期|礼拜)[一二三四五六日天])"
    r"|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?"
)

CURRENT_MAP_PROMPT_VERSION = "summary_multi_agent.current_map_prompt.v2"


def extract_json_object(raw_output: str) -> dict[str, Any]:
    """从当前事实模型输出中提取 JSON object，并拒绝其他顶层类型。"""
    text = (raw_output or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON must be an object")
    return parsed


def build_current_segment_map_prompt(
    request: SummaryRequest,
    segment: RawSegment,
) -> str:
    """为单个原文分段构造当前事实映射 Prompt。

    Prompt 要求引文必须来自该分段连续原文；这里只生成字符串，不调用模型。
    """
    return f"""你是 Summary Multi-Agent V2 的 Current Material Mapper。

任务：只分析下面这个当前原始资料分段，抽取后续摘要需要的可验证事实。当前资料优先级高于历史记忆。

硬性规则：
1. 不能补充原文没有的信息，不能把建议写成决定，不能猜负责人或完成状态。
2. 每个 item 的 evidence_quote 必须是分段原文中连续、逐字一致的短文本。
3. statement 可以规范表达，但必须与 evidence_quote 一致；不得给原文中的时间表达自行增加“上/本/这/下周”、绝对日期或完成状态。
4. category 只能是 fact、decision、constraint、task、risk、preference、question。
5. entities 只写原文明示的人物或实体；time_expression 必须逐字来自 evidence_quote，没有则为 null。原文只写“周五”时必须保留“周五”，不要猜成上周五或本周五。
6. 优先保留影响结论或执行的对象、数字、单位、范围、排除项、前后对照、条件和依赖；进展写明完成范围和剩余工作，不能把计划写成已完成。
7. 明确的承诺、分工、整改和后续验证属于任务线索；只提取原文支持的部分，缺负责人或日期时不要补造，保留预计、可能、暂定等语气。
8. 一条 item 聚焦一个可验证事实；相同事实只保留一次，独有细节、状态变化和冲突分别保留。多句共同支持时引文必须覆盖所需的连续原文；无法连续支持则拆分为独立且有据的事实，不拼造引文。
9. 没有可抽取内容时返回空 items。只返回符合以下 Schema 的 JSON，不要 Markdown，不要把 Schema 字段说明写成事实。

输出 Schema：
{json.dumps(CurrentSegmentDraft.model_json_schema(), ensure_ascii=False, separators=(",", ":"))}

请求信息：
- request_id: {request.request_id}
- source_time: {request.source_time or "未提供"}
- summary_request: {request.summary_request}
- segment_id: {segment.segment_id}
- segment_offset: [{segment.start_offset}, {segment.end_offset})

<current_segment>
{segment.text}
</current_segment>
"""


class CurrentMaterialAnalyzer:
    """并发分析全部当前原文分段，并维护当前事实的 Coverage/Evidence。

    每个分段在独立超时和有限修复次数内完成；失败写入分段错误，由 Pipeline
    后续的完整覆盖闸门决定运行不能无警告完成。最佳 grounding 结果会被保留，
    但不会修改原文或长期 Memory。
    """

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
    ) -> None:
        """初始化模型客户端、分段并发上限和本次分析的内存 trace。"""
        self._model_client = model_client
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_segment_maps)
        self._traces: list[dict[str, Any]] = []

    @property
    def traces(self) -> tuple[dict[str, Any], ...]:
        """返回按分段整理的模型调用与 grounding 审计记录。"""
        return tuple(self._traces)

    async def analyze(self, state: SummaryWorkingState) -> SummaryWorkingState:
        """并发映射所有分段并把结果合并回 M4 状态。

        分段任务通过 ``asyncio.gather`` 并发，但最终按 ordinal 排序写回，以保证
        产物稳定；单段失败只记录失败状态，所有段结束后仍由 Coverage 统一判定。
        """
        outcomes = await asyncio.gather(
            *(
                self._map_segment(state.request, segment)
                for segment in state.raw_material.segments
            )
        )
        analyses: list[CurrentSegmentAnalysis] = []
        evidence_index = dict(state.evidence_index)
        ledger = state.coverage_ledger
        self._traces = []
        for outcome in sorted(
            outcomes,
            key=lambda item: next(
                segment.ordinal
                for segment in state.raw_material.segments
                if segment.segment_id == item.segment_id
            ),
        ):
            self._traces.append(outcome.trace)
            if outcome.analysis is None:
                error = outcome.error or "current segment mapping failed"
                ledger = mark_segment_failed(
                    ledger,
                    segment_id=outcome.segment_id,
                    stage="current_fact_map",
                    error=error,
                )
                state.errors.append(f"{outcome.segment_id}: {error}")
                continue
            analyses.append(outcome.analysis)
            for evidence in outcome.evidence:
                evidence_index[evidence.evidence_id] = evidence
            ledger = mark_segment_processed(
                ledger,
                segment_id=outcome.segment_id,
                stage="current_fact_map",
                evidence_ids=(item.evidence_id for item in outcome.evidence),
            )
            state.warnings.extend(outcome.analysis.warnings)

        state.current_segment_analyses = analyses
        state.evidence_index = evidence_index
        state.coverage_ledger = ledger
        return state

    async def _map_segment(
        self,
        request: SummaryRequest,
        segment: RawSegment,
    ) -> SegmentMapOutcome:
        """在单段并发槽位内执行模型、JSON 修复和受控 grounding 重试。

        ``wait_for`` 覆盖每次模型调用，取消会向上传播并释放 semaphore；修复
        耗尽时返回最佳可 grounding 结果，或返回带错误的失败 outcome。
        """
        prompt = build_current_segment_map_prompt(request, segment)
        attempts: list[dict[str, Any]] = []
        next_prompt = prompt
        best_grounding: Optional[DraftGroundingOutcome] = None
        best_score = (-1, -1)
        async with self._semaphore:
            for attempt in range(self._config.schema_repair_attempts + 1):
                try:
                    raw_output = await asyncio.wait_for(
                        getattr(self._model_client, "complete_json", self._model_client.complete)(
                            next_prompt,
                            temperature=0.0,
                            max_tokens=self._config.current_map_max_tokens,
                        ),
                        timeout=self._config.worker_timeout_seconds,
                    )
                    attempts.append(
                        {"attempt": attempt + 1, "raw_output": raw_output, "error": None}
                    )
                    draft = CurrentSegmentDraft.model_validate(
                        extract_json_object(raw_output)
                    )
                    grounding = self._ground_draft(request, segment, draft)
                    attempts[-1]["grounding_audit"] = list(grounding.audit)
                    attempts[-1]["rejected_count"] = len(
                        grounding.rejected_items
                    )
                    score = (
                        len(grounding.analysis.items),
                        -len(grounding.rejected_items),
                    )
                    if score > best_score:
                        best_grounding = grounding
                        best_score = score
                    if not grounding.rejected_items and best_grounding is grounding:
                        return self._successful_outcome(
                            segment, grounding, prompt, attempts
                        )
                    if attempt >= self._config.schema_repair_attempts:
                        assert best_grounding is not None
                        return self._successful_outcome(
                            segment, best_grounding, prompt, attempts
                        )
                    next_prompt = self._build_grounding_repair_prompt(
                        prompt,
                        grounding.rejected_items,
                        raw_output,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempts and attempts[-1].get("attempt") == attempt + 1:
                        attempts[-1]["error"] = error
                    else:
                        attempts.append(
                            {"attempt": attempt + 1, "raw_output": None, "error": error}
                        )
                    if isinstance(exc, ContextOverflowError):
                        attempts[-1]["failure_kind"] = "context_overflow"
                        return SegmentMapOutcome(
                            segment_id=segment.segment_id,
                            analysis=None,
                            evidence=(),
                            error=error,
                            trace={
                                "segment_id": segment.segment_id,
                                "prompt_version": CURRENT_MAP_PROMPT_VERSION,
                                "prompt": prompt,
                                "attempts": attempts,
                            },
                        )
                    if attempt >= self._config.schema_repair_attempts:
                        if best_grounding is not None:
                            return self._successful_outcome(
                                segment, best_grounding, prompt, attempts
                            )
                        return SegmentMapOutcome(
                            segment_id=segment.segment_id,
                            analysis=None,
                            evidence=(),
                            error=error,
                            trace={
                                "segment_id": segment.segment_id,
                                "prompt_version": CURRENT_MAP_PROMPT_VERSION,
                                "prompt": prompt,
                                "attempts": attempts,
                            },
                        )
                    last_output = attempts[-1].get("raw_output") or "<no output>"
                    next_prompt = (
                        prompt
                        + "\n\n上一次输出未通过 JSON Schema 或证据校验。"
                        + "请修复后只返回完整 JSON。错误："
                        + error
                        + "\n上一次输出：\n"
                        + str(last_output)[:4000]
                    )
        raise AssertionError("segment map retry loop exited unexpectedly")

    @staticmethod
    def _build_grounding_repair_prompt(
        original_prompt: str,
        rejected_items: tuple[CurrentFactDraft, ...],
        previous_output: str,
    ) -> str:
        """根据被拒绝引文构造下一次完整 JSON 修复请求。"""
        rejected_json = json.dumps(
            [item.model_dump(mode="json") for item in rejected_items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            original_prompt
            + "\n\n上一次输出中有 evidence_quote 无法映射到唯一的原文连续区间。"
            + "请重新返回完整 JSON，并保留已经正确的事实。"
            + "每个 evidence_quote 必须直接复制原文；如果事实依赖相邻的多轮对话，"
            + "应复制覆盖全部事实的连续原文，或拆成多个各自有证据的 item。"
            + "不要使用语义近似引文，也不要为了通过校验而补造事实。\n"
            + "未通过项：\n"
            + rejected_json
            + "\n上一次完整输出：\n"
            + previous_output[:6000]
        )

    @staticmethod
    def _successful_outcome(
        segment: RawSegment,
        grounding: DraftGroundingOutcome,
        prompt: str,
        attempts: list[dict[str, Any]],
    ) -> SegmentMapOutcome:
        """把已通过 grounding 的分段结果包装成可写回状态的 outcome。"""
        return SegmentMapOutcome(
            segment_id=segment.segment_id,
            analysis=grounding.analysis,
            evidence=grounding.evidence,
            trace={
                "segment_id": segment.segment_id,
                "prompt_version": CURRENT_MAP_PROMPT_VERSION,
                "prompt": prompt,
                "attempts": attempts,
            },
        )

    @staticmethod
    def _ground_draft(
        request: SummaryRequest,
        segment: RawSegment,
        draft: CurrentSegmentDraft,
    ) -> DraftGroundingOutcome:
        """验证模型事实的连续引文、全局 offset、时间表达和重复项。

        这里的关键边界是“模型可提出事实，系统决定 Evidence 是否可信”；无法
        精确回映射的条目被拒绝，不用语义近似文本伪造证据。
        """
        items: list[CurrentFactItem] = []
        evidence_items: list[EvidenceRef] = []
        rejected_items: list[CurrentFactDraft] = []
        audit: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen: set[tuple[str, str, int, int]] = set()
        for item_index, draft_item in enumerate(draft.items):
            quote_match = match_evidence_quote(
                segment.text, draft_item.evidence_quote
            )
            if quote_match.grounded_quote is None:
                rejected_items.append(draft_item)
                warnings.append(
                    f"{segment.segment_id}: discarded item after controlled grounding "
                    f"({quote_match.reason}): {draft_item.statement!r}"
                )
                audit.append(
                    {
                        "item_index": item_index,
                        "status": "rejected",
                        "reason": quote_match.reason,
                        "statement": draft_item.statement,
                        "submitted_evidence_quote": draft_item.evidence_quote,
                    }
                )
                continue
            grounded_quote = quote_match.grounded_quote
            quote = grounded_quote.evidence_text
            local_start = grounded_quote.start
            local_end = grounded_quote.end
            global_start = segment.start_offset + local_start
            global_end = segment.start_offset + local_end

            statement = draft_item.statement.strip()
            quote_normalized = normalize_presentation_text(quote)
            unsupported_temporal_assertions = [
                match.group(0)
                for match in TEMPORAL_ASSERTION_PATTERN.finditer(statement)
                if normalize_presentation_text(match.group(0)) not in quote_normalized
            ]
            if unsupported_temporal_assertions:
                statement = quote
                warnings.append(
                    f"{segment.segment_id}: reset a temporally enriched statement to "
                    "its grounded source quote"
                )

            grounded_time_expression: Optional[str] = None
            time_expression_reason: Optional[str] = None
            if draft_item.time_expression:
                time_match = match_evidence_quote(
                    quote, draft_item.time_expression
                )
                if time_match.grounded_quote is not None:
                    grounded_time_expression = (
                        time_match.grounded_quote.evidence_text
                    )
                else:
                    time_expression_reason = time_match.reason
                    warnings.append(
                        f"{segment.segment_id}: discarded a time_expression that "
                        "was not grounded in its evidence quote"
                    )

            key = (
                draft_item.category.value,
                statement,
                global_start,
                global_end,
            )
            if key in seen:
                audit.append(
                    {
                        "item_index": item_index,
                        "status": "duplicate",
                        "strategy": grounded_quote.strategy,
                    }
                )
                continue
            seen.add(key)
            evidence = EvidenceRef(
                source_kind=SourceKind.CURRENT_MATERIAL,
                source_id=segment.source_id,
                segment_id=segment.segment_id,
                start_offset=global_start,
                end_offset=global_end,
                source_time=request.source_time,
                evidence_text=quote,
            )
            evidence_items.append(evidence)
            items.append(
                CurrentFactItem(
                    category=draft_item.category,
                    statement=statement,
                    evidence_quote=quote,
                    evidence_id=evidence.evidence_id,
                    entities=tuple(
                        dict.fromkeys(
                            entity.strip()
                            for entity in draft_item.entities
                            if entity.strip()
                        )
                    ),
                    time_expression=grounded_time_expression,
                )
            )
            audit.append(
                {
                    "item_index": item_index,
                    "status": "accepted",
                    "strategy": grounded_quote.strategy,
                    "start_offset": global_start,
                    "end_offset": global_end,
                    "statement_reset_to_quote": bool(
                        unsupported_temporal_assertions
                    ),
                    "unsupported_temporal_assertions": (
                        unsupported_temporal_assertions
                    ),
                    "time_expression_rejection_reason": time_expression_reason,
                }
            )
        return DraftGroundingOutcome(
            analysis=CurrentSegmentAnalysis(
                segment_id=segment.segment_id,
                start_offset=segment.start_offset,
                end_offset=segment.end_offset,
                items=tuple(items),
                warnings=tuple(warnings),
            ),
            evidence=tuple(evidence_items),
            rejected_items=tuple(rejected_items),
            audit=tuple(audit),
        )
