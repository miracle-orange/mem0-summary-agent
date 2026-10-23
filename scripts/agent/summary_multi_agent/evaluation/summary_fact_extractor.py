"""从正式摘要中抽取带原文 offset 的原子 claim。"""

from __future__ import annotations

import asyncio
import hashlib

from pydantic import Field

from ..grounding import match_evidence_quote
from ..raw_material import build_raw_material_bundle
from ..schemas import FrozenStrictModel, RunConfig
from .model_io import EvaluationModelIO
from .prompts import build_summary_fact_prompt
from .schemas import EvaluationClaimScope, EvaluationSummaryFact


# 评测 draft 模型的 Schema 由显式字段定义；使用普通注释避免改变协议描述。
# 正式摘要事实抽取模型的一条原子 claim 草案。
class SummaryFactDraft(FrozenStrictModel):
    fact: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    claimed_scope: EvaluationClaimScope = EvaluationClaimScope.UNKNOWN
    entities: tuple[str, ...] = ()


# 正式摘要事实抽取模型的批量响应。
class SummaryFactDraftList(FrozenStrictModel):
    facts: tuple[SummaryFactDraft, ...] = ()


class EvaluationSummaryFactExtractor:
    """从正式 candidate 0 抽取带摘要 offset 的原子事实。"""

    def __init__(
        self,
        *,
        model_io: EvaluationModelIO,
        config: RunConfig,
    ) -> None:
        """绑定评测模型 I/O 和分段配置，并初始化 warning/分段计数。"""
        self._model_io = model_io
        self._config = config
        self.warnings: list[str] = []
        self.segment_count = 0

    async def extract(self, summary: str) -> tuple[EvaluationSummaryFact, ...]:
        """独立分段抽取摘要 claim，grounding 后按 offset 和文本去重。"""
        bundle = build_raw_material_bundle(
            summary,
            source_id="formal_summary_candidate_0",
            max_segment_chars=self._config.evaluation_segment_max_chars,
            overlap_ratio=self._config.raw_segment_overlap_ratio,
        )
        self.segment_count = len(bundle.segments)
        semaphore = asyncio.Semaphore(
            self._config.evaluation_max_concurrent_segment_maps
        )

        async def map_segment(segment):
            """抽取并 grounding 一个正式摘要分段的原子事实。"""
            async with semaphore:
                result = await self._model_io.complete_json(
                    stage=f"evaluation.summary_facts.{segment.segment_id}",
                    prompt=build_summary_fact_prompt(segment),
                    response_model=SummaryFactDraftList,
                    max_tokens=self._config.evaluation_summary_fact_max_tokens,
                    repair_attempts=self._config.evaluation_repair_attempts,
                    timeout_seconds=self._config.evaluation_model_timeout_seconds,
                )
            facts: list[EvaluationSummaryFact] = []
            rejected = 0
            for item in result.facts:
                quote_match = match_evidence_quote(segment.text, item.evidence_quote)
                if quote_match.grounded_quote is None:
                    rejected += 1
                    continue
                grounded = quote_match.grounded_quote
                start = segment.start_offset + grounded.start
                end = segment.start_offset + grounded.end
                fact_id = "eval_summary_" + hashlib.sha256(
                    f"{start}\x1f{end}\x1f{item.fact}".encode("utf-8")
                ).hexdigest()[:20]
                facts.append(
                    EvaluationSummaryFact(
                        summary_fact_id=fact_id,
                        fact=item.fact.strip(),
                        evidence_quote=grounded.evidence_text,
                        start_offset=start,
                        end_offset=end,
                        claimed_scope=item.claimed_scope,
                        entities=tuple(dict.fromkeys(item.entities)),
                    )
                )
            if rejected:
                self.warnings.append(
                    f"summary segment {segment.segment_id}: discarded {rejected} "
                    "evaluation facts whose quotes could not be grounded"
                )
                if result.facts and not facts:
                    raise ValueError(
                        f"summary segment {segment.segment_id} returned only "
                        "ungrounded evaluation facts"
                    )
            return facts

        mapped = await asyncio.gather(
            *(map_segment(segment) for segment in bundle.segments)
        )
        retained: dict[tuple[int, int, str], EvaluationSummaryFact] = {}
        for fact in [item for group in mapped for item in group]:
            key = (fact.start_offset, fact.end_offset, fact.fact.casefold())
            retained.setdefault(key, fact)
        return tuple(
            sorted(retained.values(), key=lambda item: (item.start_offset, item.end_offset))
        )
