"""将模型生成的 Evidence quote 精确回映射到当前原文连续区间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GroundedQuote:
    """已映射到当前原文连续区间的引文及其匹配策略。"""

    start: int
    end: int
    evidence_text: str
    strategy: str


@dataclass(frozen=True)
class GroundingMatch:
    """表示引文 grounding 成功，或保存可审计的拒绝原因。"""

    grounded_quote: Optional[GroundedQuote]
    reason: Optional[str] = None


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    """生成仅忽略展示差异的文本，同时保留回到原文的字符 offset。"""
    normalized_chars: list[str] = []
    source_offsets: list[int] = []
    for source_offset, character in enumerate(text):
        if character.isspace() or character in {"\u200b", "\ufeff"}:
            continue
        codepoint = ord(character)
        normalized_character = (
            chr(codepoint - 0xFEE0)
            if 0xFF01 <= codepoint <= 0xFF5E
            else character
        )
        normalized_chars.append(normalized_character)
        source_offsets.append(source_offset)
    return "".join(normalized_chars), source_offsets


def normalize_presentation_text(text: str) -> str:
    """移除空白和展示层全角差异，用于比较而不改变原始引文。"""
    normalized, _ = _normalized_with_offsets(text)
    return normalized


def _all_occurrences(text: str, needle: str) -> list[int]:
    """返回 needle 在规范化文本中的全部起点，用于检测匹配歧义。"""
    starts: list[int] = []
    offset = 0
    while True:
        start = text.find(needle, offset)
        if start < 0:
            return starts
        starts.append(start)
        offset = start + 1


def match_evidence_quote(source_text: str, evidence_quote: str) -> GroundingMatch:
    """在不做语义模糊匹配的前提下把引文映射回原文。

    fallback 只忽略展示层 Unicode 和空白差异；任何接受的匹配都必须能回到一段
    精确、连续的原文区间。空引文、无匹配或多重匹配会返回带原因的失败结果。
    """

    quote = evidence_quote.strip()
    if not quote:
        return GroundingMatch(None, "empty_quote")

    exact_start = source_text.find(quote)
    if exact_start >= 0:
        exact_end = exact_start + len(quote)
        return GroundingMatch(
            GroundedQuote(
                start=exact_start,
                end=exact_end,
                evidence_text=source_text[exact_start:exact_end],
                strategy="exact",
            )
        )

    normalized_source, source_offsets = _normalized_with_offsets(source_text)
    normalized_quote, _ = _normalized_with_offsets(quote)
    if not normalized_quote:
        return GroundingMatch(None, "empty_quote_after_format_normalization")

    occurrences = _all_occurrences(normalized_source, normalized_quote)
    if not occurrences:
        return GroundingMatch(None, "no_format_equivalent_source_span")
    if len(occurrences) > 1:
        return GroundingMatch(None, "ambiguous_format_equivalent_source_span")

    normalized_start = occurrences[0]
    normalized_end = normalized_start + len(normalized_quote)
    source_start = source_offsets[normalized_start]
    source_end = source_offsets[normalized_end - 1] + 1
    return GroundingMatch(
        GroundedQuote(
            start=source_start,
            end=source_end,
            evidence_text=source_text[source_start:source_end],
                strategy="unique_width_whitespace_normalized",
        )
    )
