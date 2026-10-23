"""构造不可变原文分段，保证字符偏移、哈希和分段覆盖可回溯。"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from .schemas import CoverageLedger, RawMaterialBundle, RawSegment, SegmentCoverage


class IncompleteCoverageError(RuntimeError):
    """当运行试图在仍有未处理原文分段时完成时抛出。"""


def build_raw_material_bundle(
    raw_text: str,
    *,
    source_id: Optional[str] = None,
    max_segment_chars: int = 8000,
    overlap_ratio: float = 0.05,
    overlap_chars: Optional[int] = None,
) -> RawMaterialBundle:
    """把完整当前原文切成可回溯分段，并计算全局与分段哈希。

    返回的分段使用稳定字符 offset 和可选重叠；这些信息供当前事实、Evidence
    Lineage 和 Coverage Ledger 共同引用。函数只在内存中构造数据，不写文件。
    原文为空、分段大小或重叠参数非法时抛出 ``ValueError``。
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must contain non-whitespace text")
    if max_segment_chars <= 0:
        raise ValueError("max_segment_chars must be positive")
    if not 0.0 <= overlap_ratio < 0.5:
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")

    material_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    resolved_source_id = source_id or f"current_{material_hash[:16]}"
    resolved_overlap = (
        int(max_segment_chars * overlap_ratio)
        if overlap_chars is None
        else overlap_chars
    )
    if resolved_overlap < 0 or resolved_overlap >= max_segment_chars:
        raise ValueError("overlap_chars must be in [0, max_segment_chars)")

    segments: list[RawSegment] = []
    start = 0
    ordinal = 0
    while start < len(raw_text):
        end = min(start + max_segment_chars, len(raw_text))
        text = raw_text[start:end]
        segment_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        segments.append(
            RawSegment(
                segment_id=f"seg_{ordinal:04d}_{segment_hash[:12]}",
                source_id=resolved_source_id,
                ordinal=ordinal,
                start_offset=start,
                end_offset=end,
                text=text,
                content_sha256=segment_hash,
            )
        )
        if end == len(raw_text):
            break
        start = end - resolved_overlap
        ordinal += 1

    return RawMaterialBundle(
        source_id=resolved_source_id,
        raw_text=raw_text,
        content_sha256=material_hash,
        total_chars=len(raw_text),
        max_segment_chars=max_segment_chars,
        overlap_chars=resolved_overlap,
        segments=tuple(segments),
    )


def build_coverage_ledger(
    bundle: RawMaterialBundle,
    *,
    required_stages: Iterable[str],
) -> CoverageLedger:
    """为每个原文分段建立必须完成的处理阶段清单。

    阶段名称会去空白并保持首次出现顺序；返回的新账本由后续阶段通过不可变
    ``model_copy`` 更新，不能用它绕过原文分段覆盖检查。
    """
    stages = tuple(dict.fromkeys(stage.strip() for stage in required_stages if stage.strip()))
    if not stages:
        raise ValueError("required_stages must contain at least one stage")
    return CoverageLedger(
        source_id=bundle.source_id,
        segments=tuple(
            SegmentCoverage(segment_id=segment.segment_id, required_stages=stages)
            for segment in bundle.segments
        ),
    )


def mark_segment_processed(
    ledger: CoverageLedger,
    *,
    segment_id: str,
    stage: str,
    evidence_ids: Iterable[str] = (),
) -> CoverageLedger:
    """标记一个分段在指定阶段成功，并合并该阶段产生的 Evidence ID。

    该函数不原地修改账本；未知分段或未声明阶段分别抛出 ``KeyError`` 和
    ``ValueError``，从而避免审计产物出现无法解释的覆盖状态。
    """
    if not stage.strip():
        raise ValueError("stage must be non-empty")
    updated: list[SegmentCoverage] = []
    found = False
    for item in ledger.segments:
        if item.segment_id != segment_id:
            updated.append(item)
            continue
        found = True
        if stage not in item.required_stages:
            raise ValueError(f"stage {stage!r} is not required for segment {segment_id}")
        completed = tuple(dict.fromkeys((*item.completed_stages, stage)))
        merged_evidence = tuple(dict.fromkeys((*item.evidence_ids, *evidence_ids)))
        failures = dict(item.failed_stages)
        failures.pop(stage, None)
        updated.append(
            item.model_copy(
                update={
                    "completed_stages": completed,
                    "failed_stages": failures,
                    "evidence_ids": merged_evidence,
                }
            )
        )
    if not found:
        raise KeyError(f"unknown segment_id: {segment_id}")
    return ledger.model_copy(update={"segments": tuple(updated)})


def mark_segment_failed(
    ledger: CoverageLedger,
    *,
    segment_id: str,
    stage: str,
    error: str,
) -> CoverageLedger:
    """记录分段阶段失败并移除该阶段的成功标记。

    失败文本会进入 Coverage Ledger，最终由 ``assert_complete_coverage`` 阻止
    无警告完成，而不是让单个失败分段静默消失。
    """
    if not error.strip():
        raise ValueError("error must be non-empty")
    updated: list[SegmentCoverage] = []
    found = False
    for item in ledger.segments:
        if item.segment_id != segment_id:
            updated.append(item)
            continue
        found = True
        if stage not in item.required_stages:
            raise ValueError(f"stage {stage!r} is not required for segment {segment_id}")
        failures = dict(item.failed_stages)
        failures[stage] = error
        completed = tuple(value for value in item.completed_stages if value != stage)
        updated.append(
            item.model_copy(
                update={"completed_stages": completed, "failed_stages": failures}
            )
        )
    if not found:
        raise KeyError(f"unknown segment_id: {segment_id}")
    return ledger.model_copy(update={"segments": tuple(updated)})


def assert_complete_coverage(ledger: CoverageLedger) -> None:
    """在进入摘要合成前强制确认所有分段完成必需阶段。

    覆盖不完整时抛出 ``IncompleteCoverageError``，并附带每个缺失阶段，保证
    当前原文不能被隐式截断或以部分结果替代。
    """
    if ledger.is_complete:
        return
    details = []
    for item in ledger.segments:
        missing = sorted(set(item.required_stages) - set(item.completed_stages))
        if missing:
            details.append(f"{item.segment_id}: {', '.join(missing)}")
    raise IncompleteCoverageError(
        "raw material coverage is incomplete: " + "; ".join(details)
    )
