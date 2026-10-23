"""组装不可压缩的当前事实与可压缩的 Worker/历史辅助上下文。"""

from __future__ import annotations

from pydantic import ValidationError

from .context_compressor import (
    AuxiliaryContextCompressor,
    compress_memory_records,
)
from .raw_material import mark_segment_processed
from .schemas import (
    ContextCompressionMode,
    CurrentFactItem,
    SynthesisContext,
    SummaryWorkingState,
    TaskType,
)
from .tools.template import TemplateAdaptationInput, TemplateAdaptationOutput, resolve_template


def _merge_current_items(state: SummaryWorkingState) -> tuple[CurrentFactItem, ...]:
    """按原文顺序合并当前事实，并以类别、文本和 offset 去重。"""
    merged: list[CurrentFactItem] = []
    seen: set[tuple[str, str, int, int]] = set()
    for analysis in sorted(
        state.current_segment_analyses, key=lambda item: item.start_offset
    ):
        for item in analysis.items:
            evidence = state.evidence_index.get(item.evidence_id)
            if evidence is None:
                raise ValueError(f"missing current evidence: {item.evidence_id}")
            key = (
                item.category.value,
                item.statement.strip(),
                int(evidence.start_offset or 0),
                int(evidence.end_offset or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return tuple(merged)


def build_synthesis_context(
    state: SummaryWorkingState,
    *,
    compression_mode: ContextCompressionMode = ContextCompressionMode.DETERMINISTIC,
    compression_top_k_per_type: int = 10,
) -> SynthesisContext:
    """组装 Summarizer 所需上下文并写回压缩审计状态。

    当前事实和其 Evidence 必须完整覆盖全部分段；只有 Worker/历史结果作为
    辅助上下文交给 Compressor。压缩失败或 offset/coverage 不一致会抛出异常，
    从而禁止用不完整上下文生成正式摘要。
    """
    expected_segment_ids = {
        segment.segment_id for segment in state.raw_material.segments
    }
    actual_segment_ids = {
        analysis.segment_id for analysis in state.current_segment_analyses
    }
    mapper_required = any(
        "current_fact_map" in item.required_stages
        for item in state.coverage_ledger.segments
    )
    if mapper_required and actual_segment_ids != expected_segment_ids:
        missing = sorted(expected_segment_ids - actual_segment_ids)
        extra = sorted(actual_segment_ids - expected_segment_ids)
        raise ValueError(
            f"segment analysis coverage mismatch; missing={missing}, extra={extra}"
        )

    coverage_by_id = {
        segment.segment_id: segment for segment in state.coverage_ledger.segments
    }
    raw_segment_by_id = {
        segment.segment_id: segment for segment in state.raw_material.segments
    }
    analysis_by_id = {
        analysis.segment_id: analysis for analysis in state.current_segment_analyses
    }
    if not mapper_required and actual_segment_ids:
        raise ValueError("current fact map was not requested but contains analyses")
    for segment_id in expected_segment_ids:
        if not mapper_required:
            continue
        if "current_fact_map" not in coverage_by_id[segment_id].completed_stages:
            raise ValueError(f"current_fact_map is incomplete for {segment_id}")
        raw_segment = raw_segment_by_id[segment_id]
        analysis = analysis_by_id[segment_id]
        if (
            analysis.start_offset != raw_segment.start_offset
            or analysis.end_offset != raw_segment.end_offset
        ):
            raise ValueError(f"segment analysis offsets do not match {segment_id}")

    # Compressor 只能投影辅助上下文；当前原文、当前事实和 Coverage 仍由 state 保持。
    compressed = AuxiliaryContextCompressor(
        mode=compression_mode,
        top_k_per_type=compression_top_k_per_type,
    ).compress(state)
    worker_results = compressed.worker_results
    template_payload = None
    template_selection: dict[str, object] = {}
    for task_id, result in worker_results.items():
        if result.get("task_type") != TaskType.TEMPLATE_ADAPTATION.value:
            continue
        payload = result.get("payload", {})
        try:
            TemplateAdaptationOutput.model_validate(payload)
        except ValidationError:
            continue
        template_payload = dict(payload)
        template_selection = {"source": "template_worker", "task_id": task_id}
        break
    if template_payload is None:
        template_payload = resolve_template(
            TemplateAdaptationInput(
                summary_request=state.request.summary_request,
                preferences=state.request.personalization_preferences,
            )
        ).model_dump(mode="json")
        template_planned = state.plan is not None and any(
            task.task_type == TaskType.TEMPLATE_ADAPTATION for task in state.plan.tasks
        )
        template_selection = {
            "source": "default_unavailable" if template_planned else "default_not_planned",
            "task_id": None,
        }
    merged_items = _merge_current_items(state)
    synthesis_evidence_ids = {item.evidence_id for item in merged_items}
    synthesis_evidence_ids.update(compressed.evidence_ids)
    synthesis_evidence_index = {
        evidence_id: state.evidence_index[evidence_id]
        for evidence_id in sorted(synthesis_evidence_ids)
        if evidence_id in state.evidence_index
    }
    context = SynthesisContext(
        request_id=state.request.request_id,
        source_time=state.request.source_time,
        summary_request=state.request.summary_request,
        target_entities=tuple(state.request.target_entities),
        personalization_preferences=state.request.personalization_preferences,
        segment_analyses=tuple(
            sorted(state.current_segment_analyses, key=lambda item: item.start_offset)
        ),
        merged_current_items=merged_items,
        worker_results=worker_results,
        historical_records=compressed.historical_records,
        evidence_index=synthesis_evidence_index,
        template=template_payload,
        auxiliary_context_info={
            "current_fact_map_status": "completed" if mapper_required else "not_requested",
            "template_selection": template_selection,
            "policy": (
                "canonicalize historical records globally; retain all unique "
                "time/decision/constraint/task/risk records and bound only "
                "unprotected types"
            ),
            "current_material_compressed": False,
            "mode": compressed.report.mode.value,
            "input_chars": compressed.report.input_chars,
            "output_chars": compressed.report.output_chars,
            "estimated_input_tokens": compressed.report.estimated_input_tokens,
            "estimated_output_tokens": compressed.report.estimated_output_tokens,
            "compression_ratio": compressed.report.compression_ratio,
            "invariants_preserved": compressed.report.invariants_preserved,
        },
        warnings=tuple(dict.fromkeys(state.warnings)),
    )

    ledger = state.coverage_ledger
    # 合成阶段只在上下文已完成校验后登记，避免“有摘要但无分段覆盖”落盘。
    for segment in state.raw_material.segments:
        ledger = mark_segment_processed(
            ledger,
            segment_id=segment.segment_id,
            stage="synthesis_merge",
            evidence_ids=(
                item.evidence_id
                for analysis in state.current_segment_analyses
                if analysis.segment_id == segment.segment_id
                for item in analysis.items
            ),
        )
    state.coverage_ledger = ledger
    state.context_compression_report = compressed.report
    state.synthesis_context = context
    return context
