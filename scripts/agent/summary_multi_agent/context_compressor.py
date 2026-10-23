"""仅压缩历史召回等辅助上下文，并验证当前原文不变量未被改变。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .schemas import (
    AgentResult,
    AgentStatus,
    ContextCompressionAction,
    ContextCompressionDecision,
    ContextCompressionMode,
    ContextCompressionReport,
    SourceKind,
    SummaryWorkingState,
)


PROTECTED_MEMORY_TYPES = frozenset(
    {"time", "decision", "constraint", "task", "risk"}
)
CONTROL_PAYLOAD_FIELDS = frozenset(
    {
        "queries",
        "processed_segment_ids",
        "warnings",
        "selection_mode",
        "selected_observation_ids",
        "available_record_ids",
        "requested_memory_ids",
        "selected_memory_ids",
        "equivalent_record_merges",
        "planned_queries",
        "retrieval_complete",
        "incomplete_queries",
        "record_version_conflicts",
        "available_record_version_conflicts",
        "retrieval_stats",
        "available_retrieval_stats",
        "reused_search_count",
        "new_search_count",
        "execution_origin",
        "record_selection_completed",
        "record_selection_origin",
    }
)


@dataclass(frozen=True)
class CompressedAuxiliaryContext:
    """保存供 Synthesis/Review 使用的辅助投影、历史 canonical 记录和审计报告。"""

    worker_results: dict[str, dict[str, Any]]
    historical_records: dict[str, dict[str, Any]]
    evidence_ids: tuple[str, ...]
    report: ContextCompressionReport


@dataclass(frozen=True)
class _StateSnapshot:
    """保存压缩前后必须保持一致的当前材料和 Coverage 指纹。"""

    current_material_sha256: str
    segment_manifest_sha256: str
    coverage_ledger_sha256: str
    current_evidence_offsets_sha256: str


@dataclass(frozen=True)
class _RecordOccurrence:
    """记录一个 Worker 返回的历史 Memory 出现位置及其 Evidence 关联。"""

    record_ref_id: str
    source_id: str
    task_id: str
    record: dict[str, Any]
    evidence_ids: tuple[str, ...]


def _stable_json(value: Any) -> str:
    """以固定键序列化值，保证审计 hash 和 token 估算可重复。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    """计算稳定 JSON 的 SHA-256 指纹。"""
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _estimate_tokens(value: Any) -> int:
    """按 UTF-8 字节近似估算辅助上下文 token 数。"""
    encoded = _stable_json(value).encode("utf-8")
    return math.ceil(len(encoded) / 4) if encoded else 0


def _snapshot_state(state: SummaryWorkingState) -> _StateSnapshot:
    """提取压缩不允许改变的原文、分段、Coverage 和当前 Evidence offset。"""
    segment_manifest = [
        {
            "segment_id": item.segment_id,
            "source_id": item.source_id,
            "ordinal": item.ordinal,
            "start_offset": item.start_offset,
            "end_offset": item.end_offset,
            "content_sha256": item.content_sha256,
        }
        for item in state.raw_material.segments
    ]
    current_evidence_offsets = [
        {
            "evidence_id": evidence_id,
            "source_id": evidence.source_id,
            "segment_id": evidence.segment_id,
            "start_offset": evidence.start_offset,
            "end_offset": evidence.end_offset,
            "evidence_text": evidence.evidence_text,
        }
        for evidence_id, evidence in sorted(state.evidence_index.items())
        if evidence.source_kind == SourceKind.CURRENT_MATERIAL
    ]
    return _StateSnapshot(
        current_material_sha256=hashlib.sha256(
            state.request.current_material.encode("utf-8")
        ).hexdigest(),
        segment_manifest_sha256=_sha256_json(segment_manifest),
        coverage_ledger_sha256=_sha256_json(
            state.coverage_ledger.model_dump(mode="json")
        ),
        current_evidence_offsets_sha256=_sha256_json(current_evidence_offsets),
    )


def _compact_memory_record(record: dict[str, Any]) -> dict[str, Any]:
    """保留 Memory 身份、正文、分数和受控 metadata，去掉无关字段。"""
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    compact_metadata = {
        key: metadata.get(key)
        for key in (
            "typed_memory_id",
            "memory_type",
            "memory_status",
            "memory_confidence",
            "source_time",
            "source_file",
            "evidence_span",
            "entities_text",
            "topics_text",
            "structured_fields_json",
            "history_source_id",
        )
        if metadata.get(key) not in (None, "")
    }
    return {
        "memory_id": record.get("memory_id"),
        "memory_type": record.get("memory_type"),
        "content": record.get("content"),
        "score": record.get("score"),
        "metadata": compact_metadata,
        **(
            {"matched_entities": record.get("matched_entities")}
            if record.get("matched_entities")
            else {}
        ),
        **(
            {"related_entities": record.get("related_entities")}
            if record.get("related_entities")
            else {}
        ),
    }


def compress_memory_records(
    records: list[dict[str, Any]],
    *,
    top_k_per_type: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """兼容旧调用的单列表确定性 Memory 去重和按类型排序辅助函数。"""

    if top_k_per_type < 1:
        raise ValueError("top_k_per_type must be positive")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        memory_id = str(record.get("memory_id") or "").strip()
        if not memory_id:
            continue
        current = by_id.get(memory_id)
        if current is None or float(record.get("score") or 0.0) > float(
            current.get("score") or 0.0
        ):
            by_id[memory_id] = record

    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in by_id.values():
        memory_type = str(record.get("memory_type") or "unknown")
        by_type.setdefault(memory_type, []).append(record)
    retained: list[dict[str, Any]] = []
    retained_by_type: dict[str, int] = {}
    input_by_type: dict[str, int] = {}
    for memory_type in sorted(by_type):
        ranked = sorted(
            by_type[memory_type],
            key=lambda item: (
                -float(item.get("score") or 0.0),
                str(item.get("memory_id") or ""),
            ),
        )
        input_by_type[memory_type] = len(ranked)
        selected = ranked[:top_k_per_type]
        retained_by_type[memory_type] = len(selected)
        retained.extend(selected)
    return retained, {
        "input_count": len(records),
        "deduplicated_count": len(by_id),
        "retained_count": len(retained),
        "input_by_type": input_by_type,
        "retained_by_type": retained_by_type,
        "top_k_per_type": top_k_per_type,
    }


class AuxiliaryContextCompressor:
    """仅处理 Worker/历史辅助上下文，并用前后快照保护当前原文不变量。"""

    def __init__(
        self,
        *,
        mode: ContextCompressionMode,
        top_k_per_type: int,
    ) -> None:
        """设置压缩模式和未保护 Memory 类型的 top-k 上限。"""
        if top_k_per_type < 1:
            raise ValueError("top_k_per_type must be positive")
        self._mode = mode
        self._top_k_per_type = top_k_per_type

    def compress(self, state: SummaryWorkingState) -> CompressedAuxiliaryContext:
        """生成 passthrough 或 deterministic 辅助投影，不修改传入 M4 状态。

        当前材料 hash、分段 manifest、Coverage Ledger 和当前 Evidence offset 在
        前后必须完全一致；不变量破坏会抛出 ``RuntimeError``。
        """
        before = _snapshot_state(state)
        successful_results = {
            task_id: result
            for task_id, result in sorted(state.agent_results.items())
            if result.status == AgentStatus.SUCCESS
        }
        input_region = {
            task_id: result.model_dump(mode="json")
            for task_id, result in successful_results.items()
        }
        if self._mode == ContextCompressionMode.DISABLED:
            worker_results, evidence_ids, decisions, counts = self._passthrough(
                successful_results
            )
            historical_records: dict[str, dict[str, Any]] = {}
        else:
            (
                worker_results,
                historical_records,
                evidence_ids,
                decisions,
                counts,
            ) = self._compress_deterministically(successful_results)

        # 无论使用哪种模式，都在输出前重新取快照，确保压缩只改变辅助投影。
        after = _snapshot_state(state)
        if before != after:
            raise RuntimeError(
                "context compression changed current material, segment coverage, "
                "or current evidence offsets"
            )
        if before.current_material_sha256 != state.raw_material.content_sha256:
            raise RuntimeError(
                "request current material hash differs from RawMaterialBundle"
            )

        output_region = {
            "worker_results": worker_results,
            "historical_records": historical_records,
            "evidence_index": {
                evidence_id: state.evidence_index[evidence_id].model_dump(mode="json")
                for evidence_id in evidence_ids
                if evidence_id in state.evidence_index
            },
        }
        input_chars = len(_stable_json(input_region))
        output_chars = len(_stable_json(output_region))
        warnings: list[str] = list(counts.pop("warnings", ()))
        if output_chars > input_chars:
            warnings.append(
                "compressed auxiliary projection is larger than its input region"
            )
        report = ContextCompressionReport(
            mode=self._mode,
            input_chars=input_chars,
            output_chars=output_chars,
            estimated_input_tokens=_estimate_tokens(input_region),
            estimated_output_tokens=_estimate_tokens(output_region),
            compression_ratio=(output_chars / input_chars if input_chars else 1.0),
            current_material_sha256_before=before.current_material_sha256,
            current_material_sha256_after=after.current_material_sha256,
            segment_manifest_sha256_before=before.segment_manifest_sha256,
            segment_manifest_sha256_after=after.segment_manifest_sha256,
            coverage_ledger_sha256_before=before.coverage_ledger_sha256,
            coverage_ledger_sha256_after=after.coverage_ledger_sha256,
            current_evidence_offsets_sha256_before=(
                before.current_evidence_offsets_sha256
            ),
            current_evidence_offsets_sha256_after=(
                after.current_evidence_offsets_sha256
            ),
            decisions=tuple(decisions),
            warnings=tuple(dict.fromkeys(warnings)),
            **counts,
        )
        return CompressedAuxiliaryContext(
            worker_results=worker_results,
            historical_records=historical_records,
            evidence_ids=evidence_ids,
            report=report,
        )

    def _passthrough(
        self, results: dict[str, AgentResult]
    ) -> tuple[
        dict[str, dict[str, Any]],
        tuple[str, ...],
        list[ContextCompressionDecision],
        dict[str, Any],
    ]:
        """不做跨 Worker 归并，只压缩单个 payload 的冗余字段并保留所有记录。"""
        worker_results: dict[str, dict[str, Any]] = {}
        evidence_ids: list[str] = []
        decisions: list[ContextCompressionDecision] = []
        record_count = 0
        unique_record_ids: set[str] = set()
        protected_record_ids: set[str] = set()
        for task_id, result in results.items():
            payload = copy.deepcopy(result.payload)
            records = payload.get("records")
            if isinstance(records, list):
                compacted = []
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        continue
                    record_count += 1
                    memory_id = str(record.get("memory_id") or "").strip()
                    source_id = f"{task_id}:record:{index}:{memory_id or 'unkeyed'}"
                    unique_record_ids.add(memory_id or source_id)
                    memory_type = str(record.get("memory_type") or "unknown")
                    if memory_type in PROTECTED_MEMORY_TYPES:
                        protected_record_ids.add(memory_id or source_id)
                    compacted.append(_compact_memory_record(record))
                    decisions.append(
                        ContextCompressionDecision(
                            item_kind="memory_record",
                            source_ids=(source_id,),
                            evidence_ids=self._record_evidence_ids(result, memory_id),
                            action=ContextCompressionAction.RETAIN,
                            reason="compression is disabled; record retained in task payload",
                            retained_id=memory_id or source_id,
                            memory_type=memory_type,
                        )
                    )
                payload["records"] = compacted
            worker_results[task_id] = self._project_result(
                result, payload=payload, evidence_ids=tuple(
                    item.evidence_id for item in result.evidence
                )
            )
            evidence_ids.extend(item.evidence_id for item in result.evidence)
            decisions.extend(self._projection_decisions(task_id, result))
        return (
            worker_results,
            tuple(dict.fromkeys(evidence_ids)),
            decisions,
            {
                "input_record_count": record_count,
                "unique_record_count": len(unique_record_ids),
                "retained_record_count": record_count,
                "merged_record_count": 0,
                "dropped_record_count": 0,
                "protected_record_count": len(protected_record_ids),
                "warnings": (),
            },
        )

    def _compress_deterministically(
        self, results: dict[str, AgentResult]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        tuple[str, ...],
        list[ContextCompressionDecision],
        dict[str, Any],
    ]:
        """按 Memory identity 全局归并、保护关键类型并应用未保护类型 top-k。"""
        payloads: dict[str, dict[str, Any]] = {}
        occurrences: dict[str, list[_RecordOccurrence]] = {}
        decisions: list[ContextCompressionDecision] = []
        warnings: list[str] = []
        input_record_count = 0

        # 先复制并拆出 records，再进行全局 canonical 合并，避免原地污染 M4 AgentResult。
        for task_id, result in results.items():
            payload = copy.deepcopy(result.payload)
            records = payload.pop("records", [])
            if isinstance(records, list):
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        continue
                    input_record_count += 1
                    memory_id = str(record.get("memory_id") or "").strip()
                    if memory_id:
                        record_ref_id = memory_id
                    else:
                        record_ref_id = "unkeyed_" + hashlib.sha256(
                            _stable_json(_compact_memory_record(record)).encode("utf-8")
                        ).hexdigest()[:20]
                    source_id = f"{task_id}:record:{index}:{record_ref_id}"
                    occurrences.setdefault(record_ref_id, []).append(
                        _RecordOccurrence(
                            record_ref_id=record_ref_id,
                            source_id=source_id,
                            task_id=task_id,
                            record=record,
                            evidence_ids=self._record_evidence_ids(result, memory_id),
                        )
                    )
            payloads[task_id] = self._compact_control_payload(
                task_id, payload, decisions
            )
            decisions.extend(self._projection_decisions(task_id, result))

        canonical = {
            record_ref_id: self._merge_occurrences(record_ref_id, items, warnings)
            for record_ref_id, items in occurrences.items()
        }
        retained_ids = self._select_record_ids(canonical)
        historical_records = {
            record_ref_id: canonical[record_ref_id]
            for record_ref_id in sorted(retained_ids)
        }
        record_ids_by_task: dict[str, list[str]] = {
            task_id: [] for task_id in results
        }
        merged_count = 0
        dropped_count = 0
        for record_ref_id, items in sorted(occurrences.items()):
            memory_type = str(
                canonical[record_ref_id].get("memory_type") or "unknown"
            )
            all_evidence = tuple(
                dict.fromkeys(
                    evidence_id
                    for item in items
                    for evidence_id in item.evidence_ids
                )
            )
            if record_ref_id not in retained_ids:
                dropped_count += len(items)
                for item in items:
                    decisions.append(
                        ContextCompressionDecision(
                            item_kind="memory_record",
                            source_ids=(item.source_id,),
                            evidence_ids=item.evidence_ids,
                            action=ContextCompressionAction.DROP,
                            reason=(
                                "unprotected memory type exceeded deterministic "
                                "top-k budget"
                            ),
                            memory_type=memory_type,
                        )
                    )
                continue

            for task_id in dict.fromkeys(item.task_id for item in items):
                record_ids_by_task[task_id].append(record_ref_id)
            first, *duplicates = items
            decisions.append(
                ContextCompressionDecision(
                    item_kind="memory_record",
                    source_ids=(first.source_id,),
                    evidence_ids=all_evidence,
                    action=ContextCompressionAction.RETAIN,
                    reason=(
                        "unique protected record retained"
                        if memory_type in PROTECTED_MEMORY_TYPES
                        else "record retained within deterministic top-k budget"
                    ),
                    retained_id=record_ref_id,
                    memory_type=memory_type,
                )
            )
            for item in duplicates:
                merged_count += 1
                decisions.append(
                    ContextCompressionDecision(
                        item_kind="memory_record",
                        source_ids=(item.source_id,),
                        evidence_ids=item.evidence_ids,
                        action=ContextCompressionAction.MERGE,
                        reason="same memory identity merged into canonical record",
                        retained_id=record_ref_id,
                        memory_type=memory_type,
                    )
                )

        worker_results: dict[str, dict[str, Any]] = {}
        retained_evidence_ids: list[str] = []
        for task_id, result in results.items():
            payload = payloads[task_id]
            task_record_ids = tuple(dict.fromkeys(record_ids_by_task[task_id]))
            if task_record_ids:
                payload["record_ids"] = list(task_record_ids)
            evidence_ids = tuple(
                item.evidence_id
                for item in result.evidence
                if item.memory_id is None or item.memory_id in retained_ids
            )
            retained_evidence_ids.extend(evidence_ids)
            worker_results[task_id] = self._project_result(
                result, payload=payload, evidence_ids=evidence_ids
            )

        return (
            worker_results,
            historical_records,
            tuple(dict.fromkeys(retained_evidence_ids)),
            decisions,
            {
                "input_record_count": input_record_count,
                "unique_record_count": len(canonical),
                "retained_record_count": len(retained_ids),
                "merged_record_count": merged_count,
                "dropped_record_count": dropped_count,
                "protected_record_count": sum(
                    1
                    for item in canonical.values()
                    if str(item.get("memory_type") or "unknown")
                    in PROTECTED_MEMORY_TYPES
                ),
                "warnings": tuple(warnings),
            },
        )

    def _select_record_ids(
        self, canonical: dict[str, dict[str, Any]]
    ) -> set[str]:
        """选择保留的 Memory ID；time/decision/constraint/task/risk 永不截断。"""
        retained: set[str] = set()
        by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for record_ref_id, record in canonical.items():
            memory_type = str(record.get("memory_type") or "unknown")
            by_type.setdefault(memory_type, []).append((record_ref_id, record))
        for memory_type, records in by_type.items():
            ranked = sorted(
                records,
                key=lambda item: (
                    -float(item[1].get("score") or 0.0),
                    item[0],
                ),
            )
            if memory_type in PROTECTED_MEMORY_TYPES:
                retained.update(record_ref_id for record_ref_id, _ in ranked)
            else:
                retained.update(
                    record_ref_id
                    for record_ref_id, _ in ranked[: self._top_k_per_type]
                )
        return retained

    @staticmethod
    def _merge_occurrences(
        record_ref_id: str,
        items: list[_RecordOccurrence],
        warnings: list[str],
    ) -> dict[str, Any]:
        """将同一 Memory identity 的多个 Worker 结果合并为可审计 canonical 记录。"""
        ranked = sorted(
            items,
            key=lambda item: -float(item.record.get("score") or 0.0),
        )
        compacted = [_compact_memory_record(item.record) for item in ranked]
        merged = copy.deepcopy(compacted[0])
        merged["record_ref_id"] = record_ref_id
        merged["source_task_ids"] = list(
            dict.fromkeys(item.task_id for item in items)
        )
        merged["evidence_ids"] = list(
            dict.fromkeys(
                evidence_id
                for item in items
                for evidence_id in item.evidence_ids
            )
        )
        for field in ("matched_entities", "related_entities"):
            values = [
                str(value)
                for record in compacted
                for value in record.get(field, [])
                if str(value).strip()
            ]
            if values:
                merged[field] = list(dict.fromkeys(values))
        contents = list(
            dict.fromkeys(
                str(record.get("content") or "").strip()
                for record in compacted
                if str(record.get("content") or "").strip()
            )
        )
        if len(contents) > 1:
            merged["content_variants"] = contents
            warnings.append(
                f"memory {record_ref_id} had conflicting content variants; "
                "all variants were preserved"
            )
        structural_variants: list[dict[str, Any]] = []
        seen_structural_variants: set[str] = set()
        for record in compacted:
            variant = {
                "memory_type": record.get("memory_type"),
                "content": record.get("content"),
                "metadata": record.get("metadata", {}),
            }
            fingerprint = _stable_json(variant)
            if fingerprint in seen_structural_variants:
                continue
            seen_structural_variants.add(fingerprint)
            structural_variants.append(variant)
        if len(structural_variants) > 1:
            merged["record_variants"] = structural_variants
            warnings.append(
                f"memory {record_ref_id} had conflicting structural variants; "
                "all variants were preserved"
            )
        return merged

    @staticmethod
    def _compact_control_payload(
        task_id: str,
        payload: dict[str, Any],
        decisions: list[ContextCompressionDecision],
    ) -> dict[str, Any]:
        """移除仍可从 M4/Trace 追溯、但 Synthesis 不需要的控制元数据。"""
        for field in tuple(payload):
            if field not in CONTROL_PAYLOAD_FIELDS:
                continue
            payload.pop(field, None)
            decisions.append(
                ContextCompressionDecision(
                    item_kind="control_payload_field",
                    source_ids=(f"{task_id}:payload:{field}",),
                    action=ContextCompressionAction.DROP,
                    reason=(
                        "control or coverage metadata remains auditable in M4/Trace "
                        "and is not needed for synthesis"
                    ),
                )
            )
        expressions = payload.get("expressions")
        resolved = payload.get("resolved")
        if isinstance(expressions, list) and isinstance(resolved, list) and resolved:
            resolved_keys = {
                AuxiliaryContextCompressor._temporal_item_key(item)
                for item in resolved
                if isinstance(item, dict)
            }
            unresolved = [
                item
                for item in expressions
                if not isinstance(item, dict)
                or AuxiliaryContextCompressor._temporal_item_key(item)
                not in resolved_keys
            ]
            removed = len(expressions) - len(unresolved)
            if removed:
                payload["expressions"] = unresolved
                decisions.append(
                    ContextCompressionDecision(
                        item_kind="temporal_expression",
                        source_ids=(f"{task_id}:payload:expressions",),
                        action=ContextCompressionAction.MERGE,
                        reason=(
                            f"{removed} resolved expressions were already represented "
                            "by resolved temporal items"
                        ),
                        retained_id=f"{task_id}:payload:resolved",
                    )
                )
        return payload

    @staticmethod
    def _temporal_item_key(item: dict[str, Any]) -> tuple[Any, ...]:
        """生成时间表达式与解析结果的稳定匹配键。"""
        return (
            item.get("segment_id"),
            item.get("start_offset"),
            item.get("end_offset"),
            item.get("expression"),
            item.get("direction"),
        )

    @staticmethod
    def _project_result(
        result: AgentResult,
        *,
        payload: dict[str, Any],
        evidence_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """投影 AgentResult 的合成所需字段，保留状态、warning 和依赖 lineage。"""
        return {
            "task_type": result.task_type.value,
            "status": result.status.value,
            "is_empty": result.is_empty,
            "payload": payload,
            "evidence_ids": list(evidence_ids),
            "warnings": list(result.warnings),
            "consumed_result_ids": list(result.consumed_result_ids),
            "consumed_evidence_ids": list(result.consumed_evidence_ids),
        }

    @staticmethod
    def _record_evidence_ids(
        result: AgentResult, memory_id: str
    ) -> tuple[str, ...]:
        """取得某个 Memory ID 在当前 Worker 结果中的 Evidence ID。"""
        if not memory_id:
            return ()
        return tuple(
            item.evidence_id
            for item in result.evidence
            if item.memory_id == memory_id
        )

    @staticmethod
    def _projection_decisions(
        task_id: str, result: AgentResult
    ) -> list[ContextCompressionDecision]:
        """记录被排除的原始 Tool/Worker 审计，并说明其仍保留在 M4/Trace。"""
        decisions: list[ContextCompressionDecision] = []
        if result.tool_results:
            decisions.append(
                ContextCompressionDecision(
                    item_kind="tool_results",
                    source_ids=tuple(
                        item.call_id for item in result.tool_results
                    ),
                    evidence_ids=tuple(
                        evidence.evidence_id
                        for evidence in result.evidence
                        if evidence.tool_call_id
                    ),
                    action=ContextCompressionAction.DROP,
                    reason=(
                        "raw tool payloads remain in AgentResult/Trace; grounded payload "
                        "and Evidence are retained for synthesis"
                    ),
                )
            )
        if result.worker_execution is not None:
            decisions.append(
                ContextCompressionDecision(
                    item_kind="worker_execution",
                    source_ids=(f"{task_id}:worker_execution",),
                    action=ContextCompressionAction.DROP,
                    reason=(
                        "model attempts and action trace remain in AgentResult/Trace "
                        "and are excluded from synthesis context"
                    ),
                )
            )
        return decisions
