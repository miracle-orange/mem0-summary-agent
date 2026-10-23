"""按目标实体检索直接证据和一跳关系的规则 Worker。"""

from __future__ import annotations

import time
from typing import Any

from ..schemas import (
    AgentResult,
    AgentStatus,
    EvidenceRef,
    SourceKind,
    TaskType,
    ToolCall,
    ToolStatus,
)
from ..tools.registry import ToolContext
from .base import BaseWorker, WorkerContext


class EntityPersonalizationWorker(BaseWorker):
    """围绕显式目标实体检索长期证据，并在完整当前原文中定位 mentions。"""

    task_type = TaskType.ENTITY_PERSONALIZATION

    async def execute(self, context: WorkerContext) -> AgentResult:
        """执行目标实体查询并合并当前材料证据与一跳关系结果。

        没有目标实体时返回合法空成功；不会从职位、上下文或相似实体猜测身份。
        """
        self.validate_context(context)
        started = time.perf_counter()
        targets = tuple(
            dict.fromkeys(
                value.strip() for value in context.request.target_entities if value.strip()
            )
        )
        if not targets:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=AgentStatus.SUCCESS,
                is_empty=True,
                payload={"target_entities": [], "records": [], "current_mentions": []},
                warnings=("entity personalization skipped because no target entity was supplied",),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        call = ToolCall(
            tool_name="entity_query",
            arguments={
                "entities": targets,
                "include_one_hop": bool(
                    context.task.inputs.get("include_one_hop", True)
                ),
                "top_k_per_entity": int(
                    context.task.inputs.get("top_k_per_entity") or 10
                ),
            },
            timeout_seconds=min(context.task.timeout_seconds, 120),
        )
        tool_result = await context.tool_registry.execute(
            call,
            context=ToolContext(
                run_id=context.run_id,
                task_id=context.task.task_id,
                task_type=self.task_type,
            ),
            allowed_tools=context.task.allowed_tools,
            retry_attempts=context.tool_retry_attempts,
        )
        if tool_result.status != ToolStatus.SUCCESS:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=(
                    AgentStatus.TIMEOUT
                    if tool_result.status == ToolStatus.TIMEOUT
                    else AgentStatus.FAILED
                ),
                tool_results=(tool_result,),
                error=tool_result.error or "entity query failed",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
        records = [item for item in payload.get("records", []) if isinstance(item, dict)]
        current_mentions, current_evidence = self._find_current_mentions(context, targets)
        long_term_evidence = tuple(self._long_term_evidence(record) for record in records)
        warnings = tuple(str(item) for item in payload.get("warnings", []))
        return AgentResult(
            task_id=context.task.task_id,
            task_type=self.task_type,
            status=AgentStatus.SUCCESS,
            is_empty=not records and not current_mentions,
            payload={
                "target_entities": list(targets),
                "records": records,
                "related_entities": payload.get("related_entities", []),
                "current_mentions": current_mentions,
                "processed_segment_ids": [
                    segment.segment_id for segment in context.raw_material.segments
                ],
            },
            evidence=(*current_evidence, *long_term_evidence),
            tool_results=(tool_result,),
            warnings=warnings,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _find_current_mentions(
        context: WorkerContext, targets: tuple[str, ...]
    ) -> tuple[list[dict[str, Any]], tuple[EvidenceRef, ...]]:
        """在所有重叠分段中定位目标实体，并用全局 offset 去重。"""
        mentions: dict[tuple[str, int, int], dict[str, Any]] = {}
        evidence: dict[tuple[str, int, int], EvidenceRef] = {}
        for segment in context.raw_material.segments:
            for target in targets:
                position = 0
                while True:
                    local_start = segment.text.find(target, position)
                    if local_start < 0:
                        break
                    local_end = local_start + len(target)
                    global_start = segment.start_offset + local_start
                    global_end = segment.start_offset + local_end
                    key = (target, global_start, global_end)
                    mentions[key] = {
                        "entity": target,
                        "segment_id": segment.segment_id,
                        "start_offset": global_start,
                        "end_offset": global_end,
                    }
                    evidence[key] = EvidenceRef(
                        source_kind=SourceKind.CURRENT_MATERIAL,
                        source_id=context.raw_material.source_id,
                        segment_id=segment.segment_id,
                        start_offset=global_start,
                        end_offset=global_end,
                        source_time=context.request.source_time,
                        evidence_text=target,
                    )
                    position = local_end
        ordered_keys = sorted(mentions, key=lambda item: (item[1], item[0]))
        return (
            [mentions[key] for key in ordered_keys],
            tuple(evidence[key] for key in ordered_keys),
        )

    @staticmethod
    def _long_term_evidence(record: dict[str, Any]) -> EvidenceRef:
        """将实体检索记录转换为带 Memory ID 的长期 Evidence。"""
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return EvidenceRef(
            source_kind=SourceKind.LONG_TERM_MEMORY,
            memory_id=str(record["memory_id"]),
            source_id=str(metadata.get("history_source_id") or "") or None,
            source_time=str(metadata.get("source_time") or "") or None,
            source_file=str(metadata.get("source_file") or "") or None,
            evidence_text=str(metadata.get("evidence_span") or record.get("content") or "")
            or None,
        )
