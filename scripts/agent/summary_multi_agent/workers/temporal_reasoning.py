"""解析当前材料时间表达式并绑定确定性日历 Evidence 的规则 Worker。"""

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
from ..temporal_workload import (
    _direction_for_context,
    find_task_temporal_expressions,
    uses_review_temporal_scope,
)
from .base import BaseWorker, WorkerContext


class TemporalReasoningWorker(BaseWorker):
    """扫描当前原文时间表达，并通过 Calendar Tool 生成确定性时间 Evidence。"""

    task_type = TaskType.TEMPORAL_REASONING

    @staticmethod
    def _direction_for_context(segment_text: str, local_start: int) -> str:
        """代理到共享的分句时间方向判断逻辑。"""
        return _direction_for_context(segment_text, local_start)

    async def execute(self, context: WorkerContext) -> AgentResult:
        """解析任务范围内时间表达式，缺少锚点时保留原文并发出 warning。

        有锚点时每个表达式最多调用一次 Calendar Tool；工具失败返回失败结果，
        不用模型自由推理覆盖确定性解析。
        """
        self.validate_context(context)
        started = time.perf_counter()
        expressions, evidence = self._find_expressions(context)
        processed_segment_ids = self._processed_segment_ids(context, expressions)
        if not expressions:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=AgentStatus.SUCCESS,
                is_empty=True,
                payload={
                    "expressions": [],
                    "resolved": [],
                    "processed_segment_ids": processed_segment_ids,
                },
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if not context.request.source_time:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=AgentStatus.SUCCESS,
                payload={
                    "expressions": expressions,
                    "resolved": [],
                    "processed_segment_ids": processed_segment_ids,
                },
                evidence=evidence,
                warnings=(
                    "source_time is missing; relative expressions were preserved but not resolved",
                ),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        resolved: list[dict[str, Any]] = []
        tool_results = []
        warnings: list[str] = []
        max_calls = context.task.max_tool_calls
        for item in expressions[:max_calls]:
            call = ToolCall(
                tool_name="calendar_resolve",
                arguments={
                    "expression": item["expression"],
                    "source_time": context.request.source_time,
                    "direction": item["direction"],
                },
                timeout_seconds=min(context.task.timeout_seconds, 30),
            )
            result = await context.tool_registry.execute(
                call,
                context=ToolContext(
                    run_id=context.run_id,
                    task_id=context.task.task_id,
                    task_type=self.task_type,
                ),
                allowed_tools=context.task.allowed_tools,
                retry_attempts=context.tool_retry_attempts,
            )
            tool_results.append(result)
            if result.status != ToolStatus.SUCCESS:
                return AgentResult(
                    task_id=context.task.task_id,
                    task_type=self.task_type,
                    status=(
                        AgentStatus.TIMEOUT
                        if result.status == ToolStatus.TIMEOUT
                        else AgentStatus.FAILED
                    ),
                    evidence=evidence,
                    tool_results=tuple(tool_results),
                    error=result.error or "calendar resolution failed",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            payload = result.payload if isinstance(result.payload, dict) else {}
            combined = dict(item)
            combined["resolution"] = payload
            resolved.append(combined)
            if payload.get("warning"):
                warnings.append(str(payload["warning"]))

        if len(expressions) > max_calls:
            warnings.append(
                f"{len(expressions) - max_calls} temporal expressions exceeded max_tool_calls"
            )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=self.task_type,
            status=AgentStatus.SUCCESS,
            payload={
                "expressions": expressions,
                "resolved": resolved,
                "processed_segment_ids": processed_segment_ids,
            },
            evidence=evidence,
            tool_results=tuple(tool_results),
            warnings=tuple(dict.fromkeys(warnings)),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _find_expressions(
        self, context: WorkerContext
    ) -> tuple[list[dict[str, Any]], tuple[EvidenceRef, ...]]:
        """取得生产全量或 Review 限定范围的时间表达式及当前 Evidence。"""
        expressions = find_task_temporal_expressions(
            context.raw_material,
            task_id=context.task.task_id,
            inputs=context.task.inputs,
        )
        evidence = tuple(
            EvidenceRef(
                source_kind=SourceKind.CURRENT_MATERIAL,
                source_id=context.raw_material.source_id,
                segment_id=item["segment_id"],
                start_offset=item["start_offset"],
                end_offset=item["end_offset"],
                source_time=context.request.source_time,
                evidence_text=item["expression"],
            )
            for item in expressions
        )
        return expressions, evidence

    @staticmethod
    def _processed_segment_ids(
        context: WorkerContext,
        expressions: list[dict[str, Any]],
    ) -> list[str]:
        """返回该任务实际覆盖的分段 ID，区分生产扫描和 Review 子集。"""
        if not uses_review_temporal_scope(context.task.task_id, context.task.inputs):
            return [segment.segment_id for segment in context.raw_material.segments]
        return list(
            dict.fromkeys(str(item["segment_id"]) for item in expressions)
        )
