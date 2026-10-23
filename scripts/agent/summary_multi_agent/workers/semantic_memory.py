"""按计划检索并筛选历史 Typed Memory 的规则 Worker。"""

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
from ..retrieval_records import assemble_retrieved_records
from .base import BaseWorker, WorkerContext


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    """把任务输入规范化为去重且非空的字符串元组。"""
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = ()
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


class SemanticMemoryWorker(BaseWorker):
    """按计划只读检索 Typed Memory，并将结果转换为长期记忆 Evidence。"""

    task_type = TaskType.SEMANTIC_MEMORY

    async def execute(self, context: WorkerContext) -> AgentResult:
        """执行有限多 query 检索，并保真组装重复和冲突版本。

        每个工具调用通过 Tool Registry 受权限、重试和 timeout 约束；空检索是
        合法成功结果，后续 Synthesis 可继续使用当前材料。
        """
        return await self.execute_recovery(context)

    async def execute_recovery(
        self,
        context: WorkerContext,
        *,
        reused_calls: tuple[tuple[dict[str, Any], Any], ...] = (),
        remaining_tool_calls: int | None = None,
        remaining_timeout_seconds: float | None = None,
    ) -> AgentResult:
        """Execute missing rule queries while reusing successful in-run searches."""

        self.validate_context(context)
        started = time.perf_counter()
        queries = _as_string_tuple(context.task.inputs.get("queries"))
        if not queries:
            queries = (context.request.summary_request.strip() or "当前材料相关历史背景",)
        memory_types = _as_string_tuple(context.task.inputs.get("memory_types")) or (
            "time",
            "decision",
            "constraint",
            "task",
            "risk",
            "preference",
        )
        top_k = int(context.task.inputs.get("top_k_per_type") or 10)

        record_occurrences: list[dict[str, Any]] = []
        tool_results = []
        warnings: list[str] = []
        completed_keys: set[str] = set()
        executed_queries: list[str] = []
        successful_results = 0
        for arguments, result in reused_calls:
            tool_results.append(result)
            query = str(arguments.get("query") or "").strip()
            if query and query not in executed_queries:
                executed_queries.append(query)
            if result.status != ToolStatus.SUCCESS:
                continue
            completed_keys.add(self._search_key(arguments))
            successful_results += 1
            payload = result.payload if isinstance(result.payload, dict) else {}
            warnings.extend(str(value) for value in payload.get("warnings", []))
            record_occurrences.extend(
                record
                for record in payload.get("records", [])
                if isinstance(record, dict)
            )

        allowed_new_calls = (
            context.task.max_tool_calls
            if remaining_tool_calls is None
            else max(0, remaining_tool_calls)
        )
        new_calls = 0
        incomplete_queries: list[str] = []
        failed_error: str | None = None
        for query in queries:
            arguments = {
                "query": query,
                "memory_types": memory_types,
                "top_k_per_type": top_k,
            }
            if self._search_key(arguments) in completed_keys:
                continue
            if new_calls >= allowed_new_calls:
                incomplete_queries.append(query)
                continue
            elapsed = time.perf_counter() - started
            if (
                remaining_timeout_seconds is not None
                and elapsed >= remaining_timeout_seconds
            ):
                incomplete_queries.append(query)
                continue
            timeout_seconds = min(context.task.timeout_seconds, 120)
            if remaining_timeout_seconds is not None:
                timeout_seconds = max(
                    1,
                    min(timeout_seconds, int(remaining_timeout_seconds - elapsed)),
                )
            call = ToolCall(
                tool_name="typed_memory_search",
                arguments=arguments,
                timeout_seconds=timeout_seconds,
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
            new_calls += 1
            if query not in executed_queries:
                executed_queries.append(query)
            if result.status != ToolStatus.SUCCESS:
                failed_error = result.error or "typed memory search failed"
                incomplete_queries.extend(
                    item for item in queries if item not in executed_queries
                )
                break
            successful_results += 1
            payload = result.payload if isinstance(result.payload, dict) else {}
            warnings.extend(str(value) for value in payload.get("warnings", []))
            record_occurrences.extend(
                record
                for record in payload.get("records", [])
                if isinstance(record, dict)
            )

        if (failed_error and not reused_calls) or (
            successful_results == 0 and (failed_error or incomplete_queries)
        ):
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=AgentStatus.FAILED,
                tool_results=tuple(tool_results),
                error=failed_error or "semantic recovery has no budget for any search",
                warnings=(
                    "semantic recovery did not obtain a successful search result",
                ),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        records, merges, conflicts, stats = assemble_retrieved_records(
            record_occurrences
        )
        retrieval_complete = not incomplete_queries and failed_error is None
        if not retrieval_complete:
            warnings.append(
                "semantic rule recovery retained successful searches but did not "
                "complete every planned query"
            )
        if conflicts:
            warnings.append(
                "semantic retrieval returned conflicting business versions for the "
                "same memory_id; raw tool results retain every version"
            )
        evidence = tuple(self._evidence_from_record(record) for record in records)
        return AgentResult(
            task_id=context.task.task_id,
            task_type=self.task_type,
            status=AgentStatus.SUCCESS,
            is_empty=not records,
            payload={
                "queries": executed_queries,
                "planned_queries": list(queries),
                "records": records,
                "selection_mode": (
                    "rule_recovery_observation_all"
                    if reused_calls
                    else "rule_observation_all"
                ),
                "retrieval_complete": retrieval_complete,
                "incomplete_queries": list(dict.fromkeys(incomplete_queries)),
                "available_record_ids": [
                    memory_id
                    for memory_id in dict.fromkeys(
                        str(record.get("memory_id") or "")
                        for record in records
                    )
                    if memory_id
                ],
                "requested_memory_ids": None,
                "selected_memory_ids": [
                    memory_id
                    for memory_id in dict.fromkeys(
                        str(record.get("memory_id") or "")
                        for record in records
                    )
                    if memory_id
                ],
                "equivalent_record_merges": merges,
                "record_version_conflicts": conflicts,
                "retrieval_stats": stats,
                "reused_search_count": sum(
                    1 for _arguments, result in reused_calls
                    if result.status == ToolStatus.SUCCESS
                ),
                "new_search_count": new_calls,
                "execution_origin": (
                    "rule_recovery" if reused_calls else "rule"
                ),
                "record_selection_completed": False,
                "record_selection_origin": None,
            },
            evidence=evidence,
            tool_results=tuple(tool_results),
            warnings=tuple(dict.fromkeys(warnings)),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _search_key(arguments: dict[str, Any]) -> str:
        """Normalize a complete typed-memory search request for in-run reuse."""

        memory_types = arguments.get("memory_types")
        if isinstance(memory_types, (list, tuple)):
            memory_types = sorted(str(item) for item in memory_types)
        payload = {
            "query": str(arguments.get("query") or "").strip(),
            "memory_types": memory_types,
            "top_k_per_type": arguments.get("top_k_per_type"),
        }
        import json

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _evidence_from_record(record: dict[str, Any]) -> EvidenceRef:
        """将 Typed Memory 记录的 metadata 投影为 Long-term Memory Evidence。"""
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
