"""提供带 Typed Memory metadata filter 的只读历史记忆检索工具。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from pydantic import Field

from ..schemas import FrozenStrictModel, TaskType
from .registry import BaseTool, ToolContext


TYPED_MEMORY_TYPES = frozenset(
    {"time", "decision", "constraint", "task", "risk", "preference"}
)


class MemorySearchBackend(Protocol):
    """定义 Typed Memory 后端的同步搜索接口。"""

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        """按 query、Typed Memory filters 和 top-k 返回原始搜索结果。"""
        ...


# 输入/输出模型参与 Tool Schema 生成，使用普通注释说明而不改变 JSON Schema 描述。
class TypedMemorySearchInput(FrozenStrictModel):
    query: str = Field(min_length=1)
    memory_types: tuple[str, ...] = Field(min_length=1)
    top_k_per_type: int = Field(default=10, ge=1, le=100)


# 单条 Typed Memory 记录的稳定身份、类型、正文、分数和 metadata。
class MemoryRecordPayload(FrozenStrictModel):
    memory_id: str = Field(min_length=1)
    memory_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# Typed Memory Tool 的批量结果、查询数量和后端诊断。
class TypedMemorySearchOutput(FrozenStrictModel):
    records: tuple[MemoryRecordPayload, ...] = ()
    query_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def _coerce_metadata(value: Any) -> dict[str, Any]:
    """将后端 metadata 安全规整为字典，异常形态按空 metadata 处理。"""
    return dict(value) if isinstance(value, dict) else {}


def _record_content(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    """按 Typed Memory 的字段优先级取得非空记录正文。"""
    return str(
        metadata.get("content_text")
        or item.get("memory")
        or item.get("memory_text")
        or ""
    ).strip()


class TypedMemorySearchTool(BaseTool):
    # 该具体 Tool 的 __doc__ 会被 ToolRegistry 放入 Planner/Worker Prompt；保持为空。
    name = "typed_memory_search"
    input_model = TypedMemorySearchInput
    output_model = TypedMemorySearchOutput
    allowed_task_types = frozenset({TaskType.SEMANTIC_MEMORY})

    def __init__(
        self,
        backend: MemorySearchBackend,
        *,
        user_id: str,
        project: str,
        schema_version: str = "summary_typed_memory.v1",
    ) -> None:
        """保存后端和 user/project/schema 过滤身份，不创建或写入 Memory。"""
        self._backend = backend
        self._user_id = user_id
        self._project = project
        self._schema_version = schema_version

    async def execute(
        self, arguments: TypedMemorySearchInput, context: ToolContext
    ) -> TypedMemorySearchOutput:
        """按类型只读检索 Typed Memory，过滤身份/类型非法记录并按 ID 去重。"""
        del context
        invalid_types = sorted(set(arguments.memory_types) - TYPED_MEMORY_TYPES)
        if invalid_types:
            raise ValueError(f"unsupported typed memory types: {invalid_types}")

        by_id: dict[str, MemoryRecordPayload] = {}
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {
            "effective_filters": [],
            "backend_raw_count": 0,
            "accepted_count": 0,
            "discarded_by_reason": {},
        }

        def discard(reason: str) -> None:
            """累计后端记录被拒绝的原因。"""
            discarded = diagnostics["discarded_by_reason"]
            discarded[reason] = discarded.get(reason, 0) + 1

        for memory_type in arguments.memory_types:
            filters = {
                "user_id": self._user_id,
                "project": self._project,
                "memory_schema_version": self._schema_version,
                "memory_type": memory_type,
            }
            result = await asyncio.to_thread(
                self._backend.search,
                arguments.query,
                filters=filters,
                top_k=arguments.top_k_per_type,
            )
            diagnostics["effective_filters"].append(dict(filters))
            raw_results = result.get("results", []) if isinstance(result, dict) else []
            diagnostics["backend_raw_count"] += len(raw_results)
            for item in raw_results:
                if not isinstance(item, dict):
                    discard("non_object_result")
                    continue
                metadata = _coerce_metadata(item.get("metadata"))
                actual_type = str(metadata.get("memory_type") or memory_type).strip()
                if actual_type != memory_type:
                    warnings.append(
                        f"ignored result with mismatched memory_type: {actual_type}"
                    )
                    discard("memory_type_mismatch")
                    continue
                memory_id = str(
                    metadata.get("typed_memory_id") or item.get("id") or ""
                ).strip()
                content = _record_content(item, metadata)
                if not memory_id or not content:
                    warnings.append("ignored result without memory_id or content")
                    discard("missing_memory_id_or_content")
                    continue
                try:
                    score = float(item.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                payload = MemoryRecordPayload(
                    memory_id=memory_id,
                    memory_type=actual_type,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
                existing = by_id.get(memory_id)
                if existing is None or payload.score > existing.score:
                    by_id[memory_id] = payload
                    diagnostics["accepted_count"] = len(by_id)

        records = tuple(
            sorted(by_id.values(), key=lambda item: (-item.score, item.memory_id))
        )
        return TypedMemorySearchOutput(
            records=records,
            query_count=len(arguments.memory_types),
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics=diagnostics,
        )


def structured_fields_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """解析 metadata 中可选的结构化字段 JSON，非法内容返回空字典。"""
    raw = metadata.get("structured_fields_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
