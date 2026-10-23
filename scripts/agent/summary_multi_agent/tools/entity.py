"""提供实体及一跳关系的只读 Memory 查询工具。"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from pydantic import Field

from ..schemas import FrozenStrictModel, TaskType
from .memory import _coerce_metadata, _record_content
from .registry import BaseTool, ToolContext


class EntitySearchBackend(Protocol):
    """定义实体检索后端的同步最小接口，由 Tool 在线程中调用。"""

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        """按查询和身份过滤返回原始实体检索结果。"""
        ...


# 输入/输出模型参与 Tool Schema 生成，使用普通注释说明而不改变 JSON Schema 描述。
class EntityQueryInput(FrozenStrictModel):
    entities: tuple[str, ...] = Field(min_length=1)
    include_one_hop: bool = True
    top_k_per_entity: int = Field(default=10, ge=1, le=100)


# 单条实体查询结果及其直接命中/一跳关系。
class EntityMemoryPayload(FrozenStrictModel):
    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    memory_type: str = "unknown"
    matched_entities: tuple[str, ...] = ()
    related_entities: tuple[str, ...] = ()
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# Entity Tool 的批量输出、warning 和后端丢弃诊断。
class EntityQueryOutput(FrozenStrictModel):
    records: tuple[EntityMemoryPayload, ...] = ()
    matched_entities: tuple[str, ...] = ()
    related_entities: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def metadata_entities(metadata: dict[str, Any]) -> tuple[str, ...]:
    """从 metadata 的文本或序列字段提取去重实体名称。"""
    raw = metadata.get("entities_text")
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw or "").split("|")
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


class EntityQueryTool(BaseTool):
    # 该具体 Tool 的 __doc__ 会被 ToolRegistry 放入 Planner/Worker Prompt；保持为空。
    name = "entity_query"
    input_model = EntityQueryInput
    output_model = EntityQueryOutput
    allowed_task_types = frozenset({TaskType.ENTITY_PERSONALIZATION})

    def __init__(
        self,
        backend: EntitySearchBackend,
        *,
        user_id: str,
        project: str,
        schema_version: str = "summary_typed_memory.v1",
    ) -> None:
        """保存后端和固定的 user/project/schema 身份过滤条件。"""
        self._backend = backend
        self._filters = {
            "user_id": user_id,
            "project": project,
            "memory_schema_version": schema_version,
        }

    async def execute(
        self, arguments: EntityQueryInput, context: ToolContext
    ) -> EntityQueryOutput:
        """只读查询显式目标实体，并保留精确匹配记录和一跳关系诊断。"""
        del context
        targets = tuple(dict.fromkeys(value.strip() for value in arguments.entities if value.strip()))
        by_id: dict[str, EntityMemoryPayload] = {}
        warnings: list[str] = []
        related: list[str] = []
        diagnostics: dict[str, Any] = {
            "effective_filters": dict(self._filters),
            "backend_raw_count": 0,
            "accepted_count": 0,
            "discarded_by_reason": {},
        }

        def discard(reason: str) -> None:
            """累计后端记录被拒绝的原因。"""
            discarded = diagnostics["discarded_by_reason"]
            discarded[reason] = discarded.get(reason, 0) + 1

        for target in targets:
            result = await asyncio.to_thread(
                self._backend.search,
                target,
                filters=dict(self._filters),
                top_k=arguments.top_k_per_entity,
            )
            raw_results = result.get("results", []) if isinstance(result, dict) else []
            diagnostics["backend_raw_count"] += len(raw_results)
            for item in raw_results:
                if not isinstance(item, dict):
                    discard("non_object_result")
                    continue
                metadata = _coerce_metadata(item.get("metadata"))
                entities = metadata_entities(metadata)
                exact_matches = tuple(entity for entity in entities if entity == target)
                if not exact_matches:
                    discard("entity_not_in_metadata")
                    continue
                memory_id = str(
                    metadata.get("typed_memory_id") or item.get("id") or ""
                ).strip()
                content = _record_content(item, metadata)
                if not memory_id or not content:
                    warnings.append("ignored entity result without memory_id or content")
                    discard("missing_memory_id_or_content")
                    continue
                one_hop = tuple(
                    entity for entity in entities if entity not in targets and entity != target
                )
                if arguments.include_one_hop:
                    related.extend(one_hop)
                try:
                    score = float(item.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                payload = EntityMemoryPayload(
                    memory_id=memory_id,
                    content=content,
                    memory_type=str(metadata.get("memory_type") or "unknown"),
                    matched_entities=exact_matches,
                    related_entities=one_hop if arguments.include_one_hop else (),
                    score=score,
                    metadata=metadata,
                )
                existing = by_id.get(memory_id)
                if existing is None or payload.score > existing.score:
                    by_id[memory_id] = payload
                    diagnostics["accepted_count"] = len(by_id)

        return EntityQueryOutput(
            records=tuple(
                sorted(by_id.values(), key=lambda item: (-item.score, item.memory_id))
            ),
            matched_entities=targets,
            related_entities=tuple(dict.fromkeys(related)),
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics=diagnostics,
        )
