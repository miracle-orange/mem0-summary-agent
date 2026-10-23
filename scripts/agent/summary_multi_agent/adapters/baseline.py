"""将既有 Typed Memory 结果投影为 V2 的历史 Artifact Board 视图。"""

from __future__ import annotations

from typing import Any

from ..schemas import SummaryWorkingState


def build_artifact_board_from_state(state: SummaryWorkingState) -> dict[str, Any]:
    """将成功 Semantic Memory 结果投影为旧 Artifact Board 兼容视图。

    这是 V2 的兼容审计视图，不是 M4 Shared Working Memory；旧模块导入或记录
    转换失败会向调用方传播，由 Pipeline 将其作为非关键 warning 处理。
    """
    from summary_memory_artifacts import build_artifact_board
    from summary_memory_typed_memory import typed_record_from_search_result

    semantic_result = state.agent_results.get("semantic_memory")
    records = []
    if semantic_result is not None:
        raw_records = semantic_result.payload.get("records", [])
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            typed_record = typed_record_from_search_result(
                {
                    "id": record.get("memory_id"),
                    "memory": record.get("content"),
                    "score": record.get("score"),
                    "metadata": record.get("metadata") or {},
                }
            )
            if typed_record is not None:
                records.append(typed_record)
    return build_artifact_board(
        state.request.current_material,
        records,
        source_time=state.request.source_time or "",
    )
