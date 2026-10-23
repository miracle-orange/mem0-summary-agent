"""将检索到的 Typed Memory 归并为可审计的 Artifact Board。"""

from __future__ import annotations

import json
from typing import Any, Optional

from summary_memory_agent_schemas import TypedMemoryRecord


ARTIFACT_BOARD_SCHEMA_VERSION = "summary_artifact_board.v1"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _clean_text(item)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _field(record: TypedMemoryRecord, name: str) -> str:
    return _clean_text(record.structured_fields.get(name))


def _list_field(record: TypedMemoryRecord, name: str) -> list[str]:
    return _clean_list(record.structured_fields.get(name))


def _evidence_ref(record: TypedMemoryRecord) -> dict[str, Any]:
    source = record.source or {}
    return {
        "typed_memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "status": record.status,
        "confidence": record.confidence,
        "source_time": _clean_text(source.get("source_time")),
        "source_file": _clean_text(source.get("source_file")),
        "evidence_span": _clean_text(source.get("evidence_span")),
        "content": record.content,
    }


def _memory_item(record: TypedMemoryRecord) -> dict[str, Any]:
    return {
        "typed_memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "content": record.content,
        "status": record.status,
        "confidence": record.confidence,
        "structured_fields": record.structured_fields,
        "entities": record.entities,
        "topics": record.topics,
        "evidence": _evidence_ref(record),
    }


def _records_by_type(records: list[TypedMemoryRecord], memory_type: str) -> list[TypedMemoryRecord]:
    return [record for record in records if record.memory_type == memory_type]


def _items_by_status(records: list[TypedMemoryRecord], statuses: set[str]) -> list[dict[str, Any]]:
    return [_memory_item(record) for record in records if record.status in statuses]


def _time_analysis(records: list[TypedMemoryRecord]) -> dict[str, Any]:
    time_records = _records_by_type(records, "time")
    return {
        "historical_time_windows": [_memory_item(record) for record in time_records],
        "active_or_unknown_time_refs": _items_by_status(time_records, {"active", "unknown"}),
        "inactive_or_superseded_time_refs": _items_by_status(time_records, {"inactive", "superseded"}),
    }


def _decision_constraint_analysis(records: list[TypedMemoryRecord]) -> dict[str, Any]:
    decision_records = _records_by_type(records, "decision")
    constraint_records = _records_by_type(records, "constraint")
    return {
        "active_decisions": _items_by_status(decision_records, {"active"}),
        "uncertain_decisions": _items_by_status(decision_records, {"unknown"}),
        "inactive_or_superseded_decisions": _items_by_status(decision_records, {"inactive", "superseded"}),
        "active_constraints": _items_by_status(constraint_records, {"active"}),
        "uncertain_constraints": _items_by_status(constraint_records, {"unknown"}),
        "inactive_or_superseded_constraints": _items_by_status(constraint_records, {"inactive", "superseded"}),
        "rule_note": "规则版 artifact board 只按 retrieved typed memories 归并；是否被当前材料推翻需由摘要阶段结合当前材料判断。",
    }


def _task_analysis(records: list[TypedMemoryRecord]) -> dict[str, Any]:
    task_records = _records_by_type(records, "task")
    return {
        "active_or_unknown_tasks": _items_by_status(task_records, {"active", "unknown"}),
        "inactive_or_superseded_tasks": _items_by_status(task_records, {"inactive", "superseded"}),
        "owners": _collect_field_values(task_records, ("owner", "assignee", "responsible_person")),
        "deadlines": _collect_field_values(task_records, ("deadline", "due_time", "time_window")),
    }


def _risk_analysis(records: list[TypedMemoryRecord]) -> dict[str, Any]:
    risk_records = _records_by_type(records, "risk")
    return {
        "active_risks": _items_by_status(risk_records, {"active"}),
        "unknown_risks": _items_by_status(risk_records, {"unknown"}),
        "inactive_or_superseded_risks": _items_by_status(risk_records, {"inactive", "superseded"}),
    }


def _preference_analysis(records: list[TypedMemoryRecord]) -> dict[str, Any]:
    preference_records = _records_by_type(records, "preference")
    return {
        "applicable_preferences": _items_by_status(preference_records, {"active", "unknown"}),
        "inactive_or_superseded_preferences": _items_by_status(preference_records, {"inactive", "superseded"}),
        "summary_focus": _collect_list_values(preference_records, ("focus_items",)),
        "deprioritized_content": _collect_list_values(preference_records, ("deprioritized_items",)),
        "output_implications": _collect_list_values(preference_records, ("output_implications",)),
        "role_or_audience": _collect_field_values(preference_records, ("role_or_audience",)),
    }


def _collect_field_values(records: list[TypedMemoryRecord], field_names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        for field_name in field_names:
            value = _field(record, field_name)
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    return values


def _collect_list_values(records: list[TypedMemoryRecord], field_names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        for field_name in field_names:
            for value in _list_field(record, field_name):
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
    return values


def _build_warnings(records: list[TypedMemoryRecord]) -> list[str]:
    warnings: list[str] = []
    if not records:
        warnings.append("本次没有检索到 typed memories，artifact board 仅为空结构。")

    missing_evidence = [
        record.memory_id
        for record in records
        if not _clean_text((record.source or {}).get("evidence_span"))
    ]
    if missing_evidence:
        warnings.append("部分 typed memories 缺少 evidence_span，摘要阶段应避免过度推断。")
    return warnings


def build_artifact_board(
    current_content: str,
    retrieved_records: list[TypedMemoryRecord],
    source_time: str = "",
) -> dict[str, Any]:
    records = list(retrieved_records)
    return {
        "schema_version": ARTIFACT_BOARD_SCHEMA_VERSION,
        "source_time": _clean_text(source_time),
        "current_material": {
            "provided_separately": True,
            "context_char_count": len(current_content.strip()),
        },
        "retrieved_memory_count": len(records),
        "retrieved_memory_ids": [record.memory_id for record in records],
        "time_analysis": _time_analysis(records),
        "decision_constraint_analysis": _decision_constraint_analysis(records),
        "task_analysis": _task_analysis(records),
        "risk_analysis": _risk_analysis(records),
        "preference_analysis": _preference_analysis(records),
        "evidence_index": [_evidence_ref(record) for record in records],
        "warnings": _build_warnings(records),
    }


def format_artifact_board_for_summary(artifact_board: Optional[dict[str, Any]]) -> str:
    if not artifact_board:
        return "暂无 artifact board。"
    return json.dumps(artifact_board, ensure_ascii=False, indent=2)
