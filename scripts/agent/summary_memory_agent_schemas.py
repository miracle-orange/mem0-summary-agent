"""定义 Typed Memory 的固定类型、状态、字段校验和稳定标识契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


TYPED_MEMORY_SCHEMA_VERSION = "summary_typed_memory.v1"
ALLOWED_MEMORY_TYPES = ("time", "decision", "constraint", "task", "risk", "preference")
ALLOWED_MEMORY_TYPE_SET = set(ALLOWED_MEMORY_TYPES)
ALLOWED_MEMORY_STATUSES = ("active", "inactive", "unknown", "superseded")
ALLOWED_MEMORY_STATUS_SET = set(ALLOWED_MEMORY_STATUSES)
ALLOWED_CONFIDENCES = ("high", "medium", "low")
ALLOWED_CONFIDENCE_SET = set(ALLOWED_CONFIDENCES)

PREFERENCE_STRING_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "preference_type": ("preference_type", "type"),
    "scenario": ("scenario", "scenario_type", "context"),
    "role_or_audience": ("role_or_audience", "role", "audience", "target_reader", "target_audience"),
    "purpose": ("purpose", "summary_purpose", "use_purpose", "usage"),
    "evidence_basis": ("evidence_basis", "basis"),
    "inference_level": ("inference_level", "inference_type"),
}
PREFERENCE_LIST_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "focus_items": ("focus_items", "focus", "priorities", "priority_items"),
    "deprioritized_items": ("deprioritized_items", "avoid_items", "avoid", "deemphasize_items"),
    "output_implications": ("output_implications", "summary_implications", "instructions"),
}


@dataclass
class TypedMemoryRecord:
    schema_version: str
    memory_id: str
    memory_type: str
    content: str
    structured_fields: dict[str, Any]
    source: dict[str, str]
    entities: list[str]
    topics: list[str]
    status: str
    confidence: str
    embedding_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "structured_fields": self.structured_fields,
            "source": self.source,
            "entities": self.entities,
            "topics": self.topics,
            "status": self.status,
            "confidence": self.confidence,
            "embedding_text": self.embedding_text,
        }


@dataclass
class TypedRetrievalPlanItem:
    memory_types: list[str]
    query: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_types": self.memory_types,
            "query": self.query,
            "purpose": self.purpose,
        }


def stable_typed_memory_id(
    memory_type: str,
    content: str,
    source_time: str = "",
    evidence_span: str = "",
) -> str:
    identity = json.dumps(
        {
            "memory_type": normalize_text(memory_type),
            "content": normalize_text(content),
            "source_time": normalize_text(source_time),
            "evidence_span": normalize_text(evidence_span),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "tm_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = normalize_text(item)
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def normalize_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def normalize_preference_text_field(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(normalize_string_list(value))
    return normalize_text(value)


def first_preference_text_field(raw_fields: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        text = normalize_preference_text_field(raw_fields.get(alias))
        if text:
            return text
    return ""


def first_preference_list_field(raw_fields: dict[str, Any], aliases: tuple[str, ...]) -> list[str]:
    for alias in aliases:
        items = normalize_string_list(raw_fields.get(alias))
        if items:
            return items
    return []


def normalize_preference_inference_level(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"scenario_inferred", "scenario_inference", "inferred", "strongly_implied"}:
        return "scenario_inferred"
    if normalized in {"explicit", "explicitly_stated", "explicitly_mentioned"}:
        return "explicit"
    return value


def normalize_preference_structured_fields(value: Any) -> dict[str, Any]:
    raw_fields = normalize_dict(value)
    if not raw_fields:
        return {}

    structured_fields: dict[str, Any] = {}
    for field_name, aliases in PREFERENCE_STRING_FIELD_ALIASES.items():
        text = first_preference_text_field(raw_fields, aliases)
        if text:
            if field_name == "inference_level":
                text = normalize_preference_inference_level(text)
            structured_fields[field_name] = text

    for field_name, aliases in PREFERENCE_LIST_FIELD_ALIASES.items():
        items = first_preference_list_field(raw_fields, aliases)
        if items:
            structured_fields[field_name] = items

    return structured_fields


def normalize_structured_fields(memory_type: str, value: Any) -> dict[str, Any]:
    if memory_type == "preference":
        return normalize_preference_structured_fields(value)
    return normalize_dict(value)


def normalize_source(value: Any, source_time: str = "", source_file: str = "") -> dict[str, str]:
    raw_source = normalize_dict(value)
    return {
        "source_time": normalize_text(raw_source.get("source_time")) or normalize_text(source_time),
        "source_file": normalize_text(raw_source.get("source_file")) or normalize_text(source_file),
        "evidence_span": normalize_text(raw_source.get("evidence_span")),
    }


def normalize_typed_memory_record(
    raw_record: Any,
    source_time: str = "",
    source_file: str = "",
) -> Optional[TypedMemoryRecord]:
    if not isinstance(raw_record, dict):
        return None

    memory_type = normalize_text(raw_record.get("memory_type"))
    if memory_type not in ALLOWED_MEMORY_TYPE_SET:
        return None

    content = normalize_text(raw_record.get("content"))
    if not content:
        return None

    status = normalize_text(raw_record.get("status")) or "unknown"
    if status not in ALLOWED_MEMORY_STATUS_SET:
        return None

    confidence = normalize_text(raw_record.get("confidence")) or "medium"
    if confidence not in ALLOWED_CONFIDENCE_SET:
        return None

    structured_fields = normalize_structured_fields(memory_type, raw_record.get("structured_fields"))
    if memory_type == "preference" and structured_fields.get("inference_level") == "scenario_inferred":
        if status == "active":
            status = "unknown"
        if confidence == "high":
            confidence = "medium"

    source = normalize_source(raw_record.get("source"), source_time=source_time, source_file=source_file)
    memory_id = normalize_text(raw_record.get("memory_id")) or stable_typed_memory_id(
        memory_type=memory_type,
        content=content,
        source_time=source.get("source_time", ""),
        evidence_span=source.get("evidence_span", ""),
    )
    embedding_text = normalize_text(raw_record.get("embedding_text")) or content

    return TypedMemoryRecord(
        schema_version=TYPED_MEMORY_SCHEMA_VERSION,
        memory_id=memory_id,
        memory_type=memory_type,
        content=content,
        structured_fields=structured_fields,
        source=source,
        entities=normalize_string_list(raw_record.get("entities")),
        topics=normalize_string_list(raw_record.get("topics")),
        status=status,
        confidence=confidence,
        embedding_text=embedding_text,
    )


def normalize_typed_memory_records(
    raw_records: Any,
    source_time: str = "",
    source_file: str = "",
) -> list[TypedMemoryRecord]:
    if not isinstance(raw_records, list):
        raise ValueError("typed memory extraction must return a JSON array")

    records: list[TypedMemoryRecord] = []
    for raw_record in raw_records:
        record = normalize_typed_memory_record(raw_record, source_time=source_time, source_file=source_file)
        if record is not None:
            records.append(record)
    return records


def normalize_typed_retrieval_plan_item(raw_item: Any) -> TypedRetrievalPlanItem:
    if not isinstance(raw_item, dict):
        raise ValueError("retrieval plan item must be an object")

    memory_types = normalize_string_list(raw_item.get("memory_types"))
    if not memory_types:
        raise ValueError("retrieval plan item memory_types must be non-empty")
    invalid_types = [item for item in memory_types if item not in ALLOWED_MEMORY_TYPE_SET]
    if invalid_types:
        raise ValueError(f"invalid typed memory types in retrieval plan: {invalid_types}")

    query = normalize_text(raw_item.get("query"))
    if not query:
        raise ValueError("retrieval plan item query must be non-empty")

    purpose = normalize_text(raw_item.get("purpose"))
    if not purpose:
        raise ValueError("retrieval plan item purpose must be non-empty")

    return TypedRetrievalPlanItem(memory_types=memory_types, query=query, purpose=purpose)


def normalize_typed_retrieval_plan(raw_value: Any) -> list[TypedRetrievalPlanItem]:
    if not isinstance(raw_value, dict):
        raise ValueError("retrieval plan must be a JSON object")
    raw_items = raw_value.get("retrieval_plan")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("retrieval_plan must be a non-empty JSON array")
    return [normalize_typed_retrieval_plan_item(item) for item in raw_items]