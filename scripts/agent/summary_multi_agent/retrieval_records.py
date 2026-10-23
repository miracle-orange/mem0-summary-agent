"""Conservative, read-only assembly for retrieved typed-memory records."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _metadata_json_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _business_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(record)
    payload.pop("score", None)
    return payload


def retrieved_record_equivalence_key(record: dict[str, Any]) -> str | None:
    """Return a key only for same-source records with exact business fields."""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source_identity = str(
        metadata.get("history_source_id") or metadata.get("source_file") or ""
    ).strip()
    source_time = str(metadata.get("source_time") or "").strip()
    if not source_identity or not source_time:
        return None
    payload = _business_payload(record)
    payload.pop("memory_id", None)
    payload["content"] = _normalized_text(payload.get("content"))
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, dict):
        payload_metadata = deepcopy(payload_metadata)
        payload_metadata.pop("typed_memory_id", None)
        if _normalized_text(payload_metadata.get("content_text")) == payload[
            "content"
        ]:
            payload_metadata.pop("content_text", None)
        if "structured_fields_json" in payload_metadata:
            payload_metadata["structured_fields_json"] = _metadata_json_value(
                payload_metadata, "structured_fields_json"
            )
        payload["metadata"] = payload_metadata
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def assemble_retrieved_records(
    occurrences: Iterable[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Deduplicate occurrences without hiding conflicting same-ID versions.

    Returns retained records, direct-to-final exact-equivalence merges, same-ID
    version conflicts, and occurrence statistics. Similarity never removes data.
    """

    raw = [deepcopy(item) for item in occurrences]
    by_id: dict[str, list[dict[str, Any]]] = {}
    unkeyed_count = 0
    for record in raw:
        memory_id = str(record.get("memory_id") or "").strip()
        if memory_id:
            by_id.setdefault(memory_id, []).append(record)
        else:
            unkeyed_count += 1

    representatives: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for memory_id, versions in by_id.items():
        unique_business: dict[str, dict[str, Any]] = {}
        for record in versions:
            signature = json.dumps(
                _business_payload(record),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            current = unique_business.get(signature)
            if current is None or float(record.get("score") or 0.0) > float(
                current.get("score") or 0.0
            ):
                unique_business[signature] = record
        variants = list(unique_business.values())
        representative = max(
            variants,
            key=lambda item: float(item.get("score") or 0.0),
        )
        representatives.extend(variants)
        if len(variants) > 1:
            conflicts.append(
                {
                    "memory_id": memory_id,
                    "representative_memory_id": memory_id,
                    "variants": variants,
                    "reason": "same_memory_id_has_conflicting_business_versions",
                }
            )

    groups: dict[str, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for record in representatives:
        key = retrieved_record_equivalence_key(record)
        if key is None:
            ungrouped.append(record)
        else:
            groups.setdefault(key, []).append(record)

    retained = list(ungrouped)
    merges: list[dict[str, Any]] = []
    for group in groups.values():
        final = max(group, key=lambda item: float(item.get("score") or 0.0))
        retained.append(final)
        final_id = str(final.get("memory_id") or "")
        for record in group:
            removed_id = str(record.get("memory_id") or "")
            if removed_id == final_id:
                continue
            merges.append(
                {
                    "removed_memory_id": removed_id,
                    "retained_memory_id": final_id,
                    "reason": "same_source_and_exact_business_fields",
                }
            )

    retained.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("memory_id") or ""),
        )
    )
    stats = {
        "record_occurrence_count": len(raw),
        "ignored_unkeyed_record_count": unkeyed_count,
        "unique_memory_id_count": len(by_id),
        "unique_record_version_count": len(representatives),
        "retained_record_count": len(retained),
        "same_id_conflict_count": len(conflicts),
        "exact_equivalent_merge_count": len(merges),
    }
    return retained, merges, conflicts, stats
