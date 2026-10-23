"""Build lossless, reference-based model views without mutating audit objects."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable


def compact_json(value: object) -> str:
    """Serialize a model-facing payload compactly while preserving Unicode."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ": "),
        sort_keys=True,
    )


def _record_model_payload(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Separate per-search score from the lossless business record."""

    projected = deepcopy(record)
    score = projected.pop("score", None)
    metadata = projected.get("metadata")
    if isinstance(metadata, dict):
        compact_metadata = dict(metadata)
        if str(compact_metadata.get("content_text") or "").strip() == str(
            projected.get("content") or ""
        ).strip():
            compact_metadata.pop("content_text", None)
        projected["metadata"] = compact_metadata
    return projected, score


class MemoryRecordCatalog:
    """Deduplicate record serialization while retaining every occurrence and version."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._versions_by_memory_id: dict[str, list[str]] = {}
        self._occurrence_count = 0

    def add(self, raw_record: object) -> dict[str, Any] | object:
        """Return a lightweight occurrence reference or the untouched non-record value."""

        if not isinstance(raw_record, dict):
            return deepcopy(raw_record)
        memory_id = str(raw_record.get("memory_id") or "").strip()
        if not memory_id:
            return deepcopy(raw_record)
        record, score = _record_model_payload(raw_record)
        signature = compact_json(record)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        version_ref = f"{memory_id}::v_{digest}"
        if version_ref not in self._records:
            self._records[version_ref] = {
                "memory_id": memory_id,
                "record_version_ref": version_ref,
                "record": record,
            }
            self._versions_by_memory_id.setdefault(memory_id, []).append(version_ref)
        self._occurrence_count += 1
        reference: dict[str, Any] = {
            "memory_id": memory_id,
            "record_version_ref": version_ref,
        }
        if score is not None:
            reference["score"] = score
        return reference

    @property
    def payload(self) -> dict[str, Any]:
        """Return the catalog and conservative version-conflict diagnostics."""

        conflicts = [
            {"memory_id": memory_id, "record_version_refs": refs}
            for memory_id, refs in self._versions_by_memory_id.items()
            if len(refs) > 1
        ]
        return {
            "records": list(self._records.values()),
            "record_occurrence_count": self._occurrence_count,
            "unique_memory_id_count": len(self._versions_by_memory_id),
            "unique_record_version_count": len(self._records),
            "record_version_conflicts": conflicts,
            "selection_note": (
                "record_version_ref is an input-only catalog reference; terminal "
                "selected_memory_ids must use real memory_id values"
            ),
        }


class EvidenceCatalog:
    """Deduplicate dependency Evidence while preserving real evidence IDs."""

    def __init__(self) -> None:
        self._evidence: dict[str, dict[str, Any]] = {}
        self._conflicts: dict[str, list[dict[str, Any]]] = {}

    def add_many(self, values: Iterable[object]) -> list[str | dict[str, Any]]:
        refs: list[str | dict[str, Any]] = []
        for value in values:
            if not isinstance(value, dict):
                refs.append(deepcopy(value))
                continue
            evidence_id = str(value.get("evidence_id") or "").strip()
            if not evidence_id:
                refs.append(deepcopy(value))
                continue
            candidate = deepcopy(value)
            current = self._evidence.get(evidence_id)
            if current is None:
                self._evidence[evidence_id] = candidate
            elif current != candidate:
                variants = self._conflicts.setdefault(evidence_id, [current])
                if candidate not in variants:
                    variants.append(candidate)
            refs.append(evidence_id)
        return refs

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "evidence": list(self._evidence.values()),
            "evidence_conflicts": [
                {"evidence_id": evidence_id, "variants": variants}
                for evidence_id, variants in self._conflicts.items()
            ],
        }


def project_records_in_payload(
    payload: object,
    *,
    record_catalog: MemoryRecordCatalog,
) -> object:
    """Replace only the explicit top-level records list with catalog references."""

    if not isinstance(payload, dict):
        return deepcopy(payload)
    projected = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key != "diagnostics"
    }
    records = projected.get("records")
    if isinstance(records, (list, tuple)):
        projected["record_refs"] = [record_catalog.add(item) for item in records]
        projected.pop("records", None)
    return projected


def project_semantic_observations(
    observations: Iterable[object],
    *,
    tool_arguments_by_call_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one model-facing working set from raw search observations.

    The raw observations remain the authority for action validation and audit.
    A match points to an index entry with the real observation ID and retains
    its per-search score; each business record version appears only once.
    """

    arguments_by_call = tool_arguments_by_call_id or {}
    catalog = MemoryRecordCatalog()
    observation_index: list[dict[str, Any]] = []
    matches_by_version: dict[str, list[list[Any]]] = {}

    for index, observation in enumerate(observations):
        result = observation.tool_result
        payload = result.payload if isinstance(result.payload, dict) else {}
        arguments = arguments_by_call.get(result.call_id, {})
        indexed: dict[str, Any] = {
            "observation_id": observation.observation_id,
            "step_index": observation.step_index,
            "call_id": result.call_id,
            "tool_name": result.tool_name,
            "status": result.status.value,
        }
        if arguments:
            indexed["arguments"] = deepcopy(arguments)
        if result.error:
            indexed["error"] = result.error
        if isinstance(payload.get("warnings"), (list, tuple)) and payload["warnings"]:
            indexed["warnings"] = deepcopy(payload["warnings"])
        records = payload.get("records")
        if isinstance(records, (list, tuple)):
            indexed["record_count"] = len(records)
            unkeyed: list[object] = []
            for raw_record in records:
                reference = catalog.add(raw_record)
                if not isinstance(reference, dict) or "record_version_ref" not in reference:
                    unkeyed.append(reference)
                    continue
                match = [index, reference.get("score")]
                matches_by_version.setdefault(reference["record_version_ref"], []).append(match)
            if unkeyed:
                indexed["unkeyed_records"] = unkeyed
        # Preserve any unrecognized business payload without repeating records.
        other_payload = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in {"records", "warnings", "diagnostics", "query_count"}
        }
        if other_payload:
            indexed["other_payload"] = other_payload
        observation_index.append(indexed)

    catalog_payload = catalog.payload
    working_records: list[dict[str, Any]] = []
    for item in catalog_payload["records"]:
        record = deepcopy(item["record"])
        record.pop("memory_id", None)
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            # These three fields are already represented by the record header.
            for key, duplicate in (
                ("typed_memory_id", item["memory_id"]),
                ("memory_type", record.get("memory_type")),
                ("evidence_span", record.get("content")),
            ):
                if metadata.get(key) == duplicate:
                    metadata.pop(key)
            structured = metadata.get("structured_fields_json")
            if isinstance(structured, str):
                try:
                    metadata["structured_fields_json"] = json.loads(structured)
                except json.JSONDecodeError:
                    pass
        working_records.append(
            {
                "memory_id": item["memory_id"],
                "record_version_ref": item["record_version_ref"],
                "record": record,
                "matches": matches_by_version[item["record_version_ref"]],
            }
        )

    return {
        "observations": observation_index,
        "records": working_records,
        "match_columns": ["observation_index", "score"],
        "record_occurrence_count": catalog_payload["record_occurrence_count"],
        "unique_memory_id_count": catalog_payload["unique_memory_id_count"],
        "unique_record_version_count": catalog_payload["unique_record_version_count"],
        "record_version_conflicts": catalog_payload["record_version_conflicts"],
    }
