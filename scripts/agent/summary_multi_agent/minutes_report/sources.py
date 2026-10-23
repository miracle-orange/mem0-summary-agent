"""Build model-visible history aliases and resolve draft citations deterministically."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .models import (
    EntrySourceResolution,
    MinutesDocument,
    MinutesDocumentDraft,
    MinutesEntry,
    MinutesSection,
    MinutesSources,
    ResolvedSource,
)


def _marker(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_history_aliases(
    records: tuple[tuple[str, dict[str, Any]], ...],
    *,
    projector,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Group identical model-visible records under one short alias."""

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record_id, record in records:
        projected = projector(record)
        marker = _marker(projected)
        if marker not in grouped:
            grouped[marker] = {
                "projected": projected,
                "record_ids": [],
                "raw_records": [],
            }
            order.append(marker)
        source_ids = [str(record_id)]
        memory_id = record.get("memory_id")
        if memory_id not in (None, ""):
            source_ids.append(str(memory_id))
        raw_record_ids = record.get("record_ids")
        if isinstance(raw_record_ids, (list, tuple)):
            source_ids.extend(
                str(item) for item in raw_record_ids if item not in (None, "")
            )
        grouped[marker]["record_ids"].extend(source_ids)
        grouped[marker]["raw_records"].append(dict(record))

    model_records: list[dict[str, Any]] = []
    aliases: dict[str, dict[str, Any]] = {}
    for index, marker in enumerate(order, start=1):
        alias = f"H{index:03d}"
        group = grouped[marker]
        projected = dict(group["projected"])
        model_records.append({"source_ref": alias, **projected})
        aliases[alias] = {
            "record_ids": list(dict.fromkeys(group["record_ids"])),
            "content": projected.get("content", ""),
            "source_date": projected.get("source_date", ""),
            "recorded_status": projected.get("recorded_status", ""),
            "type": projected.get("type", ""),
        }
    return model_records, aliases


def normalize_document(
    draft: MinutesDocumentDraft,
    *,
    source_time: str | None,
    meeting_scene: str,
    format_kind: str,
) -> MinutesDocument:
    entry_index = 0
    sections: list[MinutesSection] = []
    for section_index, section in enumerate(draft.sections, start=1):
        entries: list[MinutesEntry] = []
        for entry in section.entries:
            entry_index += 1
            entries.append(
                MinutesEntry(
                    entry_id=f"E{entry_index:04d}",
                    **entry.model_dump(mode="python"),
                )
            )
        sections.append(
            MinutesSection(
                section_id=f"SEC{section_index:03d}",
                heading=section.heading,
                entries=tuple(entries),
            )
        )
    return MinutesDocument(
        title=draft.title,
        source_time=source_time,
        meeting_scene=meeting_scene,
        format_kind=format_kind,
        overview=draft.overview,
        sections=tuple(sections),
    )


def _quote_sources(current_material: str, quote: str) -> ResolvedSource:
    starts: list[int] = []
    cursor = 0
    while quote and (found := current_material.find(quote, cursor)) >= 0:
        starts.append(found)
        cursor = found + 1
    if len(starts) == 1:
        start = starts[0]
        return ResolvedSource(
            source_kind="current",
            status="exact",
            quote=quote,
            start_offset=start,
            end_offset=start + len(quote),
        )
    if len(starts) > 1:
        return ResolvedSource(
            source_kind="current",
            status="ambiguous",
            quote=quote,
            warning="引文在当前材料中出现多次，未猜测具体位置。",
        )
    return ResolvedSource(
        source_kind="current",
        status="missing",
        quote=quote,
        warning="引文无法在当前材料中逐字定位。",
    )


def resolve_sources(
    document: MinutesDocument,
    *,
    current_material: str,
    history_aliases: Mapping[str, Mapping[str, Any]],
) -> MinutesSources:
    entries: list[EntrySourceResolution] = []
    global_warnings: list[str] = []
    referenced_history_refs: set[str] = set()
    for section in document.sections:
        for entry in section.entries:
            sources: list[ResolvedSource] = []
            warnings: list[str] = []
            for quote in dict.fromkeys(entry.current_evidence_quotes):
                resolved = _quote_sources(current_material, quote)
                sources.append(resolved)
                if resolved.warning:
                    warnings.append(resolved.warning)
            for history_ref in dict.fromkeys(entry.history_refs):
                referenced_history_refs.add(history_ref)
                record = history_aliases.get(history_ref)
                if record is None:
                    resolved = ResolvedSource(
                        source_kind="history",
                        status="invalid_history_ref",
                        history_ref=history_ref,
                        warning="历史引用不在本次最终可见历史集合中。",
                    )
                else:
                    resolved = ResolvedSource(
                        source_kind="history",
                        status="history_record",
                        history_ref=history_ref,
                        record_ids=tuple(str(item) for item in record.get("record_ids", [])),
                        source_date=str(record.get("source_date") or ""),
                        recorded_status=str(record.get("recorded_status") or ""),
                        content=str(record.get("content") or ""),
                    )
                sources.append(resolved)
                if resolved.warning:
                    warnings.append(resolved.warning)
            if entry.kind == "history_update":
                has_current = any(
                    source.source_kind == "current" and source.status == "exact"
                    for source in sources
                )
                if not has_current:
                    warnings.append("历史状态变化缺少可唯一定位的当前材料依据，需人工核查。")
            warnings = list(dict.fromkeys(warnings))
            global_warnings.extend(f"{entry.entry_id}: {item}" for item in warnings)
            entries.append(
                EntrySourceResolution(
                    entry_id=entry.entry_id,
                    sources=tuple(sources),
                    warnings=tuple(warnings),
                )
            )
    return MinutesSources(
        history_aliases={
            key: dict(value)
            for key, value in history_aliases.items()
            if key in referenced_history_refs
        },
        entries=tuple(entries),
        warnings=tuple(dict.fromkeys(global_warnings)),
    )
