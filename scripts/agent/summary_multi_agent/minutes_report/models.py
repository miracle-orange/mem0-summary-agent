"""Contracts for the rich minutes model output and normalized report document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from ..schemas import FrozenStrictModel


MinutesEntryKind = Literal[
    "fact",
    "discussion",
    "decision",
    "action",
    "risk",
    "open_question",
    "history_update",
]
SourceResolutionStatus = Literal[
    "exact",
    "ambiguous",
    "missing",
    "history_record",
    "invalid_history_ref",
]


class MinutesEntryDraft(FrozenStrictModel):
    """One business statement returned by the rich synthesis call."""

    kind: MinutesEntryKind
    title: str = ""
    content: str = Field(min_length=1)
    owner: str = ""
    time_requirement: str = ""
    dependency: str = ""
    recorded_state: str = ""
    current_state: str = ""
    relation: str = ""
    current_evidence_quotes: tuple[str, ...] = Field(default=(), max_length=3)
    history_refs: tuple[str, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "MinutesEntryDraft":
        if self.kind == "history_update" and not self.history_refs:
            raise ValueError("history_update requires at least one history_ref")
        return self


class MinutesSectionDraft(FrozenStrictModel):
    heading: str = Field(min_length=1)
    entries: tuple[MinutesEntryDraft, ...] = Field(min_length=1)


class MinutesDocumentDraft(FrozenStrictModel):
    """Strict JSON contract produced by the rich summarizer."""

    title: str = Field(min_length=1)
    overview: tuple[str, ...] = Field(min_length=1, max_length=6)
    sections: tuple[MinutesSectionDraft, ...] = Field(min_length=1)


class MinutesEntry(MinutesEntryDraft):
    entry_id: str = Field(min_length=1)


class MinutesSection(FrozenStrictModel):
    section_id: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    entries: tuple[MinutesEntry, ...] = Field(min_length=1)


class MinutesDocument(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.minutes_document.v1"] = (
        "summary_multi_agent.minutes_document.v1"
    )
    title: str = Field(min_length=1)
    source_time: str | None = None
    meeting_scene: str = "general"
    format_kind: str = "structured"
    overview: tuple[str, ...] = Field(min_length=1, max_length=6)
    sections: tuple[MinutesSection, ...] = Field(min_length=1)


class ResolvedSource(FrozenStrictModel):
    source_kind: Literal["current", "history"]
    status: SourceResolutionStatus
    quote: str = ""
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    history_ref: str = ""
    record_ids: tuple[str, ...] = ()
    source_date: str = ""
    recorded_status: str = ""
    content: str = ""
    warning: str = ""


class EntrySourceResolution(FrozenStrictModel):
    entry_id: str = Field(min_length=1)
    sources: tuple[ResolvedSource, ...] = ()
    warnings: tuple[str, ...] = ()


class MinutesSources(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.minutes_sources.v1"] = (
        "summary_multi_agent.minutes_sources.v1"
    )
    history_aliases: dict[str, dict[str, Any]] = Field(default_factory=dict)
    entries: tuple[EntrySourceResolution, ...] = ()
    warnings: tuple[str, ...] = ()


class MinutesReportManifest(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.minutes_report_manifest.v1"] = (
        "summary_multi_agent.minutes_report_manifest.v1"
    )
    document_schema_version: str
    selected_candidate_index: int = 0
    summary_sha256: str = Field(min_length=64, max_length=64)
    output_mode: Literal["rich"] = "rich"
    artifact_status: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MinutesReportBundle:
    document: MinutesDocument
    sources: MinutesSources
    markdown: str
    html: str
    manifest: MinutesReportManifest
    warnings: tuple[str, ...] = ()
