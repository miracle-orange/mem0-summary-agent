"""Rich meeting-minutes document, source resolution, and deterministic rendering."""

from .models import (
    MinutesDocument,
    MinutesDocumentDraft,
    MinutesEntry,
    MinutesEntryDraft,
    MinutesReportBundle,
)
from .render import (
    build_report_bundle,
    render_action_risk_markdown,
    render_history_markdown,
    render_minutes_html,
    render_minutes_markdown,
    render_sources_markdown,
    render_web_report_fragment,
)
from .artifacts import persist_report_bundle

__all__ = [
    "MinutesDocument",
    "MinutesDocumentDraft",
    "MinutesEntry",
    "MinutesEntryDraft",
    "MinutesReportBundle",
    "build_report_bundle",
    "render_action_risk_markdown",
    "render_history_markdown",
    "render_minutes_html",
    "render_minutes_markdown",
    "render_sources_markdown",
    "render_web_report_fragment",
    "persist_report_bundle",
]
