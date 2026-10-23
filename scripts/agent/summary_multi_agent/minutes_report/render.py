"""Deterministic Markdown and offline HTML rendering for rich minutes."""

from __future__ import annotations

import hashlib
import html
from collections import defaultdict
from typing import Iterable

from .models import (
    MinutesDocument,
    MinutesReportBundle,
    MinutesReportManifest,
    MinutesSources,
)


_KIND_LABELS = {
    "fact": "事实",
    "discussion": "讨论与方案",
    "decision": "决定",
    "action": "行动",
    "risk": "风险",
    "open_question": "待确认",
    "history_update": "历史延续",
}


def _md_text(value: str) -> str:
    text = str(value or "")
    for source, replacement in (
        ("\\", "\\\\"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("[", "\\["),
        ("]", "\\]"),
    ):
        text = text.replace(source, replacement)
    return text


def _md_cell(value: str) -> str:
    return _md_text(value or "未明确").replace("|", "\\|").replace("\n", "<br>")


def _source_index(sources: MinutesSources) -> dict[str, object]:
    return {item.entry_id: item for item in sources.entries}


def _entry_text(entry) -> str:
    prefix = f"**{_md_text(entry.title)}**：" if entry.title else ""
    return prefix + _md_text(entry.content)


def _history_source_label(resolved) -> str:
    dates = []
    if resolved is not None:
        dates = [
            source.source_date or "来源日期未明确"
            for source in resolved.sources
            if source.source_kind == "history"
        ]
    return "、".join(dict.fromkeys(dates)) or "来源日期未明确"


def _natural_entry_text(entry, resolved=None) -> str:
    prefix = f"{entry.title}：" if entry.title else ""
    text = prefix + entry.content
    details: list[str] = []
    if entry.kind == "action":
        details.extend(
            (
                f"负责人：{entry.owner or '未明确'}",
                f"时间要求：{entry.time_requirement or '未明确'}",
                f"前提或依赖：{entry.dependency or '未明确'}",
            )
        )
    elif entry.kind == "history_update":
        details.extend(
            (
                f"历史来源日期：{_history_source_label(resolved)}",
                f"历史记录时情况：{entry.recorded_state or entry.content}",
                f"本次情况：{entry.current_state or '未明确'}",
                f"关系说明：{entry.relation or '未明确'}",
            )
        )
    if details:
        text += "（" + "；".join(details) + "）"
    return text


def render_minutes_markdown(
    document: MinutesDocument,
    sources: MinutesSources | None = None,
) -> str:
    source_map = _source_index(sources) if sources is not None else {}
    lines = [f"# {_md_text(document.title)}"]
    if document.source_time:
        lines.extend(["", f"> 材料日期：{_md_text(document.source_time)}"])
    lines.extend(["", "## 会议概览"])
    lines.extend(f"- {_md_text(item)}" for item in document.overview)

    for section in document.sections:
        lines.extend(["", f"## {_md_text(section.heading)}"])
        if document.format_kind == "natural":
            for item in section.entries:
                lines.extend(
                    [
                        "",
                        _md_text(
                            _natural_entry_text(item, source_map.get(item.entry_id))
                        ),
                    ]
                )
            continue
        normal = [
            item
            for item in section.entries
            if item.kind not in {"action", "history_update"}
        ]
        actions = [item for item in section.entries if item.kind == "action"]
        histories = [item for item in section.entries if item.kind == "history_update"]
        for item in normal:
            label = _KIND_LABELS[item.kind]
            lines.append(f"- **{label}**：{_entry_text(item)}")
        if actions:
            lines.extend(
                [
                    "",
                    "| 事项 | 负责人 | 时间要求 | 前提或依赖 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            lines.extend(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in (
                        item.content,
                        item.owner,
                        item.time_requirement,
                        item.dependency,
                    )
                )
                + " |"
                for item in actions
            )
        if histories:
            lines.extend(
                [
                    "",
                    "| 历史来源日期 | 历史记录时情况 | 本次情况 | 关系说明 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            lines.extend(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in (
                        _history_source_label(source_map.get(item.entry_id)),
                        item.recorded_state or item.content,
                        item.current_state,
                        item.relation,
                    )
                )
                + " |"
                for item in histories
            )
    return "\n".join(lines).strip() + "\n"


def render_action_risk_markdown(document: MinutesDocument) -> str:
    actions = [
        item
        for section in document.sections
        for item in section.entries
        if item.kind == "action"
    ]
    risks = [
        item
        for section in document.sections
        for item in section.entries
        if item.kind in {"risk", "open_question"}
    ]
    lines = ["## 行动与风险"]
    if actions:
        lines.extend(
            [
                "",
                "### 行动项",
                "",
                "| 事项 | 负责人 | 时间要求 | 前提或依赖 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in actions:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in (
                        item.content,
                        item.owner,
                        item.time_requirement,
                        item.dependency,
                    )
                )
                + " |"
            )
    if risks:
        lines.extend(["", "### 风险与待确认"])
        for item in risks:
            lines.append(f"- **{_KIND_LABELS[item.kind]}**：{_entry_text(item)}")
    if not actions and not risks:
        lines.extend(["", "本次纪要没有提取到有依据的行动项、风险或待确认事项。"])
    return "\n".join(lines).strip() + "\n"


def render_history_markdown(document: MinutesDocument, sources: MinutesSources) -> str:
    source_map = _source_index(sources)
    history = [
        item
        for section in document.sections
        for item in section.entries
        if item.kind == "history_update"
    ]
    lines = ["## 历史延续"]
    if not history:
        lines.extend(["", "本次成果没有形成有当前依据的历史延续或状态变化条目。"])
        return "\n".join(lines) + "\n"
    for item in history:
        resolved = source_map.get(item.entry_id)
        dates = []
        if resolved is not None:
            dates = [
                source.source_date or "来源日期未明确"
                for source in resolved.sources
                if source.source_kind == "history"
            ]
        label = "、".join(dict.fromkeys(dates)) or "来源日期未明确"
        lines.extend(
            [
                "",
                f"### {_md_text(item.title or item.content)}",
                f"- 历史来源：{_md_text(label)}",
                f"- 记录时情况：{_md_text(item.recorded_state or item.content)}",
                f"- 本次情况：{_md_text(item.current_state or '未明确')}",
                f"- 关系说明：{_md_text(item.relation or '未明确')}",
            ]
        )
        if resolved is not None:
            lines.extend(f"- 核查提示：{_md_text(warning)}" for warning in resolved.warnings)
    return "\n".join(lines).strip() + "\n"


def render_sources_markdown(document: MinutesDocument, sources: MinutesSources) -> str:
    entries = {
        item.entry_id: item
        for section in document.sections
        for item in section.entries
    }
    lines = ["## 来源查看"]
    any_source = False
    for resolution in sources.entries:
        if not resolution.sources and not resolution.warnings:
            continue
        any_source = True
        entry = entries.get(resolution.entry_id)
        lines.extend(["", f"### {_md_text((entry.title or entry.content) if entry else resolution.entry_id)}"])
        for source in resolution.sources:
            if source.source_kind == "current":
                position = (
                    f"字符 {source.start_offset}–{source.end_offset}"
                    if source.start_offset is not None
                    else "位置未唯一确定"
                )
                lines.append(
                    f"- 当前材料（{_md_text(source.status)}，{_md_text(position)}）：{_md_text(source.quote)}"
                )
            else:
                date = source.source_date or "来源日期未明确"
                state = source.recorded_status or "记录时状态未明确"
                lines.append(
                    f"- 历史记忆记录 {_md_text(source.history_ref)}（{_md_text(date)}，{_md_text(state)}）：{_md_text(source.content)}"
                )
            if source.warning:
                lines.append(f"  - 核查提示：{_md_text(source.warning)}")
        lines.extend(f"- 核查提示：{_md_text(warning)}" for warning in resolution.warnings)
    if not any_source:
        lines.extend(["", "本次成果没有可展示的条目级来源引用。"])
    return "\n".join(lines).strip() + "\n"


def _entry_html(entry, source_map: dict[str, object]) -> str:
    label = html.escape(_KIND_LABELS[entry.kind])
    title = f"<strong>{html.escape(entry.title)}</strong>：" if entry.title else ""
    body = f'<div class="mr-entry mr-{entry.kind}"><span class="mr-kind">{label}</span>{title}{html.escape(entry.content)}</div>'
    return body + _source_details_html(entry, source_map)


def _source_details_html(entry, source_map: dict[str, object]) -> str:
    source = source_map.get(entry.entry_id)
    if source is None or (not source.sources and not source.warnings):
        return ""
    details: list[str] = []
    for item in source.sources:
        if item.source_kind == "current":
            position = (
                f"字符 {item.start_offset}–{item.end_offset}"
                if item.start_offset is not None
                else "位置未唯一确定"
            )
            text = f"当前材料 · {position}：{item.quote}"
        else:
            text = (
                f"历史记忆记录 {item.history_ref} · {item.source_date or '来源日期未明确'}"
                f" · {item.recorded_status or '记录时状态未明确'}：{item.content}"
            )
        if item.warning:
            text += f"（{item.warning}）"
        details.append(f"<li>{html.escape(text)}</li>")
    details.extend(f"<li>{html.escape(item)}</li>" for item in source.warnings)
    return (
        '<details class="mr-sources"><summary>查看来源与核查状态</summary><ul>'
        + "".join(details)
        + "</ul></details>"
    )


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value or '未明确')}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="mr-table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _full_minutes_html(document: MinutesDocument, sources: MinutesSources) -> str:
    source_map = _source_index(sources)
    parts = [
        f'<header class="mr-hero"><p class="mr-eyebrow">会议纪要成果</p><h1>{html.escape(document.title)}</h1>',
        f'<p class="mr-date">材料日期：{html.escape(document.source_time or "未明确")}</p></header>',
        '<section class="mr-overview"><h2>会议概览</h2><ul>',
        "".join(f"<li>{html.escape(item)}</li>" for item in document.overview),
        "</ul></section>",
    ]
    for section in document.sections:
        if document.format_kind == "natural":
            parts.append(
                f'<section class="mr-section" id="{html.escape(section.section_id)}"><h2>{html.escape(section.heading)}</h2>'
            )
            for entry in section.entries:
                parts.append(
                    '<p class="mr-natural-entry">'
                    + html.escape(
                        _natural_entry_text(entry, source_map.get(entry.entry_id))
                    )
                    + "</p>"
                    + _source_details_html(entry, source_map)
                )
            parts.append("</section>")
            continue
        normal = [entry for entry in section.entries if entry.kind not in {"action", "history_update"}]
        actions = [entry for entry in section.entries if entry.kind == "action"]
        histories = [entry for entry in section.entries if entry.kind == "history_update"]
        parts.append(
            f'<section class="mr-section" id="{html.escape(section.section_id)}"><h2>{html.escape(section.heading)}</h2>'
        )
        parts.extend(_entry_html(entry, source_map) for entry in normal)
        if actions:
            parts.append(
                _table(
                    ("事项", "负责人", "时间要求", "前提或依赖"),
                    (
                        (entry.content, entry.owner, entry.time_requirement, entry.dependency)
                        for entry in actions
                    ),
                )
            )
            parts.extend(_source_details_html(entry, source_map) for entry in actions)
        if histories:
            parts.append(
                _table(
                    ("历史来源日期", "历史记录时情况", "本次情况", "关系说明"),
                    (
                        (
                            _history_source_label(source_map.get(entry.entry_id)),
                            entry.recorded_state or entry.content,
                            entry.current_state,
                            entry.relation,
                        )
                        for entry in histories
                    ),
                )
            )
            parts.extend(_source_details_html(entry, source_map) for entry in histories)
        parts.append("</section>")
    return "".join(parts)


def _markdown_view_html(markdown: str) -> str:
    """Render known, program-generated Markdown views without accepting arbitrary HTML."""
    lines = markdown.splitlines()
    parts: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("## "):
            if in_table:
                parts.append("</tbody></table></div>")
                in_table = False
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_table:
                parts.append("</tbody></table></div>")
                in_table = False
            parts.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| "):
            cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", " "} for cell in cells):
                continue
            if not in_table:
                parts.append('<div class="mr-table-wrap"><table><tbody>')
                in_table = True
            parts.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
        elif line.startswith("- "):
            if in_table:
                parts.append("</tbody></table></div>")
                in_table = False
            parts.append(f'<p class="mr-list">{html.escape(line[2:])}</p>')
        elif line.strip():
            parts.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        parts.append("</tbody></table></div>")
    return "".join(parts)


_STYLE = """
.mr-report{--ink:#17202a;--muted:#5d6874;--line:#d9dee5;--paper:#fff;--soft:#f5f7f9;--accent:#176b62;color:var(--ink);background:var(--paper);font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1180px;margin:auto}.mr-report *{box-sizing:border-box}.mr-shell{display:grid;grid-template-columns:220px minmax(0,1fr);gap:28px}.mr-nav{position:sticky;top:16px;align-self:start;border-right:1px solid var(--line);padding:8px 20px 16px 0}.mr-nav strong{display:block;margin-bottom:10px}.mr-nav a,.mr-mobile-nav a{display:block;color:var(--muted);text-decoration:none;padding:5px 0}.mr-mobile-nav{display:none}.mr-main{min-width:0}.mr-hero{border-bottom:2px solid var(--ink);padding:28px 0 20px}.mr-hero h1{font-size:30px;line-height:1.25;margin:2px 0 8px;letter-spacing:0}.mr-eyebrow{color:var(--accent);font-weight:700;margin:0}.mr-date{color:var(--muted);margin:0}.mr-overview{background:var(--soft);border-left:4px solid var(--accent);padding:15px 20px;margin:24px 0}.mr-section{padding:20px 0;border-bottom:1px solid var(--line);break-inside:avoid}.mr-section h2,.mr-view h2{font-size:21px;margin:0 0 12px}.mr-entry{margin:10px 0}.mr-kind{display:inline-block;color:var(--accent);font-size:12px;font-weight:700;border:1px solid #9bc5bf;border-radius:4px;padding:0 6px;margin-right:8px}.mr-table-wrap{max-width:100%;overflow-x:auto;margin:14px 0}.mr-report table{width:100%;border-collapse:collapse;min-width:620px}.mr-report th,.mr-report td{text-align:left;vertical-align:top;border:1px solid var(--line);padding:8px 10px}.mr-report th{background:var(--soft)}.mr-sources{margin:6px 0 12px;color:var(--muted);font-size:13px}.mr-sources summary{cursor:pointer;color:#375c73}.mr-tabs{margin-top:18px}.mr-tabs>input{position:absolute;opacity:0;pointer-events:none}.mr-tab-labels{display:flex;gap:4px;border-bottom:1px solid var(--line);overflow-x:auto}.mr-tab-labels label{cursor:pointer;white-space:nowrap;padding:9px 13px;color:var(--muted)}.mr-panels>.mr-view{display:none;padding:18px 0}.mr-tabs input:nth-of-type(1):checked~.mr-tab-labels label:nth-child(1),.mr-tabs input:nth-of-type(2):checked~.mr-tab-labels label:nth-child(2),.mr-tabs input:nth-of-type(3):checked~.mr-tab-labels label:nth-child(3),.mr-tabs input:nth-of-type(4):checked~.mr-tab-labels label:nth-child(4){color:var(--accent);font-weight:700;border-bottom:3px solid var(--accent)}.mr-tabs input:nth-of-type(1):checked~.mr-panels .mr-view:nth-child(1),.mr-tabs input:nth-of-type(2):checked~.mr-panels .mr-view:nth-child(2),.mr-tabs input:nth-of-type(3):checked~.mr-panels .mr-view:nth-child(3),.mr-tabs input:nth-of-type(4):checked~.mr-panels .mr-view:nth-child(4){display:block}.mr-list{padding-left:18px;position:relative}.mr-list:before{content:"•";position:absolute;left:2px;color:var(--accent)}@media(max-width:760px){.mr-shell{display:block}.mr-nav{display:none}.mr-mobile-nav{display:block;border-bottom:1px solid var(--line);padding:8px 0}.mr-mobile-nav summary{cursor:pointer;color:var(--accent);font-weight:700}.mr-mobile-nav div{padding:6px 12px}.mr-hero h1{font-size:24px}.mr-tab-labels label{padding:8px}.mr-report table{min-width:560px}}@media print{.mr-report{max-width:none}.mr-nav,.mr-mobile-nav,.mr-tab-labels{display:none}.mr-shell{display:block}.mr-panels>.mr-view{display:none!important}.mr-panels>.mr-view:nth-child(1),.mr-panels>.mr-view:nth-child(4){display:block!important}.mr-panels>.mr-view:nth-child(4){page-break-before:always}.mr-sources{display:none}.mr-section,.mr-table-wrap{break-inside:avoid}.mr-report table{min-width:0;font-size:11px}}
"""


def render_web_report_fragment(document: MinutesDocument, sources: MinutesSources) -> str:
    nav = "".join(
        f'<a href="#{html.escape(section.section_id)}">{html.escape(section.heading)}</a>'
        for section in document.sections
    )
    full = _full_minutes_html(document, sources)
    action = _markdown_view_html(render_action_risk_markdown(document))
    history = _markdown_view_html(render_history_markdown(document, sources))
    source = _markdown_view_html(render_sources_markdown(document, sources))
    tabs = """<div class="mr-tabs">
<input id="mr-tab-full" name="mr-tab" type="radio" checked><input id="mr-tab-action" name="mr-tab" type="radio"><input id="mr-tab-history" name="mr-tab" type="radio"><input id="mr-tab-source" name="mr-tab" type="radio">
<div class="mr-tab-labels"><label for="mr-tab-full">完整纪要</label><label for="mr-tab-action">行动与风险</label><label for="mr-tab-history">历史延续</label><label for="mr-tab-source">来源</label></div>
<div class="mr-panels"><div class="mr-view">FULL</div><div class="mr-view">ACTION</div><div class="mr-view">HISTORY</div><div class="mr-view">SOURCE</div></div></div>"""
    tabs = tabs.replace("FULL", full).replace("ACTION", action).replace("HISTORY", history).replace("SOURCE", source)
    mobile_nav = (
        '<details class="mr-mobile-nav"><summary>议题导航</summary><div>'
        + nav
        + "</div></details>"
    )
    return f'<style>{_STYLE}</style><div class="mr-report"><div class="mr-shell"><nav class="mr-nav"><strong>议题导航</strong>{nav}</nav><main class="mr-main">{mobile_nav}{tabs}</main></div></div>'


def render_minutes_html(document: MinutesDocument, sources: MinutesSources) -> str:
    fragment = render_web_report_fragment(document, sources)
    return "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>" + html.escape(document.title) + "</title></head><body>" + fragment + "</body></html>"


def build_report_bundle(document: MinutesDocument, sources: MinutesSources) -> MinutesReportBundle:
    markdown = render_minutes_markdown(document, sources)
    warnings = list(sources.warnings)
    artifact_status = {"minutes_document.json": "completed", "minutes_sources.json": "completed", "minutes_report.html": "completed"}
    try:
        html_output = render_minutes_html(document, sources)
    except Exception as exc:  # HTML is a non-critical derivative.
        html_output = ""
        artifact_status["minutes_report.html"] = "failed"
        warnings.append(f"minutes HTML rendering failed: {type(exc).__name__}: {exc}")
    manifest = MinutesReportManifest(
        document_schema_version=document.schema_version,
        summary_sha256=hashlib.sha256(markdown.rstrip().encode("utf-8")).hexdigest(),
        artifact_status=artifact_status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return MinutesReportBundle(
        document=document,
        sources=sources,
        markdown=markdown,
        html=html_output,
        manifest=manifest,
        warnings=tuple(dict.fromkeys(warnings)),
    )
