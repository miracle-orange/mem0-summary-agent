"""Small, deterministic projections for the meeting workspace UI."""

from __future__ import annotations

import html
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


_RUN_NAMES = {
    "ingest": "资料导入",
    "multi_agent_summarize": "会议纪要",
    "summarize": "会议纪要",
    "evaluation_compare": "纪要横评",
    "e2e_run": "完整流程",
    "e2e_ingest": "资料导入",
}
_STATUS_NAMES = {
    "queued": "排队中",
    "running": "处理中",
    "completed": "已完成",
    "completed_with_warnings": "有提示",
    "failed": "失败",
    "cancelling": "停止中",
    "cancelled": "已停止",
}


def file_date_rows(files: Any, previous: Any = None) -> list[list[str]]:
    """Keep dates attached to uploaded file positions when the selection changes."""
    paths = files if isinstance(files, (list, tuple)) else [files] if files else []
    old_rows = _rows(previous)
    dates_by_name: dict[str, deque[str]] = defaultdict(deque)
    for row in old_rows:
        if len(row) > 1:
            dates_by_name[str(row[0])].append(_date_cell(row[1]))
    result = []
    for path in paths:
        name = _upload_name(path)
        dates = dates_by_name[name]
        result.append([name, dates.popleft() if dates else ""])
    return result


def _upload_name(value: Any) -> str:
    return Path(str(getattr(value, "orig_name", None) or getattr(value, "name", value))).name


def _date_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        return value.values.tolist()
    if isinstance(value, dict):
        return value.get("data", [])
    return list(value)


def combine_source_dates(
    *, pasted: str, uploaded: Any, server_paths: str, manual_dates: str, table: Any
) -> str:
    """Translate the editable upload list into the existing ordered source-date API."""
    paths = uploaded if isinstance(uploaded, (list, tuple)) else [uploaded] if uploaded else []
    if not paths:
        return manual_dates or ""
    rows = _rows(table)
    expected = [_upload_name(path) for path in paths]
    if len(rows) != len(expected) or any(
        len(row) < 2 or str(row[0]).strip() != name
        for row, name in zip(rows, expected)
    ):
        raise ValueError("文件清单与当前上传文件不一致，请重新选择文件")
    path_count = sum(bool(line.strip()) for line in (server_paths or "").splitlines())
    pasted_count = int(bool((pasted or "").strip()))
    manual = (manual_dates or "").splitlines()
    if len(manual) > pasted_count + path_count:
        raise ValueError("高级逐份日期行数不能超过粘贴和服务器路径的资料份数")
    manual.extend("" for _ in range(pasted_count + path_count - len(manual)))
    dates = manual[:pasted_count] + [_date_cell(row[1]) for row in rows]
    dates.extend(manual[pasted_count:])
    return "\n".join(dates)


def ready_message(files: Any, pasted: str = "") -> str:
    count = len(files) if isinstance(files, (list, tuple)) else int(bool(files))
    count += int(bool((pasted or "").strip()))
    if not count:
        return '<div class="smw-ready muted">尚未添加资料</div>'
    return f'<div class="smw-ready"><span aria-hidden="true">&#10003;</span> {count} 份资料已就绪</div>'


def filtered_runs(entries: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    needle = (query or "").strip().casefold()
    return [
        item for item in entries
        if not needle or needle in " ".join(str(value) for value in item.values()).casefold()
    ]


def run_label(item: dict[str, str]) -> str:
    kind = _RUN_NAMES.get(item.get("run_type", ""), "运行")
    status = _STATUS_NAMES.get(item.get("status", ""), item.get("status", "未知"))
    project = item.get("project") or Path(item.get("output_dir") or "").name
    date = (item.get("source_time") or item.get("started_at") or "")[:10]
    return f"{project} · {kind} · {date} · {status}"


def run_list_html(entries: list[dict[str, str]], *, empty: str = "暂无运行") -> str:
    if not entries:
        return f'<div class="smw-list-empty">{html.escape(empty)}</div>'
    cards = []
    for item in entries[:12]:
        kind = _RUN_NAMES.get(item.get("run_type", ""), "运行")
        status_key = item.get("status", "")
        status = _STATUS_NAMES.get(status_key, status_key or "未知")
        project = item.get("project") or Path(item.get("output_dir") or "").name
        date = (item.get("source_time") or item.get("started_at") or "")[:10]
        cards.append(
            '<div class="smw-run-item">'
            f'<strong>{html.escape(str(project))}</strong><span class="smw-status smw-status-{html.escape(status_key)}">{html.escape(status)}</span>'
            f'<small>{html.escape(kind)} · {html.escape(date)}</small></div>'
        )
    return '<div class="smw-run-list">' + "".join(cards) + "</div>"


def run_heading_html(*, title: str, project: str, source_time: str, status: str) -> str:
    return (
        '<div class="smw-result-heading"><div>'
        f'<h2>{html.escape(title or project or "运行结果")}</h2>'
        f'<p>{html.escape(source_time or "日期未明确")} &nbsp; | &nbsp; {html.escape(project or "项目未明确")}</p>'
        '</div>'
        f'<span class="smw-status smw-status-{html.escape(status)}">{html.escape(_STATUS_NAMES.get(status, status or "未知"))}</span>'
        '</div>'
    )
