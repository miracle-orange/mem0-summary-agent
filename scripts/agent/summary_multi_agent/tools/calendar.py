"""提供无副作用的日期和相对时间确定性解析工具。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import Field

from ..schemas import FrozenStrictModel, TaskType
from .registry import BaseTool, ToolContext


WEEKDAYS = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


# 输入/输出模型参与 Tool Schema 生成，使用普通注释说明而不改变 JSON Schema 描述。
class CalendarResolveInput(FrozenStrictModel):
    expression: str = Field(min_length=1)
    source_time: str = Field(min_length=1)
    direction: Literal["future", "past", "auto"] = "auto"


# Calendar Tool 输出结构化日期、精确性、规则和 warning，不修改请求原文。
class CalendarResolveOutput(FrozenStrictModel):
    original_expression: str
    source_date: str
    normalized_date: Optional[str] = None
    normalized_datetime: Optional[str] = None
    is_exact: bool = False
    rule: Optional[str] = None
    warning: Optional[str] = None


def _parse_source_date(source_time: str) -> Optional[datetime]:
    """解析 YYYY-MM-DD 锚点；格式不合法时返回 ``None``。"""
    try:
        return datetime.strptime(source_time.strip()[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _time_suffix(expression: str) -> Optional[str]:
    """从中文时间表达式提取并校验小时、分钟后缀。"""
    match = re.search(r"(\d{1,2})\s*[:：点时]\s*(\d{1,2})?\s*分?", expression)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if "下午" in expression or "晚上" in expression:
        if hour < 12:
            hour += 12
    elif "中午" in expression and hour < 11:
        hour += 12
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def resolve_relative_date(
    expression: str,
    source_time: str,
    *,
    direction: str = "auto",
) -> CalendarResolveOutput:
    """按固定规则解析相对日、星期和绝对日期，不调用模型或修改状态。

    无法解析或星期方向不可靠时返回 warning，并保留原始表达式，避免把歧义
    时间误写成确定日期。
    """
    text = expression.strip()
    anchor = _parse_source_date(source_time)
    if anchor is None:
        return CalendarResolveOutput(
            original_expression=text,
            source_date=source_time,
            warning="source_time is not a parseable YYYY-MM-DD date",
        )

    target: Optional[datetime] = None
    rule: Optional[str] = None
    exact = True
    fixed_offsets = {
        "今天": 0,
        "今日": 0,
        "明天": 1,
        "明日": 1,
        "后天": 2,
        "昨天": -1,
        "昨日": -1,
        "前天": -2,
    }
    for token, offset in fixed_offsets.items():
        if token in text:
            target = anchor + timedelta(days=offset)
            rule = f"relative_day:{token}"
            break

    if target is None:
        offset_match = re.search(r"(\d+)\s*天(后|前|内)", text)
        if offset_match:
            days = int(offset_match.group(1))
            marker = offset_match.group(2)
            target = anchor + timedelta(days=-days if marker == "前" else days)
            rule = f"relative_day_count:{marker}"

    if target is None:
        weekday_match = re.search(
            r"(?:(上|本|这|下)?周|星期|礼拜)([一二三四五六日天])", text
        )
        if weekday_match:
            week_marker = weekday_match.group(1)
            weekday = WEEKDAYS[weekday_match.group(2)]
            week_start = anchor - timedelta(days=anchor.weekday())
            if week_marker == "上":
                week_start -= timedelta(days=7)
                rule = "weekday:previous_week"
            elif week_marker == "下":
                week_start += timedelta(days=7)
                rule = "weekday:next_week"
            elif week_marker in {"本", "这"}:
                rule = "weekday:current_week"
            else:
                candidate = week_start + timedelta(days=weekday)
                resolved_direction = direction
                if resolved_direction == "auto":
                    return CalendarResolveOutput(
                        original_expression=text,
                        source_date=anchor.strftime("%Y-%m-%d"),
                        is_exact=False,
                        rule="weekday:ambiguous_direction",
                        warning=(
                            "unmarked weekday has no reliable past/future direction; "
                            "the original expression was preserved"
                        ),
                    )
                if resolved_direction == "future" and candidate < anchor:
                    candidate += timedelta(days=7)
                elif resolved_direction == "past" and candidate > anchor:
                    candidate -= timedelta(days=7)
                target = candidate
                rule = f"weekday:nearest_{resolved_direction}"
                exact = direction != "auto"
            if target is None:
                target = week_start + timedelta(days=weekday)

    if target is None:
        absolute_match = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
        if absolute_match:
            try:
                target = datetime(
                    int(absolute_match.group(1)),
                    int(absolute_match.group(2)),
                    int(absolute_match.group(3)),
                )
                rule = "absolute_date"
            except ValueError:
                target = None

    if target is None:
        return CalendarResolveOutput(
            original_expression=text,
            source_date=anchor.strftime("%Y-%m-%d"),
            warning="expression is not supported by deterministic calendar rules",
        )

    time_suffix = _time_suffix(text)
    normalized_date = target.strftime("%Y-%m-%d")
    return CalendarResolveOutput(
        original_expression=text,
        source_date=anchor.strftime("%Y-%m-%d"),
        normalized_date=normalized_date,
        normalized_datetime=(
            f"{normalized_date}T{time_suffix}" if time_suffix else None
        ),
        is_exact=exact,
        rule=rule,
    )


class CalendarResolveTool(BaseTool):
    # 该具体 Tool 的 __doc__ 会被 ToolRegistry 放入 Planner/Worker Prompt；保持为空。
    name = "calendar_resolve"
    input_model = CalendarResolveInput
    output_model = CalendarResolveOutput
    allowed_task_types = frozenset({TaskType.TEMPORAL_REASONING})

    async def execute(
        self, arguments: CalendarResolveInput, context: ToolContext
    ) -> CalendarResolveOutput:
        """执行无副作用的确定性日期解析；ToolContext 不参与日期推断。"""
        del context
        return resolve_relative_date(
            arguments.expression,
            arguments.source_time,
            direction=arguments.direction,
        )
