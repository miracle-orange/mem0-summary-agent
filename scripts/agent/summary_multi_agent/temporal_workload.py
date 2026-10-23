"""扫描确定性时间工作量，保证 Planner 为时间解析分配足够工具预算。"""

from __future__ import annotations

import re
from typing import Any

from .schemas import RawMaterialBundle


REVIEW_TEMPORAL_SCOPE_INPUT = "review_temporal_scope"


TIME_EXPRESSION_PATTERN = re.compile(
    r"(?:今天|今日|明天|明日|后天|昨天|昨日|前天)"
    r"(?:上午|中午|下午|晚上)?(?:\d{1,2}(?:[:：点时]\d{0,2})?分?)?"
    r"|(?:(?:上|本|这|下)?周|星期|礼拜)[一二三四五六日天]"
    r"(?:之前|以前|前|之后|以后|后|开始)?"
    r"(?:上午|中午|下午|晚上)?(?:\d{1,2}(?:[:：点时]\d{0,2})?分?)?"
    r"|\d+\s*天(?:后|前|内)"
    r"|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?"
)

CLAUSE_BOUNDARY_PATTERN = re.compile(r"[。！？!?；;\n\r]")
PAST_CUE_PATTERN = re.compile(
    r"已经|曾经?|此前|先前|过去|上次|回顾|"
    r"(?:已|业已)(?:完成|提交|发送|交付|结束|确认|发布|召开|举行|补交|处理|关闭)|"
    r"(?:完成|提交|发送|发出|发到|发给|交付|结束|确认|发布|召开|举行|补交|处理|关闭|收到|回复|通知)"
    r"[^，。！？!?；;\n\r]{0,12}(?:了|过)|(?:发|交)(?:了|过)"
)
FUTURE_CUE_PATTERN = re.compile(
    r"将(?:于)?|计划|预计|拟(?:于|在|定)|(?:有待|待办|待完成|待提交|待处理|待确认)|"
    r"尚未|还(?:没|未)|需要|应当|应该|必须|截止|截至|届时|安排|准备|"
    r"请(?:于|在)"
)


def _direction_for_context(segment_text: str, local_start: int) -> str:
    """根据同一分句内最近的过去/未来提示词推断时间方向。"""
    left_boundary = 0
    for match in CLAUSE_BOUNDARY_PATTERN.finditer(segment_text, 0, local_start):
        left_boundary = match.end()
    right_match = CLAUSE_BOUNDARY_PATTERN.search(segment_text, local_start)
    right_boundary = right_match.start() if right_match else len(segment_text)
    clause = segment_text[left_boundary:right_boundary]
    expression_start = local_start - left_boundary

    def nearest_distance(pattern: re.Pattern[str]) -> int | None:
        """计算提示词到时间表达式的最近字符距离。"""
        distances: list[int] = []
        for cue in pattern.finditer(clause):
            if cue.end() <= expression_start:
                distances.append(expression_start - cue.end())
            elif cue.start() >= expression_start:
                distances.append(cue.start() - expression_start)
            else:
                distances.append(0)
        return min(distances) if distances else None

    past_distance = nearest_distance(PAST_CUE_PATTERN)
    future_distance = nearest_distance(FUTURE_CUE_PATTERN)
    if past_distance is None and future_distance is None:
        return "auto"
    if future_distance is None:
        return "past"
    if past_distance is None:
        return "future"
    if past_distance < future_distance:
        return "past"
    if future_distance < past_distance:
        return "future"
    return "auto"


def find_temporal_expressions(
    raw_material: RawMaterialBundle,
) -> list[dict[str, Any]]:
    """扫描完整原文并去除重叠分段造成的重复时间表达式。"""

    found: dict[tuple[int, int, str], dict[str, Any]] = {}
    for segment in raw_material.segments:
        for match in TIME_EXPRESSION_PATTERN.finditer(segment.text):
            expression = match.group(0).strip()
            global_start = segment.start_offset + match.start()
            global_end = segment.start_offset + match.end()
            key = (global_start, global_end, expression)
            found[key] = {
                "expression": expression,
                "segment_id": segment.segment_id,
                "start_offset": global_start,
                "end_offset": global_end,
                "direction": _direction_for_context(segment.text, match.start()),
            }
    ordered = sorted(found, key=lambda item: (item[0], item[1], item[2]))
    return [found[key] for key in ordered]


def uses_review_temporal_scope(task_id: str, inputs: dict[str, Any]) -> bool:
    """仅允许内部 ReviewVerifier 任务缩小确定性时间覆盖范围。"""

    return task_id.startswith("review_verify_") and REVIEW_TEMPORAL_SCOPE_INPUT in inputs


def find_task_temporal_expressions(
    raw_material: RawMaterialBundle,
    *,
    task_id: str,
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """返回生产全量时间覆盖，或已验证的 Review issue 子集。"""

    expressions = find_temporal_expressions(raw_material)
    if not uses_review_temporal_scope(task_id, inputs):
        return expressions

    scope = inputs.get(REVIEW_TEMPORAL_SCOPE_INPUT)
    if not isinstance(scope, dict) or not isinstance(scope.get("expressions"), list):
        raise ValueError("review temporal scope must contain an expressions list")

    known = {
        (
            item["expression"],
            item["segment_id"],
            item["start_offset"],
            item["end_offset"],
            item["direction"],
        ): item
        for item in expressions
    }
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for requested in scope["expressions"]:
        if not isinstance(requested, dict):
            raise ValueError("review temporal scope expressions must be objects")
        key = (
            str(requested.get("expression") or ""),
            str(requested.get("segment_id") or ""),
            int(requested.get("start_offset", -1)),
            int(requested.get("end_offset", -1)),
            str(requested.get("direction") or ""),
        )
        if key not in known:
            raise ValueError(
                "review temporal scope contains an expression outside the "
                "deterministic current-material scan"
            )
        if key not in seen:
            selected.append(known[key])
            seen.add(key)
    return selected
