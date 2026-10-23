"""处理 V2 模型 JSON 提取、截断检测和重复输出检测。"""

from __future__ import annotations

import json
from typing import Any


def extract_json_value(raw_output: str) -> Any:
    """从模型文本中提取 JSON object/array，并兼容 Markdown 代码围栏。"""
    text = (raw_output or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [index for token in ("{", "[") if (index := text.find(token)) >= 0]
        if not starts:
            raise ValueError("model output does not contain JSON")
        parsed, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
        return parsed


def extract_json_object(raw_output: str) -> dict[str, Any]:
    """提取并确认模型顶层 JSON 是 object，否则抛出 ``ValueError``。"""
    parsed = extract_json_value(raw_output)
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON must be an object")
    return parsed


def looks_like_truncated_json(raw_output: str | None, error: Exception) -> bool:
    """用括号平衡和 JSON 解码错误识别疑似截断，供有限修复选择策略。"""
    text = (raw_output or "").strip()
    if not text:
        return False
    if text.count("{") > text.count("}") or text.count("[") > text.count("]"):
        return True
    return isinstance(error, json.JSONDecodeError) and not text.rstrip("`").endswith("}")


def looks_like_repetitive_output(raw_output: str | None) -> bool:
    """识别解码器循环，同时不把重复的 JSON key 误判为输出重复。"""
    text = (raw_output or "").strip()
    if len(text) < 36:
        return False
    sample = text[-2400:]
    min_unit_length = 12
    max_unit_length = min(200, len(sample) // 3)
    for unit_length in range(min_unit_length, max_unit_length + 1):
        repeated_length = unit_length * 3
        for start in range(0, len(sample) - repeated_length + 1):
            unit = sample[start : start + unit_length]
            if unit.isspace():
                continue
            if sample[start + unit_length : start + repeated_length] == unit * 2:
                return True
    return False
