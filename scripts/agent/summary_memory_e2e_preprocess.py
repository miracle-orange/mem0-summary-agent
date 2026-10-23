"""End-to-end raw material preprocessing for the Summary Memory web demo.

This module intentionally stays outside the stable workflow scripts. It creates
compact base summaries for display and comparison. The web wrapper separately
feeds authoritative raw chunks into ingest and raw current material into summarize.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_CHUNK_CHAR_LIMIT = 5000
DEFAULT_CHUNK_MAX_TOKENS = 1200
DEFAULT_MERGE_MAX_TOKENS = 1600


@dataclass
class RawTextSource:
    label: str
    text: str
    source_path: Optional[Path] = None


@dataclass
class GeneratedSummary:
    label: str
    source_path: Optional[Path]
    raw_text: str
    summary_text: str
    chunk_count: int
    raw_char_count: int
    summary_char_count: int


def get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def get_preprocess_config() -> dict[str, int]:
    return {
        "chunk_char_limit": get_int_env("SMW_E2E_CHUNK_CHAR_LIMIT", DEFAULT_CHUNK_CHAR_LIMIT),
        "chunk_max_tokens": get_int_env("SMW_E2E_CHUNK_MAX_TOKENS", DEFAULT_CHUNK_MAX_TOKENS),
        "merge_max_tokens": get_int_env("SMW_E2E_MERGE_MAX_TOKENS", DEFAULT_MERGE_MAX_TOKENS),
    }


def make_path_segment(value: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    text = text.strip("._-")
    return text[:80] or fallback


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def split_text_into_chunks(text: str, chunk_char_limit: int) -> list[str]:
    # 先按段落聚合，只有超长段落才硬切，尽量保留历史材料的语义边界。
    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= chunk_char_limit:
        return [clean_text]

    paragraphs = re.split(r"\n\s*\n", clean_text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_char_limit:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            for start in range(0, len(paragraph), chunk_char_limit):
                chunks.append(paragraph[start : start + chunk_char_limit])
            continue

        next_len = current_len + len(paragraph) + (2 if current_parts else 0)
        if current_parts and next_len > chunk_char_limit:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_len = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_len = next_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def call_workflow_llm(workflow: Any, prompt: str, max_tokens: int) -> str:
    llm_call = getattr(workflow, "call_summary_llm", None)
    if llm_call is None:
        llm_call = getattr(workflow, "call_local_qwen", None)
    if llm_call is None:
        raise RuntimeError("workflow 模块缺少 call_summary_llm，无法执行 E2E 原始资料摘要预处理。")
    return (llm_call(prompt, temperature=0.0, max_tokens=max_tokens) or "").strip()


def build_source_time_hint(source_role: str, source_time: str = "") -> str:
    clean_time = (source_time or "").strip()
    if source_role != "current" or not clean_time:
        return (
            "时间处理：保留原文中的明确日期、时间段、星期、相对时间和阶段表达。"
            "历史资料可能有自己的时间锚点，不要用当前材料日期强行换算历史资料里的相对时间。"
        )

    return (
        f"当前资料时间锚点 source_time：{clean_time}\n"
        "时间处理：source_time 只作为当前资料的轻量时间锚点，用于保留和解释原文中的相对时间。"
        "基础摘要阶段不要承担最终时间裁决，不要把不确定推理写成绝对事实。"
        "如果原文没有明确“本周/下周/上周”，不要擅自把“周三前/周五后/周几开始”等改写成“本周三前/本周五后”。"
        "可以保留原表达，并在非常明确时补充候选说明；依据不足时写“具体日期待增强摘要阶段结合上下文确认”。"
    )


def build_chunk_summary_prompt(
    text: str,
    source_role: str,
    label: str,
    chunk_index: int,
    chunk_count: int,
    source_time: str = "",
) -> str:
    role_hint = "历史资料" if source_role == "history" else "当前资料"
    time_hint = build_source_time_hint(source_role, source_time)
    return f"""你是一个中文资料摘要助手。请把下面这段{role_hint}原始资料压缩成可供后续摘要记忆流程使用的基础摘要。

要求：
1. 保留事实、背景、目标、约束、关键决策、任务、负责人、时间、风险、待确认问题。
2. 去除寒暄、重复表达、无后续价值的细节。
3. 不编造原文没有的信息。
4. 用中文输出，结构清晰，优先使用条目。
5. 如果原文是会议转写，请合并同义内容并保留行动项。
6. {time_hint}

资料标签：{label}
分块：{chunk_index}/{chunk_count}

原始资料：
{text}
"""


def build_merge_summary_prompt(
    chunk_summaries: list[str],
    source_role: str,
    label: str,
    source_time: str = "",
) -> str:
    role_hint = "历史资料" if source_role == "history" else "当前资料"
    time_hint = build_source_time_hint(source_role, source_time)
    joined = "\n\n---\n\n".join(
        f"# 分块摘要 {index}\n\n{summary}" for index, summary in enumerate(chunk_summaries, start=1)
    )
    return f"""你是一个中文资料摘要助手。下面是同一份{role_hint}原始资料的多个分块摘要。

请把它们合并成一份完整、去重、可供后续摘要记忆流程使用的基础摘要。

要求：
1. 保留跨分块反复出现或前后依赖的背景、目标、决策、任务、风险、约束、待确认问题。
2. 合并重复信息，修正分块之间的顺序和指代。
3. 不新增原文和分块摘要没有的信息。
4. 用中文输出，结构清晰，优先使用条目。
5. {time_hint}

资料标签：{label}

分块摘要：
{joined}
"""


def summarize_raw_source(
    workflow: Any,
    source: RawTextSource,
    source_role: str,
    source_time: str = "",
    config: Optional[dict[str, int]] = None,
) -> GeneratedSummary:
    effective_config = dict(config or get_preprocess_config())
    chunks = split_text_into_chunks(source.text, effective_config["chunk_char_limit"])
    if not chunks:
        raise ValueError(f"原始资料为空：{source.label}")

    chunk_summaries: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = build_chunk_summary_prompt(chunk, source_role, source.label, index, len(chunks), source_time)
        chunk_summaries.append(
            call_workflow_llm(
                workflow,
                prompt,
                effective_config["chunk_max_tokens"],
            )
        )

    if len(chunk_summaries) == 1:
        summary_text = chunk_summaries[0]
    else:
        merge_prompt = build_merge_summary_prompt(chunk_summaries, source_role, source.label, source_time)
        summary_text = call_workflow_llm(
            workflow,
            merge_prompt,
            effective_config["merge_max_tokens"],
        )

    if not summary_text.strip():
        raise RuntimeError(f"基础摘要生成为空：{source.label}")

    return GeneratedSummary(
        label=source.label,
        source_path=source.source_path,
        raw_text=source.text,
        summary_text=summary_text.strip(),
        chunk_count=len(chunks),
        raw_char_count=len(source.text),
        summary_char_count=len(summary_text.strip()),
    )


def summarize_raw_history_sources(
    workflow: Any,
    sources: list[RawTextSource],
    config: Optional[dict[str, int]] = None,
) -> list[GeneratedSummary]:
    if not sources:
        raise ValueError("请提供历史原始资料。")
    return [
        summarize_raw_source(workflow, source, "history", config=config)
        for source in sources
    ]


def summarize_raw_current_source(workflow: Any, source: RawTextSource, source_time: str = "") -> GeneratedSummary:
    return summarize_raw_source(workflow, source, "current", source_time=source_time)


def save_e2e_preprocess_outputs(
    output_run_dir: Path,
    history_summaries: list[GeneratedSummary],
    current_summary: GeneratedSummary,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    history_info = write_history_preprocess_outputs(output_run_dir, history_summaries)
    current_info = write_current_preprocess_outputs(output_run_dir, current_summary)
    write_json(
        output_run_dir / "e2e_preprocess_info.json",
        {
            "mode": "e2e_raw_material_preprocess",
            "config": get_preprocess_config(),
            "metadata": metadata or {},
            "history_count": len(history_summaries),
            "history": history_info,
            "current": current_info,
        },
    )


def save_e2e_history_preprocess_outputs(
    output_run_dir: Path,
    history_summaries: list[GeneratedSummary],
    metadata: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, int]] = None,
) -> None:
    history_info = write_history_preprocess_outputs(output_run_dir, history_summaries)
    write_json(
        output_run_dir / "e2e_preprocess_info.json",
        {
            "mode": "e2e_history_raw_material_preprocess",
            "config": dict(config or get_preprocess_config()),
            "metadata": metadata or {},
            "history_count": len(history_summaries),
            "history": history_info,
            "current": None,
        },
    )


def save_e2e_current_preprocess_outputs(
    output_run_dir: Path,
    current_summary: GeneratedSummary,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    current_info = write_current_preprocess_outputs(output_run_dir, current_summary)
    write_json(
        output_run_dir / "e2e_preprocess_info.json",
        {
            "mode": "e2e_current_raw_material_preprocess",
            "config": get_preprocess_config(),
            "metadata": metadata or {},
            "history_count": 0,
            "history": [],
            "current": current_info,
        },
    )


def write_history_preprocess_outputs(
    output_run_dir: Path,
    history_summaries: list[GeneratedSummary],
) -> list[dict[str, Any]]:
    raw_history_dir = output_run_dir / "raw_history_inputs"
    generated_history_dir = output_run_dir / "generated_history_summaries"

    combined_history_parts: list[str] = []
    history_info: list[dict[str, Any]] = []
    for index, item in enumerate(history_summaries, start=1):
        segment = make_path_segment(item.label, f"history_{index}")
        raw_name = f"{index:03d}_{segment}.md"
        summary_name = f"{index:03d}_{segment}.md"

        write_text(raw_history_dir / raw_name, item.raw_text)
        write_text(generated_history_dir / summary_name, item.summary_text)
        combined_history_parts.append(f"# {item.label}\n\n{item.summary_text}")
        history_info.append(summary_to_info(item, raw_name, summary_name))

    write_text(output_run_dir / "generated_history_summaries.md", "\n\n---\n\n".join(combined_history_parts))
    return history_info


def write_current_preprocess_outputs(
    output_run_dir: Path,
    current_summary: GeneratedSummary,
) -> dict[str, Any]:
    write_text(output_run_dir / "raw_current_input.md", current_summary.raw_text)
    write_text(output_run_dir / "generated_current_summary.md", current_summary.summary_text)
    return summary_to_info(current_summary, "raw_current_input.md", "generated_current_summary.md")


def summary_to_info(item: GeneratedSummary, raw_file: str, summary_file: str) -> dict[str, Any]:
    return {
        "label": item.label,
        "source_path": str(item.source_path) if item.source_path else None,
        "raw_file": raw_file,
        "summary_file": summary_file,
        "chunk_count": item.chunk_count,
        "raw_char_count": item.raw_char_count,
        "summary_char_count": item.summary_char_count,
    }
