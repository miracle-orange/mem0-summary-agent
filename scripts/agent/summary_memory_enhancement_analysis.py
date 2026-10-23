"""为 Summary Memory Web 提供增强前后对比和时间分析，不改变主 Workflow。

本模块是 Web 侧扩展，复用既有 LLM 调用，不修改 CLI Workflow 文件。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


ANALYSIS_MAX_TOKENS = int(os.getenv("SMW_ENHANCEMENT_ANALYSIS_MAX_TOKENS", "1600"))
TEMPORAL_ANALYSIS_MAX_TOKENS = int(os.getenv("SMW_TEMPORAL_ANALYSIS_MAX_TOKENS", "1600"))
WEEKDAY_NAMES_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def parse_source_time_date(source_time: str) -> Optional[datetime]:
    clean_time = (source_time or "").strip()
    if not clean_time:
        return None

    date_part = clean_time[:10]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None


def format_date_with_weekday(value: datetime) -> str:
    return f"{value.strftime('%Y-%m-%d')}（{WEEKDAY_NAMES_ZH[value.weekday()]}）"


def format_weekday_map(week_start: datetime) -> str:
    return "，".join(format_date_with_weekday(week_start + timedelta(days=offset)) for offset in range(7))


def format_nearest_future_weekday_map(value: datetime) -> str:
    return "，".join(
        f"{WEEKDAY_NAMES_ZH[weekday]}→{format_date_with_weekday(nearest_weekday_on_or_after(value, weekday))}"
        for weekday in range(7)
    )


def format_nearest_previous_weekday_map(value: datetime) -> str:
    return "，".join(
        f"{WEEKDAY_NAMES_ZH[weekday]}→{format_date_with_weekday(nearest_weekday_before(value, weekday))}"
        for weekday in range(7)
    )


def end_of_month(value: datetime) -> datetime:
    return (value.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    candidate = value.replace(year=year, month=month, day=1)
    return candidate.replace(day=min(value.day, end_of_month(candidate).day))


def quarter_bounds(value: datetime) -> tuple[datetime, datetime]:
    quarter_start_month = ((value.month - 1) // 3) * 3 + 1
    start = value.replace(month=quarter_start_month, day=1)
    end = end_of_month(add_months(start, 2))
    return start, end


def nearest_weekday_on_or_after(value: datetime, weekday: int) -> datetime:
    days = (weekday - value.weekday()) % 7
    return value + timedelta(days=days)


def nearest_weekday_before(value: datetime, weekday: int) -> datetime:
    days = (value.weekday() - weekday) % 7
    if days == 0:
        days = 7
    return value - timedelta(days=days)


def format_source_time_anchor(source_time: str) -> str:
    clean_time = (source_time or "").strip()
    if not clean_time:
        return "未提供"

    parsed_date = parse_source_time_date(clean_time)
    if parsed_date is None:
        return clean_time
    weekday = WEEKDAY_NAMES_ZH[parsed_date.weekday()]
    if weekday in clean_time:
        return clean_time
    return f"{clean_time}（{weekday}）"


def build_time_reasoning_reference(source_time: str) -> str:
    parsed_date = parse_source_time_date(source_time)
    if parsed_date is None:
        return (
            "source_time 未提供可计算日期；仍需识别原文中的明确日期、时间段、星期、相对时间和阶段表达。"
            "只有原文或历史上下文提供可计算锚点时，才补充客观日期；否则保留原表达并写明待确认。"
        )

    previous_week_start = parsed_date - timedelta(days=parsed_date.weekday() + 7)
    current_week_start = parsed_date - timedelta(days=parsed_date.weekday())
    next_week_start = current_week_start + timedelta(days=7)
    future_week_start = parsed_date + timedelta(days=1)
    future_week_end = parsed_date + timedelta(days=7)
    month_end = end_of_month(parsed_date)
    next_month_start = add_months(parsed_date.replace(day=1), 1)
    next_month_end = end_of_month(next_month_start)
    _, quarter_end = quarter_bounds(parsed_date)
    next_quarter_start = add_months(quarter_bounds(parsed_date)[0], 3)
    _, next_quarter_end = quarter_bounds(next_quarter_start)
    year_end = parsed_date.replace(month=12, day=31)

    return "\n".join(
        [
            f"当前时间锚点：{format_date_with_weekday(parsed_date)}",
            f"上周日期：{format_weekday_map(previous_week_start)}",
            f"本周日期：{format_weekday_map(current_week_start)}",
            f"下周日期：{format_weekday_map(next_week_start)}",
            f"未来一周：{future_week_start.strftime('%Y-%m-%d')} 至 {future_week_end.strftime('%Y-%m-%d')}（按 source_time 之后 7 天理解）",
            f"未来两周：{future_week_start.strftime('%Y-%m-%d')} 至 {(parsed_date + timedelta(days=14)).strftime('%Y-%m-%d')}（按 source_time 之后 14 天理解）",
            f"未来一个月/30天内：{future_week_start.strftime('%Y-%m-%d')} 至 {(parsed_date + timedelta(days=30)).strftime('%Y-%m-%d')}（如原文指自然月，则优先按下月或月底表达处理）",
            f"本月底前：{format_date_with_weekday(month_end)} 前",
            f"下月初：{format_date_with_weekday(next_month_start)} 起的月初阶段；下月底前：{format_date_with_weekday(next_month_end)} 前",
            f"本季度末：{format_date_with_weekday(quarter_end)}；下季度：{format_date_with_weekday(next_quarter_start)} 至 {format_date_with_weekday(next_quarter_end)}",
            f"年底前：{format_date_with_weekday(year_end)} 前",
            f"最近未来星期映射（用于当前会议后的待办/计划，例如“周几前/周几后/周几开始”）：{format_nearest_future_weekday_map(parsed_date)}",
            f"最近过去星期映射（用于回顾已发生事项，例如“周几已完成/周几讨论过”）：{format_nearest_previous_weekday_map(parsed_date)}",
            "星期表达规则：先看原文是否明确“本周/下周/上周”；没有明确周次时，结合语义方向选择最近未来或最近过去。"
            "如果本周对应日期已早于 source_time 且语境是未来待办，不要硬套已过去日期；应使用最近未来对应星期或标注待确认。",
            "工作日表达参考：如出现“3个工作日内/5个工作日内”，需跳过周六、周日；无法可靠计算时保留原表达并说明需确认工作日口径。",
        ]
    )


def call_workflow_llm(workflow: Any, prompt: str, max_tokens: int) -> str:
    llm_call = getattr(workflow, "call_summary_llm", None)
    if llm_call is None:
        llm_call = getattr(workflow, "call_local_qwen", None)
    if llm_call is None:
        raise RuntimeError("workflow 模块缺少 call_summary_llm，无法生成增强对比与时间分析。")
    return (llm_call(prompt, temperature=0.0, max_tokens=max_tokens) or "").strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def style_list(summary_style: str) -> list[str]:
    if summary_style == "both":
        return ["structured", "natural"]
    if summary_style in {"structured", "natural"}:
        return [summary_style]
    raise ValueError(f"Unsupported summary style: {summary_style}")


def normalize_retrieved_memories(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        memories: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("memory") or item.get("memory_text") or item.get("text") or "").strip()
            else:
                text = str(item).strip()
            if text:
                memories.append(text)
        return memories
    if isinstance(value, dict):
        for key in ("memories", "retrieved_memories", "results"):
            if key in value:
                return normalize_retrieved_memories(value[key])
    return []


def format_memory_block(retrieved_memories: list[str]) -> str:
    if not retrieved_memories:
        return "本次没有检索到可用的历史记忆。"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(retrieved_memories, start=1))


def build_enhancement_comparison_prompt(
    baseline_summary: str,
    enhanced_summary: str,
    retrieved_memories: list[str],
    source_time: str,
    summary_style: str,
) -> str:
    memory_block = format_memory_block(retrieved_memories)
    source_time_text = format_source_time_anchor(source_time)
    time_reference = build_time_reasoning_reference(source_time)
    return f"""你是一个中文摘要评审与产品展示助手。请比较“增强前摘要”和“增强后摘要”，说明长期记忆给摘要带来了哪些实际增强。

source_time：{source_time_text}
摘要风格：{summary_style}

可用日历参考：
{time_reference}

增强前摘要（baseline）：
{baseline_summary}

增强后摘要：
{enhanced_summary}

本次检索到的历史记忆：
{memory_block}

输出要求：
1. 使用 Markdown，中文输出。
2. 不要重新生成一份新的摘要，也不要只打分；重点解释增强前后差异。
3. 明确指出增强后新增或澄清了哪些历史背景、前期约定、长期约束、任务延续、负责人、风险和待确认项。
4. 单独说明哪些增强内容来自历史记忆，哪些只是当前材料本身已有信息。
5. 特别关注时间信息：日期、阶段、相对时间、任务窗口、实际完成时间、延期/提前/按期完成、历史承诺是否已兑现或仍有效。
6. 检查增强后是否对相对时间和时间窗口做了充分补充，包括今天/明天/后天、本周/下周/上周、
   周几前/周几后、未来N天/周/月、月底前、下月初、季度末、年底前、阶段/版本窗口等。
   保留原表达并用括号补充了客观时间窗口；依据不足时是否写成“具体日期待确认”。
7. 检查星期类时间的补充日期是否与星期一致；如果增强摘要把某个星期表达补成了不匹配的日期，
   必须明确指出这是时间推理错误。
8. 如果增强摘要仍只写“周几前/周几后/本周内/下周初”等相对表达而没有补充日期或候选解释，
   请指出它没有充分利用 source_time 和日历参考。
9. 检查增强摘要是否把历史中已经完成、已兑现、延期、仍有效或已过期的时间信息用于解释当前进展、
   依赖状态、风险变化或新旧计划差异；如果只是把时间都写成未来待办，请指出增强不足。
10. 如果没有检索到历史记忆，必须明确说明本次没有可用记忆依据，不要伪造增强效果。
11. 对无法确认的推断写成“待确认”，不要把没有依据的具体日期或负责人写成事实。

请按以下结构输出：
## 总体增强结论
## 关键增强点对比
## 来自历史记忆的新增价值
## 时间与任务窗口相关增强
## 仍需确认或不能过度推断的信息
"""


def build_temporal_enrichment_prompt(
    baseline_summary: str,
    enhanced_summary: str,
    retrieved_memories: list[str],
    source_time: str,
    summary_style: str,
) -> str:
    memory_block = format_memory_block(retrieved_memories)
    source_time_text = format_source_time_anchor(source_time)
    time_reference = build_time_reasoning_reference(source_time)
    return f"""你是一个中文时间线与任务窗口分析助手。请基于增强前摘要、增强后摘要和历史记忆，专门分析“时间信息如何被增强”。

source_time：{source_time_text}
摘要风格：{summary_style}

可用日历参考：
{time_reference}

增强前摘要（baseline）：
{baseline_summary}

增强后摘要：
{enhanced_summary}

本次检索到的历史记忆：
{memory_block}

分析要求：
1. 只做时间维度分析，不要生成另一份摘要。
2. 识别当前材料的时间锚点，以及历史记忆中的关键时间点、时间段、阶段、版本、周期、相对时间表达、实际完成时间和历史承诺状态。
3. 说明增强后是否把任务、负责人、依赖、风险、阻塞与时间窗口更清楚地绑定起来。
4. 对今天/明天/后天、本周/下周/上周、周几前/周几后、未来N天/周/月、月底前、下月初、
   季度末、年底前、下阶段、下个版本等表达，请检查增强摘要是否保留原表达并用括号补充了客观时间窗口。
   只有在 source_time 或上下文足够明确时才给出推断窗口；如果依据不足，请写“（具体日期待确认）”。
   只有原文或上下文提供小时级锚点时才补充小时，不要虚构小时。
5. 对星期类时间，必须检查补充日期和星期是否一致；如果语境是当前会议后的待办，
   应检查是否按本周/下周标记或最近未来对应星期补充了具体日期；如果语境是回顾已发生事项，
   应检查是否按历史窗口或最近过去对应星期处理。
6. 如果增强摘要仍只写“周几前/周几后/本周内/下周初”等相对表达而没有补充日期或候选解释，
   请指出这是时间增强不足。
7. 判断历史时间信息是已完成/已兑现、延期/未完成、仍有效、已过期、被当前材料更新、与当前材料冲突，还是需要人工确认。
8. 检查增强摘要是否把历史完成记录用于说明当前基础或依赖关系，而不是重复写成未来任务。
9. 不要编造原文和历史记忆没有提供的具体日期、负责人或结论。

请按以下结构输出：
## 当前时间锚点
## 历史时间线与前期约定
## 时间推理与任务窗口
## 时间相关风险、冲突或延期
## 待确认时间信息
"""


def generate_enhancement_comparison(
    workflow: Any,
    baseline_summary: str,
    enhanced_summary: str,
    retrieved_memories: list[str],
    source_time: str,
    summary_style: str,
) -> str:
    prompt = build_enhancement_comparison_prompt(
        baseline_summary,
        enhanced_summary,
        retrieved_memories,
        source_time,
        summary_style,
    )
    return call_workflow_llm(workflow, prompt, ANALYSIS_MAX_TOKENS)


def generate_temporal_enrichment(
    workflow: Any,
    baseline_summary: str,
    enhanced_summary: str,
    retrieved_memories: list[str],
    source_time: str,
    summary_style: str,
) -> str:
    prompt = build_temporal_enrichment_prompt(
        baseline_summary,
        enhanced_summary,
        retrieved_memories,
        source_time,
        summary_style,
    )
    return call_workflow_llm(workflow, prompt, TEMPORAL_ANALYSIS_MAX_TOKENS)


def save_enhancement_analysis_outputs(
    workflow: Any,
    output_run_dir: Path,
    baseline_summary: str,
    source_time: str,
    summary_style: str,
    baseline_source: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    baseline = (baseline_summary or "").strip()
    if not baseline:
        raise ValueError("baseline summary 为空，无法生成增强对比。")

    retrieved_memories = normalize_retrieved_memories(read_json(output_run_dir / "retrieved_memories.json"))
    write_text(output_run_dir / "baseline_summary.md", baseline)

    comparisons: dict[str, str] = {}
    temporal_reports: dict[str, str] = {}
    generated_styles: list[str] = []

    for style in style_list(summary_style):
        enhanced_summary = read_text(output_run_dir / f"summary_{style}.md")
        if not enhanced_summary:
            continue

        comparison = generate_enhancement_comparison(
            workflow,
            baseline,
            enhanced_summary,
            retrieved_memories,
            source_time,
            style,
        )
        temporal = generate_temporal_enrichment(
            workflow,
            baseline,
            enhanced_summary,
            retrieved_memories,
            source_time,
            style,
        )

        comparisons[style] = comparison
        temporal_reports[style] = temporal
        generated_styles.append(style)
        write_text(output_run_dir / f"enhancement_comparison_{style}.md", comparison)
        write_text(output_run_dir / f"temporal_enrichment_{style}.md", temporal)

    if comparisons:
        combined_comparison = "\n\n---\n\n".join(
            f"# {style} 增强对比\n\n{text}" for style, text in comparisons.items()
        )
        write_text(output_run_dir / "enhancement_comparison.md", combined_comparison)

    if temporal_reports:
        combined_temporal = "\n\n---\n\n".join(
            f"# {style} 时间增强分析\n\n{text}" for style, text in temporal_reports.items()
        )
        write_text(output_run_dir / "temporal_enrichment.md", combined_temporal)

    info = {
        "baseline_source": baseline_source,
        "source_time": source_time,
        "summary_style": summary_style,
        "generated_styles": generated_styles,
        "retrieved_memory_count": len(retrieved_memories),
        "files": {
            "baseline_summary": "baseline_summary.md",
            "enhancement_comparison": "enhancement_comparison.md" if comparisons else None,
            "temporal_enrichment": "temporal_enrichment.md" if temporal_reports else None,
        },
        "metadata": metadata or {},
    }
    write_json(output_run_dir / "enhancement_analysis_info.json", info)
    return info
