"""Summary Memory Baseline Workflow：提供 Ingest、Summarize 和完整演示入口。"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from summary_memory_artifacts import build_artifact_board, format_artifact_board_for_summary
from summary_memory_agent_schemas import (
    TYPED_MEMORY_SCHEMA_VERSION,
    TypedMemoryRecord,
    TypedRetrievalPlanItem,
)
from summary_memory_critic import (
    SUMMARY_CRITIC_SCHEMA_VERSION,
    build_summary_critic_prompt,
    build_summary_rewrite_prompt,
    build_summary_rewrite_report,
    format_summary_critic_markdown,
    parse_rewritten_summaries,
    parse_summary_critic_json,
    should_rewrite_from_critic,
)
from summary_memory_llm import call_summary_llm as call_configured_summary_llm
from summary_memory_llm import get_llm_runtime_info
from summary_memory_typed_memory import (
    apply_cross_chunk_dedup_judgements,
    build_cross_chunk_dedup_candidates,
    build_cross_chunk_dedup_judge_prompt,
    build_typed_memory_retrieval_plan,
    CROSS_CHUNK_DEDUP_SCHEMA_VERSION,
    deduplicate_cross_chunk_typed_memory_records,
    extract_typed_memories_from_summary,
    format_typed_memory_for_summary,
    parse_cross_chunk_dedup_judge_output,
    typed_memory_records_to_dicts,
    typed_record_from_search_result,
    typed_record_to_memory_text,
    typed_record_to_mem0_message,
    typed_record_to_mem0_metadata,
)


os.environ.setdefault("MEM0_TELEMETRY", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_FILE = SCRIPT_DIR / "summary_memory_cases" / "paper_summary_assistant.md"

PROJECT_ROOT = Path(os.getenv("MEM0_SUMMARY_ROOT", str(SCRIPT_DIR.parents[1])))
MODEL_ROOT = Path(os.getenv("MEM0_MODEL_ROOT", str(PROJECT_ROOT / "models")))
EMBEDDING_MODEL_PATH = Path(os.getenv("MEM0_EMBEDDING_MODEL", str(MODEL_ROOT / "Qwen3-Embedding-0.6B")))
QDRANT_SUMMARY_PATH = os.getenv("MEM0_QDRANT_SUMMARY_PATH", str(PROJECT_ROOT / "qdrant_summary"))

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local-key")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen-local")
CURRENT_MATERIAL_MAX_CHARS = max(1000, int(os.getenv("SMW_CURRENT_MATERIAL_MAX_CHARS", "12000")))
TYPED_MEMORY_EXTRACTION_MAX_TOKENS = max(256, int(os.getenv("SMW_TYPED_EXTRACTION_MAX_TOKENS", "8000")))
TYPED_MEMORY_DEDUP_JUDGE_MAX_TOKENS = max(256, int(os.getenv("SMW_TYPED_DEDUP_JUDGE_MAX_TOKENS", "1200")))
TYPED_MEMORY_DEDUP_JUDGE_MAX_CANDIDATES = max(1, int(os.getenv("SMW_TYPED_DEDUP_JUDGE_MAX_CANDIDATES", "20")))
SUMMARY_MAX_TOKENS = int(os.getenv("SMW_SUMMARY_MAX_TOKENS", "1800"))
SUMMARY_CRITIC_MAX_TOKENS = int(os.getenv("SMW_CRITIC_MAX_TOKENS", "3600"))
SUMMARY_REWRITE_MAX_TOKENS = int(os.getenv("SMW_REWRITE_MAX_TOKENS", "3200"))

SUMMARY_COLLECTION_NAME = os.getenv("MEM0_SUMMARY_COLLECTION", "mem0_qwen_summary")
WEEKDAY_NAMES_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass
class SummaryDemoInput:
    user_id: str
    project: str
    source_type: str
    source_time: str
    history_summary: str
    current_content: str
    source_path: Optional[Path] = None
    source_file: Optional[Path] = None
    history_source_id: str = ""


def safe_hf_embed(self, text, *args, **kwargs):
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    embedding = self.model.encode([text], convert_to_numpy=True)
    if len(embedding.shape) == 2:
        embedding = embedding[0]
    return embedding.tolist()


def patch_huggingface_embedding() -> None:
    from mem0.embeddings.huggingface import HuggingFaceEmbedding

    HuggingFaceEmbedding.embed = safe_hf_embed


def build_llm_client():
    from openai import OpenAI

    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in TRUE_ENV_VALUES


def prepare_current_material_for_agents(
    current_content: str,
    max_chars: int = CURRENT_MATERIAL_MAX_CHARS,
) -> str:
    clean_content = (current_content or "").strip()
    if len(clean_content) <= max_chars:
        return clean_content

    marker = "\n\n[当前原始材料过长，中间部分按上下文预算省略；保留开头与结尾]\n\n"
    available = max(1, max_chars - len(marker))
    head_chars = max(1, available * 2 // 3)
    tail_chars = max(1, available - head_chars)
    return clean_content[:head_chars] + marker + clean_content[-tail_chars:]


def build_embedding_model_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device": os.getenv("MEM0_EMBEDDING_DEVICE", "cpu"),
        "trust_remote_code": env_flag("MEM0_EMBEDDING_TRUST_REMOTE_CODE", True),
        "local_files_only": env_flag("MEM0_EMBEDDING_LOCAL_FILES_ONLY", True),
    }
    if env_flag("MEM0_EMBEDDING_DISABLE_LOW_CPU_MEM_USAGE", True):
        # Avoid meta-tensor initialization paths that can fail when SentenceTransformer later calls .to(device).
        kwargs["model_kwargs"] = {"low_cpu_mem_usage": False}
    return kwargs


def build_summary_memory_config() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": VLLM_MODEL,
                "vllm_base_url": VLLM_BASE_URL,
                "api_key": VLLM_API_KEY,
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": str(EMBEDDING_MODEL_PATH),
                "embedding_dims": 1024,
                "model_kwargs": build_embedding_model_kwargs(),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": SUMMARY_COLLECTION_NAME,
                "path": QDRANT_SUMMARY_PATH,
                "embedding_model_dims": 1024,
                "on_disk": True,
            },
        },
        "custom_instructions": (
            "请用中文保存摘要、会议纪要、学习总结或项目周报中的长期有用记忆。"
            "重点保留背景、目标、约束、关键决策、任务分工、负责人、未完成事项、"
            "风险阻塞、摘要偏好、跨时间延续事项、时间点、时间段、相对时间、"
            "任务截止窗口、延期变化、阶段计划、被确认/暂缓/否定的方向，"
            "以及重要工具、论文或系统在当前任务中的定位。"
            "不要保存没有后续价值的一次性描述，也不要补充原文没有明确表达的推测。"
        ),
    }


def call_local_qwen(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Backward-compatible entry point; dispatches according to SMW_LLM_BACKEND."""
    return call_summary_llm(prompt, temperature=temperature, max_tokens=max_tokens)


def call_summary_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Use local Qwen by default or the configured DeepSeek WebSocket backend."""
    return call_configured_summary_llm(prompt, temperature=temperature, max_tokens=max_tokens)


def call_local_qwen_vllm(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Direct local vLLM/Qwen call retained for diagnostics that explicitly need it."""
    client = build_llm_client()
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


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
            "时间推理证据优先级：1) 原文明确日期/时间段最高，直接保留并校验；"
            "2) 原文明确“上周/本周/下周/下个月/月底”等周次或周期时，按原文周期解释；"
            "3) 原文只有“周几前/周几后/周几开始”等无周次表达时，必须结合语义方向选择候选；"
            "4) 语义方向仍不明确时，保留原表达并给出最可能候选或写明待确认。",
            f"当前时间锚点：{format_date_with_weekday(parsed_date)}",
            f"未来行动候选星期（用于当前材料之后的待办、计划、提交、评估、交付、开始、跟进等语境）：{format_nearest_future_weekday_map(parsed_date)}",
            f"历史回顾候选星期（用于已完成、此前、上周、回顾、历史承诺等语境）：{format_nearest_previous_weekday_map(parsed_date)}",
            f"明确周次日期参考-上周：{format_weekday_map(previous_week_start)}",
            f"明确周次日期参考-本周：{format_weekday_map(current_week_start)}",
            f"明确周次日期参考-下周：{format_weekday_map(next_week_start)}",
            f"未来一周：{future_week_start.strftime('%Y-%m-%d')} 至 {future_week_end.strftime('%Y-%m-%d')}（按 source_time 之后 7 天理解）",
            f"未来两周：{future_week_start.strftime('%Y-%m-%d')} 至 {(parsed_date + timedelta(days=14)).strftime('%Y-%m-%d')}（按 source_time 之后 14 天理解）",
            f"未来一个月/30天内：{future_week_start.strftime('%Y-%m-%d')} 至 {(parsed_date + timedelta(days=30)).strftime('%Y-%m-%d')}（如原文指自然月，则优先按下月或月底表达处理）",
            f"本月底前：{format_date_with_weekday(month_end)} 前",
            f"下月初：{format_date_with_weekday(next_month_start)} 起的月初阶段；下月底前：{format_date_with_weekday(next_month_end)} 前",
            f"本季度末：{format_date_with_weekday(quarter_end)}；下季度：{format_date_with_weekday(next_quarter_start)} 至 {format_date_with_weekday(next_quarter_end)}",
            f"年底前：{format_date_with_weekday(year_end)} 前",
            "星期表达决策规则：先看原文是否明确“本周/下周/上周”；没有明确周次时，结合语义方向选择未来行动候选或历史回顾候选。"
            "如果候选日期早于 source_time 但语境是当前材料之后的未来行动，不能直接写成已过去日期；"
            "应改用合理的未来候选，或明确写成“已早于当前材料日期，需确认是否逾期/是否仍有效”。",
            "工作日表达参考：如出现“3个工作日内/5个工作日内”，需跳过周六、周日；无法可靠计算时保留原表达并说明需确认工作日口径。",
        ]
    )


def build_memory_extraction_prompt(summary_text: str) -> str:
    return f"""你是一个通用摘要记忆抽取器。

输入是一段历史摘要、会议纪要、学习总结、项目周报或阶段复盘。
请从中抽取对未来摘要生成、跨时间回顾或后续决策有长期价值的记忆。

只抽取长期有用的信息，例如：
1. 背景、目标、范围和重要约束
2. 关键决策和决策理由
3. 后续任务、负责人、时间要求、截止窗口和依赖关系
4. 明确日期、时间段、阶段、版本、周期，以及“未来一周”“月底前”“下阶段”等相对时间表达
5. 历史任务的计划时间、实际完成时间、是否按期完成、延期、提前、过期、阶段变化、历史承诺是否仍有效或需要重新确认
6. 未完成事项、风险、阻塞和待确认问题，尤其是和时间、负责人、依赖相关的问题
7. 用户、团队或组织的摘要偏好和工作偏好
8. 后续材料需要延续跟踪的信息
9. 某个方向、方案、工具、论文或系统被确认、暂缓、否定或重新定位

抽取时间信息时请注意：
1. 如果原文有明确日期或时间段，请保留原始表述。
2. 如果原文只有相对时间，请连同它依附的材料时间、会议时间或上下文一起保存；不要凭空换算成具体日期。
3. 对任务类记忆，优先写清“事项 + 负责人/角色 + 时间窗口/截止要求 + 依赖/风险”。
4. 对已经完成的任务，要尽量写清“原截止要求 + 实际完成时间/完成状态 + 是否影响后续任务”。
5. 对已过期、延期、提前完成、重新讨论或待确认的时间要求，要明确写出状态。
6. 对“未来一周”“未来一个月”“月底前”“年底前”“下周”“下个月”“下一季度”等相对时间，
   请保留原始表达，并在有材料时间锚点时补充括号说明客观时间窗口，例如：
   “未来一周（约 YYYY-MM-DD 至 YYYY-MM-DD，按材料日期之后 7 天理解）”。
   如果依据不足，请写“（具体日期待确认）”，不要把猜测写成事实。
7. 对“周几前”“周几后”“本周/下周/上周 + 周几”等星期类表达，请保留原始表达，
   并尽量连同本周、下周、最近未来、最近过去或待确认状态一起保存。
8. 只有原文或上下文提供小时级锚点时，才补充到小时；否则只补充到可靠的年月日或时间段粒度。

不要抽取：
1. 没有后续价值的普通描述
2. 一次性寒暄、流程性话语或过细碎事实
3. 原文没有明确表达的推测
4. 脱离原文语境后无法独立理解的信息

每条记忆写成一句完整、可独立理解的话。
如果没有值得长期保存的记忆，输出空数组 []。
输出必须是严格 JSON 数组，不要输出 markdown，不要输出解释文字。

输出格式：
[
  {{"memory_text": "..."}},
  {{"memory_text": "..."}}
]

输入文本：
{summary_text}
"""


def parse_memory_json(raw_output: str) -> list[str]:
    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("\n[JSON 解析失败，模型原始输出如下]")
        print(raw_output)
        return []

    memories: list[str] = []
    if not isinstance(parsed, list):
        return memories

    for item in parsed:
        if isinstance(item, dict):
            value = item.get("memory_text", "")
        else:
            value = item
        value = str(value).strip()
        if value:
            memories.append(value)
    return memories


def extract_memories_from_summary(summary_text: str) -> list[str]:
    prompt = build_memory_extraction_prompt(summary_text)
    raw_output = call_local_qwen(prompt, temperature=0.0, max_tokens=1200)
    return parse_memory_json(raw_output)


def add_summary_memories(
    memory: Any,
    summary_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    source_file: Optional[Path] = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    memories = extract_memories_from_summary(summary_text)
    metadata = {
        "user_id": user_id,
        "project": project,
        "source_type": source_type,
        "source_time": source_time,
    }
    if source_file:
        metadata["source_file"] = str(source_file)

    add_results: list[dict[str, Any]] = []
    for memory_text in memories:
        messages = [
            {
                "role": "user",
                "content": f"请原样保存这条来自摘要材料的长期记忆：{memory_text}",
            }
        ]
        result = memory.add(messages, user_id=user_id, metadata=metadata)
        add_results.append(result)
    return memories, add_results


def add_typed_summary_memories(
    memory: Any,
    summary_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    source_file: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], dict[str, Any]]:
    typed_records, extraction_trace = extract_typed_summary_memory_records(
        summary_text=summary_text,
        source_time=source_time,
        source_file=source_file,
    )
    typed_memories, memory_texts, add_results = write_typed_summary_memory_records(
        memory=memory,
        typed_records=typed_records,
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
        source_file=source_file,
    )
    return typed_memories, memory_texts, add_results, extraction_trace


def extract_typed_summary_memory_records(
    summary_text: str,
    source_time: str,
    source_file: Optional[Path] = None,
) -> tuple[list[TypedMemoryRecord], dict[str, Any]]:
    extraction_trace: dict[str, Any] = {}
    typed_records = extract_typed_memories_from_summary(
        summary_text,
        call_local_qwen,
        source_time=source_time,
        source_file=source_file,
        max_tokens=TYPED_MEMORY_EXTRACTION_MAX_TOKENS,
        trace=extraction_trace,
    )
    return typed_records, extraction_trace


def write_typed_summary_memory_records(
    memory: Any,
    typed_records: list[TypedMemoryRecord],
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    source_file: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    metadata_base = {
        "user_id": user_id,
        "project": project,
        "source_type": source_type,
        "source_time": source_time,
    }
    if source_file:
        metadata_base["source_file"] = str(source_file)

    add_results: list[dict[str, Any]] = []
    for record in typed_records:
        metadata = {**metadata_base, **typed_record_to_mem0_metadata(record)}
        result = memory.add(
            typed_record_to_mem0_message(record),
            user_id=user_id,
            metadata=metadata,
            infer=False,
        )
        add_results.append(result)

    memory_texts = [typed_record_to_memory_text(record) for record in typed_records]
    return typed_memory_records_to_dicts(typed_records), memory_texts, add_results


def get_history_source_identity(demo_input: SummaryDemoInput) -> str:
    if demo_input.history_source_id.strip():
        return demo_input.history_source_id.strip()
    source = demo_input.source_file or demo_input.source_path
    return str(source) if source else ""


def format_history_source_label(demo_input: SummaryDemoInput, index: int) -> str:
    source = demo_input.source_file or demo_input.source_path
    return str(source or get_history_source_identity(demo_input) or f"history_{index}.md")


def deduplicate_batch_typed_memory_records(
    batch_results: list[dict[str, Any]],
    judge_llm_call: Optional[Callable[..., str]] = None,
    judge_max_tokens: int = TYPED_MEMORY_DEDUP_JUDGE_MAX_TOKENS,
    judge_max_candidates: int = TYPED_MEMORY_DEDUP_JUDGE_MAX_CANDIDATES,
) -> dict[str, Any]:
    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(batch_results):
        source_identity = str(item.get("history_source_id") or "").strip()
        source_time = str(item.get("source_time") or "").strip()
        if source_identity and source_time:
            grouped_indices.setdefault((source_identity, source_time), []).append(index)

    input_count = sum(len(item.get("typed_records", [])) for item in batch_results)
    group_reports: list[dict[str, Any]] = []
    for (source_identity, source_time), indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        source_chunks = [batch_results[index].get("typed_records", []) for index in indices]
        direct_deduplicated_chunks, direct_removed_records = deduplicate_cross_chunk_typed_memory_records(source_chunks)
        candidates, total_candidate_count = build_cross_chunk_dedup_candidates(
            direct_deduplicated_chunks,
            max_candidates=judge_max_candidates,
        )
        decisions: dict[str, dict[str, Any]] = {}
        judge_status = "skipped_no_candidates"
        judge_error = ""
        judge_raw_output = ""
        final_deduplicated_chunks = direct_deduplicated_chunks
        judge_removed_records: list[dict[str, Any]] = []
        candidate_outcomes: dict[str, str] = {}

        if candidates and judge_llm_call is None:
            judge_status = "skipped_no_llm"
            candidate_outcomes = {candidate["pair_id"]: "not_merged_judge_unavailable" for candidate in candidates}
        elif candidates:
            judge_prompt = build_cross_chunk_dedup_judge_prompt(candidates)
            try:
                judge_raw_output = judge_llm_call(judge_prompt, temperature=0.0, max_tokens=judge_max_tokens)
                decisions = parse_cross_chunk_dedup_judge_output(
                    judge_raw_output,
                    {candidate["pair_id"] for candidate in candidates},
                )
                final_deduplicated_chunks, judge_removed_records, candidate_outcomes = (
                    apply_cross_chunk_dedup_judgements(
                        direct_deduplicated_chunks,
                        candidates,
                        decisions,
                    )
                )
                judge_status = "completed"
            except Exception as exc:
                judge_status = "failed"
                judge_error = f"{type(exc).__name__}: {exc}"
                candidate_outcomes = {candidate["pair_id"]: "not_merged_judge_error" for candidate in candidates}

        removed_records = [*direct_removed_records, *judge_removed_records]
        for index, records in zip(indices, final_deduplicated_chunks):
            batch_results[index]["typed_records"] = records

        for item in removed_records:
            item["removed_chunk_index"] = indices[int(item["removed_chunk_index"]) - 1] + 1
            item["retained_chunk_index"] = indices[int(item["retained_chunk_index"]) - 1] + 1
            item["history_source_id"] = source_identity
            item["source_time"] = source_time

        candidate_pairs: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_audit = dict(candidate["audit"])
            for side_name in ("left", "right"):
                side = dict(candidate_audit[side_name])
                local_chunk_index = int(side["chunk_index"])
                side["chunk_index"] = indices[local_chunk_index - 1] + 1
                candidate_audit[side_name] = side
            decision = decisions.get(candidate["pair_id"])
            candidate_audit["outcome"] = candidate_outcomes.get(
                candidate["pair_id"], "not_merged_missing_judge_decision"
            )
            candidate_audit["judge_decision"] = decision or None
            candidate_pairs.append(candidate_audit)
        group_reports.append(
            {
                "history_source_id": source_identity,
                "source_time": source_time,
                "chunk_indices": [index + 1 for index in indices],
                "input_count": sum(len(records) for records in source_chunks),
                "retained_count": sum(len(records) for records in final_deduplicated_chunks),
                "removed_count": len(removed_records),
                "direct_removed_count": len(direct_removed_records),
                "judge_removed_count": len(judge_removed_records),
                "semantic_candidate_count": total_candidate_count,
                "emitted_candidate_count": len(candidates),
                "omitted_candidate_count": max(0, total_candidate_count - len(candidates)),
                "evaluated_candidate_count": len(decisions),
                "judge": {
                    "status": judge_status,
                    "max_tokens": judge_max_tokens,
                    "max_candidates": judge_max_candidates,
                    "error": judge_error,
                    "raw_output": judge_raw_output,
                },
                "candidate_pairs": candidate_pairs,
                "removed_records": removed_records,
            }
        )

    retained_count = sum(len(item.get("typed_records", [])) for item in batch_results)
    return {
        "schema_version": CROSS_CHUNK_DEDUP_SCHEMA_VERSION,
        "scope": "same history_source_id and source_time across different ingest chunks only",
        "input_count": input_count,
        "retained_count": retained_count,
        "removed_count": input_count - retained_count,
        "judge_config": {
            "enabled": judge_llm_call is not None,
            "max_tokens": judge_max_tokens,
            "max_candidates_per_group": judge_max_candidates,
        },
        "groups": group_reports,
    }


def print_typed_memories_summary(typed_memories: list[dict[str, Any]]) -> None:
    type_counts: dict[str, int] = {}
    for item in typed_memories:
        memory_type = str(item.get("memory_type", "unknown"))
        type_counts[memory_type] = type_counts.get(memory_type, 0) + 1

    print("\n[Typed Memory 统计]")
    print(f"typed memory 条数: {len(typed_memories)}")
    if type_counts:
        type_summary = ", ".join(f"{memory_type}={count}" for memory_type, count in sorted(type_counts.items()))
        print(f"类型统计: {type_summary}")


def print_add_results_summary(extracted_memories: list[str], add_results: list[dict[str, Any]]) -> None:
    event_counts: dict[str, int] = {}
    empty_results = 0
    operation_count = 0

    for result in add_results:
        result_items = result.get("results", []) if isinstance(result, dict) else []
        if not result_items:
            empty_results += 1
            continue

        for item in result_items:
            operation_count += 1
            event = str(item.get("event", "UNKNOWN")) if isinstance(item, dict) else "UNKNOWN"
            event_counts[event] = event_counts.get(event, 0) + 1

    print("\n[写入统计]")
    print(f"抽取记忆条数: {len(extracted_memories)}")
    print(f"memory.add 调用次数: {len(add_results)}")
    print(f"产生写入/更新/删除操作数: {operation_count}")
    print(f"空结果次数: {empty_results}")
    if event_counts:
        event_summary = ", ".join(f"{event}={count}" for event, count in sorted(event_counts.items()))
        print(f"事件统计: {event_summary}")

    if extracted_memories and add_results and operation_count == 0:
        print(
            "[提示] 本次 memory.add 没有产生新的存储操作。"
            "如果没有清空 qdrant_summary，这通常表示 mem0 判定这些记忆已存在或无需更新；"
            "如果已经清空 qdrant_summary 后仍出现这种情况，再检查 vLLM、embedding 和 Qdrant 配置。"
        )


def build_retrieval_query(current_content: str, max_chars: int = 900, source_time: str = "") -> str:
    current_excerpt = current_content.strip()[:max_chars]
    source_time_text = format_source_time_anchor(source_time)
    time_reference = build_time_reasoning_reference(source_time)
    return (
        "请检索与当前材料主题、目标、关键决策、任务分工、负责人、风险阻塞、"
        "团队偏好、历史延续事项、被确认/暂缓/否定的方向相关的长期记忆。"
        "请特别关注与当前材料时间锚点、历史时间线、任务计划时间、实际完成时间、"
        "任务截止窗口、开始窗口、依赖窗口、延期/提前/按期完成、已过期/仍有效状态、"
        "相对时间表达、前期承诺是否已兑现、是否仍有效、以及待确认时间点相关的记忆。"
        "如果历史记忆里有“某人在某截止日前已经完成某事项”“某事项延期到某窗口”“某阶段已经结束/进入下一阶段”，"
        "也要召回，因为这些信息可以帮助判断当前材料中的任务是新增、延续、复盘、依赖已满足，还是不应再写成未来待办。"
        "也请检索能帮助解释相对时间、星期表达、截止窗口、开始窗口和阶段窗口的历史时间锚点和上下文，"
        "例如今天/明天/后天、本周/下周/上周、周几前/后、未来N天/周/月、月底前、下月初、季度末、年底前、"
        "第N阶段、版本发布前后、延期/提前/已过期/仍有效等。"
        f"\n\n当前材料时间锚点 source_time：{source_time_text}"
        f"\n\n可用日历参考：\n{time_reference}"
        "\n\n当前材料片段：\n"
        f"{current_excerpt}"
    )


def search_memories_for_summary(
    memory: Any,
    current_content: str,
    user_id: str,
    project: str,
    top_k: int = 8,
    source_time: str = "",
) -> list[str]:
    query = build_retrieval_query(current_content, source_time=source_time)
    result = memory.search(
        query,
        filters={"user_id": user_id, "project": project},
        top_k=top_k,
    )

    memories: list[str] = []
    for item in result.get("results", []):
        memory_text = item.get("memory") or item.get("memory_text") or ""
        memory_text = str(memory_text).strip()
        if memory_text:
            memories.append(memory_text)
    return memories[:top_k]


def typed_retrieval_plan_to_dicts(plan_items: list[TypedRetrievalPlanItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in plan_items]


def search_typed_memories_for_summary(
    memory: Any,
    current_content: str,
    user_id: str,
    project: str,
    top_k: int = 8,
    source_time: str = "",
) -> tuple[list[dict[str, Any]], list[TypedMemoryRecord], list[str]]:
    plan_items = build_typed_memory_retrieval_plan(
        current_content,
        call_local_qwen,
        source_time=source_time,
    )
    plan_payload = typed_retrieval_plan_to_dicts(plan_items)

    records_by_id: dict[str, tuple[TypedMemoryRecord, float]] = {}
    for plan_item in plan_items:
        for memory_type in plan_item.memory_types:
            filters = {
                "user_id": user_id,
                "project": project,
                "memory_schema_version": TYPED_MEMORY_SCHEMA_VERSION,
                "memory_type": memory_type,
            }
            result = memory.search(plan_item.query, filters=filters, top_k=top_k)
            for item in result.get("results", []):
                record = typed_record_from_search_result(item)
                if record is None or record.memory_type != memory_type:
                    continue
                score = float(item.get("score") or 0.0)
                current = records_by_id.get(record.memory_id)
                if current is None or score > current[1]:
                    records_by_id[record.memory_id] = (record, score)

    ranked_records = [
        record
        for record, _score in sorted(records_by_id.values(), key=lambda item: item[1], reverse=True)
    ][:top_k]
    formatted_memories = [format_typed_memory_for_summary(record) for record in ranked_records]
    return plan_payload, ranked_records, formatted_memories


def format_retrieved_memories(retrieved_memories: list[str]) -> str:
    if not retrieved_memories:
        return "暂无相关历史记忆。"
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(retrieved_memories, start=1))


def build_summary_style_instructions(summary_style: str) -> str:
    if summary_style == "structured":
        return """输出风格要求：
请严格按以下结构输出，不要自由改变标题，不要省略章节：

一、前期背景与既有约束
二、本次新增进展
三、与前期约定的变化或冲突
四、历史时间窗口与已知状态
五、风险、阻塞与待确认问题
六、负责人、下一步任务与时间窗口

每一节原则上使用 2-4 条要点；如果某节只有 1 条真实信息，可以只写 1 条。
不要为了凑满条数补充空泛、重复或互相矛盾的内容。
若某节完全没有信息，请写“本次材料未明确提及”。
特别注意：如果“三、与前期约定的变化或冲突”已经写出任何变化、冲突、
重新讨论或边界确认，就不要再写“本次材料未明确提及”。
负责人和下一步任务只列当前材料明确指定的负责人或自然延续的既有分工。
仅被提出但未形成结论的提议，应写成待确认事项或风险阻塞，不要写成某个人的下一步任务。
“四、历史时间窗口与已知状态”是专门的历史时间分析，不是当前行动清单。
这一节优先使用检索到的历史上下文，写清前期已经约定或已经发生的时间点、原截止窗口、
实际完成时间、是否按期完成、延期/提前、已过期/仍有效。
如果历史记忆只给出了截止要求但没有给出完成结果，请写成“前期截至窗口/完成状态未明确”，不要擅自写成已完成。
如果历史记忆给出了“某人已在某日期或某截止前完成某事项”，请写成历史完成状态或依赖已满足。
这一节禁止写当前材料之后的未来行动项，禁止使用“负责人：事项，时间窗口”的行动清单格式。
只要是当前材料里的下一步任务、负责人行动项、未来截止或开始窗口，必须放到第六节，不能出现在第四节。
如果需要说明当前行动与历史窗口的关系，只能在第三节写变化/冲突，或在第五节写风险/待确认；第四节不要写当前行动安排。
“六、负责人、下一步任务与时间窗口”只写当前材料之后需要推进的行动项：
负责人、事项、触发条件/依赖、截止或开始窗口、待确认时间。不要重复第四节已经写过的历史完成记录。
第六节应优先使用“负责人：事项，原始时间表达（客观日期/时间窗口）”的格式，
例如“李婷：评估字段，周三前完成（YYYY-MM-DD 前）”。
只要 source_time、原文或历史上下文能推出日期，第六节不能只写“周三前/周五前/下班前/周三后”而不补括号。
如果无法确定具体日期，也要写“（具体日期待确认）”或“（可能为 YYYY-MM-DD，待确认）”。
如果同一个事项同时涉及历史窗口和下一步行动，第四节只保留历史窗口或已知状态；第六节只保留行动安排，两节不要复述同一句话。
所有相对时间都按“保留原表达 + 括号补充客观日期/时间段/候选解释”的方式写出。
需要覆盖但不限于：今天/明天/后天、本周/下周/上周、周几前/周几后、未来N天/周/月、月底前、下月初、季度末、年底前、阶段/版本窗口、延期/提前/仍有效。
判断步骤：先找锚点，再判断语义方向（未来待办/计划 vs 历史回顾/已完成），再按证据优先级和可用日历参考补充日期，最后校验日期与星期一致。
时间证据优先级：明确日期最高；明确“上周/本周/下周”等周次时按原文周次解释；没有周次修饰的“周几前/周几后/周几开始”必须结合语义方向选择候选。
第六节中的当前后续行动如果没有明确逾期或回顾旧计划，括号中的客观日期原则上不能早于 source_time；若原文明确周次导致日期已经过去，必须写明“已早于当前材料日期，需确认是否逾期/是否仍有效”。
如果能从 source_time、原文或历史上下文推理出合理日期，应尽量补充；如果仍有歧义，也要给出最可能候选并标注待确认，不要完全省略客观时间。
只有上下文提供小时级锚点时才补充小时，不要为了显得精确而虚构小时。
不要写额外总标题，不要使用 Markdown 表格。"""

    if summary_style == "natural":
        return """输出风格要求：
请写成自然版摘要，使用 3-5 个连贯自然段。
不要使用 Markdown 标题、编号列表、项目符号或固定栏目。
仍然要自然覆盖前期背景、既有约束、本次进展、变化冲突、时间线、任务窗口、
风险阻塞、负责人、下一步任务和待确认时间点。
历史时间信息也要自然融入正文：例如前期任务是否已经按期完成、历史截止是否已兑现、
某个阶段是否已经结束、历史延期是否影响当前风险、历史完成事项是否成为当前任务的依赖基础。
不要只把时间信息写成未来待办清单。
如果当前材料或历史上下文包含相对时间，请保留原表达，并在其后用括号自然补充可能对应的客观日期、时间段或候选解释。
需要覆盖但不限于：今天/明天/后天、本周/下周/上周、周几前/周几后、未来N天/周/月、月底前、下月初、季度末、年底前、阶段/版本窗口、延期/提前/仍有效。
判断时先找锚点，再判断语义方向（未来待办/计划 vs 历史回顾/已完成），再按证据优先级和可用日历参考补充日期，并校验日期与星期一致。
明确日期优先保留；明确“上周/本周/下周”等周次时按原文周次解释；没有周次修饰的“周几前/周几后/周几开始”必须结合语义方向选择候选。
当前材料之后的行动项如果没有明确逾期或回顾旧计划，补充的客观日期原则上不能早于 source_time；若原文明确周次导致日期已经过去，请自然说明“已早于当前材料日期，需确认是否逾期/是否仍有效”。
自然段中可以用一句话串联多个行动项，但每个带相对时间的行动项都必须紧跟括号补充客观日期、时间窗口或待确认说明。
不要把原文没有明确周次的“周三前/周五前/周三后”等擅自改写为“本周三前/本周五前/本周三后”；如果需要解释周次，请放在括号里作为候选或依据。
不要因为是 natural 风格就省略行动项时间括号；natural 只是不使用固定标题，不代表可以放松时间准确性。
如果能推理出合理日期，应尽量补充；如果仍有歧义，也要给出最可能候选并标注待确认，不要完全省略客观时间。
只有上下文提供小时级锚点时才补充小时，不要为了显得精确而虚构小时。
语气像正式周报或会议纪要，避免显得像检查清单。"""

    raise ValueError(f"Unsupported summary style: {summary_style}")


def summary_temperature(summary_style: str) -> float:
    if summary_style == "structured":
        return 0.0
    if summary_style == "natural":
        return 0.0
    raise ValueError(f"Unsupported summary style: {summary_style}")


def build_augmented_summary_prompt(
    current_content: str,
    retrieved_memories: list[str],
    summary_style: str,
    source_time: str = "",
    artifact_board: Optional[dict[str, Any]] = None,
) -> str:
    memory_block = format_retrieved_memories(retrieved_memories)
    artifact_block = format_artifact_board_for_summary(artifact_board)
    style_instructions = build_summary_style_instructions(summary_style)
    source_time_text = format_source_time_anchor(source_time)
    time_reference = build_time_reasoning_reference(source_time)
    return f"""你将根据当前原始材料和有证据的历史记忆生成一份增强摘要。

当前材料时间锚点 source_time：
{source_time_text}

可用日历参考：
{time_reference}

当前原始材料（当前事实的最高优先级）：
{current_content}

以下是本次运行根据 retrieved typed memories 归并出的 artifact board。
它用于整理历史时间窗口、前期决策、既有约束、任务状态、风险和偏好线索，不替代当前原始材料：
{artifact_block}

以下是 artifact board 的 typed memory 证据来源。
它们只供你理解前期约定、既有边界、任务分工和延续事项。
请优先根据 artifact board 做状态判断，并在需要核对事实时参考这些证据；不要生硬堆砌：
{memory_block}

生成要求：
1. 当前事实必须以当前原始材料为准；artifact board 用于理解历史上下文，typed memory 用于核对历史证据。发生冲突时，不得用 artifact board 或 typed memory 覆盖当前原始材料。
2. 摘要要衔接历史背景、历史目标、历史决策和长期约束。
3. 明确当前材料相对于前期约定、既有任务和长期约束的新增进展、变化或冲突。
4. 突出本次新增决策、负责人、后续任务、风险阻塞和未完成事项。
5. 必须主动利用时间信息：当前材料时间锚点、历史时间线、明确日期、时间段、阶段、版本、
   相对时间、星期表达、历史计划时间、历史实际完成时间、任务截止窗口、开始窗口、阶段窗口、
   延期/提前/按期完成、已过期/仍有效状态。
   时间信息不只用于未来待办，也要用于判断历史任务是否已经完成、前期承诺是否已兑现、
   当前材料是否处在某个历史阶段之后、哪些历史截止已经成为当前工作的前提或约束。
6. 对所有时间表达使用统一解析流程：
   - 识别原始时间表达：明确日期、今天/明天/后天、本周/下周/上周、周几前/后、未来N天/周/月、月底前、下月初、季度末、年底前、阶段/版本窗口、延期/提前/仍有效等。
   - 确定时间锚点：优先使用原文明确日期，其次使用 source_time，再结合历史上下文。
   - 判断语义方向：待办、计划、需要完成、开始、交付、评估、提交等通常指未来；已完成、上周、回顾、此前、历史承诺等通常指过去或历史窗口。
   - 输出时保留原表达，并在括号中补充推理后的客观日期、时间段或候选解释。
   - 最后校验日期与星期、月份、季度是否一致。
7. 时间解析必须遵循证据优先级：
   - 原文已有明确日期或明确时间段时，直接保留并校验，不要重新推断。
   - 原文明确写出“上周/本周/下周/下个月/月底”等周次或周期时，按原文周期解释；如果该日期已早于 source_time 且语境是当前后续行动，请写明“已早于当前材料日期，需确认是否逾期/是否仍有效”，不要擅自改成别的周次。
   - 原文只有“周几前/周几后/周几开始”等无周次表达时，必须结合语义方向选择候选：未来行动使用不早于 source_time 的合理候选，历史回顾使用过去或历史窗口候选。
   - 如果方向仍不明确，保留原表达并给出最可能候选或“具体日期待确认”，不要把猜测写成绝对事实。
8. 对当前材料之后的行动项，括号中的客观日期原则上不能早于 source_time，除非摘要同时明确说明该事项是在回顾旧计划、已经逾期或需要补救。
   例如 source_time 是周四时，无周次修饰的“周三前完成”若语境是后续待办，应按未来候选解释；只有原文明确“本周三前”或“已在周三完成”时，才可使用已过去的日期，并说明状态。
9. 如果能从 source_time、原文或历史上下文推理出合理日期，应尽量补充，不要因为表达是相对时间就省略客观时间。
   对“周几后/周几前/本周内/下周初”等表达，也要补成“原表达（YYYY-MM-DD 后/前/起，按……理解）”或给出候选。
10. 如果存在歧义，输出最可能候选并标注原因或待确认；不要把多个可能周次中的一个强写成绝对事实。
11. 只有原文或上下文有小时级锚点时才补充小时；否则补充到日期、日期范围、月份、季度或阶段粒度。
12. 使用历史时间信息时，请区分四类状态：
   - 已完成/已兑现：写成历史进展或当前前提，不要再写成未来待办。
   - 延期/未完成：写清原窗口、当前状态和对后续任务的影响。
   - 仍有效：写清它对当前材料的约束或依赖关系。
   - 已过期/被更新：写清新旧时间要求的变化或冲突。
13. 如果历史记忆显示某人已在某日期或截止日前完成了某事项，最终摘要应把它用于解释当前材料的进展基础、
    依赖已满足或风险变化；不要只在“下一步任务”里重复未来计划。
14. 结构化摘要生成前，必须先把时间信息分桶：
    - 历史窗口/历史状态/历史完成记录/历史承诺是否兑现 → 放入“四、历史时间窗口与已知状态”。
    - 当前材料之后的新行动、负责人、下一步、未来截止或开始窗口 → 放入“六、负责人、下一步任务与时间窗口”，
      并按“负责人：事项，原始时间表达（客观日期/时间窗口）”写出。
    - 风险、阻塞、待确认输入条件或未决范围 → 放入“五、风险、阻塞与待确认问题”。
    不要在第四节按负责人逐条复制第六节的行动清单。
    第四节不得出现当前材料中的下一步任务、负责人行动项、未来截止或开始窗口；
    即使这些行动项与历史窗口有关，也不能把行动项本身写进第四节。
    第六节是行动清单，必须尽量补足每个行动项的客观时间括号；不要只保留“周三前/周五前/下班前/周三后”等原始相对时间。
15. 如果历史记忆只有“前期某事项需在某日期前完成/一周内完成”，但没有说明是否完成，
    结构化摘要第四节应写成“前期截至窗口，完成状态未明确/需由当前材料确认”，
    而不是把它写成当前仍需完成的下一步任务。
16. 如果当前材料与前期约定不一致，请明确说明变化，不要强行调和。
17. 不要把历史中已经暂缓、否定或仅待讨论的方向写成当前已决定事项。
   如果当前材料只是再次讨论某个历史上暂缓或否定的方向，但没有明确推翻旧决策，
   请写清“本次重新讨论，但尚未推翻此前决定”。
18. 生成前必须主动检查“历史约束/暂缓事项/否定事项”和“当前材料中新提议或重新讨论事项”之间的关系。
   如果当前材料重新提出历史中已暂缓、否定或受限制的方向，例如新增指标、自动发送、
   外部网页抓取、自动同步文件、扩大权限、过度确认实验结论等，即使当前材料没有明确说“冲突”，
   也必须在摘要中说明它与此前约定的关系。
19. 对结构化摘要，如果存在上述重新讨论或潜在冲突，“三、与前期约定的变化或冲突”不能写
   “本次材料未明确提及”，而应写明“本次重新讨论，但尚未推翻此前决定”或相应事实。
   对自然化摘要，也要用自然语言写出同样判断。
20. 如果已经识别出任何变化、冲突、重新讨论、边界确认或风险延续，不要再同时写
   “本次材料未明确提及与前期约定的变化或冲突”。该兜底句只能在完全没有相关信息时使用。
21. 不要把“重新讨论但未决定”的事项分配成负责人任务，除非当前材料明确指定了负责人。
   例如某人只是提出 arXiv 自动抓取、自动发送回复或新增指标，不等于他已经负责推进该事项。
   在“负责人和下一步任务”中，这类事项应写成“负责人未明确，待团队进一步评估/确认”，
   不要写成“提出者需继续推进/探讨/确认该事项”。
22. 涉及数字、日期、星期、样本量、负责人姓名、任务状态等事实时，必须与当前材料或历史上下文一致。
    不要自行改写、补全或省略关键数字；例如当前材料写“12 份”，最终摘要也必须保留为“12 份”。
23. 不要在最终摘要正文中使用“artifact board”“typed memory”“历史记忆”“检索到的记忆”“长期记忆”等系统内部术语。
   请改写为“此前约定”“前期决策”“既有约束”“前期分工”“历史背景”等自然表达。
24. 语言简洁，结构清晰，优先使用中文。

{style_instructions}
"""


def generate_augmented_summary(prompt: str, summary_style: str) -> str:
    return call_local_qwen(prompt, temperature=summary_temperature(summary_style), max_tokens=SUMMARY_MAX_TOKENS)


SUMMARY_SCORE_BASE_SCORE = 100

SUMMARY_SCORE_RULES: list[dict[str, Any]] = [
    {
        "id": "completeness_core_points",
        "name": "完整性-核心信息覆盖",
        "applies_to": "all",
        "max_deduction": 10,
        "judge_prompt": "检查摘要是否覆盖主题、关键结论、关键事实，不可明显遗漏核心信息。",
        "deduction_guide": "无明显遗漏扣0；轻微遗漏扣2-3；核心遗漏扣4-5。",
    },
    {
        "id": "structure_template_alignment",
        "name": "结构-模板一致性",
        "applies_to": "all",
        "max_deduction": 15,
        "judge_prompt": "检查摘要是否按模板结构组织，标题层级是否清晰。",
        "deduction_guide": "结构清晰扣0；个别标题缺失扣2-5；结构混乱扣6-10。",
    },
    {
        "id": "factual_consistency",
        "name": "准确性-事实一致",
        "applies_to": "all",
        "max_deduction": 25,
        "judge_prompt": "检查摘要表述是否与原文事实一致，避免幻觉和无依据扩写。",
        "deduction_guide": "事实一致扣0；轻微偏差扣5-12；明显事实错误扣13-25。",
    },
    {
        "id": "fluency",
        "name": "流畅性",
        "applies_to": "all",
        "max_deduction": 20,
        "judge_prompt": "检查摘要表述是否流畅，避免语法错误和废话信息。",
        "deduction_guide": "输出流畅扣0；轻微语法错误和废话信息扣5-12；严重语法错误和废话信息扣13-20。",
    },
    {
        "id": "quality",
        "name": "质量性检查",
        "applies_to": "all",
        "max_deduction": 15,
        "judge_prompt": "检查摘要表述是否质量低下，避免机械重复和乱码。",
        "deduction_guide": "无机械重复和乱码扣0；轻微机械重复和乱码扣3-8；严重机械重复和乱码扣9-15。",
    },
    {
        "id": "richness",
        "name": "内容丰富度检查",
        "applies_to": "all",
        "max_deduction": 5,
        "judge_prompt": "检查摘要表述是否过于简略，避免丰富度不足。",
        "deduction_guide": "内容详实扣0；内容简略扣5。",
    },
    {
        "id": "safety",
        "name": "安全性检查",
        "applies_to": "all",
        "max_deduction": 10,
        "judge_prompt": "检查摘要表述是否安全，避免黄赌毒、血腥暴力、政治/文化/宗教敏感、消极情绪等内容。",
        "deduction_guide": "内容安全扣0；轻微不安全扣2-5；明显不安全扣6-10。",
    },
    {
        "id": "meeting_action_items",
        "name": "会议-行动项可执行性",
        "applies_to": "meeting",
        "max_deduction": 25,
        "judge_prompt": "检查是否明确行动项、负责人、时间节点，强调可执行性。",
        "deduction_guide": "完整可执行扣0；缺少1项关键要素扣8-15；缺少多项扣16-25。",
    },
    {
        "id": "interview_qa_signal",
        "name": "访谈-问答要点呈现",
        "applies_to": "interview",
        "max_deduction": 20,
        "judge_prompt": "检查是否体现问答关系、观点与依据，避免只罗列结论。",
        "deduction_guide": "问答清晰扣0；问答关系弱扣6-12；缺少问答结构扣13-20。",
    },
]


def select_summary_score_rules(score_type: str) -> list[dict[str, Any]]:
    applies_to = "all" if score_type == "general" else score_type
    return [rule for rule in SUMMARY_SCORE_RULES if rule["applies_to"] in {"all", applies_to}]


def build_summary_score_prompt(
    current_content: str,
    retrieved_memories: list[str],
    summary_text: str,
    summary_style: str,
    score_type: str,
) -> str:
    rules = select_summary_score_rules(score_type)
    rule_block = json.dumps(rules, ensure_ascii=False, indent=2)
    memory_block = format_retrieved_memories(retrieved_memories)
    return f"""你是一个严格但公平的中文摘要质量评审器。

请根据评分规则，对生成摘要进行扣分制评分。基础分是 {SUMMARY_SCORE_BASE_SCORE} 分。
每条规则只能扣 0 到 max_deduction 之间的整数分，不要超过该规则上限。

评分时请同时参考：
1. 当前材料，这是主要事实来源。
2. 历史上下文，它用于判断摘要中衔接前期约定、任务分工和长期约束的内容是否有依据。
3. 摘要输出风格：structured 表示固定栏目结构；natural 表示自然段结构。
   对 natural 摘要，不要仅因为没有 Markdown 标题或编号列表而扣“结构-模板一致性”。

当前材料：
{current_content}

历史上下文：
{memory_block}

被评估摘要风格：{summary_style}
评分类型：{score_type}

被评估摘要：
{summary_text}

评分规则：
{rule_block}

输出必须是严格 JSON 对象，不要输出 markdown，不要输出解释文字。
输出格式：
{{
  "base_score": 100,
  "score_type": "{score_type}",
  "summary_style": "{summary_style}",
  "total_deduction": 0,
  "final_score": 100,
  "rule_results": [
    {{
      "id": "rule_id",
      "name": "规则名称",
      "deduction": 0,
      "max_deduction": 10,
      "reason": "简要说明扣分理由；如果不扣分，说明达标原因"
    }}
  ],
  "overall_comment": "一句话总体评价"
}}
"""


def normalize_score_result(raw_result: dict[str, Any], summary_style: str, score_type: str) -> dict[str, Any]:
    rules = select_summary_score_rules(score_type)
    rule_by_id = {rule["id"]: rule for rule in rules}
    raw_rule_results = raw_result.get("rule_results", [])
    raw_rule_by_id = {
        str(item.get("id", "")): item for item in raw_rule_results if isinstance(item, dict)
    } if isinstance(raw_rule_results, list) else {}

    normalized_rules: list[dict[str, Any]] = []
    total_deduction = 0
    for rule in rules:
        raw_rule = raw_rule_by_id.get(rule["id"], {})
        try:
            deduction = int(raw_rule.get("deduction", 0))
        except (TypeError, ValueError):
            deduction = 0
        deduction = max(0, min(deduction, int(rule["max_deduction"])))
        total_deduction += deduction
        normalized_rules.append(
            {
                "id": rule["id"],
                "name": rule["name"],
                "applies_to": rule["applies_to"],
                "deduction": deduction,
                "max_deduction": rule["max_deduction"],
                "reason": str(raw_rule.get("reason", "")).strip(),
            }
        )

    final_score = max(0, SUMMARY_SCORE_BASE_SCORE - total_deduction)
    return {
        "base_score": SUMMARY_SCORE_BASE_SCORE,
        "score_type": score_type,
        "summary_style": summary_style,
        "total_deduction": total_deduction,
        "final_score": final_score,
        "rule_results": normalized_rules,
        "overall_comment": str(raw_result.get("overall_comment", "")).strip(),
        "raw_final_score": raw_result.get("final_score"),
        "raw_total_deduction": raw_result.get("total_deduction"),
        "rule_count": len(rule_by_id),
    }


def parse_summary_score_json(raw_output: str, summary_style: str, score_type: str) -> dict[str, Any]:
    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "base_score": SUMMARY_SCORE_BASE_SCORE,
            "score_type": score_type,
            "summary_style": summary_style,
            "total_deduction": None,
            "final_score": None,
            "rule_results": [],
            "overall_comment": "评分 JSON 解析失败，请查看 raw_output。",
            "raw_output": raw_output,
            "parse_error": True,
        }

    if not isinstance(parsed, dict):
        return {
            "base_score": SUMMARY_SCORE_BASE_SCORE,
            "score_type": score_type,
            "summary_style": summary_style,
            "total_deduction": None,
            "final_score": None,
            "rule_results": [],
            "overall_comment": "评分输出不是 JSON 对象。",
            "raw_output": raw_output,
            "parse_error": True,
        }

    return normalize_score_result(parsed, summary_style, score_type)


def score_augmented_summary(
    current_content: str,
    retrieved_memories: list[str],
    summary_text: str,
    summary_style: str,
    score_type: str,
) -> dict[str, Any]:
    prompt = build_summary_score_prompt(
        current_content=current_content,
        retrieved_memories=retrieved_memories,
        summary_text=summary_text,
        summary_style=summary_style,
        score_type=score_type,
    )
    raw_output = call_local_qwen(prompt, temperature=0.0, max_tokens=2200)
    return parse_summary_score_json(raw_output, summary_style, score_type)


def read_text_file(path: Path) -> str:
    return path.expanduser().read_text(encoding="utf-8-sig").strip()


SECTION_ALIASES = {
    "history": "history_summary",
    "history_summary": "history_summary",
    "history_material": "history_summary",
    "history_content": "history_summary",
    "历史摘要": "history_summary",
    "历史材料": "history_summary",
    "current": "current_content",
    "current_content": "current_content",
    "current_material": "current_content",
    "当前材料": "current_content",
    "本次材料": "current_content",
}


def detect_case_section(line: str) -> Optional[str]:
    stripped = line.strip()
    if stripped.startswith("#"):
        section_name = stripped.lstrip("#").strip()
    elif stripped.startswith("===") and stripped.endswith("==="):
        section_name = stripped.strip("= ").strip()
    else:
        return None

    normalized = re.sub(r"[\s-]+", "_", section_name.lower())
    return SECTION_ALIASES.get(normalized)


def parse_case_file(path: Path) -> SummaryDemoInput:
    text = read_text_file(path)
    metadata: dict[str, str] = {}
    sections = {"history_summary": [], "current_content": []}
    current_section: Optional[str] = None
    metadata_keys = {"user_id", "project", "source_type", "source_time"}

    for line in text.splitlines():
        section = detect_case_section(line)
        if section:
            current_section = section
            continue

        if current_section:
            sections[current_section].append(line)
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
        if match:
            key = match.group(1).strip().replace("-", "_")
            if key in metadata_keys:
                metadata[key] = match.group(2).strip()

    history_summary = "\n".join(sections["history_summary"]).strip()
    current_content = "\n".join(sections["current_content"]).strip()
    missing = [
        name
        for name, value in {
            "history_summary": history_summary,
            "current_content": current_content,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"{path} 缺少必要段落：{', '.join(missing)}")

    defaults = build_builtin_demo_input()
    return SummaryDemoInput(
        user_id=metadata.get("user_id", defaults.user_id),
        project=metadata.get("project", defaults.project),
        source_type=metadata.get("source_type", defaults.source_type),
        source_time=metadata.get("source_time", defaults.source_time),
        history_summary=history_summary,
        current_content=current_content,
        source_path=path,
        source_file=path,
    )


def build_builtin_demo_input() -> SummaryDemoInput:
    return SummaryDemoInput(
        user_id="team_research_ops",
        project="paper_summary_assistant",
        source_type="project_weekly_summary",
        source_time="2026-04-10",
        history_summary="""
本周项目例会复盘了“论文阅读与实验追踪助手”的第一版范围。
团队确认第一版只面向组内论文阅读、实验记录和周会纪要，不对外开放协作空间。
为了降低合规风险，第一版暂不接入外部网页抓取，也不自动同步个人网盘文件；
所有输入材料都由研究员手动上传到项目目录。
摘要质量标准被重新明确：每份摘要必须区分“论文原文结论”“组内实验观察”和“待验证假设”，
不能把待验证假设写成已经成立的结论。
刘洋负责整理论文元数据字段，包括标题、作者、年份、任务、数据集和关键指标；
赵敏负责设计实验记录模板，要求保留失败实验和负结果。
团队还约定，后续周会摘要要优先突出研究结论变化、实验阻塞、负责人和下周验证计划。
""".strip(),
        current_content="""
本周周会讨论了论文阅读与实验追踪助手的开发进展。
刘洋表示论文元数据字段已经整理出第一版，但在抽取 benchmark 名称时发现同一数据集存在多种缩写，
需要增加别名映射和人工确认字段。
赵敏展示了新的实验记录模板，大家认为失败实验记录的位置不够明显，
建议把“负结果原因”和“下一步修正计划”放到模板首页。
王珂提出是否可以接入 arXiv 页面自动抓取论文摘要，以减少研究员手动上传负担；
会议中没有形成最终结论。
本周实验方面，检索增强摘要模型在两个任务上提升了人工评分，
但样本量只有 12 份，陈博士提醒不能把这视为稳定结论，
下周需要补充更多论文和失败案例。
""".strip(),
    )


def build_file_demo_input(args: argparse.Namespace) -> SummaryDemoInput:
    if not args.history_file or not args.current_file:
        raise ValueError("--history-file 和 --current-file 必须一起使用")

    defaults = build_builtin_demo_input()
    return SummaryDemoInput(
        user_id=args.user_id or defaults.user_id,
        project=args.project or defaults.project,
        source_type=args.source_type or defaults.source_type,
        source_time=args.source_time or defaults.source_time,
        history_summary=read_text_file(args.history_file),
        current_content=read_text_file(args.current_file),
        source_path=None,
        source_file=None,
    )


def apply_metadata_overrides(demo_input: SummaryDemoInput, args: argparse.Namespace) -> SummaryDemoInput:
    return SummaryDemoInput(
        user_id=args.user_id or demo_input.user_id,
        project=args.project or demo_input.project,
        source_type=args.source_type or demo_input.source_type,
        source_time=args.source_time or demo_input.source_time,
        history_summary=demo_input.history_summary,
        current_content=demo_input.current_content,
        source_path=demo_input.source_path,
        source_file=demo_input.source_file,
    )


def load_demo_input(args: argparse.Namespace) -> SummaryDemoInput:
    if args.history_dir:
        raise ValueError("--history-dir 只支持 ingest 命令")

    if args.case_file and (args.history_file or args.current_file):
        raise ValueError("--case-file 不能和 --history-file / --current-file 同时使用")

    if args.case_file:
        demo_input = parse_case_file(args.case_file)
    elif args.history_file or args.current_file:
        demo_input = build_file_demo_input(args)
    elif DEFAULT_CASE_FILE.exists():
        demo_input = parse_case_file(DEFAULT_CASE_FILE)
    else:
        demo_input = build_builtin_demo_input()

    return apply_metadata_overrides(demo_input, args)


def list_history_dir_files(history_dir: Path) -> list[Path]:
    directory = history_dir.expanduser()
    if not directory.exists():
        raise ValueError(f"--history-dir 不存在：{directory}")
    if not directory.is_dir():
        raise ValueError(f"--history-dir 必须是目录：{directory}")

    history_files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not history_files:
        raise ValueError(f"--history-dir 中没有找到 .txt 或 .md 文件：{directory}")
    return history_files


def build_ingest_history_file_input(args: argparse.Namespace, history_file: Path) -> SummaryDemoInput:
    defaults = build_builtin_demo_input()
    return SummaryDemoInput(
        user_id=args.user_id or defaults.user_id,
        project=args.project or defaults.project,
        source_type=args.source_type or defaults.source_type,
        source_time=args.source_time or defaults.source_time,
        history_summary=read_text_file(history_file),
        current_content="",
        source_path=history_file,
        source_file=history_file,
    )


def load_ingest_input(args: argparse.Namespace) -> SummaryDemoInput:
    if args.history_dir:
        raise ValueError("--history-dir 批量导入请通过 load_ingest_inputs 处理")
    if args.case_file and (args.history_file or args.current_file):
        raise ValueError("--case-file 不能和 --history-file / --current-file 同时使用")
    if args.current_file:
        raise ValueError("ingest 命令不使用 --current-file，请改用 --history-file 或 --history-dir")

    if args.case_file:
        demo_input = parse_case_file(args.case_file)
    elif args.history_file:
        demo_input = build_ingest_history_file_input(args, args.history_file)
    elif DEFAULT_CASE_FILE.exists():
        demo_input = parse_case_file(DEFAULT_CASE_FILE)
    else:
        demo_input = build_builtin_demo_input()

    return apply_metadata_overrides(demo_input, args)


def load_ingest_inputs(args: argparse.Namespace) -> list[SummaryDemoInput]:
    if args.history_dir:
        if args.case_file or args.history_file or args.current_file:
            raise ValueError("--history-dir 不能和 --case-file / --history-file / --current-file 同时使用")
        return [
            apply_metadata_overrides(build_ingest_history_file_input(args, path), args)
            for path in list_history_dir_files(args.history_dir)
        ]

    return [load_ingest_input(args)]


def load_summarize_input(args: argparse.Namespace) -> SummaryDemoInput:
    if args.history_dir:
        raise ValueError("--history-dir 只支持 ingest 命令")
    if args.case_file and (args.history_file or args.current_file):
        raise ValueError("--case-file 不能和 --history-file / --current-file 同时使用")
    if args.history_file:
        raise ValueError("summarize 命令不使用 --history-file，请改用 --current-file")

    if args.case_file:
        demo_input = parse_case_file(args.case_file)
    elif args.current_file:
        defaults = build_builtin_demo_input()
        demo_input = SummaryDemoInput(
            user_id=args.user_id or defaults.user_id,
            project=args.project or defaults.project,
            source_type=args.source_type or defaults.source_type,
            source_time=args.source_time or defaults.source_time,
            history_summary="",
            current_content=read_text_file(args.current_file),
            source_path=args.current_file,
            source_file=args.current_file,
        )
    elif DEFAULT_CASE_FILE.exists():
        demo_input = parse_case_file(DEFAULT_CASE_FILE)
    else:
        demo_input = build_builtin_demo_input()

    return apply_metadata_overrides(demo_input, args)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def list_case_files() -> None:
    case_dir = DEFAULT_CASE_FILE.parent
    if not case_dir.exists():
        print(f"未找到样例目录：{case_dir}")
        return

    case_files = sorted({*case_dir.glob("*.md"), *case_dir.glob("*.txt")})
    for path in case_files:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local mem0 summary-memory demo with built-in or external .txt/.md inputs."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "ingest", "summarize"),
        default="run",
        help="run=完整演示，ingest=只写入历史记忆，summarize=只读取历史记忆并生成摘要",
    )
    parser.add_argument("--case-file", type=Path, help="单个样例文件，格式见 scripts/summary_memory_cases/*.md 或 *.txt")
    parser.add_argument("--history-file", type=Path, help="历史摘要/周报/会议纪要 .txt 或 .md 文件")
    parser.add_argument("--history-dir", type=Path, help="批量导入历史材料目录，仅支持 ingest；非递归读取 .txt / .md")
    parser.add_argument("--current-file", type=Path, help="当前材料 .txt 或 .md 文件")
    parser.add_argument("--output-dir", type=Path, help="保存本次运行产物的目录；每次运行会自动创建时间子目录")
    parser.add_argument("--user-id", help="覆盖样例中的 user_id")
    parser.add_argument("--project", help="覆盖样例中的 project")
    parser.add_argument("--source-type", help="覆盖样例中的 source_type")
    parser.add_argument("--source-time", help="覆盖样例中的 source_time")
    parser.add_argument("--top-k", type=int, default=8, help="检索历史记忆条数，默认 8")
    parser.add_argument(
        "--summary-style",
        choices=("structured", "natural", "both"),
        default="structured",
        help="增强摘要输出风格：structured=固定栏目，natural=自然段，both=同时输出两版",
    )
    parser.add_argument("--score-summary", action="store_true", help="对生成的增强摘要进行 LLM 评分，仅支持 run/summarize")
    parser.add_argument(
        "--rewrite-from-critic",
        action="store_true",
        help="当 Summary Critic 存在 high/medium issue 时生成 rewrite draft；不覆盖原摘要，仅支持 run/summarize",
    )
    parser.add_argument(
        "--score-type",
        choices=("general", "meeting", "interview"),
        default="general",
        help="评分规则类型：general=只使用通用规则，meeting/interview=额外启用对应场景规则",
    )
    parser.add_argument("--list-cases", action="store_true", help="列出可用样例文件，不运行模型")
    return parser.parse_args()


def close_summary_memory(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    client = getattr(vector_store, "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            print(f"[Qdrant close warning] {exc}")


def build_summary_memory() -> Any:
    from mem0 import Memory

    patch_huggingface_embedding()
    return Memory.from_config(build_summary_memory_config())


def make_path_segment(value: str, fallback: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    segment = segment.strip("._-")
    return segment or fallback


def create_output_run_dir(output_dir: Optional[Path], command: str, project: str) -> Optional[Path]:
    if not output_dir:
        return None

    root = output_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}_{make_path_segment(command, 'run')}_{make_path_segment(project, 'project')}"
    candidate = root / base_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base_name}-{suffix}"
        suffix += 1

    candidate.mkdir(parents=True)
    return candidate


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def write_json_file(path: Path, data: Any) -> None:
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_run_info(
    command: str,
    demo_input: SummaryDemoInput,
    top_k: Optional[int] = None,
    summary_style: Optional[str] = None,
    score_summary: bool = False,
    score_type: Optional[str] = None,
    sources: Optional[list[Path]] = None,
) -> dict[str, Any]:
    run_info: dict[str, Any] = {
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": demo_input.user_id,
        "project": demo_input.project,
        "source_type": demo_input.source_type,
        "source_time": demo_input.source_time,
        "source_path": demo_input.source_path,
        "source_file": demo_input.source_file,
        "project_root": PROJECT_ROOT,
        "model_root": MODEL_ROOT,
        "qdrant_summary_path": QDRANT_SUMMARY_PATH,
        "summary_collection_name": SUMMARY_COLLECTION_NAME,
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
        "typed_memory_extraction_max_tokens": TYPED_MEMORY_EXTRACTION_MAX_TOKENS,
        "typed_memory_dedup_judge_max_tokens": TYPED_MEMORY_DEDUP_JUDGE_MAX_TOKENS,
        "typed_memory_dedup_judge_max_candidates": TYPED_MEMORY_DEDUP_JUDGE_MAX_CANDIDATES,
        "llm_runtime": get_llm_runtime_info(),
    }
    if top_k is not None:
        run_info["top_k"] = top_k
    if summary_style is not None:
        run_info["summary_style"] = summary_style
    run_info["score_summary"] = score_summary
    if score_type is not None:
        run_info["score_type"] = score_type
    if sources is not None:
        run_info["sources"] = sources
        run_info["source_count"] = len(sources)
    return run_info


def save_summary_files(output_run_dir: Path, summaries: dict[str, str]) -> None:
    for style, text in summaries.items():
        write_text_file(output_run_dir / f"summary_{style}.md", text)


def save_report(
    output_run_dir: Path,
    command: str,
    demo_input: SummaryDemoInput,
    extracted_memories: Optional[Any] = None,
    add_results: Optional[Any] = None,
    retrieved_memories: Optional[list[str]] = None,
    summaries: Optional[dict[str, str]] = None,
    summary_scores: Optional[dict[str, Any]] = None,
    summary_prompts: Optional[dict[str, str]] = None,
    summary_critic_report: Optional[dict[str, Any]] = None,
) -> None:
    lines = [
        "# Summary Memory Demo Report",
        "",
        f"- Command: {command}",
        f"- User ID: {demo_input.user_id}",
        f"- Project: {demo_input.project}",
        f"- Source Type: {demo_input.source_type}",
        f"- Source Time: {demo_input.source_time}",
    ]
    if demo_input.source_path:
        lines.append(f"- Source Path: {demo_input.source_path}")

    if extracted_memories is not None:
        if isinstance(extracted_memories, list) and extracted_memories and isinstance(extracted_memories[0], dict):
            count = sum(len(item.get("memories", [])) for item in extracted_memories)
        else:
            count = len(extracted_memories)
        lines.append(f"- Extracted Memories: {count}")
    if add_results is not None:
        if (
            isinstance(add_results, list)
            and add_results
            and isinstance(add_results[0], dict)
            and "add_results" in add_results[0]
        ):
            count = sum(len(item.get("add_results", [])) for item in add_results)
        else:
            count = len(add_results)
        lines.append(f"- Add Result Calls: {count}")
    if retrieved_memories is not None:
        lines.append(f"- Retrieved Memories: {len(retrieved_memories)}")
    if summaries is not None:
        lines.append(f"- Summaries: {', '.join(summaries)}")
    if summary_prompts is not None:
        lines.append("- Augmented Summary Prompt: augmented_summary_prompt.txt")
    if summary_critic_report is not None:
        lines.append("- Summary Critic Prompt: summary_critic_prompt.txt")
        lines.append("- Summary Rewrite Draft: summary_rewrite_report.json / summary_rewritten_*.md")
        lines.append(
            "- Summary Critic: "
            f"passed={summary_critic_report.get('passed')} "
            f"issues={summary_critic_report.get('issue_count')}"
        )
    if summary_scores is not None:
        score_parts = [
            f"{style}={result.get('final_score')}"
            for style, result in summary_scores.items()
            if isinstance(result, dict)
        ]
        lines.append(f"- Summary Scores: {', '.join(score_parts)}")

    write_text_file(output_run_dir / "report.md", "\n".join(lines))


def print_augmented_summaries(
    current_content: str,
    retrieved_memories: list[str],
    summary_style: str,
    section_number: str,
    source_time: str = "",
    artifact_board: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    styles = ["structured", "natural"] if summary_style == "both" else [summary_style]
    style_labels = {
        "structured": "结构化",
        "natural": "自然化",
    }

    summaries: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for idx, style in enumerate(styles, start=1):
        if summary_style == "both":
            print_section(f"{section_number}{chr(64 + idx)}. 生成{style_labels[style]}增强摘要")
        else:
            print_section(f"{section_number}. 生成{style_labels[style]}增强摘要")

        prompt = build_augmented_summary_prompt(
            current_content,
            retrieved_memories,
            style,
            source_time=source_time,
            artifact_board=artifact_board,
        )
        prompts[style] = prompt
        augmented_summary = generate_augmented_summary(prompt, style)
        summaries[style] = augmented_summary
        print(augmented_summary)
    return summaries, prompts


def format_augmented_summary_prompts(prompts: dict[str, str]) -> str:
    if not prompts:
        return ""

    ordered_styles = [style for style in ("structured", "natural") if style in prompts]
    ordered_styles.extend(style for style in prompts if style not in ordered_styles)
    if len(ordered_styles) == 1:
        return prompts[ordered_styles[0]]

    sections = []
    for style in ordered_styles:
        sections.append(f"===== {style} augmented summary prompt =====\n\n{prompts[style]}")
    return "\n\n".join(sections)


def print_summary_scores(
    current_content: str,
    retrieved_memories: list[str],
    summaries: dict[str, str],
    score_type: str,
    section_number: str,
) -> dict[str, Any]:
    print_section(f"{section_number}. 生成摘要评分")
    score_results: dict[str, Any] = {}
    for style, summary_text in summaries.items():
        result = score_augmented_summary(
            current_content=current_content,
            retrieved_memories=retrieved_memories,
            summary_text=summary_text,
            summary_style=style,
            score_type=score_type,
        )
        score_results[style] = result
        print(f"\n[{style}] final_score: {result.get('final_score')} / {SUMMARY_SCORE_BASE_SCORE}")
        print(f"[{style}] total_deduction: {result.get('total_deduction')}")
        overall_comment = result.get("overall_comment")
        if overall_comment:
            print(f"[{style}] overall_comment: {overall_comment}")

        for item in result.get("rule_results", []):
            print(
                f"- {item.get('id')} {item.get('name')}: "
                f"-{item.get('deduction')}/{item.get('max_deduction')}，{item.get('reason')}"
            )
    return score_results


def generate_summary_critic_report(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
) -> tuple[dict[str, Any], str, str]:
    prompt = build_summary_critic_prompt(
        current_content=current_content,
        source_time=source_time,
        artifact_board=artifact_board,
        retrieved_typed_memories=retrieved_typed_memories,
        summaries=summaries,
    )
    try:
        raw_output = call_local_qwen(prompt, temperature=0.0, max_tokens=SUMMARY_CRITIC_MAX_TOKENS)
        report = parse_summary_critic_json(raw_output)
    except Exception as exc:  # noqa: BLE001 - keep summary outputs available if critic fails.
        report = {
            "schema_version": SUMMARY_CRITIC_SCHEMA_VERSION,
            "passed": False,
            "issue_count": 1,
            "summary_results": {},
            "issues": [
                {
                    "issue_id": "issue_critic_runtime_error",
                    "severity": "high",
                    "issue_type": "critic_runtime_error",
                    "summary_style": "all",
                    "location": "summary_critic",
                    "problem": f"Summary Critic 调用失败：{exc}",
                    "source_evidence": "",
                    "summary_quote": "",
                    "evidence_refs": [],
                    "conflicting_artifact": "",
                    "suggested_fix": "检查 vLLM 服务、critic max_tokens 或重试；摘要文件仍会正常落盘。",
                    "required_action": "mark_unclear",
                    "rewrite_instruction": "检查 vLLM 服务、critic max_tokens 或重试；摘要文件仍会正常落盘。",
                    "needs_retrieval": False,
                    "route_to": "unknown",
                }
            ],
            "checked_items": [],
            "warnings": ["summary critic runtime failed"],
            "runtime_error": True,
        }
    return report, format_summary_critic_markdown(report), prompt


def print_summary_critic_report(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
    section_number: str,
) -> tuple[dict[str, Any], str, str]:
    print_section(f"{section_number}. Summary Critic 检查")
    report, report_markdown, critic_prompt = generate_summary_critic_report(
        current_content=current_content,
        source_time=source_time,
        artifact_board=artifact_board,
        retrieved_typed_memories=retrieved_typed_memories,
        summaries=summaries,
    )
    print(report_markdown)
    return report, report_markdown, critic_prompt



def generate_summary_rewrite_draft(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
    critic_report: dict[str, Any],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    prompt = build_summary_rewrite_prompt(
        current_content=current_content,
        source_time=source_time,
        artifact_board=artifact_board,
        retrieved_typed_memories=retrieved_typed_memories,
        summaries=summaries,
        critic_report=critic_report,
    )
    try:
        raw_output = call_local_qwen(prompt, temperature=0.0, max_tokens=SUMMARY_REWRITE_MAX_TOKENS)
        rewritten = parse_rewritten_summaries(raw_output, summaries)
        if not rewritten:
            raise ValueError("rewrite output did not contain structured/natural sections")
        report = build_summary_rewrite_report(
            critic_report=critic_report,
            rewritten_summaries=rewritten,
            enabled=True,
            triggered=True,
        )
    except Exception as exc:  # noqa: BLE001 - keep original summaries if rewrite fails.
        rewritten = {}
        report = build_summary_rewrite_report(
            critic_report=critic_report,
            rewritten_summaries={},
            enabled=True,
            triggered=True,
            notes=f"Rewrite draft failed: {exc}. Original summaries were kept untouched.",
        )
        report["runtime_error"] = True
        report["error"] = str(exc)
    return rewritten, prompt, report


def print_summary_rewrite_draft(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
    critic_report: dict[str, Any],
    section_number: str,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    print_section(f"{section_number}. Summary Rewrite Draft（不覆盖原摘要）")
    if not should_rewrite_from_critic(critic_report):
        prompt = ""
        report = build_summary_rewrite_report(
            critic_report=critic_report or {},
            rewritten_summaries={},
            enabled=True,
            triggered=False,
            notes="No high/medium critic issues; rewrite draft skipped.",
        )
        print("[Summary Rewrite] skipped: no high/medium issues")
        return {}, prompt, report

    rewritten, prompt, report = generate_summary_rewrite_draft(
        current_content=current_content,
        source_time=source_time,
        artifact_board=artifact_board,
        retrieved_typed_memories=retrieved_typed_memories,
        summaries=summaries,
        critic_report=critic_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for style, text in rewritten.items():
        print(f"\n[Rewritten {style}]\n{text}")
    return rewritten, prompt, report


def save_rewrite_files(
    output_run_dir: Path,
    rewritten_summaries: dict[str, str],
    rewrite_prompt: Optional[str],
    rewrite_report: Optional[dict[str, Any]],
) -> None:
    if rewrite_prompt:
        write_text_file(output_run_dir / "summary_rewrite_prompt.txt", rewrite_prompt)
    if rewrite_report is not None:
        write_json_file(output_run_dir / "summary_rewrite_report.json", rewrite_report)
    if "structured" in rewritten_summaries:
        write_text_file(output_run_dir / "summary_rewritten_structured.md", rewritten_summaries["structured"])
    if "natural" in rewritten_summaries:
        write_text_file(output_run_dir / "summary_rewritten_natural.md", rewritten_summaries["natural"])

def print_run_info(
    mode: str,
    demo_input: SummaryDemoInput,
    top_k: Optional[int] = None,
    summary_style: Optional[str] = None,
    score_summary: bool = False,
    score_type: Optional[str] = None,
    output_run_dir: Optional[Path] = None,
) -> None:
    print_section("运行信息")
    print(f"MODE: {mode}")
    if demo_input.source_path:
        print(f"SOURCE_PATH: {demo_input.source_path}")
    print(f"USER_ID: {demo_input.user_id}")
    print(f"PROJECT: {demo_input.project}")
    print(f"SOURCE_TYPE: {demo_input.source_type}")
    print(f"SOURCE_TIME: {demo_input.source_time}")
    if top_k is not None:
        print(f"TOP_K: {top_k}")
    if summary_style is not None:
        print(f"SUMMARY_STYLE: {summary_style}")
    print(f"SCORE_SUMMARY: {score_summary}")
    if score_summary and score_type:
        print(f"SCORE_TYPE: {score_type}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MODEL_ROOT: {MODEL_ROOT}")
    print(f"QDRANT_SUMMARY_PATH: {QDRANT_SUMMARY_PATH}")
    print(f"SUMMARY_COLLECTION_NAME: {SUMMARY_COLLECTION_NAME}")
    print(f"VLLM_BASE_URL: {VLLM_BASE_URL}")
    print(f"VLLM_MODEL: {VLLM_MODEL}")
    print(f"TYPED_MEMORY_EXTRACTION_MAX_TOKENS: {TYPED_MEMORY_EXTRACTION_MAX_TOKENS}")
    print(f"TYPED_MEMORY_DEDUP_JUDGE_MAX_TOKENS: {TYPED_MEMORY_DEDUP_JUDGE_MAX_TOKENS}")
    print(f"TYPED_MEMORY_DEDUP_JUDGE_MAX_CANDIDATES: {TYPED_MEMORY_DEDUP_JUDGE_MAX_CANDIDATES}")
    print(f"LLM_RUNTIME: {json.dumps(get_llm_runtime_info(), ensure_ascii=False)}")
    if output_run_dir:
        print(f"OUTPUT_DIR: {output_run_dir}")


def save_typed_extraction_trace(
    output_dir: Path,
    trace: Optional[dict[str, Any]],
    stem: str = "typed_memory_extraction",
) -> None:
    if not trace:
        return

    prompt = str(trace.get("prompt", ""))
    raw_output = str(trace.get("raw_output", ""))
    if prompt:
        write_text_file(output_dir / f"{stem}_prompt.txt", prompt)
    if raw_output:
        write_text_file(output_dir / f"{stem}_raw_output.txt", raw_output)
    write_json_file(
        output_dir / f"{stem}_trace.json",
        {
            "max_tokens": trace.get("max_tokens"),
            "raw_output_chars": len(raw_output),
            "raw_record_count": trace.get("raw_record_count"),
            "accepted_record_count": trace.get("accepted_record_count"),
            "rejected_record_count": trace.get("rejected_record_count"),
        },
    )


def save_ingest_outputs(
    output_run_dir: Optional[Path],
    demo_input: SummaryDemoInput,
    extracted_memories: list[str],
    add_results: list[dict[str, Any]],
    typed_memories: Optional[list[dict[str, Any]]] = None,
    typed_extraction_trace: Optional[dict[str, Any]] = None,
) -> None:
    if not output_run_dir:
        return

    write_json_file(output_run_dir / "run_info.json", build_run_info("ingest", demo_input))
    write_text_file(output_run_dir / "history_input.md", demo_input.history_summary)
    if typed_memories is not None:
        write_json_file(output_run_dir / "typed_memories.json", typed_memories)
    save_typed_extraction_trace(output_run_dir, typed_extraction_trace)
    write_json_file(output_run_dir / "extracted_memories.json", extracted_memories)
    write_json_file(output_run_dir / "add_results.json", add_results)
    save_report(
        output_run_dir=output_run_dir,
        command="ingest",
        demo_input=demo_input,
        extracted_memories=extracted_memories,
        add_results=add_results,
    )


def save_batch_ingest_outputs(
    output_run_dir: Optional[Path],
    demo_inputs: list[SummaryDemoInput],
    batch_results: list[dict[str, Any]],
    typed_memory_dedup_report: Optional[dict[str, Any]] = None,
) -> None:
    if not output_run_dir:
        return

    first_input = demo_inputs[0]
    sources = [demo_input.source_file or demo_input.source_path for demo_input in demo_inputs]
    write_json_file(
        output_run_dir / "run_info.json",
        build_run_info("ingest", first_input, sources=[path for path in sources if path is not None]),
    )

    history_input_parts = []
    history_inputs_dir = output_run_dir / "history_inputs"
    history_inputs_dir.mkdir(parents=True, exist_ok=True)
    for idx, demo_input in enumerate(demo_inputs, start=1):
        source_name = demo_input.source_file.name if demo_input.source_file else (
            get_history_source_identity(demo_input) or f"history_{idx}.md"
        )
        output_name = f"{idx:03d}_{make_path_segment(source_name, f'history_{idx}')}.md"
        write_text_file(history_inputs_dir / output_name, demo_input.history_summary)
        history_input_parts.append(f"# {source_name}\n\n{demo_input.history_summary}")

    extracted_payload = [
        {
            "source_file": item["source_file"],
            "history_source_id": item.get("history_source_id", ""),
            "memories": item["memories"],
        }
        for item in batch_results
    ]
    typed_payload = [
        {
            "source_file": item["source_file"],
            "history_source_id": item.get("history_source_id", ""),
            "typed_memories": item.get("typed_memories", []),
        }
        for item in batch_results
    ]
    add_payload = [
        {"source_file": item["source_file"], "add_results": item["add_results"]} for item in batch_results
    ]

    write_text_file(output_run_dir / "history_input.md", "\n\n---\n\n".join(history_input_parts))
    write_json_file(output_run_dir / "typed_memories.json", typed_payload)
    write_json_file(output_run_dir / "extracted_memories.json", extracted_payload)
    write_json_file(output_run_dir / "add_results.json", add_payload)
    batch_results_without_traces = [
        {key: value for key, value in item.items() if key not in {"extraction_trace", "typed_records"}}
        for item in batch_results
    ]
    write_json_file(output_run_dir / "batch_results.json", batch_results_without_traces)
    if typed_memory_dedup_report is not None:
        write_json_file(output_run_dir / "typed_memory_dedup_report.json", typed_memory_dedup_report)
    traces_dir = output_run_dir / "typed_memory_extraction_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(batch_results, start=1):
        source = item.get("source_file")
        source_name = Path(str(source)).name if source else f"history_{index}.md"
        stem = f"{index:03d}_{make_path_segment(source_name, f'history_{index}')}_typed_memory_extraction"
        save_typed_extraction_trace(traces_dir, item.get("extraction_trace"), stem=stem)
    save_report(
        output_run_dir=output_run_dir,
        command="ingest",
        demo_input=first_input,
        extracted_memories=batch_results_without_traces,
        add_results=batch_results_without_traces,
    )


def run_ingest(demo_inputs: list[SummaryDemoInput], output_run_dir: Optional[Path] = None) -> None:
    memory = build_summary_memory()
    try:
        if len(demo_inputs) == 1:
            demo_input = demo_inputs[0]
            print_section("1. 从历史材料抽取 typed memories 并写入库")
            typed_memories, extracted_memories, add_results, extraction_trace = add_typed_summary_memories(
                memory=memory,
                summary_text=demo_input.history_summary,
                user_id=demo_input.user_id,
                project=demo_input.project,
                source_type=demo_input.source_type,
                source_time=demo_input.source_time,
                source_file=demo_input.source_file,
            )
            for idx, item in enumerate(extracted_memories, start=1):
                print(f"{idx}. {item}")
            print_typed_memories_summary(typed_memories)

            print_section("2. 写入 mem0 / Qdrant 的结果")
            for result in add_results:
                print(result)
            print_add_results_summary(extracted_memories, add_results)

            save_ingest_outputs(
                output_run_dir,
                demo_input,
                extracted_memories,
                add_results,
                typed_memories,
                extraction_trace,
            )
            print_run_info("ingest", demo_input, output_run_dir=output_run_dir)
            return

        print_section("1. 批量从历史材料抽取 typed memories")
        batch_results: list[dict[str, Any]] = []
        for file_idx, demo_input in enumerate(demo_inputs, start=1):
            source_label = format_history_source_label(demo_input, file_idx)
            print_section(f"1.{file_idx}. {source_label}")
            typed_records, extraction_trace = extract_typed_summary_memory_records(
                summary_text=demo_input.history_summary,
                source_time=demo_input.source_time,
                source_file=demo_input.source_file,
            )
            extracted_memories = [typed_record_to_memory_text(record) for record in typed_records]
            typed_memories = typed_memory_records_to_dicts(typed_records)
            for idx, item in enumerate(extracted_memories, start=1):
                print(f"{idx}. {item}")
            print_typed_memories_summary(typed_memories)
            batch_results.append(
                {
                    "source_file": demo_input.source_file or demo_input.source_path,
                    "history_source_id": get_history_source_identity(demo_input),
                    "source_time": demo_input.source_time,
                    "typed_records": typed_records,
                    "typed_memories": typed_memories,
                    "extraction_trace": extraction_trace,
                }
            )

        typed_memory_dedup_report = deduplicate_batch_typed_memory_records(
            batch_results,
            judge_llm_call=call_summary_llm,
        )
        print_section("2. 同源历史分块 typed memory 去重")
        print(f"去重前 typed memory 总数：{typed_memory_dedup_report['input_count']}")
        print(f"保留 typed memory 总数：{typed_memory_dedup_report['retained_count']}")
        print(f"移除跨 chunk 近重复数：{typed_memory_dedup_report['removed_count']}")
        for group in typed_memory_dedup_report["groups"]:
            if group["semantic_candidate_count"]:
                print(
                    f"- {group['history_source_id']} / {group['source_time']}: "
                    f"语义候选 {group['semantic_candidate_count']} 条，"
                    f"Judge={group['judge']['status']}，"
                    f"有效裁决 {group['evaluated_candidate_count']} 条"
                )
            if group["removed_count"]:
                print(
                    f"- {group['history_source_id']} / {group['source_time']}: "
                    f"移除 {group['removed_count']} 条"
                )
        if not typed_memory_dedup_report["groups"]:
            print("没有可应用去重的同源多 chunk 组；保留所有抽取记录。")

        print_section("3. 写入去重后的 typed memories 到 mem0 / Qdrant")
        for file_idx, (demo_input, item) in enumerate(zip(demo_inputs, batch_results), start=1):
            source_label = format_history_source_label(demo_input, file_idx)
            typed_records = item["typed_records"]
            typed_memories, extracted_memories, add_results = write_typed_summary_memory_records(
                memory=memory,
                typed_records=typed_records,
                user_id=demo_input.user_id,
                project=demo_input.project,
                source_type=demo_input.source_type,
                source_time=demo_input.source_time,
                source_file=demo_input.source_file,
            )
            item["typed_memories"] = typed_memories
            item["memories"] = extracted_memories
            item["add_results"] = add_results
            print_section(f"3.{file_idx}. {source_label}")
            for result in add_results:
                print(result)
            print_add_results_summary(extracted_memories, add_results)

        print_section("4. 批量写入汇总")
        total_typed_memories = sum(len(item.get("typed_memories", [])) for item in batch_results)
        total_memories = sum(len(item["memories"]) for item in batch_results)
        total_calls = sum(len(item["add_results"]) for item in batch_results)
        print(f"历史材料文件数：{len(demo_inputs)}")
        print(f"typed memory 总数：{total_typed_memories}")
        print(f"抽取记忆总数：{total_memories}")
        print(f"memory.add 调用总数：{total_calls}")

        save_batch_ingest_outputs(
            output_run_dir,
            demo_inputs,
            batch_results,
            typed_memory_dedup_report=typed_memory_dedup_report,
        )
        print_run_info("ingest", demo_inputs[0], output_run_dir=output_run_dir)
    finally:
        close_summary_memory(memory)


def save_summarize_outputs(
    output_run_dir: Optional[Path],
    demo_input: SummaryDemoInput,
    retrieved_memories: list[str],
    summaries: dict[str, str],
    summary_scores: Optional[dict[str, Any]],
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    typed_retrieval_plan: Optional[list[dict[str, Any]]] = None,
    retrieved_typed_memories: Optional[list[dict[str, Any]]] = None,
    artifact_board: Optional[dict[str, Any]] = None,
    summary_prompts: Optional[dict[str, str]] = None,
    summary_critic_report: Optional[dict[str, Any]] = None,
    summary_critic_markdown: Optional[str] = None,
    summary_critic_prompt: Optional[str] = None,
    rewritten_summaries: Optional[dict[str, str]] = None,
    summary_rewrite_prompt: Optional[str] = None,
    summary_rewrite_report: Optional[dict[str, Any]] = None,
) -> None:
    if not output_run_dir:
        return

    write_json_file(
        output_run_dir / "run_info.json",
        build_run_info("summarize", demo_input, top_k, summary_style, score_summary, score_type),
    )
    write_text_file(output_run_dir / "current_input.md", demo_input.current_content)
    if typed_retrieval_plan is not None:
        write_json_file(output_run_dir / "typed_retrieval_plan.json", typed_retrieval_plan)
    if retrieved_typed_memories is not None:
        write_json_file(output_run_dir / "retrieved_typed_memories.json", retrieved_typed_memories)
    if artifact_board is not None:
        write_json_file(output_run_dir / "artifact_board.json", artifact_board)
    write_json_file(output_run_dir / "retrieved_memories.json", retrieved_memories)
    if summary_prompts is not None:
        write_text_file(output_run_dir / "augmented_summary_prompt.txt", format_augmented_summary_prompts(summary_prompts))
    if summary_critic_prompt is not None:
        write_text_file(output_run_dir / "summary_critic_prompt.txt", summary_critic_prompt)
    if summary_critic_report is not None:
        write_json_file(output_run_dir / "summary_critic_report.json", summary_critic_report)
    if summary_critic_markdown is not None:
        write_text_file(output_run_dir / "summary_critic_report.md", summary_critic_markdown)
    save_summary_files(output_run_dir, summaries)
    if rewritten_summaries or summary_rewrite_prompt or summary_rewrite_report is not None:
        save_rewrite_files(
            output_run_dir,
            rewritten_summaries or {},
            summary_rewrite_prompt,
            summary_rewrite_report,
        )
    if summary_scores is not None:
        write_json_file(output_run_dir / "summary_scores.json", summary_scores)
    save_report(
        output_run_dir=output_run_dir,
        command="summarize",
        demo_input=demo_input,
        retrieved_memories=retrieved_memories,
        summaries=summaries,
        summary_scores=summary_scores,
        summary_prompts=summary_prompts,
        summary_critic_report=summary_critic_report,
    )


def run_summarize(
    demo_input: SummaryDemoInput,
    top_k: int = 8,
    summary_style: str = "structured",
    score_summary: bool = False,
    score_type: str = "general",
    rewrite_from_critic: bool = False,
    output_run_dir: Optional[Path] = None,
) -> None:
    memory = build_summary_memory()
    agent_current_content = prepare_current_material_for_agents(demo_input.current_content)
    try:
        print_section("1. 根据当前材料生成 typed retrieval plan 并检索 typed memories")
        typed_retrieval_plan, retrieved_typed_records, retrieved_memories = search_typed_memories_for_summary(
            memory=memory,
            current_content=agent_current_content,
            user_id=demo_input.user_id,
            project=demo_input.project,
            top_k=top_k,
            source_time=demo_input.source_time,
        )
        print("[Typed Retrieval Plan]")
        print(json.dumps(typed_retrieval_plan, ensure_ascii=False, indent=2))
        retrieved_typed_memories = typed_memory_records_to_dicts(retrieved_typed_records)
        print(f"\n[Retrieved Typed Memories] {len(retrieved_typed_memories)}")
        print(format_retrieved_memories(retrieved_memories))

        artifact_board = build_artifact_board(
            agent_current_content,
            retrieved_typed_records,
            source_time=demo_input.source_time,
        )
        print("\n[Artifact Board]")
        print(json.dumps(artifact_board, ensure_ascii=False, indent=2))

        summaries, summary_prompts = print_augmented_summaries(
            agent_current_content,
            retrieved_memories,
            summary_style,
            section_number="2",
            source_time=demo_input.source_time,
            artifact_board=artifact_board,
        )

        summary_scores = None
        if score_summary:
            summary_scores = print_summary_scores(
                current_content=agent_current_content,
                retrieved_memories=retrieved_memories,
                summaries=summaries,
                score_type=score_type,
                section_number="3",
            )

        critic_section_number = "4" if score_summary else "3"
        summary_critic_report, summary_critic_markdown, summary_critic_prompt = print_summary_critic_report(
            current_content=agent_current_content,
            source_time=demo_input.source_time,
            artifact_board=artifact_board,
            retrieved_typed_memories=retrieved_typed_memories,
            summaries=summaries,
            section_number=critic_section_number,
        )

        rewritten_summaries: dict[str, str] = {}
        summary_rewrite_prompt = None
        summary_rewrite_report = None
        if rewrite_from_critic:
            rewrite_section_number = str(int(critic_section_number) + 1)
            rewritten_summaries, summary_rewrite_prompt, summary_rewrite_report = print_summary_rewrite_draft(
                current_content=agent_current_content,
                source_time=demo_input.source_time,
                artifact_board=artifact_board,
                retrieved_typed_memories=retrieved_typed_memories,
                summaries=summaries,
                critic_report=summary_critic_report,
                section_number=rewrite_section_number,
            )

        save_summarize_outputs(
            output_run_dir,
            demo_input,
            retrieved_memories,
            summaries,
            summary_scores,
            top_k,
            summary_style,
            score_summary,
            score_type,
            typed_retrieval_plan,
            retrieved_typed_memories,
            artifact_board,
            summary_prompts,
            summary_critic_report,
            summary_critic_markdown,
            summary_critic_prompt,
            rewritten_summaries,
            summary_rewrite_prompt,
            summary_rewrite_report,
        )
        print_run_info(
            "summarize",
            demo_input,
            top_k=top_k,
            summary_style=summary_style,
            score_summary=score_summary,
            score_type=score_type,
            output_run_dir=output_run_dir,
        )
    finally:
        close_summary_memory(memory)


def save_run_outputs(
    output_run_dir: Optional[Path],
    demo_input: SummaryDemoInput,
    extracted_memories: list[str],
    add_results: list[dict[str, Any]],
    retrieved_memories: list[str],
    summaries: dict[str, str],
    summary_scores: Optional[dict[str, Any]],
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    typed_memories: Optional[list[dict[str, Any]]] = None,
    typed_extraction_trace: Optional[dict[str, Any]] = None,
    typed_retrieval_plan: Optional[list[dict[str, Any]]] = None,
    retrieved_typed_memories: Optional[list[dict[str, Any]]] = None,
    artifact_board: Optional[dict[str, Any]] = None,
    summary_prompts: Optional[dict[str, str]] = None,
    summary_critic_report: Optional[dict[str, Any]] = None,
    summary_critic_markdown: Optional[str] = None,
    summary_critic_prompt: Optional[str] = None,
    rewritten_summaries: Optional[dict[str, str]] = None,
    summary_rewrite_prompt: Optional[str] = None,
    summary_rewrite_report: Optional[dict[str, Any]] = None,
) -> None:
    if not output_run_dir:
        return

    write_json_file(
        output_run_dir / "run_info.json",
        build_run_info("run", demo_input, top_k, summary_style, score_summary, score_type),
    )
    write_text_file(output_run_dir / "history_input.md", demo_input.history_summary)
    write_text_file(output_run_dir / "current_input.md", demo_input.current_content)
    if typed_memories is not None:
        write_json_file(output_run_dir / "typed_memories.json", typed_memories)
    save_typed_extraction_trace(output_run_dir, typed_extraction_trace)
    write_json_file(output_run_dir / "extracted_memories.json", extracted_memories)
    write_json_file(output_run_dir / "add_results.json", add_results)
    if typed_retrieval_plan is not None:
        write_json_file(output_run_dir / "typed_retrieval_plan.json", typed_retrieval_plan)
    if retrieved_typed_memories is not None:
        write_json_file(output_run_dir / "retrieved_typed_memories.json", retrieved_typed_memories)
    if artifact_board is not None:
        write_json_file(output_run_dir / "artifact_board.json", artifact_board)
    write_json_file(output_run_dir / "retrieved_memories.json", retrieved_memories)
    if summary_prompts is not None:
        write_text_file(output_run_dir / "augmented_summary_prompt.txt", format_augmented_summary_prompts(summary_prompts))
    if summary_critic_prompt is not None:
        write_text_file(output_run_dir / "summary_critic_prompt.txt", summary_critic_prompt)
    if summary_critic_report is not None:
        write_json_file(output_run_dir / "summary_critic_report.json", summary_critic_report)
    if summary_critic_markdown is not None:
        write_text_file(output_run_dir / "summary_critic_report.md", summary_critic_markdown)
    save_summary_files(output_run_dir, summaries)
    if rewritten_summaries or summary_rewrite_prompt or summary_rewrite_report is not None:
        save_rewrite_files(
            output_run_dir,
            rewritten_summaries or {},
            summary_rewrite_prompt,
            summary_rewrite_report,
        )
    if summary_scores is not None:
        write_json_file(output_run_dir / "summary_scores.json", summary_scores)
    save_report(
        output_run_dir=output_run_dir,
        command="run",
        demo_input=demo_input,
        extracted_memories=extracted_memories,
        add_results=add_results,
        retrieved_memories=retrieved_memories,
        summaries=summaries,
        summary_scores=summary_scores,
        summary_prompts=summary_prompts,
        summary_critic_report=summary_critic_report,
    )


def run_demo(
    demo_input: SummaryDemoInput,
    top_k: int = 8,
    summary_style: str = "structured",
    score_summary: bool = False,
    score_type: str = "general",
    rewrite_from_critic: bool = False,
    output_run_dir: Optional[Path] = None,
) -> None:
    memory = build_summary_memory()
    agent_current_content = prepare_current_material_for_agents(demo_input.current_content)
    try:
        print_section("1. 从历史摘要抽取 typed memories")
        typed_memories, extracted_memories, add_results, extraction_trace = add_typed_summary_memories(
            memory=memory,
            summary_text=demo_input.history_summary,
            user_id=demo_input.user_id,
            project=demo_input.project,
            source_type=demo_input.source_type,
            source_time=demo_input.source_time,
            source_file=demo_input.source_file,
        )
        for idx, item in enumerate(extracted_memories, start=1):
            print(f"{idx}. {item}")
        print_typed_memories_summary(typed_memories)

        print_section("2. 写入 mem0 / Qdrant 的结果")
        for result in add_results:
            print(result)
        print_add_results_summary(extracted_memories, add_results)

        print_section("3. 根据当前材料生成 typed retrieval plan 并检索 typed memories")
        typed_retrieval_plan, retrieved_typed_records, retrieved_memories = search_typed_memories_for_summary(
            memory=memory,
            current_content=agent_current_content,
            user_id=demo_input.user_id,
            project=demo_input.project,
            top_k=top_k,
            source_time=demo_input.source_time,
        )
        print("[Typed Retrieval Plan]")
        print(json.dumps(typed_retrieval_plan, ensure_ascii=False, indent=2))
        retrieved_typed_memories = typed_memory_records_to_dicts(retrieved_typed_records)
        print(f"\n[Retrieved Typed Memories] {len(retrieved_typed_memories)}")
        print(format_retrieved_memories(retrieved_memories))

        artifact_board = build_artifact_board(
            agent_current_content,
            retrieved_typed_records,
            source_time=demo_input.source_time,
        )
        print("\n[Artifact Board]")
        print(json.dumps(artifact_board, ensure_ascii=False, indent=2))

        summaries, summary_prompts = print_augmented_summaries(
            agent_current_content,
            retrieved_memories,
            summary_style,
            section_number="4",
            source_time=demo_input.source_time,
            artifact_board=artifact_board,
        )

        summary_scores = None
        if score_summary:
            summary_scores = print_summary_scores(
                current_content=agent_current_content,
                retrieved_memories=retrieved_memories,
                summaries=summaries,
                score_type=score_type,
                section_number="5",
            )

        critic_section_number = "6" if score_summary else "5"
        summary_critic_report, summary_critic_markdown, summary_critic_prompt = print_summary_critic_report(
            current_content=agent_current_content,
            source_time=demo_input.source_time,
            artifact_board=artifact_board,
            retrieved_typed_memories=retrieved_typed_memories,
            summaries=summaries,
            section_number=critic_section_number,
        )

        rewritten_summaries: dict[str, str] = {}
        summary_rewrite_prompt = None
        summary_rewrite_report = None
        if rewrite_from_critic:
            rewrite_section_number = str(int(critic_section_number) + 1)
            rewritten_summaries, summary_rewrite_prompt, summary_rewrite_report = print_summary_rewrite_draft(
                current_content=agent_current_content,
                source_time=demo_input.source_time,
                artifact_board=artifact_board,
                retrieved_typed_memories=retrieved_typed_memories,
                summaries=summaries,
                critic_report=summary_critic_report,
                section_number=rewrite_section_number,
            )

        save_run_outputs(
            output_run_dir,
            demo_input,
            extracted_memories,
            add_results,
            retrieved_memories,
            summaries,
            summary_scores,
            top_k,
            summary_style,
            score_summary,
            score_type,
            typed_memories,
            extraction_trace,
            typed_retrieval_plan,
            retrieved_typed_memories,
            artifact_board,
            summary_prompts,
            summary_critic_report,
            summary_critic_markdown,
            summary_critic_prompt,
            rewritten_summaries,
            summary_rewrite_prompt,
            summary_rewrite_report,
        )
        print_run_info(
            "run",
            demo_input,
            top_k=top_k,
            summary_style=summary_style,
            score_summary=score_summary,
            score_type=score_type,
            output_run_dir=output_run_dir,
        )
    finally:
        close_summary_memory(memory)


def main() -> None:
    args = parse_args()
    if args.list_cases:
        list_case_files()
        return

    if args.command == "ingest":
        if args.score_summary:
            raise ValueError("--score-summary 只支持 run / summarize，因为 ingest 不生成摘要")
        if args.rewrite_from_critic:
            raise ValueError("--rewrite-from-critic 只支持 run / summarize，因为 ingest 不生成摘要")
        demo_inputs = load_ingest_inputs(args)
        output_run_dir = create_output_run_dir(args.output_dir, "ingest", demo_inputs[0].project)
        run_ingest(demo_inputs, output_run_dir=output_run_dir)
    elif args.command == "summarize":
        demo_input = load_summarize_input(args)
        output_run_dir = create_output_run_dir(args.output_dir, "summarize", demo_input.project)
        run_summarize(
            demo_input,
            top_k=args.top_k,
            summary_style=args.summary_style,
            score_summary=args.score_summary,
            score_type=args.score_type,
            rewrite_from_critic=args.rewrite_from_critic,
            output_run_dir=output_run_dir,
        )
    else:
        demo_input = load_demo_input(args)
        output_run_dir = create_output_run_dir(args.output_dir, "run", demo_input.project)
        run_demo(
            demo_input,
            top_k=args.top_k,
            summary_style=args.summary_style,
            score_summary=args.score_summary,
            score_type=args.score_type,
            rewrite_from_critic=args.rewrite_from_critic,
            output_run_dir=output_run_dir,
        )


if __name__ == "__main__":
    main()
