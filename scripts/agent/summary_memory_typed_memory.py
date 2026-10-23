"""负责 Typed Memory 抽取、六类类型约束、跨 chunk 去重和检索计划。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

from summary_memory_agent_schemas import (
    ALLOWED_MEMORY_TYPES,
    TYPED_MEMORY_SCHEMA_VERSION,
    TypedMemoryRecord,
    TypedRetrievalPlanItem,
    normalize_typed_memory_record,
    normalize_typed_memory_records,
    normalize_typed_retrieval_plan,
)


LLMCaller = Callable[..., str]
CROSS_CHUNK_DEDUP_SCHEMA_VERSION = "summary_typed_memory_cross_chunk_dedup.v3"
CROSS_CHUNK_DEDUP_CONTENT_CANDIDATE_SIMILARITY = 0.60
CROSS_CHUNK_DEDUP_EVIDENCE_CANDIDATE_SIMILARITY = 0.80
CROSS_CHUNK_DEDUP_MAX_PAIRS_PER_RECORD = 3
_DEADLINE_DATETIME_RE = re.compile(
    r"^\s*(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
    r"(?:[Tt\s]+)(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    r"(?:\.\d+)?\s*$"
)


def extract_json_value(raw_output: str) -> Any:
    text = (raw_output or "").strip()
    if not text:
        raise ValueError("LLM output is empty")

    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM output does not contain valid JSON")


def coerce_typed_memory_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("typed_memories", "memories", "memory_records", "records", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        if value.get("memory_type") and value.get("content"):
            return [value]
        keys = ", ".join(sorted(str(key) for key in value.keys()))
        raise ValueError(f"typed memory extraction JSON object does not contain a memory array; keys={keys}")
    raise ValueError("typed memory extraction must return a JSON array")


def build_typed_memory_extraction_prompt(
    summary_text: str, source_time: str = "", source_file: str = "",
    *, json_object_output: bool = False,
) -> str:
    allowed_types = "、".join(ALLOWED_MEMORY_TYPES)
    source_time_text = source_time.strip() or "未提供"
    source_file_text = source_file.strip() or "未提供"
    output_rule = (
        '顶层输出必须是严格 JSON 对象，格式为 {"typed_memories": [...]}。'
        if json_object_output else
        '顶层输出必须是严格 JSON 数组，以 [ 开始，以 ] 结束；不要输出 {"typed_memories": [...]} 这种外层对象。'
    )
    example_open = '{"typed_memories": [' if json_object_output else '['
    example_close = ']}' if json_object_output else ']'
    example_note = (
        '注意最外层必须是含 typed_memories 数组的对象'
        if json_object_output else '注意最外层必须是数组'
    )
    return f"""你是 Summary Multi-Agent 的 typed memory extraction 模块。

你的任务是从历史材料中抽取后续 Agent 可检索、可推理的长期历史知识。

硬性规则：
1. memory_type 只能从这 6 类中选择：{allowed_types}。
2. 不允许创造其他 memory_type，不允许输出 fact_metric、open_question、meeting、weekly_report、learning_note 等类型。
3. 只抽取对未来摘要增强、时间判断、决策约束判断、任务负责人判断、风险判断或摘要偏好有长期价值的信息。
4. 不保存无后续价值的一次性描述。
5. 不编造原文没有的信息；证据不足时 status 或 confidence 要降低。
6. 每条记忆必须保留 evidence_span，尽量引用原文中的短证据片段。
7. 同一个底层事实在本批输出中只保留一次；不要因为原文重复、换一种措辞或同时出现在上下文与结尾提醒中而重复输出。
8. 负责人、日期或状态缺失时保持缺失，不得为了让字段完整而补造。
9. {output_rule}
10. 不要 Markdown，不要解释文字。

6 类 typed memory 的边界：
- time：历史截止时间、计划时间、实际完成时间、任务时间窗口、相对时间归一、延期、完成状态未知。
- decision：已确认、暂缓、否定、重新讨论、被推翻或待确认的决策。
- constraint：当前阶段不做什么、范围边界、安全/部署/默认行为约束、禁止误写的范围。
- task：负责人、任务、协作人、截止时间、完成状态、依赖条件、是否仍在继续。
- risk：风险、阻塞、外部依赖、技术依赖、环境限制、待确认条件、风险是否解除。
- preference：从输入资料中抽取的场景、角色、目标读者、摘要用途或输出视角偏好；它描述在该资料场景下后续摘要应优先关注什么、弱化什么、如何组织信息。

preference 的额外规则：
1. preference 仍然必须来自输入资料，不来自用户反馈、运行配置或外部用户画像。
2. 如果资料明确提到目标读者、使用角色、摘要用途、输出视角或表达取舍，可以抽取 preference。
3. 如果资料没有明说，但场景和角色需求非常强，可以抽取场景推理型 preference；此时 structured_fields.inference_level 必须为 scenario_inferred，status 应为 unknown，confidence 应为 medium 或 low。
4. 如果资料只是普通事实、任务、决策、约束或风险，不要强行输出 preference。
5. 不要因为一个场景可能存在多个角色，就自由枚举所有可能角色的 preference；只抽取资料中明确或强烈隐含的视角。
6. scenario、role_or_audience、purpose 应使用资料中的真实线索或通用短语，不要写死为面试、课堂、项目复盘等少数示例，也不要创造外部 persona。

顶层输出示例，{example_note}：
{example_open}
  {{
    "schema_version": "{TYPED_MEMORY_SCHEMA_VERSION}",
    "memory_id": "",
    "memory_type": "time|decision|constraint|task|risk|preference",
    "content": "给后续 Agent 阅读的中文自然语言记忆",
    "structured_fields": {{}},
    "source": {{
      "source_time": "{source_time_text}",
      "source_file": "{source_file_text}",
      "evidence_span": "来自原文的短证据片段"
    }},
    "entities": [],
    "topics": [],
    "status": "active|inactive|unknown|superseded",
    "confidence": "high|medium|low",
    "embedding_text": "用于向量检索的中文文本，包含类型、主题、实体、关键状态和内容"
  }}
{example_close}

structured_fields 应根据 memory_type 填写关键字段：
- time: owner, task, raw_time, normalized_time, deadline, time_window, completion_status
- decision: topic, decision_status, options, scope
- constraint: topic, scope, validity
- task: owner, task, deadline, status, dependencies
- risk: topic, risk_type, status, dependency
- preference: preference_type, scenario, role_or_audience, purpose, focus_items, deprioritized_items, output_implications, evidence_basis, inference_level
preference 字段说明：scenario、role_or_audience、purpose 是开放短语；对场景推理型 preference，必须填写 inference_level=scenario_inferred。

材料 source_time：{source_time_text}
材料 source_file：{source_file_text}

历史材料：
{summary_text}
"""


def parse_typed_memory_json(raw_output: str, source_time: str = "", source_file: str = "") -> list[TypedMemoryRecord]:
    value = extract_json_value(raw_output)
    records_value = coerce_typed_memory_array(value)
    return normalize_typed_memory_records(records_value, source_time=source_time, source_file=source_file)


def extract_typed_memories_from_summary(
    summary_text: str,
    llm_call: LLMCaller,
    source_time: str = "",
    source_file: Optional[Path] = None,
    max_tokens: int = 8000,
    trace: Optional[dict[str, Any]] = None,
    json_object_output: bool = False,
) -> list[TypedMemoryRecord]:
    source_file_text = str(source_file) if source_file else ""
    prompt = build_typed_memory_extraction_prompt(
        summary_text,
        source_time=source_time,
        source_file=source_file_text,
        json_object_output=json_object_output,
    )
    raw_output = llm_call(prompt, temperature=0.0, max_tokens=max_tokens)
    raw_records = coerce_typed_memory_array(extract_json_value(raw_output))
    typed_records = normalize_typed_memory_records(
        raw_records,
        source_time=source_time,
        source_file=source_file_text,
    )
    if trace is not None:
        trace.update(
            {
                "prompt": prompt,
                "raw_output": raw_output,
                "max_tokens": max_tokens,
                "raw_record_count": len(raw_records),
                "accepted_record_count": len(typed_records),
                "rejected_record_count": len(raw_records) - len(typed_records),
            }
        )
    return typed_records


def typed_record_to_memory_text(record: TypedMemoryRecord) -> str:
    return record.content


def typed_record_to_mem0_message(record: TypedMemoryRecord) -> list[dict[str, str]]:
    text = record.embedding_text.strip() or record.content
    return [{"role": "user", "content": text}]


def typed_record_to_mem0_metadata(record: TypedMemoryRecord) -> dict[str, str]:
    source = record.source or {}
    return {
        "memory_schema_version": record.schema_version,
        "typed_memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "memory_status": record.status,
        "memory_confidence": record.confidence,
        "source_time": source.get("source_time", ""),
        "source_file": source.get("source_file", ""),
        "evidence_span": source.get("evidence_span", ""),
        "entities_text": " | ".join(record.entities),
        "topics_text": " | ".join(record.topics),
        "structured_fields_json": json.dumps(record.structured_fields, ensure_ascii=False, sort_keys=True),
        "content_text": record.content,
    }


def typed_memory_records_to_dicts(records: list[TypedMemoryRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def _normalize_dedup_text(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _structured_text(record: TypedMemoryRecord, field_name: str) -> str:
    return _normalize_dedup_text(record.structured_fields.get(field_name, ""))


def _record_text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _record_deadline_value(record: TypedMemoryRecord) -> str:
    return str(
        record.structured_fields.get("deadline") or record.structured_fields.get("normalized_time") or ""
    ).strip()


def canonicalize_deadline_for_dedup(value: Any) -> str:
    """Return a conservative comparison key without changing stored model output.

    A zero-second ISO-like timestamp is equivalent to the same minute without a
    seconds component. Date-only and non-absolute time expressions intentionally
    return an empty key so they cannot broaden semantic-dedup candidates.
    """

    match = _DEADLINE_DATETIME_RE.match(str(value or ""))
    if not match:
        return ""

    values = match.groupdict()
    try:
        timestamp = datetime(
            int(values["year"]),
            int(values["month"]),
            int(values["day"]),
            int(values["hour"]),
            int(values["minute"]),
            int(values["second"] or 0),
        )
    except ValueError:
        return ""

    if values["second"] and timestamp.second != 0:
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    return timestamp.strftime("%Y-%m-%dT%H:%M")


def _record_deadline_key(record: TypedMemoryRecord) -> str:
    return canonicalize_deadline_for_dedup(_record_deadline_value(record))


def _record_work_status(record: TypedMemoryRecord) -> str:
    for field_name in ("completion_status", "status"):
        value = _normalize_dedup_text(record.structured_fields.get(field_name, ""))
        if value:
            return value
    return ""


def _cross_chunk_duplicate_reason(
    kept_record: TypedMemoryRecord,
    candidate_record: TypedMemoryRecord,
) -> tuple[Optional[str], float]:
    if kept_record.memory_type != candidate_record.memory_type:
        return None, 0.0

    kept_content = _normalize_dedup_text(kept_record.content)
    candidate_content = _normalize_dedup_text(candidate_record.content)
    if (
        kept_content
        and kept_content == candidate_content
        and _record_direct_business_signature(kept_record)
        == _record_direct_business_signature(candidate_record)
    ):
        return "same_type_exact_content", 1.0

    content_similarity = _record_text_similarity(kept_content, candidate_content)
    return None, content_similarity


def _normalize_business_value(value: Any) -> Any:
    """为确定性一致性检查规范字符串空白，同时保留日期精度和结构。"""

    if isinstance(value, dict):
        return {
            str(key): _normalize_business_value(item)
            for key, item in sorted(value.items())
            if item not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_business_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _record_direct_business_signature(record: TypedMemoryRecord) -> dict[str, Any]:
    """返回直接去重所需完整业务签名；缺字段与有字段不会被视为相同。"""

    return {
        "structured_fields": _normalize_business_value(record.structured_fields),
        "entities": sorted(_normalize_dedup_text(item) for item in record.entities),
        "topics": sorted(_normalize_dedup_text(item) for item in record.topics),
        "status": record.status,
    }


def _record_completeness_key(record: TypedMemoryRecord) -> tuple[int, int, int, int, int]:
    source = record.source or {}
    structured_value_count = sum(1 for value in record.structured_fields.values() if value)
    confidence_rank = {"high": 3, "medium": 2, "low": 1}.get(record.confidence, 0)
    return (
        len(_normalize_dedup_text(record.content)),
        len(_normalize_dedup_text(source.get("evidence_span", ""))),
        structured_value_count,
        len(record.entities) + len(record.topics),
        confidence_rank,
    )


def deduplicate_cross_chunk_typed_memory_records(
    chunk_records: list[list[TypedMemoryRecord]],
) -> tuple[list[list[TypedMemoryRecord]], list[dict[str, Any]]]:
    # 这是同源分块的写前局部去重，不查询历史库，也不承担跨运行生命周期管理。
    """Keep the more complete record for conservative same-source cross-chunk duplicates."""

    deduplicated_records: list[list[TypedMemoryRecord]] = [[] for _ in chunk_records]
    kept_records: list[tuple[int, TypedMemoryRecord]] = []
    removed_records: list[dict[str, Any]] = []

    for candidate_chunk_index, records in enumerate(chunk_records):
        for candidate_record in records:
            matched_index: Optional[int] = None
            match_reason: Optional[str] = None
            match_similarity = 0.0
            for index, (kept_chunk_index, kept_record) in enumerate(kept_records):
                if kept_chunk_index == candidate_chunk_index:
                    continue
                reason, similarity = _cross_chunk_duplicate_reason(kept_record, candidate_record)
                if reason:
                    matched_index = index
                    match_reason = reason
                    match_similarity = similarity
                    break

            if matched_index is None:
                kept_records.append((candidate_chunk_index, candidate_record))
                deduplicated_records[candidate_chunk_index].append(candidate_record)
                continue

            kept_chunk_index, kept_record = kept_records[matched_index]
            if _record_completeness_key(candidate_record) > _record_completeness_key(kept_record):
                deduplicated_records[kept_chunk_index].remove(kept_record)
                deduplicated_records[candidate_chunk_index].append(candidate_record)
                kept_records[matched_index] = (candidate_chunk_index, candidate_record)
                removed_record = kept_record
                retained_record = candidate_record
                removed_chunk_index = kept_chunk_index
                retained_chunk_index = candidate_chunk_index
            else:
                removed_record = candidate_record
                retained_record = kept_record
                removed_chunk_index = candidate_chunk_index
                retained_chunk_index = kept_chunk_index

            removed_records.append(
                {
                    "reason": match_reason,
                    "similarity": round(match_similarity, 3),
                    "removed_chunk_index": removed_chunk_index + 1,
                    "retained_chunk_index": retained_chunk_index + 1,
                    "removed_memory_id": removed_record.memory_id,
                    "retained_memory_id": retained_record.memory_id,
                    "removed_record": removed_record.to_dict(),
                    "retained_record": retained_record.to_dict(),
                }
            )

    return deduplicated_records, removed_records


def _preview_dedup_text(value: Any, limit: int = 320) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _dedup_candidate_record_payload(record: TypedMemoryRecord, chunk_index: int, preview: bool) -> dict[str, Any]:
    source = record.source or {}
    deadline_value = _record_deadline_value(record)
    task_value = str(record.structured_fields.get("task", "") or "")
    transform = _preview_dedup_text if preview else lambda value: str(value or "").strip()
    payload = {
        "memory_id": record.memory_id,
        "chunk_index": chunk_index + 1,
        "memory_type": record.memory_type,
        "content": transform(record.content),
        "owner": str(record.structured_fields.get("owner", "") or ""),
        "task": transform(task_value),
        "deadline": deadline_value,
        "canonical_deadline": _record_deadline_key(record),
        "work_status": _record_work_status(record),
        "evidence_span": transform(source.get("evidence_span", "")),
        "topics": record.topics,
    }
    if not preview:
        payload["structured_fields"] = record.structured_fields
    return payload


def _build_dedup_candidate(
    pair_number: int,
    left_chunk_index: int,
    left_record: TypedMemoryRecord,
    right_chunk_index: int,
    right_record: TypedMemoryRecord,
) -> dict[str, Any]:
    left_source = left_record.source or {}
    right_source = right_record.source or {}
    left_task = _structured_text(left_record, "task")
    right_task = _structured_text(right_record, "task")
    left_content = _normalize_dedup_text(left_record.content)
    right_content = _normalize_dedup_text(right_record.content)
    left_evidence = _normalize_dedup_text(left_source.get("evidence_span", ""))
    right_evidence = _normalize_dedup_text(right_source.get("evidence_span", ""))
    return {
        "pair_id": f"candidate_{pair_number}",
        "left_locator": (left_chunk_index, left_record.memory_id),
        "right_locator": (right_chunk_index, right_record.memory_id),
        "audit": {
            "pair_id": f"candidate_{pair_number}",
            "memory_type": left_record.memory_type,
            "business_anchors": {
                "left_owner": str(left_record.structured_fields.get("owner", "") or ""),
                "right_owner": str(right_record.structured_fields.get("owner", "") or ""),
                "left_canonical_deadline": _record_deadline_key(left_record),
                "right_canonical_deadline": _record_deadline_key(right_record),
                "left_work_status": _record_work_status(left_record),
                "right_work_status": _record_work_status(right_record),
            },
            "signals": {
                "task_similarity": round(_record_text_similarity(left_task, right_task), 3),
                "content_similarity": round(_record_text_similarity(left_content, right_content), 3),
                "evidence_similarity": round(_record_text_similarity(left_evidence, right_evidence), 3),
            },
            "left": _dedup_candidate_record_payload(left_record, left_chunk_index, preview=True),
            "right": _dedup_candidate_record_payload(right_record, right_chunk_index, preview=True),
        },
        "judge_input": {
            "pair_id": f"candidate_{pair_number}",
            "memory_type": left_record.memory_type,
            "business_anchors": {
                "left_owner": str(left_record.structured_fields.get("owner", "") or ""),
                "right_owner": str(right_record.structured_fields.get("owner", "") or ""),
                "left_canonical_deadline": _record_deadline_key(left_record),
                "right_canonical_deadline": _record_deadline_key(right_record),
            },
            "left": _dedup_candidate_record_payload(left_record, left_chunk_index, preview=False),
            "right": _dedup_candidate_record_payload(right_record, right_chunk_index, preview=False),
        },
    }


def build_cross_chunk_dedup_candidates(
    chunk_records: list[list[TypedMemoryRecord]],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return bounded semantic-dedup candidates after deterministic deduplication."""

    ranked_pairs: list[
        tuple[
            float,
            int,
            TypedMemoryRecord,
            int,
            TypedMemoryRecord,
        ]
    ] = []
    for left_chunk_index, left_records in enumerate(chunk_records):
        for right_chunk_index in range(left_chunk_index + 1, len(chunk_records)):
            for left_record in left_records:
                for right_record in chunk_records[right_chunk_index]:
                    if left_record.memory_type != right_record.memory_type:
                        continue
                    left_source = left_record.source or {}
                    right_source = right_record.source or {}
                    content_similarity = _record_text_similarity(
                        _normalize_dedup_text(left_record.content),
                        _normalize_dedup_text(right_record.content),
                    )
                    left_task = _structured_text(left_record, "task")
                    right_task = _structured_text(right_record, "task")
                    task_similarity = _record_text_similarity(
                        left_task, right_task
                    )
                    evidence_similarity = _record_text_similarity(
                        _normalize_dedup_text(
                            left_source.get("evidence_span", "")
                        ),
                        _normalize_dedup_text(
                            right_source.get("evidence_span", "")
                        ),
                    )
                    same_task = bool(left_task and left_task == right_task)
                    if not (
                        content_similarity
                        >= CROSS_CHUNK_DEDUP_CONTENT_CANDIDATE_SIMILARITY
                        or task_similarity
                        >= CROSS_CHUNK_DEDUP_CONTENT_CANDIDATE_SIMILARITY
                        or evidence_similarity
                        >= CROSS_CHUNK_DEDUP_EVIDENCE_CANDIDATE_SIMILARITY
                        or same_task
                    ):
                        continue
                    ranked_pairs.append(
                        (
                            max(
                                content_similarity,
                                task_similarity,
                                evidence_similarity,
                                1.0 if same_task else 0.0,
                            ),
                            left_chunk_index,
                            left_record,
                            right_chunk_index,
                            right_record,
                        )
                    )
    total_candidate_count = len(ranked_pairs)
    ranked_pairs.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2].memory_id,
            item[3],
            item[4].memory_id,
        )
    )
    pair_counts: Counter[tuple[int, str]] = Counter()
    selected_pairs: list[
        tuple[float, int, TypedMemoryRecord, int, TypedMemoryRecord]
    ] = []
    for pair in ranked_pairs:
        _score, left_chunk, left_record, right_chunk, right_record = pair
        left_locator = (left_chunk, left_record.memory_id)
        right_locator = (right_chunk, right_record.memory_id)
        if (
            pair_counts[left_locator] >= CROSS_CHUNK_DEDUP_MAX_PAIRS_PER_RECORD
            or pair_counts[right_locator]
            >= CROSS_CHUNK_DEDUP_MAX_PAIRS_PER_RECORD
        ):
            continue
        pair_counts[left_locator] += 1
        pair_counts[right_locator] += 1
        selected_pairs.append(pair)
    candidates = [
        _build_dedup_candidate(
            pair_number,
            left_chunk,
            left_record,
            right_chunk,
            right_record,
        )
        for pair_number, (
            _score,
            left_chunk,
            left_record,
            right_chunk,
            right_record,
        ) in enumerate(selected_pairs[:max_candidates], start=1)
    ]
    return candidates, total_candidate_count


def build_cross_chunk_dedup_judge_prompt(candidates: list[dict[str, Any]]) -> str:
    candidate_inputs = [candidate["judge_input"] for candidate in candidates]
    candidate_json = json.dumps(candidate_inputs, ensure_ascii=False, indent=2)
    return f"""你是 Summary Multi-Agent 的 typed memory 去重裁决器。

每一对记录都来自同一历史原始材料、同一材料时间锚点、不同 chunk 且 memory_type 相同。程序只依据文本、任务或证据相似度召回候选；负责人、日期、状态可能缺失或不同，不能把候选资格当成重复结论。

你的任务是判断每一对是否陈述同一个底层历史事实。详细说明与后续简称可以是同一事实；仅仅同一负责人、同一截止时间、同一主题或同一发送目标，不足以判为同一事实。

规则：
1. 只能依据每对记录的 content、task、evidence_span、topics 和结构化字段判断，不得补充材料外事实。
2. 只有在确信两条记录是同一底层事实的详细版与简版时，same_underlying_fact 才为 true；该规则适用于全部六类记忆。
3. 任何不确定、数字或否定含义不同、两项不同待办、状态变化、日期精度不同、互补信息、冲突或证据不足的情况都返回 false。
4. reason 用一句中文说明判断依据，不要复述全部材料。
5. 输出必须是严格 JSON 对象，不要 Markdown 或额外解释。

输出格式：
{{
  "decisions": [
    {{
      "pair_id": "candidate_1",
      "same_underlying_fact": true,
      "confidence": "high|medium|low",
      "reason": "简短依据"
    }}
  ]
}}

候选对：
{candidate_json}
"""


def parse_cross_chunk_dedup_judge_output(
    raw_output: str,
    candidate_ids: set[str],
) -> dict[str, dict[str, Any]]:
    value = extract_json_value(raw_output)
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        raise ValueError("cross-chunk dedup judge output must contain a decisions array")

    decisions: dict[str, dict[str, Any]] = {}
    for item in value["decisions"]:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id", "")).strip()
        same_underlying_fact = item.get("same_underlying_fact")
        if pair_id not in candidate_ids or not isinstance(same_underlying_fact, bool):
            continue
        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        decisions[pair_id] = {
            "same_underlying_fact": same_underlying_fact,
            "confidence": confidence,
            "reason": _preview_dedup_text(item.get("reason", ""), limit=500),
        }
    return decisions


def apply_cross_chunk_dedup_judgements(
    chunk_records: list[list[TypedMemoryRecord]],
    candidates: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[list[list[TypedMemoryRecord]], list[dict[str, Any]], dict[str, str]]:
    """逐对应用高置信裁决；不通过传递关系合并未直接裁决的记录。"""

    deduplicated_records = [list(records) for records in chunk_records]
    record_lookup: dict[tuple[int, str], TypedMemoryRecord] = {
        (chunk_index, record.memory_id): record
        for chunk_index, records in enumerate(deduplicated_records)
        for record in records
    }
    outcomes: dict[str, str] = {}
    removed_records: list[dict[str, Any]] = []
    active = set(record_lookup)
    for candidate in candidates:
        pair_id = candidate["pair_id"]
        decision = decisions.get(pair_id)
        if decision is None:
            outcomes[pair_id] = "not_merged_missing_judge_decision"
            continue
        if decision["same_underlying_fact"] is not True:
            outcomes[pair_id] = "not_merged_judge_false"
            continue
        if decision["confidence"] != "high":
            outcomes[pair_id] = "not_merged_judge_low_confidence"
            continue
        left_locator = candidate["left_locator"]
        right_locator = candidate["right_locator"]
        if left_locator not in active or right_locator not in active:
            outcomes[pair_id] = "not_merged_transitive_path"
            continue
        left_record = record_lookup[left_locator]
        right_record = record_lookup[right_locator]
        retained_locator, removed_locator = sorted(
            (left_locator, right_locator),
            key=lambda locator: _record_completeness_key(record_lookup[locator]),
            reverse=True,
        )
        retained_record = record_lookup[retained_locator]
        removed_record = record_lookup[removed_locator]
        if not _judge_merge_is_programmatically_safe(
            retained_record, removed_record
        ):
            outcomes[pair_id] = "not_merged_business_field_conflict"
            continue
        deduplicated_records[removed_locator[0]].remove(removed_record)
        active.remove(removed_locator)
        outcomes[pair_id] = "merged_by_judge"
        removed_records.append(
            {
                "reason": "same_underlying_fact_high_confidence_no_conflict",
                "similarity": candidate["audit"].get("signals"),
                "judge_pair_ids": [pair_id],
                "removed_chunk_index": removed_locator[0] + 1,
                "retained_chunk_index": retained_locator[0] + 1,
                "removed_memory_id": removed_record.memory_id,
                "retained_memory_id": retained_record.memory_id,
                "removed_record": removed_record.to_dict(),
                "retained_record": retained_record.to_dict(),
            }
        )
    return deduplicated_records, removed_records, outcomes


def _number_tokens(value: Any) -> tuple[str, ...]:
    """提取数字信号，防止 10ms/2ms 等相似文本被误合并。"""

    return tuple(re.findall(r"\d+(?:\.\d+)?", str(value or "")))


def _has_negation(value: Any) -> bool:
    """识别中英文常见否定标记，作为保守冲突边界。"""

    text = str(value or "").casefold()
    return any(mark in text for mark in ("不", "未", "无", "取消", "否定", "not", "no "))


def _comparable_structured_value(field_name: str, value: Any) -> Any:
    """对零秒日期时间做分钟级等价，其余日期精度保持不同。"""

    if field_name in {"deadline", "normalized_time"}:
        canonical = canonicalize_deadline_for_dedup(value)
        return canonical or _normalize_business_value(value)
    return _normalize_business_value(value)


def _judge_merge_is_programmatically_safe(
    retained: TypedMemoryRecord,
    removed: TypedMemoryRecord,
) -> bool:
    """确认被删记录没有保留记录缺失的独有结构信息或明确冲突。"""

    if retained.memory_type != removed.memory_type:
        return False
    if retained.status != removed.status:
        return False
    if _number_tokens(retained.content) != _number_tokens(removed.content):
        return False
    if _has_negation(retained.content) != _has_negation(removed.content):
        return False
    retained_fields = retained.structured_fields
    for field_name, removed_value in removed.structured_fields.items():
        if removed_value in (None, "", [], {}):
            continue
        retained_value = retained_fields.get(field_name)
        if retained_value in (None, "", [], {}):
            return False
        left = _comparable_structured_value(field_name, retained_value)
        right = _comparable_structured_value(field_name, removed_value)
        if left == right:
            continue
        if field_name in {"task", "topic", "scope", "dependency", "purpose"}:
            left_text = _normalize_dedup_text(retained_value)
            right_text = _normalize_dedup_text(removed_value)
            if (
                right_text
                and right_text in left_text
                and _number_tokens(left_text) == _number_tokens(right_text)
                and _has_negation(left_text) == _has_negation(right_text)
            ):
                continue
        return False
    retained_entities = {_normalize_dedup_text(item) for item in retained.entities}
    removed_entities = {_normalize_dedup_text(item) for item in removed.entities}
    retained_topics = {_normalize_dedup_text(item) for item in retained.topics}
    removed_topics = {_normalize_dedup_text(item) for item in removed.topics}
    return removed_entities.issubset(retained_entities) and removed_topics.issubset(
        retained_topics
    )


def build_typed_memory_retrieval_plan_prompt(
    current_content: str,
    source_time: str = "",
    max_chars: int = 12000,
) -> str:
    allowed_types = "、".join(ALLOWED_MEMORY_TYPES)
    current_excerpt = current_content.strip()[:max_chars]
    source_time_text = source_time.strip() or "未提供"
    return f"""你是 Summary Multi-Agent 的 retrieval planner。

你需要根据当前材料判断：为了生成长期记忆增强摘要，应该检索哪几类 typed memory。

硬性规则：
1. memory_types 只能从这 6 类中选择：{allowed_types}。
2. 不允许创造其他类型，不允许输出 fact_metric、open_question、meeting、weekly_report、learning_note 等类型。
3. 用户不会手动指定类型，你必须根据当前材料自己判断。
4. 如果当前材料涉及相对时间、截止、延期、完成状态，检索 time/task。
5. 如果当前材料涉及重新讨论、是否支持、暂不、不做、范围变化，检索 decision/constraint。
6. 如果当前材料涉及负责人、待办、推进、交付，检索 task/time。
7. 如果当前材料涉及风险、阻塞、依赖、不确定，检索 risk/constraint/task。
8. 如果当前材料体现了特定场景、角色、目标读者、摘要用途、输出视角、复盘/汇报/学习/评估/决策等使用意图，或需要判断不同视角下的信息取舍，检索 preference；query 必须包含当前材料中实际出现或强烈隐含的场景/角色/用途线索，不要泛泛检索“摘要偏好”。
9. 每个 retrieval_plan item 必须有非空 memory_types、query、purpose。
10. 输出必须是严格 JSON 对象，不要 Markdown，不要解释文字。

输出格式：
{{
  "retrieval_plan": [
    {{
      "memory_types": ["time", "task"],
      "query": "用于检索 typed memory 的中文查询",
      "purpose": "为什么需要检索这些类型"
    }}
  ]
}}

当前材料 source_time：{source_time_text}

当前材料：
{current_excerpt}
"""


def parse_typed_retrieval_plan_json(raw_output: str) -> list[TypedRetrievalPlanItem]:
    value = extract_json_value(raw_output)
    return normalize_typed_retrieval_plan(value)


def build_typed_memory_retrieval_plan(
    current_content: str,
    llm_call: LLMCaller,
    source_time: str = "",
    max_tokens: int = 1200,
    attempts: int = 2,
) -> list[TypedRetrievalPlanItem]:
    last_error: Optional[Exception] = None
    prompt = build_typed_memory_retrieval_plan_prompt(current_content, source_time=source_time)
    for attempt in range(1, attempts + 1):
        retry_hint = ""
        if attempt > 1:
            retry_hint = (
                "\n\n上一次输出不合法。请严格只使用固定类型 "
                f"{', '.join(ALLOWED_MEMORY_TYPES)}，并输出严格 JSON。"
            )
        raw_output = llm_call(prompt + retry_hint, temperature=0.0, max_tokens=max_tokens)
        try:
            return parse_typed_retrieval_plan_json(raw_output)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"typed retrieval plan is invalid after {attempts} attempts: {last_error}")


def typed_record_from_search_result(item: dict[str, Any]) -> Optional[TypedMemoryRecord]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if not metadata:
        return None

    structured_fields: dict[str, Any] = {}
    raw_fields = metadata.get("structured_fields_json", "")
    if raw_fields:
        try:
            parsed = json.loads(str(raw_fields))
            if isinstance(parsed, dict):
                structured_fields = parsed
        except json.JSONDecodeError:
            structured_fields = {}

    raw_record = {
        "schema_version": metadata.get("memory_schema_version") or TYPED_MEMORY_SCHEMA_VERSION,
        "memory_id": metadata.get("typed_memory_id") or item.get("id"),
        "memory_type": metadata.get("memory_type"),
        "content": metadata.get("content_text") or item.get("memory") or item.get("memory_text") or "",
        "structured_fields": structured_fields,
        "source": {
            "source_time": metadata.get("source_time", ""),
            "source_file": metadata.get("source_file", ""),
            "evidence_span": metadata.get("evidence_span", ""),
        },
        "entities": str(metadata.get("entities_text", "")).split("|") if metadata.get("entities_text") else [],
        "topics": str(metadata.get("topics_text", "")).split("|") if metadata.get("topics_text") else [],
        "status": metadata.get("memory_status") or "unknown",
        "confidence": metadata.get("memory_confidence") or "medium",
        "embedding_text": item.get("memory") or item.get("memory_text") or "",
    }
    return normalize_typed_memory_record(raw_record)


def format_typed_memory_for_summary(record: TypedMemoryRecord) -> str:
    source = record.source or {}
    fields_text = json.dumps(record.structured_fields, ensure_ascii=False, sort_keys=True)
    return (
        f"[{record.memory_type} | {record.status} | {record.confidence}]\n"
        f"content: {record.content}\n"
        f"source_time: {source.get('source_time', '')}\n"
        f"evidence: {source.get('evidence_span', '')}\n"
        f"structured_fields: {fields_text}"
    )
