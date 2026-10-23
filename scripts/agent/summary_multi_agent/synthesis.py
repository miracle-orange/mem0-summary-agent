"""基于当前事实、历史辅助证据和模板上下文生成候选摘要。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .adapters.model import ModelClient
from .adapters.model import is_context_overflow_error
from .minutes_report import MinutesReportBundle, build_report_bundle
from .minutes_report.models import MinutesDocumentDraft
from .minutes_report.prompt import (
    RICH_SYNTHESIS_PROMPT_VERSION,
    build_rich_synthesis_prompt,
)
from .minutes_report.sources import (
    build_history_aliases,
    normalize_document,
    resolve_sources,
)
from .model_io import extract_json_object
from .model_input_audit import ModelInputAuditRecorder
from .schemas import MinutesOutputMode, RunConfig, SynthesisContext
from .tools.template import project_template


SYNTHESIS_PROMPT_VERSION = "summary_multi_agent.synthesis_prompt.v4"
MODEL_INPUT_REPORT_VERSION = "summary_multi_agent.model_input_report.v1"


@dataclass(frozen=True)
class SynthesisOutcome:
    """保存正式候选摘要、实际 Prompt、模型输入报告和 warning。"""

    summary: str
    prompt: str
    warnings: tuple[str, ...] = ()
    model_input_report: dict[str, Any] | None = None
    report_bundle: MinutesReportBundle | None = None


SUMMARY_META_PREAMBLE_MARKERS = (
    "synthesiscontext",
    "<internal_synthesis_input>",
    "根据您提供",
    "根据你提供",
    "重新生成",
    "校验问题",
    "以下是",
)


def remove_summary_meta_preamble(summary: str) -> tuple[str, bool]:
    """只移除摘要正文前短小的实现元前言，并返回是否发生移除。"""

    text = summary.strip()
    if not text:
        return text, False
    lines = text.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None or lines[first].lstrip().startswith("#"):
        return text, False

    end = first
    while end < len(lines):
        stripped = lines[end].strip()
        if end > first and (not stripped or stripped.startswith("#")):
            break
        end += 1
    paragraph = " ".join(line.strip() for line in lines[first:end]).strip()
    lowered = paragraph.lower()
    is_meta = (
        len(paragraph) <= 300
        and "摘要" in paragraph
        and any(marker in lowered for marker in SUMMARY_META_PREAMBLE_MARKERS)
    )
    if not is_meta:
        return text, False

    cleaned = "\n".join(lines[end:]).lstrip()
    if not cleaned:
        return text, False
    return cleaned, True


def _compact_json(value: Any) -> str:
    """生成稳定的 UTF-8 友好紧凑 JSON，减少纯排版 token。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def visible_historical_records(
    context: SynthesisContext,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """返回 Evaluation 与 Summarizer 共同认定的历史业务记录集合。

    确定性压缩模式把记录放在 ``historical_records``；关闭压缩时记录仍在
    Worker payload 中。这里按既有顺序取并集，不改变检索、Top-K 或内容。
    """

    records: dict[str, dict[str, Any]] = {
        str(record_id): dict(record)
        for record_id, record in context.historical_records.items()
        if isinstance(record, Mapping)
    }
    for task_id, result in context.worker_results.items():
        if not isinstance(result, Mapping):
            continue
        payload = result.get("payload")
        if not isinstance(payload, Mapping):
            continue
        payload_records = payload.get("records")
        if not isinstance(payload_records, list):
            continue
        for index, record in enumerate(payload_records):
            if not isinstance(record, Mapping):
                continue
            record_id = str(
                record.get("memory_id")
                or record.get("record_ref_id")
                or f"{task_id}:record:{index}"
            )
            records.setdefault(record_id, dict(record))
    return tuple(records.items())


def _parsed_structured_fields(value: Any) -> Any:
    """解析 Memory 中已有的结构化业务字段，不重新抽取或概括。"""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _project_record_variant(record: Mapping[str, Any]) -> dict[str, Any]:
    """移除单条历史记录的审计字段，保留业务语义和记录时状态。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    content = record.get("content") or metadata.get("evidence_span") or ""
    projected: dict[str, Any] = {
        "type": record.get("memory_type") or metadata.get("memory_type"),
        "content": content,
        "source_date": metadata.get("source_time") or record.get("source_time"),
        "recorded_status": metadata.get("memory_status") or record.get("status"),
    }
    structured = _parsed_structured_fields(
        metadata.get("structured_fields_json")
        if "structured_fields_json" in metadata
        else record.get("structured_fields")
    )
    if structured not in (None, "", {}, []):
        projected["structured_fields"] = structured
    for source_key, target_key in (
        ("matched_entities", "matched_entities"),
        ("related_entities", "related_entities"),
        ("entities_text", "entities"),
        ("topics_text", "topics"),
    ):
        value = record.get(source_key, metadata.get(source_key))
        if value not in (None, "", [], {}, ()):
            projected[target_key] = value
    return {key: value for key, value in projected.items() if value not in (None, "")}


def _project_history_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """构造供模型阅读的历史记录，不暴露 ID、分数、offset 或 lineage。"""

    projected = _project_record_variant(record)
    canonical_marker = _compact_json(projected)
    content_variants = record.get("content_variants")
    if isinstance(content_variants, list):
        unique_variants = list(
            dict.fromkeys(
                str(item).strip()
                for item in content_variants
                if str(item).strip()
                and str(item).strip() != str(projected.get("content", "")).strip()
            )
        )
        if unique_variants:
            projected["content_variants"] = unique_variants
            projected["conflict_note"] = "同一历史来源存在不同内容版本，请谨慎表述。"

    record_variants = record.get("record_variants")
    if isinstance(record_variants, list):
        variants = [
            _project_record_variant(item)
            for item in record_variants
            if isinstance(item, Mapping)
        ]
        distinct: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in variants:
            marker = _compact_json(item)
            if marker in seen or marker == canonical_marker:
                continue
            seen.add(marker)
            distinct.append(item)
        if distinct:
            projected["record_variants"] = distinct
            projected["conflict_note"] = "同一历史来源存在不同结构版本，请保留差异。"
    return projected


def _project_temporal_note(value: Any) -> str | None:
    """仅保留会影响日期解释的简短 Temporal 提示。"""

    note = str(value).strip()
    if not note:
        return None
    lowered = note.lower()
    semantic_markers = (
        "time",
        "date",
        "temporal",
        "unresolved",
        "ambiguous",
        "conflict",
        "时间",
        "日期",
        "未解析",
        "歧义",
        "冲突",
        "模糊",
    )
    if not any(marker in lowered for marker in semantic_markers):
        return None
    return note if len(note) <= 240 else f"{note[:237]}..."


def _text_anchor(
    current_material: str,
    *,
    start_offset: Any,
    end_offset: Any,
    radius: int = 60,
) -> str | None:
    """为当前时间表达式保留最小原文定位短句。"""

    if not isinstance(start_offset, int) or not isinstance(end_offset, int):
        return None
    if start_offset < 0 or end_offset < start_offset:
        return None
    start = max(0, start_offset - radius)
    end = min(len(current_material), end_offset + radius)
    anchor = current_material[start:end].strip()
    return anchor or None


def _project_temporal_context(
    context: SynthesisContext,
    current_material: str,
) -> dict[str, Any]:
    """投影当前材料的时间结果；历史 Memory 日期不在此重新处理。"""

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    notes: list[str] = []
    seen_resolved: set[str] = set()
    seen_unresolved: set[str] = set()

    for result in context.worker_results.values():
        if not isinstance(result, Mapping):
            continue
        if result.get("task_type") != "temporal_reasoning":
            continue
        payload = result.get("payload")
        if not isinstance(payload, Mapping):
            continue

        resolved_markers: set[tuple[Any, Any, Any]] = set()
        result_resolved = payload.get("resolved")
        if isinstance(result_resolved, list):
            for item in result_resolved:
                if not isinstance(item, Mapping):
                    continue
                start_offset = item.get("start_offset")
                end_offset = item.get("end_offset")
                expression = item.get("expression")
                projected = {
                    "expression": expression,
                    "direction": item.get("direction"),
                    "segment_id": item.get("segment_id"),
                    "source_time": item.get("source_time") or context.source_time,
                    "resolution": item.get("resolution"),
                    "normalized_date": item.get("normalized_date"),
                    "confidence": item.get("confidence"),
                    "anchor": _text_anchor(
                        current_material,
                        start_offset=start_offset,
                        end_offset=end_offset,
                    ),
                }
                projected = {
                    key: value
                    for key, value in projected.items()
                    if value not in (None, "", {}, [])
                }
                marker = _compact_json(projected)
                if marker not in seen_resolved:
                    seen_resolved.add(marker)
                    resolved.append(projected)
                resolved_markers.add((expression, start_offset, end_offset))

        expressions = payload.get("expressions")
        if isinstance(expressions, list):
            for item in expressions:
                if not isinstance(item, Mapping):
                    continue
                marker_tuple = (
                    item.get("expression"),
                    item.get("start_offset"),
                    item.get("end_offset"),
                )
                if marker_tuple in resolved_markers:
                    continue
                projected = {
                    "expression": item.get("expression"),
                    "direction": item.get("direction"),
                    "segment_id": item.get("segment_id"),
                    "source_time": context.source_time,
                    "status": "unresolved",
                    "anchor": _text_anchor(
                        current_material,
                        start_offset=item.get("start_offset"),
                        end_offset=item.get("end_offset"),
                    ),
                }
                projected = {
                    key: value
                    for key, value in projected.items()
                    if value not in (None, "", {}, [])
                }
                marker = _compact_json(projected)
                if marker not in seen_unresolved:
                    seen_unresolved.add(marker)
                    unresolved.append(projected)

        warnings = result.get("warnings")
        if isinstance(warnings, (list, tuple)):
            notes.extend(
                note
                for note in (_project_temporal_note(item) for item in warnings)
                if note is not None
            )

    return {
        "resolved": resolved,
        "unresolved_or_ambiguous": unresolved,
        "notes": list(dict.fromkeys(notes)),
    }


def _project_output_template(template: Mapping[str, Any]) -> dict[str, Any]:
    """只向 Summarizer 暴露最终格式、章节和写作要求。"""

    return project_template(dict(template))


def build_synthesis_model_input(
    context: SynthesisContext,
    *,
    current_material: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """纯函数：从完整审计上下文构造精简的 Summarizer 模型视图。"""

    visible_records = visible_historical_records(context)
    model_input = {
        "summary_task": {
            "summary_request": context.summary_request,
            "source_time": context.source_time,
            "target_entities": list(context.target_entities),
            "personalization_preferences": context.personalization_preferences,
        },
        "historical_context": [
            _project_history_record(record) for _, record in visible_records
        ],
        "current_temporal_results": _project_temporal_context(
            context, current_material
        ),
        "output_template": _project_output_template(context.template),
    }
    return model_input, tuple(record_id for record_id, _ in visible_records)


def build_legacy_summary_synthesis_prompt(context: SynthesisContext) -> str:
    """保留 V2 完整上下文 Prompt，供旧内部调用和输入对比使用。"""

    context_json = json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"""你是 Summary Multi-Agent V2 的 Summarizer Agent。

请根据下面经过结构校验的内部事实输入生成最终中文摘要。

事实优先级：
1. current segment analyses 中有当前原文 evidence_quote 支持的事实；
2. historical_records 中按 memory identity 全局归并、并由 record_ids 关联回 Worker 的长期历史证据；
3. 模板和个性化偏好只控制表达，不得改写事实。

硬性规则：
- 覆盖用户 summary_request，并优先突出 target_entities。
- 当前事实与历史背景必须区分；历史未知状态不能写成已完成。
- 历史 Memory 中“今天/明天/周五”等相对时间必须以该条 Memory metadata.source_time 为锚点，不能以本轮 request source_time 重新解释；若 structured_fields_json 已有绝对 deadline，应优先使用绝对时间。
- 当前原文已明确某个历史任务完成、取消或变更时，当前状态覆盖历史 pending 状态；不得在同一摘要中同时写成“待确认”和“已完成”。
- 决定、建议、问题、风险和任务不能相互混写。
- 负责人必须有证据；仅提出意见的人不能自动写成负责人。
- Temporal Worker 是时间语义的唯一推理权威。优先使用它在 worker_results 中给出的 normalized_date、source_time、原始时间表达、evidence_id 和 confidence。
- Summarizer 可以基于 Temporal Worker 的已核验结果解释时间关系、组织时间线和保留必要的原始表达，但不得重新计算相对日期、用本轮 request source_time 覆盖历史 Memory 的时间锚点，或自行改写 normalized_date。
- 不得把 Current Mapper 自行改写的相对时间与 Temporal Worker 日期机械拼接；只有 Temporal Worker 已确认二者来自同一证据且方向明确时才能组合。
- Temporal Worker 未解析、报告冲突或方向不明确时，保留原始时间表达并标为待确认，不得猜测或自行补日期。
- Worker warning 或证据冲突必须转化为谨慎表述或待确认项。
- historical_records 的 content_variants/record_variants 表示同一来源身份出现了不同内容或结构版本，必须保留冲突或谨慎表述，不能擅自择一覆盖。
- 不要输出 evidence_id、JSON、分析过程或“根据上下文”等元叙述。
- 最终输出必须直接从摘要正文或第一个 Markdown 标题开始，不得添加确认、致谢、生成说明或引导语。
- 不得向用户暴露 SynthesisContext、Schema、Prompt、校验问题、Agent/Worker 调用过程等内部实现；如果当前原始材料本身明确讨论这些概念，才可把它们作为有证据支持的业务事实写入正文。
- 只输出最终摘要正文。

<internal_synthesis_input>
{context_json}
</internal_synthesis_input>
"""


def _length_stats(text: str) -> dict[str, Any]:
    """按字符和 UTF-8 字节记录长度，并显式标注 token 为估算。"""

    byte_count = len(text.encode("utf-8"))
    return {
        "chars": len(text),
        "utf8_bytes": byte_count,
        "estimated_tokens": (byte_count + 3) // 4,
        "token_estimator": "utf8_bytes_div_4",
    }


def _build_model_input_report(
    *,
    context: SynthesisContext,
    current_material: str,
    prompt: str,
    sections: Mapping[str, str],
    history_record_ids: tuple[str, ...],
    prompt_version: str = SYNTHESIS_PROMPT_VERSION,
    mode: str = "projected",
) -> dict[str, Any]:
    """记录实际模型输入口径，不把报告本身送入模型。"""

    legacy_prompt = build_legacy_summary_synthesis_prompt(context)
    prompt_stats = _length_stats(prompt)
    legacy_stats = _length_stats(legacy_prompt)
    legacy_chars = int(legacy_stats["chars"])
    reduction_ratio = (
        (legacy_chars - int(prompt_stats["chars"])) / legacy_chars
        if legacy_chars
        else 0.0
    )
    return {
        "schema_version": MODEL_INPUT_REPORT_VERSION,
        "prompt_version": prompt_version,
        "mode": mode,
        "current_material_sha256": hashlib.sha256(
            current_material.encode("utf-8")
        ).hexdigest(),
        "history_record_ids": list(history_record_ids),
        "template_selection": {
            **context.auxiliary_context_info.get(
                "template_selection", {"source": "provided_context"}
            ),
            "template_name": context.template.get("template_name"),
            "meeting_scene": context.template.get("meeting_scene", "general"),
            "format_kind": context.template.get("format_kind"),
        },
        "sections": {name: _length_stats(value) for name, value in sections.items()},
        "totals": {
            "projected": prompt_stats,
            "legacy_full_context": legacy_stats,
        },
        "char_reduction_ratio": reduction_ratio,
    }


def build_summary_synthesis_prompt(
    context: SynthesisContext,
    *,
    current_material: str | None = None,
) -> str:
    """构造实际 Summarizer Prompt；未提供原文时保留旧内部调用路径。"""

    if current_material is None:
        return build_legacy_summary_synthesis_prompt(context)
    prompt, _ = build_projected_summary_synthesis_prompt(
        context,
        current_material=current_material,
    )
    return prompt


def build_projected_summary_synthesis_prompt(
    context: SynthesisContext,
    *,
    current_material: str,
) -> tuple[str, dict[str, Any]]:
    """构造独立模型视图和只读输入报告，不修改 ``context``。"""

    model_input, history_record_ids = build_synthesis_model_input(
        context,
        current_material=current_material,
    )
    sections = {
        "summary_task": _compact_json(model_input["summary_task"]),
        "current_material": current_material,
        "historical_context": _compact_json(model_input["historical_context"]),
        "current_temporal_results": _compact_json(
            model_input["current_temporal_results"]
        ),
        "output_template": _compact_json(model_input["output_template"]),
    }
    instructions = """你是 Summary Multi-Agent V2 的 Summarizer Agent。

请只根据以下五个输入分区生成最终中文摘要。审计 ID、检索分数、offset、工具轨迹和运行统计未提供给你，它们不属于摘要事实。

事实优先级：
1. current_material 是本轮完整当前原文，也是当前事实的直接来源；
2. current_temporal_results 提供当前材料中需要推算的日期结果；原文明示的绝对日期可直接引用；
3. historical_context 是与本轮相关的历史背景，状态均为记录时状态；
4. output_template 和显式偏好只控制表达，不得改写事实。

硬性规则：
- 覆盖 summary_task.summary_request，并优先突出 target_entities。
- 当前事实与历史背景必须区分；引用历史时标明已有来源日期及记录时状态；缺少日期写为未明确，历史未知状态不能写成已完成。历史未被本次提及不代表取消或完成，未检索到不代表首次提出。
- 历史记录的 source_date 来自已有 Memory metadata.source_time；source_date、recorded_status 和 structured_fields 直接按已有值解释，不使用本轮 source_time 重新计算历史相对时间。
- 当前原文明示历史任务完成、取消或变更时，当前状态覆盖历史记录时状态；不得同时写成互相矛盾的状态。
- content_variants、record_variants 或 conflict_note 表示实质差异，必须保留冲突或谨慎表述，不能擅自择一覆盖。
- 决定、明确要求、建议、计划、问题和风险不能相互混写；保留“预计、可能、暂定、待确认”等语气，条件未满足的方案不能写成已执行。负责人必须有证据，仅提出意见的人或 target_entities 不能自动写成负责人。
- 原文明示的绝对日期可以直接引用；需要推算的当前日期只使用 current_temporal_results 中对应同一事项的已解析结果。未解析、有歧义或有冲突时保留原始表达并标为待确认，不得自行补日期或把某事项日期移给另一事项。
- 记录时状态不等于最新业务状态；Memory 的 active/inactive 等管理标记不能直接解释为任务完成或取消。
- 按业务议题组织完整语句，不按发言人流水账。每个议题按材料实际内容交代事实、讨论或方案及当前结论，不强制填满小节或凑足议题数量。
- 保留影响判断或执行的具体对象、数字、单位、范围、排除项、前后对照、条件和依赖。成组对照保留对应关系，不能只概括为“有所提升”“持续优化”。
- 已完成部分与剩余工作分开表达；“本周推进”不能写成“本周完成”，“当前一批”不能扩大为全部工作；不要将分散且缺少关联依据的信息拼成同一承诺。
- 后续行动写清事项、负责人、时间要求与前提；只列有依据的任务，缺失字段写“未明确”，不从发言身份推定分工。
- 默认按议题适度详实，不规定最低字数。概览简述结论，正文展开必要依据，行动项集中写可执行安排，避免跨章节重复展开同一事实；优先保留关键事实并写完正文，不用套话扩充篇幅。
- 用户在 summary_request 中明确的格式和详略要求优先于预置样式，事实约束始终有效。其余情况遵循 output_template 的章节和写作要求；无内容的历史、风险、决定或行动项整节省略，不重复输出模板说明。
- 不输出 JSON、分析过程、内部实现或“根据上下文”等元叙述。
- 最终输出直接从摘要正文或第一个 Markdown 标题开始，不得添加确认、致谢、生成说明或引导语；只输出最终摘要正文。"""
    prompt = f"""{instructions}

<summary_task>
{sections["summary_task"]}
</summary_task>

<current_material>
{sections["current_material"]}
</current_material>

<historical_context>
{sections["historical_context"]}
</historical_context>

<current_temporal_results>
{sections["current_temporal_results"]}
</current_temporal_results>

<output_template>
{sections["output_template"]}
</output_template>
"""
    report = _build_model_input_report(
        context=context,
        current_material=current_material,
        prompt=prompt,
        sections={"instructions": instructions, **sections},
        history_record_ids=history_record_ids,
    )
    return prompt, report


def build_rich_summary_synthesis_prompt(
    context: SynthesisContext,
    *,
    current_material: str,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    """Build the rich model view with short aliases for visible history records."""

    model_input, history_record_ids = build_synthesis_model_input(
        context,
        current_material=current_material,
    )
    model_history, history_aliases = build_history_aliases(
        visible_historical_records(context),
        projector=_project_history_record,
    )
    model_input["historical_context"] = model_history
    sections = {
        "summary_task": _compact_json(model_input["summary_task"]),
        "current_material": current_material,
        "historical_context": _compact_json(model_input["historical_context"]),
        "current_temporal_results": _compact_json(
            model_input["current_temporal_results"]
        ),
        "output_template": _compact_json(model_input["output_template"]),
    }
    prompt = build_rich_synthesis_prompt(sections)
    report = _build_model_input_report(
        context=context,
        current_material=current_material,
        prompt=prompt,
        sections=sections,
        history_record_ids=history_record_ids,
        prompt_version=RICH_SYNTHESIS_PROMPT_VERSION,
        mode="rich_projected",
    )
    report["rich_history_aliases"] = history_aliases
    return prompt, report, history_aliases, sections


class SummarySynthesizer:
    """调用模型合成摘要正文，并保留只做输出规范化的 trace。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        model_input_audit: ModelInputAuditRecorder | None = None,
    ) -> None:
        """保存模型客户端和摘要 token/timeout 配置。"""
        self._model_client = model_client
        self._config = config
        self._model_input_audit = model_input_audit
        self._synthesis_trace: list[dict[str, object]] = []
        self._last_prompt: str | None = None
        self._last_model_input_report: dict[str, Any] | None = None

    @property
    def synthesis_trace(self) -> tuple[dict[str, object], ...]:
        """返回最近一次摘要生成的规范化审计记录。"""
        return tuple(self._synthesis_trace)

    @property
    def last_prompt(self) -> str | None:
        """返回最近一次实际提交或准备提交给模型的 Prompt。"""

        return self._last_prompt

    @property
    def last_model_input_report(self) -> dict[str, Any] | None:
        """返回最近一次投影输入报告，模型调用失败时仍可审计。"""

        return self._last_model_input_report

    async def generate(
        self,
        context: SynthesisContext,
        *,
        current_material: str | None = None,
    ) -> SynthesisOutcome:
        """生成 candidate 0；正式 Pipeline 传原文，旧调用保留 V2 Prompt。"""

        if (
            self._config.minutes_output_mode == MinutesOutputMode.RICH
            and current_material is not None
        ):
            return await self._generate_rich(
                context,
                current_material=current_material,
            )

        model_input_report: dict[str, Any] | None = None
        if current_material is None:
            prompt = build_legacy_summary_synthesis_prompt(context)
        else:
            prompt, model_input_report = build_projected_summary_synthesis_prompt(
                context,
                current_material=current_material,
            )
        self._last_prompt = prompt
        self._last_model_input_report = model_input_report
        self._synthesis_trace = []
        call_id: str | None = None
        if self._model_input_audit is not None:
            if current_material is None:
                sections = {"legacy_synthesis_prompt": prompt}
                prompt_version = "summary_multi_agent.synthesis_prompt.v2"
            else:
                model_view, _ = build_synthesis_model_input(
                    context,
                    current_material=current_material,
                )
                sections = {
                    "current_material": current_material,
                    "summary_task": _compact_json(model_view["summary_task"]),
                    "historical_context": _compact_json(
                        model_view["historical_context"]
                    ),
                    "current_temporal_results": _compact_json(
                        model_view["current_temporal_results"]
                    ),
                    "output_template": _compact_json(
                        model_view["output_template"]
                    ),
                }
                prompt_version = SYNTHESIS_PROMPT_VERSION
            call_id = self._model_input_audit.start_call(
                stage="summarizer",
                prompt_version=prompt_version,
                prompt=prompt,
                sections=sections,
                max_tokens=self._config.summary_max_tokens,
            )
        started = time.perf_counter()
        try:
            raw_summary = (
                await asyncio.wait_for(
                    self._model_client.complete(
                        prompt,
                        temperature=0.0,
                        max_tokens=self._config.summary_max_tokens,
                    ),
                    timeout=self._config.worker_timeout_seconds,
                )
            ).strip()
        except BaseException as exc:
            if call_id is not None and self._model_input_audit is not None:
                self._model_input_audit.finish_call(
                    call_id,
                    status=(
                        "cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "failed"
                    ),
                    model_call_duration_ms=int(
                        (time.perf_counter() - started) * 1000
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        if call_id is not None and self._model_input_audit is not None:
            self._model_input_audit.finish_call(
                call_id,
                status="completed",
                model_call_duration_ms=int(
                    (time.perf_counter() - started) * 1000
                ),
            )
        summary, meta_preamble_removed = remove_summary_meta_preamble(raw_summary)
        if not summary:
            raise RuntimeError("Summarizer returned empty content")

        self._synthesis_trace.append(
            {
                "attempt": 1,
                "candidate_summary": summary,
                "raw_candidate_summary": raw_summary,
                "normalizations": (
                    ["removed_meta_preamble"] if meta_preamble_removed else []
                ),
                "issues": [],
                "temporal_authority": "temporal_worker",
                "validation_mode": "output_normalization_only",
                "prompt_version": (
                    SYNTHESIS_PROMPT_VERSION
                    if current_material is not None
                    else "summary_multi_agent.synthesis_prompt.v2"
                ),
            }
        )
        warnings = []
        if meta_preamble_removed:
            warnings.append("summary meta preamble was removed by the output guard")
        return SynthesisOutcome(
            summary=summary,
            prompt=prompt,
            warnings=tuple(warnings),
            model_input_report=model_input_report,
        )

    async def _generate_rich(
        self,
        context: SynthesisContext,
        *,
        current_material: str,
    ) -> SynthesisOutcome:
        """Generate one structured document and derive every rich report view from it."""

        prompt, model_input_report, history_aliases, audit_sections = (
            build_rich_summary_synthesis_prompt(
                context,
                current_material=current_material,
            )
        )
        self._last_prompt = prompt
        self._last_model_input_report = model_input_report
        self._synthesis_trace = []
        attempts: list[dict[str, Any]] = []
        next_prompt = prompt
        draft: MinutesDocumentDraft | None = None

        for attempt_index in range(2):
            raw_output: str | None = None
            call_id: str | None = None
            started = time.perf_counter()
            self._last_prompt = next_prompt
            if self._model_input_audit is not None:
                call_id = self._model_input_audit.start_call(
                    stage="summarizer",
                    prompt_version=RICH_SYNTHESIS_PROMPT_VERSION,
                    prompt=next_prompt,
                    sections=(
                        audit_sections
                        if attempt_index == 0
                        else {
                            "base_rich_synthesis_prompt": prompt,
                            "repair_instruction": next_prompt[len(prompt) :],
                        }
                    ),
                    max_tokens=self._config.rich_summary_max_tokens,
                    attempt_index=attempt_index + 1,
                )
            try:
                raw_output = await asyncio.wait_for(
                    getattr(self._model_client, "complete_json", self._model_client.complete)(
                        next_prompt,
                        temperature=0.0,
                        max_tokens=self._config.rich_summary_max_tokens,
                    ),
                    timeout=self._config.worker_timeout_seconds,
                )
                draft = MinutesDocumentDraft.model_validate(
                    extract_json_object(raw_output)
                )
                if call_id is not None and self._model_input_audit is not None:
                    self._model_input_audit.finish_call(
                        call_id,
                        status="completed",
                        model_call_duration_ms=int(
                            (time.perf_counter() - started) * 1000
                        ),
                    )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_output": raw_output,
                        "error": None,
                    }
                )
                break
            except BaseException as exc:
                if call_id is not None and self._model_input_audit is not None:
                    self._model_input_audit.finish_call(
                        call_id,
                        status=(
                            "cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "failed"
                        ),
                        model_call_duration_ms=int(
                            (time.perf_counter() - started) * 1000
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                failure_kind = "validation_error"
                if isinstance(exc, asyncio.TimeoutError):
                    failure_kind = "timeout"
                elif is_context_overflow_error(exc):
                    failure_kind = "context_overflow"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_output": raw_output,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failure_kind": failure_kind,
                    }
                )
                if (
                    isinstance(exc, (asyncio.CancelledError, asyncio.TimeoutError))
                    or is_context_overflow_error(exc)
                    or not isinstance(exc, Exception)
                ):
                    self._synthesis_trace.append(
                        {
                            "prompt_version": RICH_SYNTHESIS_PROMPT_VERSION,
                            "validation_mode": "strict_json_with_one_repair",
                            "attempts": attempts,
                        }
                    )
                    raise
                if attempt_index >= 1:
                    self._synthesis_trace.append(
                        {
                            "prompt_version": RICH_SYNTHESIS_PROMPT_VERSION,
                            "validation_mode": "strict_json_with_one_repair",
                            "attempts": attempts,
                        }
                    )
                    raise RuntimeError(
                        "rich minutes synthesis did not produce a valid document"
                    ) from exc
                next_prompt = (
                    prompt
                    + "\n\n上一次输出未通过 JSON 或结构校验。"
                    + "请从头返回一份完整、紧凑且符合原契约的 JSON；"
                    + "不得删除已有事实来规避校验，不得输出 Markdown。"
                    + f"\n校验错误：{type(exc).__name__}: {exc}"
                    + "\n上一次输出：\n"
                    + str(raw_output or "<no output>")[:6000]
                )

        if draft is None:
            raise RuntimeError("rich minutes synthesis returned no document")

        template = project_template(context.template)
        document = normalize_document(
            draft,
            source_time=context.source_time,
            meeting_scene=str(template.get("meeting_scene") or "general"),
            format_kind=str(template.get("format_kind") or "structured"),
        )
        sources = resolve_sources(
            document,
            current_material=current_material,
            history_aliases=history_aliases,
        )
        bundle = build_report_bundle(document, sources)
        self._synthesis_trace.append(
            {
                "prompt_version": RICH_SYNTHESIS_PROMPT_VERSION,
                "validation_mode": "strict_json_with_one_repair",
                "attempts": attempts,
                "document_schema_version": document.schema_version,
                "source_warning_count": len(sources.warnings),
            }
        )
        return SynthesisOutcome(
            summary=bundle.markdown.strip(),
            prompt=prompt,
            warnings=bundle.warnings,
            model_input_report=model_input_report,
            report_bundle=bundle,
        )
