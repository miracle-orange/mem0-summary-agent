"""发现、归一化并裁决摘要问题；Red 产出候选而不是最终修订。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import Field

from ..adapters.model import ModelClient
from ..grounding import match_evidence_quote
from ..model_input_audit import ModelInputAuditRecorder
from ..schemas import (
    EvidenceRef,
    FrozenStrictModel,
    RedReviewReport,
    ReviewAction,
    ReviewIssue,
    ReviewIssueDisposition,
    ReviewSeverity,
    RunConfig,
    SourceKind,
    SummaryWorkingState,
)
from .model_io import complete_validated_json


# Review draft 模型通过显式字段校验，使用普通注释避免改变模型协议描述。
# Red 单个问题候选草案。
class RedDraftIssue(FrozenStrictModel):
    issue_type: str = Field(min_length=1)
    severity: ReviewSeverity
    section: str = "unknown"
    description: str = Field(min_length=1)
    summary_quote: str = ""
    evidence_ids: tuple[str, ...] = ()
    evidence_quotes: tuple[str, ...] = ()
    target_entities: tuple[str, ...] = ()
    related_task_ids: tuple[str, ...] = ()
    recommended_action: ReviewAction
    repair_instruction: str = ""
    route_to: str = "summary_writer"
    needs_verification: bool = False
    verification_query: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# Red 全局或分段审核的模型响应草案。
class RedDraftReport(FrozenStrictModel):
    issues: tuple[RedDraftIssue, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_red_draft_report(
    payload: Any,
    *,
    segment_mode: bool,
) -> RedDraftReport:
    """兼容已知 Red 输出形态，并将其归一化为严格报告。"""
    normalization: set[str] = set()
    report_warnings: list[str] = []
    if isinstance(payload, list):
        raw_issues = payload
        normalization.add("root_array_to_report")
    elif isinstance(payload, dict) and "issues" in payload:
        extra_report_keys = set(payload) - {"issues", "warnings"}
        if extra_report_keys:
            raise ValueError(
                "Red report contains unknown fields: "
                + ", ".join(sorted(extra_report_keys))
            )
        raw_issues = payload.get("issues", [])
        raw_warnings = payload.get("warnings", [])
        if isinstance(raw_warnings, str):
            raw_warnings = [raw_warnings]
            normalization.add("warning_string_to_list")
        if not isinstance(raw_warnings, (list, tuple)):
            raise ValueError("Red report warnings must be a list")
        report_warnings.extend(str(item) for item in raw_warnings if str(item))
    elif isinstance(payload, dict):
        raw_issues = [payload]
        normalization.add("single_issue_to_report")
    else:
        raise ValueError("Red output must be a report object, issue object, or issue array")

    if isinstance(raw_issues, dict):
        raw_issues = [raw_issues]
        normalization.add("single_issue_to_list")
    if not isinstance(raw_issues, (list, tuple)):
        raise ValueError("Red report issues must be a list")

    normalized_issues: list[dict[str, Any]] = []
    sequence_fields = (
        "evidence_ids",
        "evidence_quotes",
        "target_entities",
        "related_task_ids",
    )
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, dict):
            raise ValueError("each Red issue must be an object")
        issue = dict(raw_issue)
        if "target_section" in issue:
            if "section" not in issue:
                issue["section"] = issue["target_section"]
            issue.pop("target_section")
            normalization.add("target_section_to_section")
        for field_name in sequence_fields:
            value = issue.get(field_name)
            if isinstance(value, str):
                issue[field_name] = [value]
                normalization.add(f"{field_name}_string_to_list")
        if not issue.get("issue_type"):
            issue["issue_type"] = "critical_omission" if segment_mode else "other"
            normalization.add("default_issue_type")
        if not issue.get("severity"):
            issue["severity"] = ReviewSeverity.MEDIUM.value
            normalization.add("default_medium_severity")
        if not issue.get("recommended_action"):
            issue["recommended_action"] = (
                ReviewAction.ADD.value
                if segment_mode
                else ReviewAction.MODIFY.value
            )
            normalization.add("default_recommended_action")
        if segment_mode and issue.get("recommended_action") != ReviewAction.ADD.value:
            issue["recommended_action"] = ReviewAction.ADD.value
            normalization.add("segment_action_to_add")
        normalized_issues.append(issue)

    if normalization:
        report_warnings.append(
            "Red normalized known model output variants: "
            + ", ".join(sorted(normalization))
        )
    return RedDraftReport.model_validate(
        {
            "issues": normalized_issues,
            "warnings": list(dict.fromkeys(report_warnings)),
        }
    )


# 裁决 draft 只承载模型输出字段；不增加类 Docstring 以保持协议描述稳定。
# Red 对单个候选 Issue 的保留/驳回决定。
class RedAdjudicationDraftDecision(FrozenStrictModel):
    issue_id: str = Field(min_length=1)
    keep: bool
    reason_code: Literal[
        "confirmed",
        "already_covered",
        "duplicate",
        "speculative",
        "nonessential",
        "unsupported",
    ]
    reason: str = Field(min_length=1, max_length=160)


# Red 批量 Issue 裁决响应。
class RedAdjudicationDraftReport(FrozenStrictModel):
    decisions: tuple[RedAdjudicationDraftDecision, ...] = ()
    warnings: tuple[str, ...] = ()


def _json(value: Any) -> str:
    """将 Red 输入稳定序列化为 Prompt 片段。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _red_compact_retry_hint(max_candidates: int) -> str:
    """生成限制 issue 数量和字段长度的 Red 重试提示。"""
    return f"""
Red 紧凑重试约束：
- 最多输出 {max_candidates} 个 issue；只保留会改变结论或执行的信息。
- description、summary_quote、repair_instruction、verification_query 各不超过 60 个汉字。
- 不写日期推导过程，只写一次最终冲突；已有 evidence_id 才引用，没有则留空并请求验证。
- 空数组和空字符串字段可以省略；没有可靠 issue 时返回 {{"issues": [], "warnings": []}}。
"""


_REVIEW_CONTROL_PAYLOAD_FIELDS = frozenset(
    {
        "queries",
        "planned_queries",
        "processed_segment_ids",
        "selection_mode",
        "selected_observation_ids",
        "available_record_ids",
        "requested_memory_ids",
        "selected_memory_ids",
        "equivalent_record_merges",
        "record_version_conflicts",
        "available_record_version_conflicts",
        "retrieval_stats",
        "available_retrieval_stats",
        "retrieval_complete",
        "incomplete_queries",
        "reused_search_count",
        "new_search_count",
        "execution_origin",
        "record_selection_completed",
        "record_selection_origin",
        "record_ids",
        "warnings",
    }
)


def _build_global_review_context(state: SummaryWorkingState) -> dict[str, Any]:
    """从 SynthesisContext 构造去重后的全局 Review 辅助视图。"""
    context = state.synthesis_context
    if context is None:
        raise ValueError("Red Agent requires SynthesisContext")

    evidence_ids_by_memory: dict[str, list[str]] = {}
    historical_evidence: dict[str, dict[str, Any]] = {}
    for evidence_id, evidence in context.evidence_index.items():
        if evidence.source_kind == SourceKind.CURRENT_MATERIAL:
            continue
        historical_evidence[evidence_id] = {
            "source_kind": evidence.source_kind.value,
            "memory_id": evidence.memory_id,
            "source_time": evidence.source_time,
            "source_file": evidence.source_file,
            "evidence_text": evidence.evidence_text,
        }
        if evidence.memory_id:
            evidence_ids_by_memory.setdefault(evidence.memory_id, []).append(
                evidence_id
            )

    historical_records: dict[str, dict[str, Any]] = {
        record_ref_id: dict(record)
        for record_ref_id, record in context.historical_records.items()
    }
    for record_ref_id, record in historical_records.items():
        memory_id = str(record.get("memory_id") or "").strip()
        record["evidence_ids"] = list(
            dict.fromkeys(
                [
                    *record.get("evidence_ids", []),
                    *evidence_ids_by_memory.get(memory_id, []),
                ]
            )
        )
        record.setdefault("record_ref_id", record_ref_id)
    worker_findings: dict[str, dict[str, Any]] = {}
    for task_id, result in context.worker_results.items():
        payload = dict(result.get("payload") or {})
        records = payload.pop("records", [])
        for field in _REVIEW_CONTROL_PAYLOAD_FIELDS:
            payload.pop(field, None)
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                memory_id = str(record.get("memory_id") or "").strip()
                if not memory_id:
                    continue
                enriched = dict(record)
                enriched["evidence_ids"] = evidence_ids_by_memory.get(memory_id, [])
                current = historical_records.get(memory_id)
                if current is None or float(enriched.get("score") or 0.0) > float(
                    current.get("score") or 0.0
                ):
                    historical_records[memory_id] = enriched
        if payload:
            worker_findings[task_id] = {
                "task_type": result.get("task_type"),
                "payload": payload,
                "warnings": result.get("warnings", []),
            }

    return {
        "current_facts": [
            item.model_dump(mode="json") for item in context.merged_current_items
        ],
        "historical_records": [
            historical_records[memory_id]
            for memory_id in sorted(historical_records)
        ],
        "historical_and_tool_evidence": historical_evidence,
        "worker_findings": worker_findings,
        "template": context.template,
    }


def build_global_red_prompt(
    state: SummaryWorkingState,
    candidate_summary: str,
    round_index: int,
    max_candidates: int,
) -> str:
    """构造全局 Red 审核 Prompt，不直接修改候选摘要。"""
    review_context = _build_global_review_context(state)
    return f"""你是 Summary Multi-Agent V2 的 Red Agent，只检查，不修改摘要。
这是第 {round_index} 轮全局审核。请检查事实一致性、决定/建议混淆、责任人、完成状态、历史引用、模板、重复、无证据生成以及用户摘要要求覆盖。时间语义由 Temporal Worker 负责，本轮不重新归一化或裁决日期。

硬性规则：
1. 当前事实以 current evidence 为最高优先级；历史 Memory 只能作为历史上下文。
2. 只报告会影响结论、任务、风险、约束、状态、人物视角或用户要求的具体问题，不为追求数量罗列所有细节。
3. evidence_ids 只能复制 CompactReviewContext 中真实存在的 evidence_id；current_facts 的 item_id 不是证据 id。如引用 Current Fact，必须复制同一条的 evidence_id。缺少证据时置空并设置 needs_verification=true。全局审核的 evidence_quotes 必须为空，不得复制 evidence_text。
4. 需要补查时 action=verify，并给出 verification_query 与 route_to；不得假装已经找到证据。
5. 必须先按语义检查候选摘要全部章节。摘要已表达相同任务、责任人、状态、决定、风险或结论时，不得因措辞、章节或详细程度不同而报告遗漏。
6. 先在内部按 critical/high/medium/low 排序，只输出最重要的最多 {max_candidates} 个 issue；低优先级候选直接省略，不得为了列全而输出半截 JSON。
7. description、summary_quote 和 repair_instruction 各不超过 120 个汉字，不在 description 中重复 evidence 原文。
8. 同一句或同一日期推导不得在任何字段中重复；只写一次最终判断。
9. 不要生成 time_error；不要在本轮对相对时间进行日期归一化、日期正确性判断或时间修复。
10. target_entities 只是审核关注对象，不是责任人或任务关联证据；只有 Evidence 或摘要原句明确出现该实体时，才能把任务、状态或决定归到该实体。
10a. 关键数字、范围、排除项、条件与成组对照只有在遗漏会改变结论或执行时才报告；重点检查将局部完成扩大为全部完成、将计划推进改成截止承诺、将历史记录写成本次决定。
10b. 按用户明确格式及当前选定模板审核，允许无内容章节省略、自然叙述和正常同义表达；不因议题数量、段数或排版偏好而报错。历史来源日期缺失时允许明确说明未知，Memory 管理标记不等于业务完成状态。
11. 你不得生成修订摘要。只返回完整严格 JSON，不要 Markdown。

允许 action：add/delete/modify/verify/mark_uncertain/regenerate_section。
route_to 优先使用：summary_writer/semantic_memory/entity_personalization/template_adaptation。

请求：
{_json({
    "source_time": state.request.source_time,
    "summary_request": state.request.summary_request,
    "target_entities": state.request.target_entities,
    "personalization_preferences": state.request.personalization_preferences,
})}

CompactReviewContext（已去掉逐段分析和当前 evidence 的重复副本）：
{_json(review_context)}

待审核候选摘要：
<candidate_summary>
{candidate_summary}
</candidate_summary>

输出格式：
{{
  "issues": [
    {{
      "issue_type": "unsupported_fact|critical_omission|state_error|owner_error|decision_error|constraint_error|history_error|template_error|repetition|other",
      "severity": "critical|high|medium|low",
      "section": "章节或位置",
      "description": "不超过120字的具体问题",
      "summary_quote": "摘要原句；遗漏问题可为空",
      "evidence_ids": ["输入中存在的 evidence_id"],
      "evidence_quotes": [],
      "target_entities": ["该 issue 直接涉及的人物或实体；没有则为空"],
      "related_task_ids": [],
      "recommended_action": "add|delete|modify|verify|mark_uncertain|regenerate_section",
      "repair_instruction": "不超过120字的局部修复指令",
      "route_to": "summary_writer|semantic_memory|entity_personalization|temporal_reasoning|template_adaptation",
      "needs_verification": false,
      "verification_query": "需要补查时填写",
      "confidence": 0.0
    }}
  ],
  "warnings": []
}}
"""


def build_segment_red_prompt(
    state: SummaryWorkingState,
    candidate_summary: str,
    segment_id: str,
    segment_text: str,
    mapped_items: list[dict[str, Any]],
    round_index: int,
    max_candidates: int,
) -> str:
    """构造单原文分段的遗漏审核 Prompt。"""
    return f"""你是 Summary Multi-Agent V2 的 Red Agent，正在执行第 {round_index} 轮逐段遗漏审核。
只检查下面这个当前原始资料 segment 是否有与用户摘要要求直接相关、但在候选摘要中缺失或写错的关键信息。不要修改摘要。

硬性规则：
1. 不要求摘要包含原文每个细节；仅报告会影响决定、任务、责任人、截止时间、状态、约束、风险、关键结论、目标人物或用户明确要求的遗漏/错误。
2. evidence_quotes 必须逐字复制本 segment 中支持判断的连续原文。多轮对话共同支持一个事实时，应复制覆盖相关轮次的连续原文，或拆成多个 issue。
3. 不得使用语义近似引文。evidence_ids 留空，由系统在原文中定位后生成。
4. 必须按语义检查候选摘要全部章节；同一责任人、任务、截止时间、状态或结论已经表达时，不得因措辞不同、缺少 evidence 引用或细节更短而重复报 issue。
5. target_entities 只表示摘要关注对象，不能据此推测某项任务“可能涉及”该对象；没有直接证据的关联不是 issue。
6. 示例、补充解释和课堂逐字过程只有在用户明确要求或会改变结论/执行时才是关键遗漏。
7. 先在内部按 critical/high/medium/low 排序，只输出最重要的最多 {max_candidates} 个 issue；低优先级候选直接省略。
8. description 和 repair_instruction 各不超过 120 个汉字，使用最短且足够定位的逐字 evidence quote。
9. 不要生成 time_error；时间归一化和日期正确性已经由 Temporal Worker 完成，不在 Red 中重复审核；target_entities 不是责任人证据。
9a. 保留关键数字的对象、单位、范围与条件；仅当其缺失影响判断或执行才报告。任务范围扩大或把预计改成确定承诺属于事实问题；无需为了增加篇幅而补回非关键细节。
10. 只返回完整严格 JSON，不要 Markdown。

请求：
{_json({
    "source_time": state.request.source_time,
    "summary_request": state.request.summary_request,
    "target_entities": state.request.target_entities,
})}

segment_id: {segment_id}
当前 Mapper 已保留条目：
{_json(mapped_items)}

待审核候选摘要：
<candidate_summary>
{candidate_summary}
</candidate_summary>

当前原始 segment：
<current_segment>
{segment_text}
</current_segment>

输出格式与全局 Red 相同。遗漏通常使用 issue_type=critical_omission、recommended_action=add、route_to=summary_writer，并在 evidence_quotes 中提供原文。
"""


def build_red_adjudication_prompt(
    state: SummaryWorkingState,
    candidate_summary: str,
    candidate_issues: list[dict[str, Any]],
    evidence_index: dict[str, dict[str, Any]],
    current_fact_catalog: list[dict[str, Any]],
    round_index: int,
) -> str:
    """构造候选 Issue 的统一裁决 Prompt，供 Blue 只处理保留项。"""
    return f"""你是 Summary Multi-Agent V2 的 Red Issue Adjudicator。
这是第 {round_index} 轮候选 issue 的统一裁决。你只判断每个 issue 是否值得进入 Blue，不修改摘要，也不新增 issue。

硬性规则：
1. 对候选摘要做全篇语义覆盖检查，不要求逐字相同；摘要已表达相同任务、责任人、状态、时间、决定、风险或结论时必须 dismiss 为 already_covered。
2. 当前事实与历史事实冲突、历史相对时间使用错误 source_time、完成状态冲突、无证据责任人和会改变执行的遗漏应 keep。
3. 仅因摘要没有显示 evidence_id、没有展开次要例子、没有逐字复述课堂过程，不构成 issue。
4. “可能涉及目标人物”等推测性关联必须 dismiss 为 speculative。
5. 重复 issue 只保留信息最完整、证据最直接的一项，其余 dismiss 为 duplicate。
6. origin=segment 的候选只证明该 segment 中存在某项事实，不能仅凭一个 segment 未出现其他事实就声称摘要内容虚构、错误或应删除；必须同时对照完整 Current Fact Catalog。
7. 当前材料中明确的过去态执行陈述（如“已提交”“发了”“完成了”）可作为任务已执行/完成证据；没有冲突证据时，不得仅因缺少额外第三方确认而报告状态错误。
8. 角色称谓和姓名明确指向同一人时（如“课代表陈一诺”），同一动作、时间和对象不得拆成两个独立任务。
9. 对每个 omission/owner issue，必须逐字检查候选摘要中最接近的责任人、动作、对象、状态和时间；只缺非关键修饰时 dismiss 为 already_covered。reason 必须引用候选中的最接近短语。
10. 同一 Evidence 产生相反的责任人修改建议时，若 Current Fact Catalog 不能证明角色与姓名冲突，应 dismiss 为 speculative 或 duplicate，不得同时 keep。
11. target_entities 不是事实关联证据；Evidence 和候选原句都未出现目标实体时，必须 dismiss 该实体关联为 speculative。
12. 只能使用输入 issue_id；每个输入 issue_id 恰好返回一项 decision。reason 必须简短。
13. 只返回严格 JSON，不要 Markdown。

请求：
{_json({
    "source_time": state.request.source_time,
    "summary_request": state.request.summary_request,
    "target_entities": state.request.target_entities,
})}

待审核候选摘要：
<candidate_summary>
{candidate_summary}
</candidate_summary>

候选 issues（origin 仅表示来源）：
{_json(candidate_issues)}

去重后的相关 Evidence Index：
{_json(evidence_index)}

完整 Current Fact Catalog（跨全部 segment，仅用于裁决，不得新增 issue）：
{_json(current_fact_catalog)}

输出格式：
{{
  "decisions": [
    {{
      "issue_id": "输入 issue_id",
      "keep": true,
      "reason_code": "confirmed|already_covered|duplicate|speculative|nonessential|unsupported",
      "reason": "不超过160字符的裁决理由"
    }}
  ],
  "warnings": []
}}
"""


class RedAgent:
    """发现、grounding、去重并裁决摘要问题，产出供 Blue 使用的 Issue。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        model_input_audit: ModelInputAuditRecorder | None = None,
    ) -> None:
        """保存模型客户端、分段审核并发上限和 trace 容器。"""
        self._model_client = model_client
        self._config = config
        self._model_input_audit = model_input_audit
        self._semaphore = asyncio.Semaphore(
            config.review_max_concurrent_segment_audits
        )
        self.traces: list[dict[str, Any]] = []

    async def review(
        self,
        state: SummaryWorkingState,
        candidate_summary: str,
        *,
        candidate_index: int,
        round_index: int,
    ) -> RedReviewReport:
        """合并全局/逐段审核，归一化并 fail-closed 裁决有界 Issue。"""
        global_report = await self._global_review(
            state, candidate_summary, round_index
        )
        segment_reports = await self._segment_reviews(
            state, candidate_summary, round_index
        )
        warnings = list(global_report.warnings)
        draft_entries: list[tuple[RedDraftIssue, Optional[str], str]] = [
            (issue, None, "global") for issue in global_report.issues
        ]
        for segment_id, report in segment_reports:
            warnings.extend(report.warnings)
            draft_entries.extend(
                (issue, segment_id, f"segment:{segment_id}")
                for issue in report.issues
            )

        candidate_issues: list[ReviewIssue] = []
        issue_origins: dict[str, str] = {}
        seen: set[tuple[Any, ...]] = set()
        for draft, segment_id, origin in draft_entries:
            issue, issue_warnings = self._normalize_issue(
                state, draft, segment_id=segment_id
            )
            warnings.extend(issue_warnings)
            if issue is None:
                continue
            trace_anchor = (
                issue.summary_quote.strip(),
                tuple(sorted(issue.evidence_ids)),
            )
            if not trace_anchor[0] and not trace_anchor[1]:
                trace_anchor = (issue.description.strip().casefold(), ())
            key = (
                issue.issue_type,
                trace_anchor,
                issue.recommended_action.value,
            )
            if key in seen:
                continue
            seen.add(key)
            candidate_issues.append(issue)
            issue_origins[issue.issue_id] = origin

        adjudication = None
        if candidate_issues:
            try:
                adjudication = await self._adjudicate_issues(
                    state,
                    candidate_summary,
                    candidate_issues,
                    issue_origins,
                    round_index,
                )
                warnings.extend(adjudication.warnings)
            except Exception as exc:
                warnings.append(
                    "Red issue adjudication failed; applying bounded fail-closed "
                    f"fallback: {type(exc).__name__}: {exc}"
                )

        known_issue_ids = {issue.issue_id for issue in candidate_issues}
        decisions: dict[str, RedAdjudicationDraftDecision] = {}
        if adjudication is not None:
            for decision in adjudication.decisions:
                if decision.issue_id not in known_issue_ids:
                    warnings.append(
                        "Red adjudication discarded unknown issue id: "
                        + decision.issue_id
                    )
                    continue
                if decision.issue_id in decisions:
                    warnings.append(
                        "Red adjudication discarded duplicate decision for: "
                        + decision.issue_id
                    )
                    continue
                decisions[decision.issue_id] = decision

        kept: list[ReviewIssue] = []
        disposition_reasons: dict[str, tuple[str, str, str]] = {}
        for issue in candidate_issues:
            decision = decisions.get(issue.issue_id)
            keep = decision.keep if decision is not None else True
            reason = (
                decision.reason
                if decision is not None
                else "adjudicator omitted this issue; retained fail-closed"
            )
            reason_code = (
                decision.reason_code
                if decision is not None
                else "adjudicator_missing"
            )
            if keep:
                kept.append(issue)
                disposition_reasons[issue.issue_id] = (
                    "kept",
                    reason_code,
                    reason,
                )
            else:
                disposition_reasons[issue.issue_id] = (
                    "dismissed",
                    reason_code,
                    reason,
                )

        kept.sort(
            key=lambda issue: self._issue_priority(
                issue, issue_origins[issue.issue_id]
            )
        )
        limit = self._config.review_max_issues_per_round
        issues = kept[:limit]
        deferred_issues = kept[limit:]
        for issue in deferred_issues:
            disposition_reasons[issue.issue_id] = (
                "deferred",
                "per_round_budget",
                f"deferred by per-round issue budget {limit}",
            )
        if deferred_issues:
            warnings.append(
                f"Red deferred {len(deferred_issues)} adjudicated issues "
                f"after applying per-round budget {limit}"
            )

        issue_dispositions = tuple(
            ReviewIssueDisposition(
                issue_id=issue.issue_id,
                disposition=disposition_reasons[issue.issue_id][0],
                reason_code=disposition_reasons[issue.issue_id][1],
                reason=disposition_reasons[issue.issue_id][2],
            )
            for issue in candidate_issues
        )
        return RedReviewReport(
            round_index=round_index,
            candidate_index=candidate_index,
            passed=not issues and not deferred_issues,
            candidate_issue_count=len(candidate_issues),
            issues=tuple(issues),
            deferred_issues=tuple(deferred_issues),
            issue_dispositions=issue_dispositions,
            checked_segment_ids=tuple(
                segment.segment_id for segment in state.raw_material.segments
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    async def _adjudicate_issues(
        self,
        state: SummaryWorkingState,
        candidate_summary: str,
        issues: list[ReviewIssue],
        origins: dict[str, str],
        round_index: int,
    ) -> RedAdjudicationDraftReport:
        """按配置批量裁决候选 Issue，单批失败时保留其问题。"""
        batch_size = self._config.review_red_adjudication_batch_size
        batch_count = (len(issues) + batch_size - 1) // batch_size
        decisions: list[RedAdjudicationDraftDecision] = []
        warnings: list[str] = []
        for offset in range(0, len(issues), batch_size):
            batch = issues[offset : offset + batch_size]
            batch_index = offset // batch_size + 1
            try:
                report = await self._adjudicate_issue_batch(
                    state,
                    candidate_summary,
                    batch,
                    origins,
                    round_index,
                    batch_index=batch_index,
                    batch_count=batch_count,
                )
            except Exception as exc:
                warnings.append(
                    "Red issue adjudication failed for batch; affected issues will "
                    "be retained fail-closed "
                    f"(batch {batch_index}/{batch_count}): "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            decisions.extend(report.decisions)
            warnings.extend(report.warnings)
        return RedAdjudicationDraftReport(
            decisions=tuple(decisions),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    async def _adjudicate_issue_batch(
        self,
        state: SummaryWorkingState,
        candidate_summary: str,
        issues: list[ReviewIssue],
        origins: dict[str, str],
        round_index: int,
        *,
        batch_index: int,
        batch_count: int,
    ) -> RedAdjudicationDraftReport:
        """调用模型裁决一个 Issue 批次，并限制其只能使用输入 ID。"""
        payload = []
        evidence_index: dict[str, dict[str, Any]] = {}
        for issue in issues:
            for evidence_id in issue.evidence_ids:
                if evidence_id not in state.evidence_index:
                    continue
                evidence = state.evidence_index[evidence_id]
                evidence_index[evidence_id] = {
                    "evidence_id": evidence_id,
                    "source_kind": evidence.source_kind.value,
                    "source_time": evidence.source_time,
                    "memory_id": evidence.memory_id,
                    "evidence_text": evidence.evidence_text,
                }
            payload.append(
                {
                    **issue.model_dump(mode="json"),
                    "origin": origins[issue.issue_id],
                }
            )
        prompt = build_red_adjudication_prompt(
            state,
            candidate_summary,
            payload,
            evidence_index,
            [
                {
                    "item_id": item.item_id,
                    "category": item.category.value,
                    "statement": item.statement,
                    "evidence_id": item.evidence_id,
                    "entities": list(item.entities),
                    "time_expression": item.time_expression,
                }
                for item in (
                    state.synthesis_context.merged_current_items
                    if state.synthesis_context is not None
                    else ()
                )
            ],
            round_index,
        )
        return await complete_validated_json(
            model_client=self._model_client,
            prompt=prompt,
            validator=RedAdjudicationDraftReport.model_validate,
            max_tokens=self._config.review_red_adjudication_max_tokens,
            timeout_seconds=self._config.worker_timeout_seconds,
            repair_attempts=self._config.schema_repair_attempts,
            trace=self.traces,
            trace_metadata={
                "stage": "red_adjudication",
                "round_index": round_index,
                "candidate_issue_count": len(issues),
                "batch_index": batch_index,
                "batch_count": batch_count,
            },
            compact_retry_hint=(
                "\n只为输入中的每个 issue_id 返回一项短 decision；"
                "reason 不超过 60 个汉字，不得新增 issue_id。"
            ),
            model_input_audit=self._model_input_audit,
        )

    def _bound_draft_report(
        self,
        report: RedDraftReport,
        *,
        stage: str,
    ) -> RedDraftReport:
        """按严重度和出现顺序限制单次审核输出数量。"""
        limit = self._config.review_red_max_candidates_per_audit
        if len(report.issues) <= limit:
            return report
        severity_rank = {
            ReviewSeverity.CRITICAL: 0,
            ReviewSeverity.HIGH: 1,
            ReviewSeverity.MEDIUM: 2,
            ReviewSeverity.LOW: 3,
        }
        ranked = sorted(
            enumerate(report.issues),
            key=lambda item: (severity_rank[item[1].severity], item[0]),
        )
        retained = tuple(issue for _, issue in ranked[:limit])
        warning = (
            f"Red {stage} returned {len(report.issues)} candidates; retained "
            f"the highest-priority {limit} by per-audit budget"
        )
        return RedDraftReport(
            issues=retained,
            warnings=tuple(dict.fromkeys((*report.warnings, warning))),
        )

    @staticmethod
    def _issue_priority(issue: ReviewIssue, origin: str) -> tuple[Any, ...]:
        """为保留和延迟排序生成确定性 Issue 优先级。"""
        severity_rank = {
            ReviewSeverity.CRITICAL: 0,
            ReviewSeverity.HIGH: 1,
            ReviewSeverity.MEDIUM: 2,
            ReviewSeverity.LOW: 3,
        }
        return (
            severity_rank[issue.severity],
            0 if origin == "global" else 1,
            0 if issue.evidence_ids else 1,
            issue.issue_id,
        )

    async def _global_review(
        self,
        state: SummaryWorkingState,
        candidate_summary: str,
        round_index: int,
    ) -> RedDraftReport:
        """执行一次全局 Red 模型审核并应用单次候选上限。"""
        prompt = build_global_red_prompt(
            state,
            candidate_summary,
            round_index,
            self._config.review_red_max_candidates_per_audit,
        )
        report = await complete_validated_json(
            model_client=self._model_client,
            prompt=prompt,
            validator=lambda value: validate_red_draft_report(
                value,
                segment_mode=False,
            ),
            max_tokens=self._config.review_red_max_tokens,
            timeout_seconds=self._config.worker_timeout_seconds,
            repair_attempts=self._config.schema_repair_attempts,
            trace=self.traces,
            trace_metadata={
                "stage": "red_global",
                "round_index": round_index,
                "max_candidates_per_audit": (
                    self._config.review_red_max_candidates_per_audit
                ),
            },
            compact_retry_hint=_red_compact_retry_hint(
                self._config.review_red_max_candidates_per_audit
            ),
            model_input_audit=self._model_input_audit,
        )
        return self._bound_draft_report(report, stage="global audit")

    async def _segment_reviews(
        self,
        state: SummaryWorkingState,
        candidate_summary: str,
        round_index: int,
    ) -> list[tuple[str, RedDraftReport]]:
        """并发审核所有当前原文分段，并返回带 segment ID 的报告。"""
        analyses = {
            analysis.segment_id: analysis
            for analysis in state.current_segment_analyses
        }

        async def review_segment(segment) -> tuple[str, RedDraftReport]:
            """审核一个分段；并发由 Red 的 semaphore 控制。"""
            analysis = analyses[segment.segment_id]
            prompt = build_segment_red_prompt(
                state,
                candidate_summary,
                segment.segment_id,
                segment.text,
                [item.model_dump(mode="json") for item in analysis.items],
                round_index,
                self._config.review_red_max_candidates_per_audit,
            )
            async with self._semaphore:
                report = await complete_validated_json(
                    model_client=self._model_client,
                    prompt=prompt,
                    validator=lambda value: validate_red_draft_report(
                        value,
                        segment_mode=True,
                    ),
                    max_tokens=self._config.review_red_max_tokens,
                    timeout_seconds=self._config.worker_timeout_seconds,
                    repair_attempts=self._config.schema_repair_attempts,
                    trace=self.traces,
                    trace_metadata={
                        "stage": "red_segment",
                        "round_index": round_index,
                        "segment_id": segment.segment_id,
                        "max_candidates_per_audit": (
                            self._config.review_red_max_candidates_per_audit
                        ),
                    },
                    compact_retry_hint=_red_compact_retry_hint(
                        self._config.review_red_max_candidates_per_audit
                    ),
                    model_input_audit=self._model_input_audit,
                )
            return segment.segment_id, self._bound_draft_report(
                report,
                stage=f"segment audit {segment.segment_id}",
            )

        return list(
            await asyncio.gather(
                *(review_segment(segment) for segment in state.raw_material.segments)
            )
        )

    def _normalize_issue(
        self,
        state: SummaryWorkingState,
        draft: RedDraftIssue,
        *,
        segment_id: Optional[str],
    ) -> tuple[Optional[ReviewIssue], list[str]]:
        """把模型 Issue 映射到真实 Evidence/segment，并拒绝越权实体或时间问题。"""
        warnings: list[str] = []
        current_fact_aliases = {
            item.item_id: item.evidence_id
            for analysis in state.current_segment_analyses
            for item in analysis.items
            if item.evidence_id in state.evidence_index
        }
        evidence_ids: list[str] = []
        resolved_aliases: list[tuple[str, str]] = []
        invalid_ids: set[str] = set()
        for requested_id in draft.evidence_ids:
            if requested_id in state.evidence_index:
                evidence_ids.append(requested_id)
                continue
            resolved_id = current_fact_aliases.get(requested_id)
            if resolved_id is not None:
                evidence_ids.append(resolved_id)
                resolved_aliases.append((requested_id, resolved_id))
                continue
            invalid_ids.add(requested_id)
        if resolved_aliases:
            warnings.append(
                "Red resolved Current Fact ids to evidence ids: "
                + ", ".join(
                    f"{item_id}->{evidence_id}"
                    for item_id, evidence_id in resolved_aliases
                )
            )
        if invalid_ids:
            warnings.append(
                "Red discarded unknown evidence ids: "
                + ", ".join(sorted(invalid_ids))
            )

        segment_ids: list[str] = []
        if segment_id is not None:
            segment = next(
                item
                for item in state.raw_material.segments
                if item.segment_id == segment_id
            )
            for quote in draft.evidence_quotes:
                match = match_evidence_quote(segment.text, quote)
                grounded = match.grounded_quote
                if grounded is None:
                    warnings.append(
                        f"Red could not ground evidence quote in {segment_id}: "
                        f"{match.reason}"
                    )
                    continue
                global_start = segment.start_offset + grounded.start
                global_end = segment.start_offset + grounded.end
                existing = next(
                    (
                        item
                        for item in state.evidence_index.values()
                        if item.source_kind == SourceKind.CURRENT_MATERIAL
                        and item.source_id == segment.source_id
                        and item.start_offset == global_start
                        and item.end_offset == global_end
                    ),
                    None,
                )
                evidence = existing or EvidenceRef(
                    source_kind=SourceKind.CURRENT_MATERIAL,
                    source_id=segment.source_id,
                    segment_id=segment.segment_id,
                    start_offset=global_start,
                    end_offset=global_end,
                    source_time=state.request.source_time,
                    evidence_text=grounded.evidence_text,
                )
                state.evidence_index[evidence.evidence_id] = evidence
                evidence_ids.append(evidence.evidence_id)
                segment_ids.append(segment.segment_id)

        evidence_ids = list(dict.fromkeys(evidence_ids))
        segment_ids.extend(
            evidence.segment_id
            for evidence_id in evidence_ids
            if (evidence := state.evidence_index.get(evidence_id)) is not None
            and evidence.segment_id is not None
        )
        segment_ids = list(dict.fromkeys(segment_ids))
        requested_target_entities = tuple(
            dict.fromkeys(
                entity.strip()
                for entity in draft.target_entities
                if entity.strip()
            )
        )
        support_text = "\n".join(
            (
                draft.summary_quote,
                *(
                    evidence.evidence_text
                    for evidence_id in evidence_ids
                    if (evidence := state.evidence_index.get(evidence_id)) is not None
                ),
            )
        )
        claimed_target_entities = set(requested_target_entities)
        claimed_target_entities.update(
            entity
            for entity in state.request.target_entities
            if entity
            and (
                entity in draft.description
                or entity in draft.repair_instruction
            )
        )
        unsupported_target_entities = sorted(
            entity
            for entity in claimed_target_entities
            if entity not in support_text
        )
        target_entities = tuple(
            entity
            for entity in requested_target_entities
            if entity not in unsupported_target_entities
        )
        if unsupported_target_entities:
            warnings.append(
                "Red removed target entities without direct issue evidence: "
                + ", ".join(unsupported_target_entities)
            )
        unsupported_entity_claims = [
            entity
            for entity in unsupported_target_entities
            if entity in draft.description or entity in draft.repair_instruction
        ]
        if unsupported_entity_claims:
            warnings.append(
                "Red discarded issue with ungrounded target-entity association: "
                + ", ".join(unsupported_entity_claims)
            )
            return None, warnings

        if draft.issue_type == "time_error":
            warnings.append(
                "Red ignored time_error because temporal semantics are owned by Temporal Worker"
            )
            return None, warnings
        needs_verification = (
            draft.needs_verification
            or draft.recommended_action == ReviewAction.VERIFY
        )
        evidence_required_actions = {
            ReviewAction.ADD,
            ReviewAction.MODIFY,
            ReviewAction.REGENERATE_SECTION,
        }
        evidence_optional_issue_types = {"template_error", "repetition", "style_error"}
        if (
            draft.recommended_action in evidence_required_actions
            and not evidence_ids
            and draft.issue_type not in evidence_optional_issue_types
        ):
            needs_verification = True
            warnings.append(
                f"Red issue {draft.issue_type} requires verification because no evidence was grounded"
            )
        stable_material = "|".join(
            [
                draft.issue_type,
                draft.description.strip(),
                draft.summary_quote.strip(),
                *sorted(evidence_ids),
            ]
        )
        issue_id = "review_issue_" + hashlib.sha256(
            stable_material.encode("utf-8")
        ).hexdigest()[:16]
        return (
            ReviewIssue(
                issue_id=issue_id,
                issue_type=draft.issue_type,
                severity=draft.severity,
                section=draft.section,
                description=draft.description,
                summary_quote=draft.summary_quote,
                recommended_action=draft.recommended_action,
                repair_instruction=draft.repair_instruction,
                route_to=draft.route_to,
                evidence_ids=tuple(evidence_ids),
                segment_ids=tuple(segment_ids),
                audit_origin="segment" if segment_id is not None else "global",
                target_entities=target_entities,
                related_task_ids=draft.related_task_ids,
                needs_verification=needs_verification,
                verification_query=draft.verification_query,
                confidence=draft.confidence,
            ),
            warnings,
        )
