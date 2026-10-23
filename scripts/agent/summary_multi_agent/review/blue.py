"""根据已裁决 Issue 和 Evidence 生成受限的 Review 修订候选。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from ..adapters.model import ModelClient
from ..model_input_audit import ModelInputAuditRecorder
from ..tools.template import project_template
from ..schemas import (
    BlueRepairRecord,
    BlueReviewResult,
    FrozenStrictModel,
    RedReviewReport,
    ReviewAction,
    ReviewSeverity,
    ReviewVerificationResult,
    RunConfig,
    SummaryWorkingState,
)
from .model_io import complete_validated_json


# Review draft 模型的 Schema 不通过 __doc__ 参与协议，使用普通注释避免改变输出描述。
# Blue 对单个 Issue 的修复草案。
class BlueDraftRepair(FrozenStrictModel):
    issue_id: str = Field(min_length=1)
    action: ReviewAction
    applied: bool
    rationale: str = Field(min_length=1, max_length=240)
    evidence_ids: tuple[str, ...] = ()


# Blue 一轮完整候选修订草案。
class BlueDraftResult(FrozenStrictModel):
    candidate_summary: str = Field(min_length=1)
    repairs: tuple[BlueDraftRepair, ...] = ()
    unresolved_issue_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _json(value: Any) -> str:
    """将 Review 输入稳定序列化为 Prompt 片段。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_blue_prompt(
    state: SummaryWorkingState,
    source_candidate: str,
    red_report: RedReviewReport,
    verification_results: tuple[ReviewVerificationResult, ...],
    round_index: int,
) -> str:
    """构造只允许依据 Red Issue 和已验证 Evidence 局部修订的 Prompt。"""
    actionable = [
        issue.model_dump(mode="json")
        for issue in red_report.issues
        if issue.severity
        in {ReviewSeverity.CRITICAL, ReviewSeverity.HIGH, ReviewSeverity.MEDIUM}
    ]
    relevant_evidence_ids = {
        evidence_id
        for issue in red_report.issues
        for evidence_id in issue.evidence_ids
    }
    relevant_evidence_ids.update(
        evidence_id
        for result in verification_results
        for evidence_id in result.evidence_ids
    )
    evidence = {
        evidence_id: state.evidence_index[evidence_id].model_dump(mode="json")
        for evidence_id in sorted(relevant_evidence_ids)
        if evidence_id in state.evidence_index
    }
    return f"""你是 Summary Multi-Agent V2 的 Blue Agent，执行第 {round_index} 轮定向修复。
你只能根据 Red 的 critical/high/medium Issue 和已提供证据修复候选摘要。输出完整修复候选，但不得借机重写无关正确章节。

硬性规则：
1. 每项改动必须对应一个 issue_id，并在 repairs 中记录。
2. ADD/MODIFY/REGENERATE_SECTION 必须有当前 Evidence Index 或 VERIFY 成功结果支持。
3. VERIFY 为 not_found/failed 时，不得假装补查成功；只能保持原文、删除无证据断言或标为待确认。
4. 优先局部 ADD/DELETE/MODIFY/MARK_UNCERTAIN；只有章节结构本身错误才 REGENERATE_SECTION。
5. 不得改变未被 Issue 指出的日期、责任人、状态、决定或事实。
6. repairs 只记录真正改变候选正文的 issue；若摘要已经包含该内容，不得声称 applied=true，应放入 unresolved_issue_ids 交回下一轮核验。
6a. 同一个 issue_id 不得同时出现在 applied=true 的 repairs 和 unresolved_issue_ids 中。
7. 每个 rationale 使用不超过80个汉字的短句，不重复 evidence 原文；保持原摘要篇幅和结构，不扩写无关内容。
8. 若本轮无法在输出预算内可靠处理某项 issue，保持相关原文并把 issue_id 放入 unresolved_issue_ids，不得输出半截 JSON。
9. audit_origin=segment 的 issue 只用于补充该 segment 中有直接证据的遗漏，不足以证明候选中其他内容虚构；不得据此 DELETE、MARK_UNCERTAIN 或 REGENERATE_SECTION。
10. 不要修改时间语义或自行归一化日期；时间事实以 Temporal Worker 已提供的结构化结果为准。Red 不应生成时间正确性 Issue。
11. target_entities 不是责任人证据；不得把 Evidence 中只写角色的任务改派给目标人物。
11a. 局部修复保留数字、范围、否定、条件和不确定语气；历史事实保持来源与记录时状态，不将计划提升为承诺。章节修复遵循下方模板及用户显式格式，允许无内容章节省略和自然叙述，不为凑栏目扩写。
12. 只返回严格 JSON，不要在 JSON 外输出 Markdown 或解释。

请求：
{_json({
    "source_time": state.request.source_time,
    "summary_request": state.request.summary_request,
    "target_entities": state.request.target_entities,
    "personalization_preferences": state.request.personalization_preferences,
})}

原候选摘要：
<source_candidate>
{source_candidate}
</source_candidate>

选定模板（只约束表达，不能作为事实证据）：
<output_template>
{json.dumps(project_template(state.synthesis_context.template if state.synthesis_context else {}), ensure_ascii=False, separators=(",", ":"))}
</output_template>

可处理 Issues：
{_json(actionable)}

VERIFY 结果：
{_json([item.model_dump(mode="json") for item in verification_results])}

相关 Evidence Index：
{_json(evidence)}

输出格式：
{{
  "candidate_summary": "完整修复候选",
  "repairs": [
    {{
      "issue_id": "Red issue_id",
      "action": "add|delete|modify|verify|mark_uncertain|regenerate_section",
      "applied": true,
      "rationale": "本次局部修改及证据依据",
      "evidence_ids": ["实际使用的 evidence_id"]
    }}
  ],
  "unresolved_issue_ids": [],
  "warnings": []
}}
"""


class BlueAgent:
    """根据已裁决 Issue 生成证据约束的候选修订，不直接替换正式摘要。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        model_input_audit: ModelInputAuditRecorder | None = None,
    ) -> None:
        """保存模型客户端和 Review 配置，并初始化本轮模型 trace。"""
        self._model_client = model_client
        self._config = config
        self._model_input_audit = model_input_audit
        self.traces: list[dict[str, Any]] = []

    async def repair(
        self,
        state: SummaryWorkingState,
        source_candidate: str,
        red_report: RedReviewReport,
        verification_results: tuple[ReviewVerificationResult, ...],
        *,
        source_candidate_index: int,
        round_index: int,
    ) -> BlueReviewResult:
        """调用模型生成候选修订，再由本地规则过滤越权或破坏性改动。"""
        prompt = build_blue_prompt(
            state,
            source_candidate,
            red_report,
            verification_results,
            round_index,
        )
        draft = await complete_validated_json(
            model_client=self._model_client,
            prompt=prompt,
            validator=BlueDraftResult.model_validate,
            max_tokens=self._config.review_blue_max_tokens,
            timeout_seconds=self._config.worker_timeout_seconds,
            repair_attempts=self._config.schema_repair_attempts,
            trace=self.traces,
            trace_metadata={"stage": "blue", "round_index": round_index},
            model_input_audit=self._model_input_audit,
        )
        return self._normalize(
            state,
            source_candidate,
            red_report,
            verification_results,
            draft,
            source_candidate_index=source_candidate_index,
            round_index=round_index,
        )

    @staticmethod
    def _normalize(
        state: SummaryWorkingState,
        source_candidate: str,
        red_report: RedReviewReport,
        verification_results: tuple[ReviewVerificationResult, ...],
        draft: BlueDraftResult,
        *,
        source_candidate_index: int,
        round_index: int,
    ) -> BlueReviewResult:
        """校验 Issue/Evidence 关联、阻断危险修订并保留 unresolved 状态。"""
        issues = {
            issue.issue_id: issue
            for issue in red_report.issues
            if issue.severity
            in {ReviewSeverity.CRITICAL, ReviewSeverity.HIGH, ReviewSeverity.MEDIUM}
        }
        verifications = {item.issue_id: item for item in verification_results}
        warnings = list(draft.warnings)
        repairs: list[BlueRepairRecord] = []
        handled: set[str] = set()
        discard_candidate = False
        for repair in draft.repairs:
            issue = issues.get(repair.issue_id)
            if issue is None:
                warnings.append(
                    f"Blue discarded repair for unknown/non-actionable issue {repair.issue_id}"
                )
                continue
            allowed_evidence = set(issue.evidence_ids)
            verification = verifications.get(issue.issue_id)
            if verification is not None:
                allowed_evidence.update(verification.evidence_ids)
            evidence_ids = tuple(
                evidence_id
                for evidence_id in dict.fromkeys(repair.evidence_ids)
                if evidence_id in allowed_evidence
                and evidence_id in state.evidence_index
            )
            evidence_required_actions = {
                ReviewAction.ADD,
                ReviewAction.MODIFY,
                ReviewAction.REGENERATE_SECTION,
            }
            if repair.action in evidence_required_actions and not evidence_ids:
                evidence_ids = tuple(
                    evidence_id
                    for evidence_id in sorted(allowed_evidence)
                    if evidence_id in state.evidence_index
                )
            applied = repair.applied
            if (
                issue.audit_origin == "segment"
                and repair.action
                in {
                    ReviewAction.DELETE,
                    ReviewAction.MARK_UNCERTAIN,
                    ReviewAction.REGENERATE_SECTION,
                }
                and applied
            ):
                applied = False
                discard_candidate = True
                warnings.append(
                    f"Blue blocked destructive repair from segment-only issue {issue.issue_id}"
                )
            if repair.action == ReviewAction.VERIFY and applied:
                applied = False
                warnings.append(
                    f"Blue cannot apply VERIFY for {issue.issue_id}; "
                    "verification is handled by Orchestrator"
                )
            if (
                issue.needs_verification
                and (verification is None or verification.status != "verified")
                and repair.action
                in {
                    ReviewAction.ADD,
                    ReviewAction.MODIFY,
                    ReviewAction.REGENERATE_SECTION,
                }
            ):
                applied = False
                warnings.append(
                    f"Blue blocked unsupported repair for {issue.issue_id}"
                )
            if repair.action in evidence_required_actions and not evidence_ids:
                applied = False
                warnings.append(
                    f"Blue blocked ungrounded repair for {issue.issue_id}"
                )
            repairs.append(
                BlueRepairRecord(
                    issue_id=issue.issue_id,
                    action=repair.action,
                    applied=applied,
                    rationale=repair.rationale,
                    evidence_ids=evidence_ids,
                )
            )
            handled.add(issue.issue_id)

        unresolved = set(draft.unresolved_issue_ids)
        unresolved.update(set(issues) - handled)
        unresolved.update(
            item.issue_id
            for item in repairs
            if not item.applied
        )
        unknown_unresolved = unresolved - set(issues)
        if unknown_unresolved:
            warnings.append(
                "Blue discarded unknown unresolved issue ids: "
                + ", ".join(sorted(unknown_unresolved))
            )
        unresolved.intersection_update(issues)
        candidate_summary = draft.candidate_summary.strip()
        if discard_candidate:
            candidate_summary = source_candidate
            repairs = [item.model_copy(update={"applied": False}) for item in repairs]
            unresolved.update(issues)
            warnings.append(
                "Blue candidate was discarded because segment-only evidence "
                "cannot support destructive changes"
            )
        content_change_actions = {
            ReviewAction.ADD,
            ReviewAction.DELETE,
            ReviewAction.MODIFY,
            ReviewAction.MARK_UNCERTAIN,
            ReviewAction.REGENERATE_SECTION,
        }
        has_traceable_content_change = any(
            item.applied and item.action in content_change_actions
            for item in repairs
        )
        if has_traceable_content_change and candidate_summary == source_candidate:
            unchanged_applied_ids = {
                item.issue_id
                for item in repairs
                if item.applied and item.action in content_change_actions
            }
            repairs = [
                item.model_copy(update={"applied": False})
                if item.issue_id in unchanged_applied_ids
                else item
                for item in repairs
            ]
            unresolved.update(unchanged_applied_ids)
            has_traceable_content_change = False
            warnings.append(
                "Blue normalized applied repairs whose candidate text did not change"
            )
        if not has_traceable_content_change:
            if candidate_summary != source_candidate:
                warnings.append(
                    "Blue candidate changes were discarded because no traceable content repair was applied"
                )
            candidate_summary = source_candidate
        applied_issue_ids = {
            item.issue_id
            for item in repairs
            if item.applied and item.action in content_change_actions
        }
        stale_unresolved = unresolved.intersection(applied_issue_ids)
        if stale_unresolved:
            unresolved.difference_update(stale_unresolved)
            warnings.append(
                "Blue removed stale unresolved issue ids after guarded repairs were applied: "
                + ", ".join(sorted(stale_unresolved))
            )
        return BlueReviewResult(
            round_index=round_index,
            source_candidate_index=source_candidate_index,
            candidate_summary=candidate_summary,
            repairs=tuple(repairs),
            unresolved_issue_ids=tuple(sorted(unresolved)),
            warnings=tuple(dict.fromkeys(warnings)),
        )
