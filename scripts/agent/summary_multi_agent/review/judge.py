"""独立评价 Review 候选，只允许对已建立的 Issue 作终止判断。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field

from ..adapters.model import ModelClient
from ..model_input_audit import ModelInputAuditRecorder
from ..schemas import (
    BlueReviewResult,
    FrozenStrictModel,
    JudgeDecision,
    JudgeDisposition,
    RedReviewReport,
    ReviewVerificationResult,
    RunConfig,
    SummaryWorkingState,
)
from .model_io import complete_validated_json


JUDGE_DIMENSIONS = (
    "factual_consistency",
    "coverage",
    "personalization",
    "historical_memory_use",
    "template_adherence",
    "conciseness",
)


# Review draft 模型的 Schema 不通过 __doc__ 参与协议，使用普通注释避免改变输出描述。
# Judge 的候选评分和处置草案。
class JudgeDraftDecision(FrozenStrictModel):
    disposition: JudgeDisposition
    score: float = Field(ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    remaining_issue_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _json(value: Any) -> str:
    """将 Judge 输入稳定序列化为 Prompt 片段。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_judge_prompt(
    state: SummaryWorkingState,
    previous_candidate: str,
    candidate_summary: str,
    red_report: RedReviewReport,
    blue_result: BlueReviewResult | None,
    verification_results: tuple[ReviewVerificationResult, ...],
    round_index: int,
) -> str:
    """构造独立评价候选质量、剩余 Issue 和真实改进幅度的 Prompt。"""
    raw_segment_index = [
        {
            "segment_id": segment.segment_id,
            "start_offset": segment.start_offset,
            "end_offset": segment.end_offset,
        }
        for segment in state.raw_material.segments
    ]
    selected_red_report = {
        "round_index": red_report.round_index,
        "candidate_index": red_report.candidate_index,
        "issues": [item.model_dump(mode="json") for item in red_report.issues],
        "deferred_issue_count": len(red_report.deferred_issues),
        "note": "deferred issues are handled by the outer loop and are not valid remaining_issue_ids in this Judge call",
    }
    blue_payload = blue_result.model_dump(mode="json") if blue_result else None
    if (
        isinstance(blue_payload, dict)
        and blue_payload.get("candidate_summary") == candidate_summary
    ):
        blue_payload = dict(blue_payload)
        blue_payload.pop("candidate_summary", None)
        blue_payload["candidate_summary_ref"] = "candidate_summary section"
    return f"""你是 Summary Multi-Agent V2 的 Judge。请独立评价第 {round_index} 轮候选，不能修改摘要，也不能仅凭 Blue 声称“已修复”就通过。

必须检查：事实一致性、关键覆盖、个性化、历史知识使用、模板遵循、简洁性、剩余严重 Issue，以及相对上一候选是否真实改善。时间语义以 Temporal Worker 的结构化结果为准，不在 Judge 中重新归一化或裁决日期。

硬性规则：
1. 对照完整当前原始材料、segment 位置索引、Evidence、Red Issue 和 Blue 修复记录。
2. remaining_issue_ids 只能引用本轮“Selected Red Report”中 issues 的 issue_id；不得引用 deferred issue、上一轮 issue 或自行生成 id。
3. critical/high 未解决时不得 accept；证据不足且无法继续修时 stop_with_warnings。
4. repair 仅用于仍有可修复 high/medium Issue 的情况。
5. 你可以因独立发现的问题降低维度分数，但不得把不在 Selected Red Report 中的新问题伪装成 remaining_issue_id；下一轮 Red 会重新审核。
6. Grounded Current Fact Catalog 汇总了跨全部 segment 的已定位事实；其中已有的任务不得因只查看某个局部 segment 而判为虚构。
6a. 评估关键对象、数字、范围、条件及状态是否忠实保留；不得把篇幅增加当作质量提高，也不得把正常详略或同义表达当作错误。
6b. 模板遵循以用户明确格式和选定模板为准；无内容章节省略、自然叙述和按材料调整议题数量均合法。核对历史与当前边界、负责人依据及行动项范围，禁止将预计或计划改成承诺。
7. dimension_scores 的每个值必须在 0 到 1；只返回严格 JSON。

请求：
{_json({
    "source_time": state.request.source_time,
    "summary_request": state.request.summary_request,
    "target_entities": state.request.target_entities,
    "template": state.synthesis_context.template if state.synthesis_context else {},
})}

完整当前原始材料：
<current_material>
{state.request.current_material}
</current_material>

当前材料 segment 位置索引：
{_json(raw_segment_index)}

Grounded Current Fact Catalog：
{_json([
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
])}

上一候选：
<previous_candidate>
{previous_candidate}
</previous_candidate>

本轮待评价候选：
<candidate_summary>
{candidate_summary}
</candidate_summary>

Selected Red Report：
{_json(selected_red_report)}

Blue 结果：
{_json(blue_payload)}

VERIFY 结果：
{_json([item.model_dump(mode="json") for item in verification_results])}

输出格式：
{{
  "disposition": "accept|repair|stop_with_warnings",
  "score": 0.0,
  "dimension_scores": {{
    "factual_consistency": 0.0,
    "coverage": 0.0,
    "personalization": 0.0,
    "historical_memory_use": 0.0,
    "template_adherence": 0.0,
    "conciseness": 0.0
  }},
  "rationale": "独立判断理由",
  "remaining_issue_ids": [],
  "warnings": []
}}
"""


class JudgeAgent:
    """独立评价 Review 候选，只能对本轮已建立 Issue 作终止判断。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        model_input_audit: ModelInputAuditRecorder | None = None,
    ) -> None:
        """保存模型客户端和 Judge 参数，并初始化模型 trace。"""
        self._model_client = model_client
        self._config = config
        self._model_input_audit = model_input_audit
        self.traces: list[dict[str, Any]] = []

    async def judge(
        self,
        state: SummaryWorkingState,
        previous_candidate: str,
        candidate_summary: str,
        red_report: RedReviewReport,
        blue_result: BlueReviewResult | None,
        verification_results: tuple[ReviewVerificationResult, ...],
        *,
        candidate_index: int,
        round_index: int,
        previous_score: float,
    ) -> JudgeDecision:
        """调用 Judge 模型并过滤未知 remaining issue，返回带候选 hash 的决策。"""
        prompt = build_judge_prompt(
            state,
            previous_candidate,
            candidate_summary,
            red_report,
            blue_result,
            verification_results,
            round_index,
        )
        draft = await complete_validated_json(
            model_client=self._model_client,
            prompt=prompt,
            validator=self._validate_draft,
            max_tokens=self._config.review_judge_max_tokens,
            timeout_seconds=self._config.worker_timeout_seconds,
            repair_attempts=self._config.schema_repair_attempts,
            trace=self.traces,
            trace_metadata={"stage": "judge", "round_index": round_index},
            model_input_audit=self._model_input_audit,
        )
        known_issue_ids = {issue.issue_id for issue in red_report.issues}
        remaining = tuple(
            issue_id
            for issue_id in dict.fromkeys(draft.remaining_issue_ids)
            if issue_id in known_issue_ids
        )
        warnings = list(draft.warnings)
        unknown = set(draft.remaining_issue_ids) - set(remaining)
        if unknown:
            warnings.append(
                "Judge discarded unknown remaining issue ids: "
                + ", ".join(sorted(unknown))
            )
        dimensions = {
            name: float(draft.dimension_scores.get(name, draft.score))
            for name in JUDGE_DIMENSIONS
        }
        return JudgeDecision(
            round_index=round_index,
            candidate_index=candidate_index,
            disposition=draft.disposition,
            score=draft.score,
            rationale=draft.rationale,
            dimension_scores=dimensions,
            remaining_issue_ids=remaining,
            improvement=draft.score - previous_score,
            candidate_hash=hashlib.sha256(
                candidate_summary.encode("utf-8")
            ).hexdigest(),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _validate_draft(value: Any) -> JudgeDraftDecision:
        """把模型 JSON 严格校验为 Judge draft。"""
        return JudgeDraftDecision.model_validate(value)
