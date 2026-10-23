"""通过只读 Worker/Tool 为 Review Issue 补充可追溯证据。"""

from __future__ import annotations

import re

from ..orchestrator import SummaryOrchestrator
from ..schemas import (
    AgentStatus,
    ReviewIssue,
    ReviewVerificationResult,
    RunConfig,
    SubTask,
    SummaryWorkingState,
    TaskType,
    WorkerMode,
)


class ReviewVerifier:
    """通过受限 ad-hoc Worker/Tool 为 Review Issue 补充可追溯 Evidence。"""

    def __init__(
        self,
        *,
        orchestrator: SummaryOrchestrator,
        config: RunConfig,
    ) -> None:
        """绑定主 Orchestrator 和用于构造验证任务的运行配置。"""
        self._orchestrator = orchestrator
        self._config = config

    async def verify(
        self,
        state: SummaryWorkingState,
        issue: ReviewIssue,
        *,
        round_index: int,
    ) -> ReviewVerificationResult:
        """按 issue 路由执行验证，并将新 Evidence 写入 M4 索引。"""
        if not issue.needs_verification:
            return ReviewVerificationResult(
                issue_id=issue.issue_id,
                route_to=issue.route_to,
                status="not_required",
                evidence_ids=issue.evidence_ids,
            )

        task = self._build_task(issue, round_index, state=state)
        if task is None:
            if issue.evidence_ids:
                return ReviewVerificationResult(
                    issue_id=issue.issue_id,
                    route_to=issue.route_to,
                    status="verified",
                    evidence_ids=issue.evidence_ids,
                    warning="verification used already-grounded current evidence",
                )
            return ReviewVerificationResult(
                issue_id=issue.issue_id,
                route_to=issue.route_to,
                status="not_found",
                warning="no compatible Worker/Tool verification route is available",
            )

        result = await self._orchestrator.execute_ad_hoc_task(
            state,
            task,
            trace_scope=f"review_round_{round_index}",
        )
        if result.status != AgentStatus.SUCCESS:
            return ReviewVerificationResult(
                issue_id=issue.issue_id,
                route_to=issue.route_to,
                status="failed",
                warning=result.error or "verification task failed",
            )

        for evidence in result.evidence:
            state.evidence_index[evidence.evidence_id] = evidence
        evidence_ids = tuple(
            dict.fromkeys((*issue.evidence_ids, *(item.evidence_id for item in result.evidence)))
        )
        found = bool(result.evidence) or self._payload_has_result(result.payload)
        return ReviewVerificationResult(
            issue_id=issue.issue_id,
            route_to=issue.route_to,
            status="verified" if found else "not_found",
            evidence_ids=evidence_ids,
            payload={
                "task_id": result.task_id,
                "task_type": result.task_type.value,
                "is_empty": result.is_empty,
                "result": result.payload,
            },
            warning=(
                None if found else "verification route completed without supporting evidence"
            ),
        )

    def _build_task(
        self,
        issue: ReviewIssue,
        round_index: int,
        *,
        state: SummaryWorkingState | None = None,
    ) -> SubTask | None:
        """把可支持的 Issue route 转换成不进入主 DAG 的非关键任务。"""
        safe_issue_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", issue.issue_id)
        common = {
            "task_id": f"review_verify_{round_index}_{safe_issue_id}",
            "description": f"Verify review issue: {issue.description}",
            "critical": False,
            "timeout_seconds": self._config.worker_timeout_seconds,
            "max_model_calls": (
                0
                if self._config.worker_mode == WorkerMode.RULE
                else self._config.worker_max_model_calls
            ),
        }
        if issue.route_to == TaskType.SEMANTIC_MEMORY.value:
            query = issue.verification_query.strip() or issue.description
            return SubTask(
                **common,
                task_type=TaskType.SEMANTIC_MEMORY,
                allowed_tools=("typed_memory_search",),
                inputs={
                    "queries": [query],
                    "memory_types": [
                        "time",
                        "decision",
                        "constraint",
                        "task",
                        "risk",
                        "preference",
                    ],
                    "top_k_per_type": self._config.retrieval_top_k_per_plan_item,
                },
                max_tool_calls=1,
            )
        if issue.route_to == TaskType.ENTITY_PERSONALIZATION.value:
            return SubTask(
                **common,
                task_type=TaskType.ENTITY_PERSONALIZATION,
                allowed_tools=("entity_query",),
                inputs={"include_one_hop": True},
                max_tool_calls=1,
            )
        if issue.route_to == TaskType.TEMPLATE_ADAPTATION.value:
            return SubTask(
                **common,
                task_type=TaskType.TEMPLATE_ADAPTATION,
                allowed_tools=("template_adaptation",),
                inputs={
                    "requested_style": None,
                },
                max_tool_calls=1,
            )
        return None

    @staticmethod
    def _payload_has_result(payload: dict) -> bool:
        """判断验证 payload 是否含有非空的结构化支持结果。"""
        result_keys = ("records", "resolved", "matches")
        present_result_keys = [key for key in result_keys if key in payload]
        if present_result_keys:
            return any(
                isinstance(payload.get(key), list) and bool(payload[key])
                for key in present_result_keys
            )
        return bool(payload) and not bool(payload.get("is_empty"))
