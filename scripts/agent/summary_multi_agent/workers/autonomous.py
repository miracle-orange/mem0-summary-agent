"""将模型动作和 Observation 组装为受 Evidence 约束的领域结果。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from ..schemas import (
    AgentResult,
    AgentStatus,
    EvidenceRef,
    TaskType,
    ToolResult,
    ToolStatus,
    WorkerActionType,
    WorkerExecutionAudit,
    WorkerLoopOutcome,
    WorkerMode,
    WorkerObservation,
    WorkerTerminationReason,
    WorkerToolCallDraft,
)
from ..retrieval_records import assemble_retrieved_records
from .action_loop import WorkerActionLoop
from .base import BaseWorker, WorkerContext
from .entity_personalization import EntityPersonalizationWorker
from .semantic_memory import SemanticMemoryWorker
from .template_adaptation import TemplateAdaptationWorker
from .temporal_reasoning import TemporalReasoningWorker


SUCCESS_TERMINATIONS = frozenset(
    {
        WorkerTerminationReason.GOAL_SATISFIED,
        WorkerTerminationReason.LEGAL_EMPTY_RESULT,
        WorkerTerminationReason.PARTIAL_WITH_EVIDENCE,
    }
)


class WorkerResultAssemblyError(ValueError):
    """表示模型循环结果无法安全组装为领域 AgentResult。"""


class AutonomousWorker(BaseWorker):
    """执行一个 capability 的模型循环，并组装受 Evidence 约束的领域结果。"""

    def __init__(
        self,
        *,
        task_type: TaskType,
        action_loop: WorkerActionLoop,
        rule_worker: BaseWorker,
        fallback_to_rule: bool,
    ) -> None:
        """绑定任务类型、行动循环、规则 Worker 及是否允许规则 fallback。"""
        if rule_worker.task_type != task_type:
            raise ValueError("autonomous worker and rule fallback types must match")
        self.task_type = task_type
        self._action_loop = action_loop
        self._rule_worker = rule_worker
        self._fallback_to_rule = fallback_to_rule

    async def execute(self, context: WorkerContext) -> AgentResult:
        """运行模型 Action Loop；无法可信装配时按配置失败或降级到规则 Worker。"""
        self.validate_context(context)
        started = time.perf_counter()
        outcome = await self._action_loop.run(
            context,
            prior_results=context.prior_results,
        )
        try:
            result = self._assemble_result(context, outcome, started=started)
        except WorkerResultAssemblyError as exc:
            return await self._fallback_or_failure(
                context,
                outcome,
                reason=f"{type(exc).__name__}: {exc}",
                started=started,
            )
        if result is not None:
            return result
        return await self._fallback_or_failure(
            context,
            outcome,
            reason=outcome.error or outcome.termination_reason.value,
            started=started,
        )

    def _assemble_result(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        *,
        started: float,
    ) -> AgentResult | None:
        """仅从 terminal action 选中的成功 Observation 组装领域 AgentResult。"""
        if outcome.termination_reason not in SUCCESS_TERMINATIONS:
            return None
        if outcome.final_action is None:
            raise WorkerResultAssemblyError("successful loop has no terminal action")

        selected = self._selected_observations(outcome)
        if self.task_type == TaskType.SEMANTIC_MEMORY:
            result = self._assemble_semantic(context, outcome, selected)
        elif self.task_type == TaskType.ENTITY_PERSONALIZATION:
            result = self._assemble_entity(context, outcome, selected)
        elif self.task_type == TaskType.TEMPLATE_ADAPTATION:
            result = self._assemble_template(context, outcome, selected)
        elif self.task_type == TaskType.TEMPORAL_REASONING:
            result = self._assemble_temporal(context, outcome, selected)
        else:
            raise WorkerResultAssemblyError(
                f"unsupported autonomous task type: {self.task_type.value}"
            )

        if (
            outcome.final_action.action == WorkerActionType.COMPLETE_EMPTY
            and not result.is_empty
        ):
            raise WorkerResultAssemblyError(
                "complete_empty contradicts the assembled non-empty result"
            )

        consumed_evidence = self._consumed_prior_evidence(
            context, outcome.consumed_prior_evidence_ids
        )
        merged_evidence = tuple(
            {
                item.evidence_id: item
                for item in (*result.evidence, *consumed_evidence)
            }.values()
        )
        return result.model_copy(
            update={
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "evidence": merged_evidence,
                "consumed_result_ids": outcome.consumed_prior_result_ids,
                "consumed_evidence_ids": tuple(
                    item.evidence_id for item in consumed_evidence
                ),
                "worker_execution": self._audit(
                    context, outcome, WorkerMode.MODEL
                ),
            }
        )

    async def _fallback_or_failure(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        *,
        reason: str,
        started: float,
    ) -> AgentResult:
        """在模型结果不可用时执行受审计的规则 fallback，或返回失败/超时结果。"""
        if self._fallback_to_rule:
            fallback_new_tool_calls = 0
            if (
                self.task_type == TaskType.SEMANTIC_MEMORY
                and isinstance(self._rule_worker, SemanticMemoryWorker)
            ):
                reused_calls = self._semantic_search_calls(outcome)
                remaining_tool_calls = max(
                    0,
                    context.task.max_tool_calls
                    - outcome.final_budget.tool_calls_used,
                )
                remaining_timeout = max(
                    0.0,
                    context.task.timeout_seconds
                    - (time.perf_counter() - started),
                )
                fallback = await self._rule_worker.execute_recovery(
                    context,
                    reused_calls=reused_calls,
                    remaining_tool_calls=remaining_tool_calls,
                    remaining_timeout_seconds=remaining_timeout,
                )
                payload = fallback.payload if isinstance(fallback.payload, dict) else {}
                fallback_new_tool_calls = int(payload.get("new_search_count") or 0)
            else:
                fallback = await self._rule_worker.execute(context)
            warning = (
                "autonomous worker used rule fallback after "
                f"{outcome.termination_reason.value}: {reason}"
            )
            if context.task.dependencies:
                warning += (
                    "; rule fallback did not apply declared dependency results"
                )
            return fallback.model_copy(
                update={
                    "warnings": tuple(
                        dict.fromkeys((*fallback.warnings, warning))
                    ),
                    "duration_ms": int(
                        (time.perf_counter() - started) * 1000
                    ),
                    "worker_execution": self._audit(
                        context,
                        outcome,
                        WorkerMode.MODEL_WITH_RULE_FALLBACK,
                        fallback_reason=reason,
                        fallback_tool_calls_used=fallback_new_tool_calls,
                    ),
                }
            )

        status = (
            AgentStatus.TIMEOUT
            if outcome.termination_reason == WorkerTerminationReason.TIMEOUT
            else AgentStatus.FAILED
        )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=context.task.task_type,
            status=status,
            tool_results=tuple(
                item.tool_result for item in outcome.observations
            ),
            warnings=outcome.warnings,
            error=reason,
            duration_ms=int((time.perf_counter() - started) * 1000),
            consumed_result_ids=outcome.consumed_prior_result_ids,
            consumed_evidence_ids=tuple(
                item.evidence_id
                for item in self._consumed_prior_evidence(
                    context, outcome.consumed_prior_evidence_ids
                )
            ),
            worker_execution=self._audit(
                context, outcome, WorkerMode.MODEL
            ),
        )

    @staticmethod
    def _audit(
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        mode: WorkerMode,
        *,
        fallback_reason: str | None = None,
        fallback_tool_calls_used: int = 0,
    ) -> WorkerExecutionAudit:
        """把循环步骤、预算、Observation、依赖消费和 fallback 原因写入审计。"""
        selected = (
            outcome.final_action.selected_observation_ids
            if outcome.final_action is not None
            else ()
        )
        final_budget = outcome.final_budget
        if fallback_tool_calls_used:
            final_budget = final_budget.model_copy(
                update={
                    "tool_calls_used": (
                        final_budget.tool_calls_used + fallback_tool_calls_used
                    )
                }
            )
        return WorkerExecutionAudit(
            mode=mode,
            termination_reason=outcome.termination_reason,
            steps=outcome.steps,
            observations=outcome.observations,
            model_attempts=outcome.model_attempts,
            final_budget=final_budget,
            selected_observation_ids=selected,
            available_prior_result_ids=tuple(
                item.task_id for item in context.prior_results
            ),
            consumed_prior_result_ids=outcome.consumed_prior_result_ids,
            consumed_prior_evidence_ids=outcome.consumed_prior_evidence_ids,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _semantic_search_calls(
        outcome: WorkerLoopOutcome,
    ) -> tuple[tuple[dict[str, Any], ToolResult], ...]:
        """Recover real search arguments and results from completed model steps."""

        values: list[tuple[dict[str, Any], ToolResult]] = []
        for step in outcome.steps:
            if step.action.action != WorkerActionType.CALL_TOOL:
                continue
            for draft, result in zip(step.action.tool_calls, step.tool_results):
                if draft.tool_name != "typed_memory_search":
                    continue
                values.append((dict(draft.arguments), result))
        return tuple(values)

    @staticmethod
    def _consumed_prior_evidence(
        context: WorkerContext,
        consumed_evidence_ids: tuple[str, ...],
    ) -> tuple[EvidenceRef, ...]:
        """从已声明消费的前序结果中提取对应 Evidence，拒绝越权证据。"""
        consumed = set(consumed_evidence_ids)
        return tuple(
            {
                evidence.evidence_id: evidence
                for result in context.prior_results
                for evidence in result.evidence
                if evidence.evidence_id in consumed
            }.values()
        )

    @staticmethod
    def _selected_observations(
        outcome: WorkerLoopOutcome,
    ) -> tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...]:
        """按 terminal action 选择模型实际用于组装结果的工具 Observation。"""
        if outcome.final_action is None:
            return ()
        by_id: dict[str, tuple[WorkerObservation, WorkerToolCallDraft]] = {}
        observations = {item.observation_id: item for item in outcome.observations}
        for step in outcome.steps:
            if step.action.action != WorkerActionType.CALL_TOOL:
                continue
            for observation_id, draft in zip(
                step.observation_ids, step.action.tool_calls
            ):
                observation = observations[observation_id]
                by_id[observation_id] = (observation, draft)
        return tuple(
            by_id[observation_id]
            for observation_id in outcome.final_action.selected_observation_ids
            if observation_id in by_id
        )

    @staticmethod
    def _require_tool(
        selected: tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...],
        tool_name: str,
    ) -> tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...]:
        """确认选中的 Observation 全部来自指定工具且执行成功。"""
        if not selected:
            raise WorkerResultAssemblyError(
                f"terminal action did not select a {tool_name} observation"
            )
        wrong = [
            item.tool_result.tool_name
            for item, _draft in selected
            if item.tool_result.tool_name != tool_name
        ]
        if wrong:
            raise WorkerResultAssemblyError(
                f"selected observations contain unexpected tools: {wrong}"
            )
        if any(
            item.tool_result.status != ToolStatus.SUCCESS
            for item, _draft in selected
        ):
            raise WorkerResultAssemblyError(
                "selected observations must contain successful tool results"
            )
        return selected

    @staticmethod
    def _all_tool_results(outcome: WorkerLoopOutcome) -> tuple[ToolResult, ...]:
        """从循环 Observation 展开完整工具结果，供 AgentResult 审计保留。"""
        return tuple(item.tool_result for item in outcome.observations)

    @staticmethod
    def _warnings(
        outcome: WorkerLoopOutcome, *payloads: dict[str, Any]
    ) -> tuple[str, ...]:
        """合并循环和领域 payload warning，并保持首次出现顺序。"""
        values = list(outcome.warnings)
        for payload in payloads:
            raw = payload.get("warnings", [])
            if isinstance(raw, (list, tuple)):
                values.extend(str(item) for item in raw if str(item).strip())
        if outcome.termination_reason == WorkerTerminationReason.PARTIAL_WITH_EVIDENCE:
            values.append("autonomous worker returned a partial evidence-backed result")
        return tuple(dict.fromkeys(values))

    def _assemble_semantic(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        selected: tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...],
    ) -> AgentResult:
        """合并选中的 Typed Memory Observation，并按 Memory ID/分数生成 Evidence。"""
        selected = self._require_tool(selected, "typed_memory_search")
        record_occurrences: list[dict[str, Any]] = []
        queries: list[str] = []
        payloads: list[dict[str, Any]] = []
        for observation, draft in selected:
            payload = (
                observation.tool_result.payload
                if isinstance(observation.tool_result.payload, dict)
                else {}
            )
            payloads.append(payload)
            query = str(draft.arguments.get("query") or "").strip()
            if query:
                queries.append(query)
            for raw_record in payload.get("records", []):
                if not isinstance(raw_record, dict):
                    continue
                memory_id = str(raw_record.get("memory_id") or "").strip()
                if not memory_id:
                    continue
                record_occurrences.append(raw_record)
        available_record_ids = tuple(
            dict.fromkeys(
                str(record.get("memory_id") or "")
                for record in record_occurrences
                if str(record.get("memory_id") or "").strip()
            )
        )
        (
            _available_records,
            _available_merges,
            available_version_conflicts,
            available_retrieval_stats,
        ) = assemble_retrieved_records(record_occurrences)
        requested_memory_ids = outcome.final_action.selected_memory_ids
        selection_mode = "observation_all"
        if requested_memory_ids is not None:
            selection_mode = (
                "explicit_empty" if not requested_memory_ids else "record_ids"
            )
            requested = set(requested_memory_ids)
            record_occurrences = [
                record
                for record in record_occurrences
                if str(record.get("memory_id") or "") in requested
            ]
        (
            records,
            equivalent_merges,
            version_conflicts,
            retrieval_stats,
        ) = assemble_retrieved_records(
            record_occurrences
        )
        evidence = tuple(
            SemanticMemoryWorker._evidence_from_record(record)
            for record in records
        )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=context.task.task_type,
            status=AgentStatus.SUCCESS,
            is_empty=not records,
            payload={
                "queries": list(dict.fromkeys(queries)),
                "records": records,
                "selection_mode": selection_mode,
                "selected_observation_ids": list(
                    outcome.final_action.selected_observation_ids
                ),
                "available_record_ids": list(available_record_ids),
                "requested_memory_ids": (
                    list(requested_memory_ids)
                    if requested_memory_ids is not None
                    else None
                ),
                "selected_memory_ids": [
                    memory_id
                    for memory_id in dict.fromkeys(
                        str(record.get("memory_id") or "")
                        for record in records
                    )
                    if memory_id
                ],
                "equivalent_record_merges": equivalent_merges,
                "record_version_conflicts": version_conflicts,
                "retrieval_stats": retrieval_stats,
                "available_retrieval_stats": available_retrieval_stats,
                "available_record_version_conflicts": (
                    available_version_conflicts
                ),
                "retrieval_complete": True,
                "execution_origin": "model_selection",
                "record_selection_completed": (
                    requested_memory_ids is not None
                ),
                "record_selection_origin": (
                    "model" if requested_memory_ids is not None else None
                ),
            },
            evidence=evidence,
            tool_results=self._all_tool_results(outcome),
            warnings=self._warnings(outcome, *payloads),
        )


    def _assemble_entity(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        selected: tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...],
    ) -> AgentResult:
        """合并目标实体的历史 Observation 与当前原文 mentions。"""
        targets = tuple(
            dict.fromkeys(
                value.strip()
                for value in context.request.target_entities
                if value.strip()
            )
        )
        current_mentions, current_evidence = (
            EntityPersonalizationWorker._find_current_mentions(context, targets)
        )
        payloads: list[dict[str, Any]] = []
        records_by_id: dict[str, dict[str, Any]] = {}
        related: list[str] = []
        if selected:
            for observation, _draft in self._require_tool(
                selected, "entity_query"
            ):
                payload = (
                    observation.tool_result.payload
                    if isinstance(observation.tool_result.payload, dict)
                    else {}
                )
                payloads.append(payload)
                related.extend(
                    str(item)
                    for item in payload.get("related_entities", [])
                    if str(item).strip()
                )
                for raw_record in payload.get("records", []):
                    if not isinstance(raw_record, dict):
                        continue
                    memory_id = str(raw_record.get("memory_id") or "").strip()
                    if not memory_id:
                        continue
                    current = records_by_id.get(memory_id)
                    if current is None or float(
                        raw_record.get("score") or 0.0
                    ) > float(current.get("score") or 0.0):
                        records_by_id[memory_id] = raw_record
        elif not current_mentions:
            raise WorkerResultAssemblyError(
                "entity task has neither selected observations nor current mentions"
            )
        records = sorted(
            records_by_id.values(),
            key=lambda item: (
                -float(item.get("score") or 0.0),
                str(item.get("memory_id") or ""),
            ),
        )
        long_term_evidence = tuple(
            EntityPersonalizationWorker._long_term_evidence(record)
            for record in records
        )
        is_empty = not records and not current_mentions
        boundary_warnings: list[str] = []
        if (
            outcome.final_action is not None
            and outcome.final_action.action == WorkerActionType.COMPLETE_EMPTY
            and not is_empty
        ):
            if current_mentions and not records:
                boundary_warnings.append(
                    "当前材料包含目标实体证据，但长期 Memory 为空；保留当前证据并完成实体任务。"
                )
            else:
                raise WorkerResultAssemblyError(
                    "complete_empty contradicts grounded entity mentions"
                )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=context.task.task_type,
            status=AgentStatus.SUCCESS,
            is_empty=is_empty,
            payload={
                "target_entities": list(targets),
                "records": records,
                "related_entities": list(dict.fromkeys(related)),
                "current_mentions": current_mentions,
                "processed_segment_ids": [
                    segment.segment_id for segment in context.raw_material.segments
                ],
            },
            evidence=(*current_evidence, *long_term_evidence),
            tool_results=self._all_tool_results(outcome),
            warnings=self._warnings(outcome, *payloads, {"warnings": boundary_warnings}),
        )

    def _assemble_template(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        selected: tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...],
    ) -> AgentResult:
        """验证唯一模板 Observation，并将其 payload 投影为成功结果。"""
        selected = self._require_tool(selected, "template_adaptation")
        if len(selected) != 1:
            raise WorkerResultAssemblyError(
                "template task must select exactly one observation"
            )
        payload = selected[0][0].tool_result.payload
        if not isinstance(payload, dict) or not payload.get("template_name"):
            raise WorkerResultAssemblyError(
                "template observation does not contain a valid template"
            )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=context.task.task_type,
            status=AgentStatus.SUCCESS,
            payload=payload,
            tool_results=self._all_tool_results(outcome),
            warnings=self._warnings(outcome, payload),
        )

    def _assemble_temporal(
        self,
        context: WorkerContext,
        outcome: WorkerLoopOutcome,
        selected: tuple[tuple[WorkerObservation, WorkerToolCallDraft], ...],
    ) -> AgentResult:
        """核对每个 Calendar Observation 与确定性时间表达式，再生成时间结果。"""
        expressions, evidence = TemporalReasoningWorker()._find_expressions(context)
        processed_segment_ids = TemporalReasoningWorker._processed_segment_ids(
            context, expressions
        )
        if not expressions:
            if outcome.final_action is not None and (
                outcome.final_action.action != WorkerActionType.COMPLETE_EMPTY
            ):
                raise WorkerResultAssemblyError(
                    "temporal task claimed work although no expression exists"
                )
            return AgentResult(
                task_id=context.task.task_id,
                task_type=context.task.task_type,
                status=AgentStatus.SUCCESS,
                is_empty=True,
                payload={
                    "expressions": [],
                    "resolved": [],
                    "processed_segment_ids": processed_segment_ids,
                },
                tool_results=self._all_tool_results(outcome),
                warnings=self._warnings(outcome),
            )
        if not context.request.source_time:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=context.task.task_type,
                status=AgentStatus.SUCCESS,
                payload={
                    "expressions": expressions,
                    "resolved": [],
                    "processed_segment_ids": processed_segment_ids,
                },
                evidence=evidence,
                tool_results=self._all_tool_results(outcome),
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *self._warnings(outcome),
                            "source_time is missing; relative expressions were preserved but not resolved",
                        )
                    )
                ),
            )

        selected = self._require_tool(selected, "calendar_resolve")
        remaining: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in expressions:
            remaining[(item["expression"], item["direction"])].append(item)
        resolved: list[dict[str, Any]] = []
        payloads: list[dict[str, Any]] = []
        for observation, draft in selected:
            arguments = draft.arguments
            expression = str(arguments.get("expression") or "")
            direction = str(arguments.get("direction") or "auto")
            if str(arguments.get("source_time") or "") != context.request.source_time:
                raise WorkerResultAssemblyError(
                    "calendar call changed the request source_time"
                )
            candidates = remaining.get((expression, direction), [])
            if not candidates:
                raise WorkerResultAssemblyError(
                    "calendar observation does not match an unresolved expression: "
                    f"{expression!r}/{direction}"
                )
            item = candidates.pop(0)
            payload = (
                observation.tool_result.payload
                if isinstance(observation.tool_result.payload, dict)
                else {}
            )
            if payload.get("original_expression") != expression:
                raise WorkerResultAssemblyError(
                    "calendar output expression does not match its tool call"
                )
            payloads.append(payload)
            combined = dict(item)
            combined["resolution"] = payload
            resolved.append(combined)

        unresolved_count = sum(len(items) for items in remaining.values())
        if unresolved_count:
            if context.task.critical or (
                outcome.final_action is not None
                and outcome.final_action.action == WorkerActionType.COMPLETE
            ):
                raise WorkerResultAssemblyError(
                    f"temporal worker left {unresolved_count} expressions unresolved"
                )
        warnings = list(self._warnings(outcome, *payloads))
        if unresolved_count:
            warnings.append(
                f"autonomous temporal worker left {unresolved_count} expressions unresolved"
            )
        return AgentResult(
            task_id=context.task.task_id,
            task_type=context.task.task_type,
            status=AgentStatus.SUCCESS,
            payload={
                "expressions": expressions,
                "resolved": sorted(
                    resolved,
                    key=lambda item: (
                        int(item.get("start_offset") or 0),
                        str(item.get("expression") or ""),
                    ),
                ),
                "processed_segment_ids": processed_segment_ids,
            },
            evidence=evidence,
            tool_results=self._all_tool_results(outcome),
            warnings=tuple(dict.fromkeys(warnings)),
        )


def build_autonomous_worker(
    *,
    task_type: TaskType,
    action_loop: WorkerActionLoop,
    fallback_to_rule: bool,
) -> AutonomousWorker:
    """为指定 TaskType 创建带对应规则 fallback 的自主 Worker。"""
    rule_workers: dict[TaskType, type[BaseWorker]] = {
        TaskType.SEMANTIC_MEMORY: SemanticMemoryWorker,
        TaskType.ENTITY_PERSONALIZATION: EntityPersonalizationWorker,
        TaskType.TEMPLATE_ADAPTATION: TemplateAdaptationWorker,
        TaskType.TEMPORAL_REASONING: TemporalReasoningWorker,
    }
    return AutonomousWorker(
        task_type=task_type,
        action_loop=action_loop,
        rule_worker=rule_workers[task_type](),
        fallback_to_rule=fallback_to_rule,
    )
