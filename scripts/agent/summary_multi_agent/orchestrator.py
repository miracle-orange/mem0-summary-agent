"""DAG Worker 编排器，负责依赖调度、失败传播、重规划和运行状态收集。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .agent_pool import AgentPool
from .dag import topological_layers
from .planner import (
    Planner,
    build_planner_input,
    build_worker_capabilities,
    validate_plan_against_input,
)
from .schemas import (
    AgentResult,
    AgentStatus,
    ExecutionPlan,
    RunConfig,
    RunStatus,
    SubTask,
    SummaryWorkingState,
    TaskRuntimeState,
    WorkerMode,
)
from .tools.registry import ToolRegistry
from .tracing import TraceRecorder
from .working_memory import save_working_state_checkpoint
from .workers.base import WorkerContext


class SummaryOrchestrator:
    """按 Planner DAG 安全调度 Worker，并维护 M4、失败传播和 Replanning。

    同一拓扑层的无依赖任务可并发执行；Worker、工具和模型的超时/取消都会
    经过 lease/finally 释放资源。成功结果、Evidence Lineage 和调用预算统一
    写入 Shared Working Memory，供下游任务和下一轮 Replanning 消费。
    """

    def __init__(
        self,
        *,
        planner: Planner,
        agent_pool: AgentPool,
        tool_registry: ToolRegistry,
        config: RunConfig,
        output_dir: Optional[Path] = None,
        trace_recorder: Optional[TraceRecorder] = None,
    ) -> None:
        """装配 Planner、Worker 池、Tool Registry、并发信号量和 Trace。"""
        self._planner = planner
        self._agent_pool = agent_pool
        self._tool_registry = tool_registry
        self._config = config
        self._output_dir = output_dir
        self._trace = trace_recorder or TraceRecorder(
            output_dir / "execution_trace.jsonl" if output_dir else None
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrent_workers)
        self._tool_capabilities = tool_registry.planner_capabilities
        self._worker_capabilities = build_worker_capabilities(
            agent_pool.registered_task_types,
            self._tool_capabilities,
            max_tasks_per_plan=config.planner_max_tasks_per_capability,
            execution_mode=config.worker_mode,
            min_model_calls=(
                0
                if config.worker_mode == WorkerMode.RULE
                else min(
                    config.worker_max_model_calls,
                    2 * (1 + config.worker_action_repair_attempts),
                )
            ),
            max_model_calls=(
                0
                if config.worker_mode == WorkerMode.RULE
                else config.worker_max_model_calls
            ),
        )

    async def run_workers(self, state: SummaryWorkingState) -> SummaryWorkingState:
        """运行完整 Worker 阶段，并把全局超时/异常转换为失败状态。"""
        await self._set_status(state, RunStatus.PLANNING)
        try:
            await asyncio.wait_for(
                self._run_worker_pipeline(state),
                timeout=self._config.global_timeout_seconds,
            )
        except asyncio.TimeoutError:
            state.errors.append(
                f"orchestrator exceeded global timeout of "
                f"{self._config.global_timeout_seconds} seconds"
            )
            await self._set_status(state, RunStatus.FAILED)
            await self._trace.emit(
                "run.global_timeout",
                run_id=state.run_id,
                payload={"timeout_seconds": self._config.global_timeout_seconds},
            )
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            await self._set_status(state, RunStatus.FAILED)
            await self._trace.emit(
                "run.failed",
                run_id=state.run_id,
                payload={"error": state.errors[-1]},
            )
        await self._checkpoint(state)
        return state

    async def execute_ad_hoc_task(
        self,
        state: SummaryWorkingState,
        task: SubTask,
        *,
        trace_scope: str,
    ) -> AgentResult:
        """执行受限的 Review/Verification 任务，但不改变主 DAG。"""

        await self._trace.emit(
            "task.ad_hoc_started",
            run_id=state.run_id,
            payload={
                "scope": trace_scope,
                "task_id": task.task_id,
                "task_type": task.task_type.value,
            },
        )
        try:
            async with self._semaphore:
                async with self._agent_pool.lease(task.task_type) as worker:
                    result = await asyncio.wait_for(
                        worker.execute(
                            WorkerContext(
                                run_id=state.run_id,
                                request=state.request,
                                raw_material=state.raw_material,
                                task=task,
                                tool_registry=self._tool_registry,
                                tool_retry_attempts=self._config.tool_retry_attempts,
                            )
                        ),
                        timeout=task.timeout_seconds,
                    )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status=AgentStatus.TIMEOUT,
                error=f"ad-hoc worker timed out after {task.timeout_seconds} seconds",
            )
        except Exception as exc:
            result = AgentResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        await self._trace.emit(
            "task.ad_hoc_finished",
            run_id=state.run_id,
            payload={
                "scope": trace_scope,
                "task_id": task.task_id,
                "status": result.status.value,
                "error": result.error,
                **self._worker_trace_payload(result),
            },
        )
        return result

    async def _run_worker_pipeline(self, state: SummaryWorkingState) -> None:
        """创建计划、执行 DAG，并在关键失败时有限调用 Replanner。"""
        planner_input = build_planner_input(
            request=state.request,
            raw_material=state.raw_material,
            worker_capabilities=self._worker_capabilities,
            tool_capabilities=self._tool_capabilities,
            config=self._config,
        )
        plan = validate_plan_against_input(
            await self._planner.create_plan(planner_input), planner_input
        )
        self._collect_planner_warnings(state)
        self._install_plan(state, plan)
        await self._trace.emit(
            "plan.created",
            run_id=state.run_id,
            payload={
                "plan_id": plan.plan_id,
                "source": plan.source.value,
                "plan_version": plan.plan_version,
                "task_ids": [t.task_id for t in plan.tasks],
            },
        )
        await self._checkpoint(state)

        while True:
            await self._set_status(state, RunStatus.EXECUTING)
            await self._execute_plan(state, plan)
            await self._set_status(state, RunStatus.COLLECTING)
            self._collect_evidence(state)
            critical_failures = self._critical_failures(plan, state.agent_results)
            if not critical_failures:
                self._collect_worker_warnings(state)
                self._record_noncritical_failures(state, plan)
                return
            if state.replan_round >= self._config.max_replan_rounds:
                self._collect_worker_warnings(state)
                state.errors.append(
                    "critical tasks failed after maximum replan rounds: "
                    + ", ".join(critical_failures)
                )
                await self._set_status(state, RunStatus.FAILED)
                return

            state.replan_round += 1
            await self._set_status(state, RunStatus.REPLANNING)
            await self._trace.emit(
                "plan.replanning",
                run_id=state.run_id,
                payload={
                    "round": state.replan_round,
                    "critical_failures": critical_failures,
                },
            )
            failure_reasons = tuple(
                self._render_failure_reason(task_id, state.agent_results)
                for task_id in critical_failures
            )
            current_task_ids = {task.task_id for task in plan.tasks}
            planner_input = build_planner_input(
                request=state.request,
                raw_material=state.raw_material,
                worker_capabilities=self._worker_capabilities,
                tool_capabilities=self._tool_capabilities,
                config=self._config,
                previous_plan=plan,
                results={
                    task_id: result
                    for task_id, result in state.agent_results.items()
                    if task_id in current_task_ids
                },
                plan_history=tuple(state.plan_history),
                task_attempts={
                    task_id: runtime.attempts
                    for task_id, runtime in state.task_states.items()
                    if task_id in current_task_ids
                },
                worker_model_calls_used=state.worker_model_calls_used,
                worker_tool_calls_used=state.worker_tool_calls_used,
                replan_round=state.replan_round,
                failure_reasons=failure_reasons,
            )
            plan = validate_plan_against_input(
                await self._planner.replan(planner_input), planner_input
            )
            self._collect_planner_warnings(state)
            self._install_plan(state, plan, preserve_existing=True)
            await self._trace.emit(
                "plan.replanned",
                run_id=state.run_id,
                payload={
                    "plan_id": plan.plan_id,
                    "parent_plan_id": plan.parent_plan_id,
                    "plan_version": plan.plan_version,
                    "replan_round": plan.replan_round,
                    "preserved_task_ids": list(plan.preserved_task_ids),
                    "task_ids": [task.task_id for task in plan.tasks],
                    "worker_model_calls_used": state.worker_model_calls_used,
                    "worker_tool_calls_used": state.worker_tool_calls_used,
                },
            )
            await self._checkpoint(state)

    def _install_plan(
        self,
        state: SummaryWorkingState,
        plan: ExecutionPlan,
        *,
        preserve_existing: bool = False,
    ) -> None:
        """安装当前计划，并在 Replanning 时保留成功任务及其结果。"""
        state.plan = plan
        if not any(item.plan_id == plan.plan_id for item in state.plan_history):
            state.plan_history.append(plan)
        if preserve_existing:
            current_task_ids = {task.task_id for task in plan.tasks}
            state.agent_results = {
                task_id: result
                for task_id, result in state.agent_results.items()
                if task_id in current_task_ids
            }
        new_states: dict[str, TaskRuntimeState] = {}
        for task in plan.tasks:
            existing = state.task_states.get(task.task_id) if preserve_existing else None
            existing_result = state.agent_results.get(task.task_id)
            same_task_type = existing is not None and existing.task_type == task.task_type
            if (
                existing
                and same_task_type
                and existing.status == AgentStatus.SUCCESS
                and existing_result is not None
                and existing_result.task_type == task.task_type
            ):
                new_states[task.task_id] = existing
            else:
                attempts = existing.attempts if existing else 0
                if existing_result is not None and existing_result.task_type != task.task_type:
                    state.agent_results.pop(task.task_id, None)
                new_states[task.task_id] = TaskRuntimeState(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    status=AgentStatus.PENDING,
                    attempts=attempts,
                    last_error=existing.last_error if existing else None,
                )
        state.task_states = new_states

    async def _execute_plan(
        self, state: SummaryWorkingState, plan: ExecutionPlan
    ) -> None:
        """按拓扑层执行计划，先把失败依赖转换为 SKIPPED 再启动可运行任务。"""
        for layer_index, layer in enumerate(topological_layers(plan)):
            pending = [
                task
                for task in layer
                if state.agent_results.get(task.task_id, None) is None
                or state.agent_results[task.task_id].status != AgentStatus.SUCCESS
            ]
            runnable: list[SubTask] = []
            for task in pending:
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency not in state.agent_results
                    or state.agent_results[dependency].status != AgentStatus.SUCCESS
                ]
                if failed_dependencies:
                    result = AgentResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        status=AgentStatus.SKIPPED,
                        error=(
                            "dependency tasks did not succeed: "
                            + ", ".join(failed_dependencies)
                        ),
                    )
                    self._store_result(state, task, result)
                else:
                    runnable.append(task)
            await self._trace.emit(
                "dag.layer_started",
                run_id=state.run_id,
                payload={
                    "layer_index": layer_index,
                    "task_ids": [task.task_id for task in runnable],
                },
            )
            if runnable:
                # 同层任务彼此无未完成依赖，可以并发；结果仍按 runnable 顺序写回以保持稳定审计。
                results = await asyncio.gather(
                    *(self._execute_task(state, task) for task in runnable)
                )
                for task, result in zip(runnable, results):
                    self._store_result(state, task, result)
            await self._checkpoint(state)

    async def _execute_task(
        self, state: SummaryWorkingState, task: SubTask
    ) -> AgentResult:
        """在 Worker lease 和任务 timeout 内执行一个任务并生成终态结果。"""
        runtime = state.task_states[task.task_id]
        state.task_states[task.task_id] = runtime.model_copy(
            update={"status": AgentStatus.RUNNING, "attempts": runtime.attempts + 1}
        )
        await self._trace.emit(
            "task.started",
            run_id=state.run_id,
            payload={
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "dependency_task_ids": list(task.dependencies),
                "dependency_evidence_count": sum(
                    len(state.agent_results[dependency].evidence)
                    for dependency in task.dependencies
                    if dependency in state.agent_results
                ),
            },
        )
        try:
            async with self._semaphore:
                async with self._agent_pool.lease(task.task_type) as worker:
                    result = await asyncio.wait_for(
                        worker.execute(
                            WorkerContext(
                                run_id=state.run_id,
                                request=state.request,
                                raw_material=state.raw_material,
                                task=task,
                                tool_registry=self._tool_registry,
                                tool_retry_attempts=self._config.tool_retry_attempts,
                                prior_results=tuple(
                                    state.agent_results[dependency]
                                    for dependency in task.dependencies
                                    if dependency in state.agent_results
                                ),
                            )
                        ),
                        timeout=task.timeout_seconds,
                    )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status=AgentStatus.TIMEOUT,
                error=f"worker timed out after {task.timeout_seconds} seconds",
            )
        except Exception as exc:
            result = AgentResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        await self._trace.emit(
            "task.finished",
            run_id=state.run_id,
            payload={
                "task_id": task.task_id,
                "status": result.status.value,
                "error": result.error,
                **self._worker_trace_payload(result),
            },
        )
        return result

    @staticmethod
    def _worker_trace_payload(result: AgentResult) -> dict[str, object]:
        """把 Worker 执行审计压缩为写入 DAG Trace 的统计投影。"""
        execution = result.worker_execution
        if execution is None:
            return {
                "worker_mode": WorkerMode.RULE.value,
                "consumed_prior_result_ids": list(result.consumed_result_ids),
                "consumed_evidence_ids": list(result.consumed_evidence_ids),
            }
        budget = execution.final_budget
        return {
            "worker_mode": execution.mode.value,
            "worker_termination": (
                execution.termination_reason.value
                if execution.termination_reason is not None
                else None
            ),
            "worker_steps": len(execution.steps),
            "worker_model_calls": (
                budget.model_calls_used if budget is not None else 0
            ),
            "worker_tool_calls": (
                budget.tool_calls_used if budget is not None else 0
            ),
            "worker_fallback_reason": execution.fallback_reason,
            "available_prior_result_ids": list(
                execution.available_prior_result_ids
            ),
            "consumed_prior_result_ids": list(
                execution.consumed_prior_result_ids
            ),
            "consumed_evidence_ids": list(result.consumed_evidence_ids),
        }

    @staticmethod
    def _store_result(
        state: SummaryWorkingState, task: SubTask, result: AgentResult
    ) -> None:
        """校验结果身份与 Evidence Lineage 后写入 M4 并累计调用预算。"""
        if result.task_id != task.task_id or result.task_type != task.task_type:
            raise ValueError(
                "worker result identity does not match its planned task: "
                f"expected=({task.task_id}, {task.task_type.value}), "
                f"actual=({result.task_id}, {result.task_type.value})"
            )
        dependency_ids = set(task.dependencies)
        unknown_consumed_results = set(result.consumed_result_ids) - dependency_ids
        if unknown_consumed_results:
            raise ValueError(
                "worker result consumed undeclared dependency results: "
                f"{sorted(unknown_consumed_results)}"
            )
        upstream_evidence_ids = {
            evidence.evidence_id
            for dependency in result.consumed_result_ids
            for evidence in state.agent_results[dependency].evidence
        }
        unknown_consumed_evidence = set(
            result.consumed_evidence_ids
        ) - upstream_evidence_ids
        if unknown_consumed_evidence:
            raise ValueError(
                "worker result consumed evidence outside its dependency results: "
                f"{sorted(unknown_consumed_evidence)}"
            )
        execution = result.worker_execution
        if execution is not None:
            if set(execution.available_prior_result_ids) != dependency_ids:
                raise ValueError(
                    "worker execution prior-result scope does not match task dependencies"
                )
            if execution.mode == WorkerMode.MODEL:
                if set(execution.consumed_prior_result_ids) != set(
                    result.consumed_result_ids
                ):
                    raise ValueError(
                        "model worker result lineage does not match its execution audit"
                    )
                if set(execution.consumed_prior_evidence_ids) != set(
                    result.consumed_evidence_ids
                ):
                    raise ValueError(
                        "model worker evidence lineage does not match its execution audit"
                    )
                if (
                    result.status == AgentStatus.SUCCESS
                    and set(result.consumed_result_ids) != dependency_ids
                ):
                    raise ValueError(
                        "successful model worker did not consume every declared dependency"
                    )
        if execution is not None and execution.final_budget is not None:
            state.worker_model_calls_used += execution.final_budget.model_calls_used
            state.worker_tool_calls_used += max(
                execution.final_budget.tool_calls_used,
                len(result.tool_results),
            )
        else:
            state.worker_tool_calls_used += len(result.tool_results)
        state.agent_results[task.task_id] = result
        runtime = state.task_states[task.task_id]
        state.task_states[task.task_id] = runtime.model_copy(
            update={"status": result.status, "last_error": result.error}
        )

    @staticmethod
    def _collect_evidence(state: SummaryWorkingState) -> None:
        """把各 Worker 结果中的 Evidence 合并到运行级索引。"""
        evidence = dict(state.evidence_index)
        for result in state.agent_results.values():
            for item in result.evidence:
                evidence[item.evidence_id] = item
        state.evidence_index = evidence

    @staticmethod
    def _collect_worker_warnings(state: SummaryWorkingState) -> None:
        """把 Worker warning 按 task id 归并到运行级 warning 列表。"""
        for task_id, result in state.agent_results.items():
            for warning in result.warnings:
                rendered = f"{task_id}: {warning}"
                if rendered not in state.warnings:
                    state.warnings.append(rendered)

    def _collect_planner_warnings(self, state: SummaryWorkingState) -> None:
        """消费 Planner 的 fallback/修复 warning，避免重复写入状态。"""
        consume = getattr(self._planner, "consume_warnings", None)
        if not callable(consume):
            return
        for warning in consume():
            rendered = f"planner: {warning}"
            if rendered not in state.warnings:
                state.warnings.append(rendered)

    @staticmethod
    def _critical_failures(
        plan: ExecutionPlan, results: dict[str, AgentResult]
    ) -> list[str]:
        """返回当前计划中缺失或未成功的关键任务 ID。"""
        failures = []
        for task in plan.tasks:
            if not task.critical:
                continue
            result = results.get(task.task_id)
            if result is None or result.status != AgentStatus.SUCCESS:
                failures.append(task.task_id)
        return failures

    @staticmethod
    def _render_failure_reason(
        task_id: str, results: dict[str, AgentResult]
    ) -> str:
        """把任务失败结果渲染为 Replanning 可消费的简短诊断。"""
        result = results.get(task_id)
        if result is None:
            return f"{task_id}: missing result"
        detail = result.error or result.status.value
        return f"{task_id}: {detail}"

    @staticmethod
    def _record_noncritical_failures(
        state: SummaryWorkingState, plan: ExecutionPlan
    ) -> None:
        """将非关键任务失败记录为 warning，而不阻断完整摘要阶段。"""
        for task in plan.tasks:
            if task.critical:
                continue
            result = state.agent_results.get(task.task_id)
            if result is None or result.status != AgentStatus.SUCCESS:
                warning = f"non-critical task {task.task_id} did not succeed"
                if warning not in state.warnings:
                    state.warnings.append(warning)

    async def _set_status(
        self, state: SummaryWorkingState, status: RunStatus
    ) -> None:
        """更新 M4 运行状态并追加对应的状态变更 Trace。"""
        state.status = status
        await self._trace.emit(
            "run.status_changed",
            run_id=state.run_id,
            payload={"status": status.value},
        )

    async def _checkpoint(self, state: SummaryWorkingState) -> None:
        """在配置了输出目录时异步保存当前 M4 checkpoint。"""
        if self._output_dir is not None:
            await asyncio.to_thread(
                save_working_state_checkpoint, state, self._output_dir
            )
