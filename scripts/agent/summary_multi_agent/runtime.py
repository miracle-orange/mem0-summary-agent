"""装配 V2 生产运行时：Tool、Worker、Planner、Orchestrator、Summarizer 和 Review。"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

from .adapters import (
    ContextWindowModelClient,
    SyncCallableModelClient,
    build_artifact_board_from_state,
)
from .adapters.model import AsyncOpenAICompatibleModelClient
from .agent_pool import AgentPool
from .current_material import CurrentMaterialAnalyzer
from .evaluation import SummaryEvaluator
from .evaluation.lite import LiteSummaryEvaluator
from .model_input_audit import ModelInputAuditRecorder
from .orchestrator import SummaryOrchestrator
from .pipeline import MultiAgentSummaryPipeline
from .planner import ModelDrivenPlanner, RuleBasedPlanner
from .review.blue import BlueAgent
from .review.judge import JudgeAgent
from .review.loop import ReviewLoop
from .review.red import RedAgent
from .review.verify import ReviewVerifier
from .schemas import EvaluationMode, PlannerMode, RunConfig, TaskType, WorkerMode
from .synthesis import SummarySynthesizer
from .tools import (
    CalendarResolveTool,
    EntityQueryTool,
    TemplateAdaptationTool,
    ToolRegistry,
    TypedMemorySearchTool,
)
from .tracing import TraceRecorder
from .workers import (
    EntityPersonalizationWorker,
    SemanticMemoryWorker,
    TemplateAdaptationWorker,
    TemporalReasoningWorker,
    WorkerActionLoop,
    WorkerActionLoopConfig,
    build_autonomous_worker,
)


def build_multi_agent_v2_pipeline(
    *,
    memory: Any,
    llm_call: Any,
    user_id: str,
    project: str,
    output_dir: Path,
    config: RunConfig,
) -> MultiAgentSummaryPipeline:
    """装配 V2 的 Tool、Worker、Planner、Orchestrator、Review 和 Evaluation。

    ``memory`` 只通过只读工具暴露给 Summarize；实际长期 Memory 生命周期由外层
    CLI/Web 入口管理。返回的 Pipeline 共享一个模型适配器和 Trace，但不在装配
    阶段执行模型调用或写入长期 Store。
    """
    from summary_memory_llm import OPENAI_COMPATIBLE_BACKEND, call_summary_llm, get_llm_backend

    external_client = (
        AsyncOpenAICompatibleModelClient(trace_path=output_dir / "external_model_calls.jsonl")
        if llm_call is call_summary_llm and get_llm_backend() == OPENAI_COMPATIBLE_BACKEND
        else None
    )
    model_client = ContextWindowModelClient(
        external_client or SyncCallableModelClient(llm_call),
        context_window_tokens=config.model_context_window_tokens,
        safety_margin_tokens=config.model_context_safety_margin_tokens,
    )
    model_input_audit = ModelInputAuditRecorder(
        output_dir,
        context_window_tokens=config.model_context_window_tokens,
        safety_margin_tokens=config.model_context_safety_margin_tokens,
    )
    tool_registry = ToolRegistry(
        [
            TypedMemorySearchTool(memory, user_id=user_id, project=project),
            EntityQueryTool(memory, user_id=user_id, project=project),
            TemplateAdaptationTool(),
            CalendarResolveTool(),
        ]
    )
    worker_factories = {
        TaskType.SEMANTIC_MEMORY: SemanticMemoryWorker,
        TaskType.ENTITY_PERSONALIZATION: EntityPersonalizationWorker,
        TaskType.TEMPLATE_ADAPTATION: TemplateAdaptationWorker,
        TaskType.TEMPORAL_REASONING: TemporalReasoningWorker,
    }
    if config.worker_mode != WorkerMode.RULE:
        action_loop = WorkerActionLoop(
            model_client=model_client,
            config=WorkerActionLoopConfig(
                max_steps=config.worker_max_steps,
                max_model_calls=config.worker_max_model_calls,
                max_tool_calls=config.temporal_max_tool_calls,
                max_tokens=config.worker_action_max_tokens,
                repair_attempts=config.worker_action_repair_attempts,
                model_timeout_seconds=config.worker_model_timeout_seconds,
            ),
            model_semaphore=asyncio.Semaphore(
                config.max_concurrent_model_calls
            ),
            model_input_audit=model_input_audit,
        )
        worker_factories = {
            task_type: partial(
                build_autonomous_worker,
                task_type=task_type,
                action_loop=action_loop,
                fallback_to_rule=(
                    config.worker_mode
                    == WorkerMode.MODEL_WITH_RULE_FALLBACK
                ),
            )
            for task_type in worker_factories
        }
    agent_pool = AgentPool(
        worker_factories,
        max_total=config.max_concurrent_workers,
    )
    trace_recorder = TraceRecorder(output_dir / "execution_trace.jsonl")
    rule_planner = RuleBasedPlanner(
        retrieval_top_k_per_plan_item=config.retrieval_top_k_per_plan_item,
        temporal_max_tool_calls=config.temporal_max_tool_calls,
        worker_max_model_calls=(
            0
            if config.worker_mode == WorkerMode.RULE
            else config.worker_max_model_calls
        ),
        task_timeout_seconds=min(
            config.worker_timeout_seconds,
            config.planner_max_task_timeout_seconds,
        ),
    )
    planner = rule_planner
    if config.planner_mode == PlannerMode.MODEL:
        planner = ModelDrivenPlanner(
            model_client=model_client,
            fallback_planner=rule_planner,
            output_dir=output_dir,
            max_tokens=config.planner_max_tokens,
            repair_max_tokens=config.planner_repair_max_tokens,
            repair_attempts=config.planner_repair_attempts,
            timeout_seconds=config.planner_timeout_seconds,
            replanner_max_tokens=config.replanner_max_tokens,
            replanner_repair_max_tokens=config.replanner_repair_max_tokens,
            replanner_repair_attempts=config.replanner_repair_attempts,
            replanner_timeout_seconds=config.replanner_timeout_seconds,
            model_input_audit=model_input_audit,
        )
    orchestrator = SummaryOrchestrator(
        planner=planner,
        agent_pool=agent_pool,
        tool_registry=tool_registry,
        config=config,
        output_dir=output_dir,
        trace_recorder=trace_recorder,
    )
    analyzer = CurrentMaterialAnalyzer(model_client=model_client, config=config)
    synthesizer = SummarySynthesizer(
        model_client=model_client,
        config=config,
        model_input_audit=model_input_audit,
    )
    reviewer = None
    if config.review_enabled:
        reviewer = ReviewLoop(
            red_agent=RedAgent(
                model_client=model_client,
                config=config,
                model_input_audit=model_input_audit,
            ),
            blue_agent=BlueAgent(
                model_client=model_client,
                config=config,
                model_input_audit=model_input_audit,
            ),
            judge_agent=JudgeAgent(
                model_client=model_client,
                config=config,
                model_input_audit=model_input_audit,
            ),
            verifier=ReviewVerifier(orchestrator=orchestrator, config=config),
            config=config,
            trace_recorder=trace_recorder,
        )
    evaluator = None
    if config.evaluation_enabled:
        evaluator = (
            LiteSummaryEvaluator(
                model_client=model_client,
                config=config,
                output_dir=output_dir,
            )
            if config.evaluation_mode == EvaluationMode.LITE
            else SummaryEvaluator(model_client=model_client, config=config)
        )
    return MultiAgentSummaryPipeline(
        model_closer=external_client.aclose if external_client else None,
        orchestrator=orchestrator,
        current_material_analyzer=analyzer,
        synthesizer=synthesizer,
        config=config,
        output_dir=output_dir,
        trace_recorder=trace_recorder,
        artifact_board_builder=build_artifact_board_from_state,
        reviewer=reviewer,
        evaluator=evaluator,
    )
