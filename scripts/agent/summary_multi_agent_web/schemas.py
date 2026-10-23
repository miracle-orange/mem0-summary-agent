"""定义 Web 请求、E2E 阶段、运行状态和 Inspector 结果契约。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from summary_multi_agent.schemas import (
    ContextCompressionMode,
    EvaluationMode,
    LiteEvaluationConfig,
    MinutesOutputMode,
    PlannerMode,
    RunConfig,
    SummaryRequest,
    WorkerMode,
)
from summary_multi_agent.tools.template import MeetingScene
from summary_multi_agent.profiles import CLOUD_FAST_DEFAULTS


# Web 请求/产物模型的字段构成外部契约；以下普通注释不改变 JSON Schema。
# Web 状态机向 UI 暴露的运行状态。
class WebRunStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLING = "cancelling"


class WebStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# Web 摘要请求及其 V2 Planner/Worker/Review/Evaluation 参数。
class WebSummarizeRequest(WebStrictModel):
    execution_profile: Literal["legacy", "cloud_fast"] = "legacy"
    current_material: str = Field(min_length=1)
    summary_request: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    source_time: str = ""
    target_entities: list[str] = Field(default_factory=list)
    summary_style: str = "structured"
    meeting_scene: MeetingScene | Literal["auto"] = "auto"
    minutes_output_mode: MinutesOutputMode = MinutesOutputMode.CLASSIC
    personalization_preferences: dict[str, Any] = Field(default_factory=dict)
    qdrant_path: str = ""
    collection: str = ""

    planner_mode: PlannerMode = PlannerMode.MODEL
    planner_max_tokens: int = Field(default=1600, ge=128)
    planner_repair_max_tokens: int = Field(default=1600, ge=128)
    planner_repair_attempts: int = Field(default=2, ge=0, le=5)
    planner_timeout_seconds: int = Field(default=180, ge=1)
    replanner_max_tokens: int = Field(default=1800, ge=128)
    replanner_repair_max_tokens: int = Field(default=1800, ge=128)
    replanner_repair_attempts: int = Field(default=2, ge=0, le=5)
    replanner_timeout_seconds: int = Field(default=180, ge=1)
    worker_mode: WorkerMode = WorkerMode.MODEL_WITH_RULE_FALLBACK
    worker_max_steps: int = Field(default=6, ge=1, le=32)
    worker_max_model_calls: int = Field(default=4, ge=2, le=16)
    worker_action_max_tokens: int = Field(default=3000, ge=128)
    worker_action_repair_attempts: int = Field(default=1, ge=0, le=5)
    worker_model_timeout_seconds: int = Field(default=180, ge=1)
    context_compression_mode: ContextCompressionMode = (
        ContextCompressionMode.DETERMINISTIC
    )
    context_compression_top_k_per_type: int = Field(default=10, ge=1, le=100)
    retrieval_top_k_per_plan_item: int = Field(default=10, ge=1, le=100)
    enable_review: bool = False
    max_review_rounds: int = Field(default=3, ge=0, le=10)
    max_replan_rounds: int = Field(default=2, ge=0, le=10)
    max_concurrent_workers: int = Field(default=4, ge=1, le=32)
    max_concurrent_segment_maps: int = Field(default=2, ge=1, le=16)
    max_concurrent_model_calls: int = Field(default=1, ge=1, le=16)
    worker_timeout_seconds: int = Field(default=180, ge=1)
    global_timeout_seconds: int = Field(default=600, ge=1)
    model_context_window_tokens: int | None = Field(default=None, ge=1024)
    model_context_safety_margin_tokens: int = Field(default=1024, ge=0)
    planner_max_tasks: int = Field(default=16, ge=1, le=16)
    planner_max_depth: int = Field(default=8, ge=1, le=16)
    planner_max_tasks_per_capability: int = Field(default=4, ge=1, le=16)
    planner_max_total_model_calls: int = Field(default=64, ge=0)
    planner_max_total_tool_calls: int = Field(default=2048, ge=0)
    planner_max_task_timeout_seconds: int = Field(default=600, ge=1)
    raw_segment_max_chars: int = Field(default=3000, ge=100)
    raw_segment_overlap_ratio: float = Field(default=0.05, ge=0.0, lt=0.5)
    current_map_max_tokens: int = Field(default=2400, ge=128)
    summary_max_tokens: int = Field(default=2400, ge=128)
    rich_summary_max_tokens: int = Field(default=6000, ge=256)
    temporal_max_tool_calls: int = Field(default=128, ge=1, le=1024)
    schema_repair_attempts: int = Field(default=2, ge=0, le=10)
    tool_retry_attempts: int = Field(default=2, ge=1, le=10)
    review_red_max_tokens: int = Field(default=2400, ge=128)
    review_red_max_candidates_per_audit: int = Field(default=6, ge=1, le=16)
    review_red_adjudication_max_tokens: int = Field(default=1600, ge=128)
    review_red_adjudication_batch_size: int = Field(default=3, ge=1, le=16)
    review_blue_max_tokens: int = Field(default=3000, ge=128)
    review_judge_max_tokens: int = Field(default=1600, ge=128)
    review_max_concurrent_segment_audits: int = Field(default=1, ge=1, le=16)
    review_max_issues_per_round: int = Field(default=6, ge=1, le=32)
    review_accept_score: float = Field(default=0.85, ge=0.0, le=1.0)
    review_min_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    review_text_change_convergence_threshold: float = Field(
        default=0.02, ge=0.0, le=1.0
    )
    enable_evaluation: bool = False
    evaluation_mode: EvaluationMode = EvaluationMode.STRICT
    lite_evaluation: LiteEvaluationConfig = Field(default_factory=LiteEvaluationConfig)
    evaluation_segment_max_chars: int = Field(default=1000, ge=100)
    evaluation_source_fact_max_tokens: int = Field(default=3000, ge=128, le=3000)
    evaluation_summary_fact_max_tokens: int = Field(default=3000, ge=128, le=3000)
    evaluation_match_max_tokens: int = Field(default=3000, ge=128, le=3000)
    evaluation_repair_attempts: int = Field(default=2, ge=0, le=5)
    evaluation_timeout_seconds: int = Field(default=600, ge=1)
    evaluation_model_timeout_seconds: int = Field(default=180, ge=1)
    evaluation_max_model_calls: int = Field(default=48, ge=1)
    evaluation_max_concurrent_segment_maps: int = Field(default=1, ge=1, le=16)
    evaluation_match_batch_size: int = Field(default=6, ge=1, le=50)
    evaluation_history_batch_size: int = Field(default=12, ge=1, le=50)
    evaluation_match_full_scan_threshold: int = Field(default=48, ge=1)
    evaluation_match_candidate_limit_per_fact: int = Field(default=12, ge=1)
    evaluation_match_candidate_union_limit: int = Field(default=48, ge=1)
    evaluation_personalization_batch_size: int = Field(default=24, ge=1, le=50)
    evaluation_personalization_max_tokens: int = Field(default=2400, ge=128, le=3000)
    evaluation_personalization_max_model_calls: int = Field(default=24, ge=1)
    evaluation_personalization_timeout_seconds: int = Field(default=300, ge=1)

    @model_validator(mode="before")
    @classmethod
    def apply_execution_profile(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("execution_profile") != "cloud_fast":
            return value
        result = dict(value)
        for key, default in CLOUD_FAST_DEFAULTS.items():
            result.setdefault(key, default)
        result.setdefault("lite_evaluation", LiteEvaluationConfig.cloud_fast())
        return result

    @field_validator("current_material", "summary_request", "user_id", "project")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """拒绝空白请求字段，避免后台任务启动后才暴露输入错误。"""
        if not value.strip():
            raise ValueError("text fields must contain non-whitespace text")
        return value

    @field_validator("summary_style")
    @classmethod
    def validate_summary_style(cls, value: str) -> str:
        """将摘要风格限制为核心 Tool 支持的两种值。"""
        normalized = value.strip()
        if normalized not in {"structured", "natural"}:
            raise ValueError("summary_style must be structured or natural")
        return normalized

    @field_validator("target_entities")
    @classmethod
    def normalize_target_entities(cls, value: list[str]) -> list[str]:
        """清理目标实体空白项，保持 Web 到核心请求的输入稳定。"""
        return [item.strip() for item in value if item.strip()]


# 一份可供 Ingest 的历史原始资料。
class WebHistorySource(WebStrictModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_path: str | None = None
    source_time: str | None = None

    @field_validator("label", "text")
    @classmethod
    def reject_blank_source_fields(cls, value: str) -> str:
        """拒绝空白历史资料标签或正文。"""
        if not value.strip():
            raise ValueError("history source fields must contain non-whitespace text")
        return value


class WebIngestConfig(WebStrictModel):
    """历史资料 Typed Memory Ingest 的请求级配置。"""

    chunk_char_limit: int | None = Field(default=None, ge=100)
    chunk_max_tokens: int | None = Field(default=None, ge=128)
    merge_max_tokens: int | None = Field(default=None, ge=128)
    typed_extraction_max_tokens: int | None = Field(default=None, ge=256)
    dedup_judge_max_tokens: int | None = Field(default=None, ge=256)
    dedup_judge_max_candidates: int | None = Field(default=None, ge=1)


# 独立历史 Ingest 请求及 Store identity。
class WebIngestRequest(WebStrictModel):
    execution_profile: Literal["legacy", "cloud_fast"] = "legacy"
    history_sources: list[WebHistorySource] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    source_time: str = ""
    source_type: str = "history"
    qdrant_path: str = ""
    collection: str = ""
    config: WebIngestConfig = Field(default_factory=WebIngestConfig)

    @field_validator("user_id", "project", "source_type")
    @classmethod
    def reject_blank_identity(cls, value: str) -> str:
        """拒绝空白 Ingest 身份和来源类型。"""
        if not value.strip():
            raise ValueError("identity fields must contain non-whitespace text")
        return value


class WebIngestResult(WebStrictModel):
    """独立历史入库运行的可审计结果。"""

    status: WebRunStatus
    typed_memory_count: int = 0
    add_call_count: int = 0
    operation_count: int = 0
    chunk_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    store_probe: dict[str, Any] = Field(default_factory=dict)
    readback: dict[str, Any] = Field(default_factory=dict)


# E2E Ingest 或 Ingest 后 Summarize 的组合请求。
class WebE2ERequest(WebStrictModel):
    execution_profile: Literal["legacy", "cloud_fast"] = "legacy"
    run_type: Literal["e2e_ingest", "e2e_run"]
    history_sources: list[WebHistorySource] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    source_time: str = ""
    summary: WebSummarizeRequest | None = None
    source_type: str = "history"
    qdrant_path: str = ""
    collection: str = ""

    @model_validator(mode="after")
    def validate_e2e_shape(self) -> "WebE2ERequest":
        """确保 E2E 的 Ingest/Summarize 身份、时间和 Store 配置一致。"""
        if not self.user_id.strip() or not self.project.strip():
            raise ValueError("E2E identity fields must contain non-whitespace text")
        if not self.source_type.strip():
            raise ValueError("source_type must contain non-whitespace text")
        if self.run_type == "e2e_run" and self.summary is None:
            raise ValueError("e2e_run requires a summarize request")
        if self.summary is not None:
            if (
                self.summary.user_id != self.user_id
                or self.summary.project != self.project
            ):
                raise ValueError("E2E summary and ingest identities must match")
            if (
                self.summary.qdrant_path
                and self.qdrant_path
                and (self.summary.qdrant_path != self.qdrant_path)
            ):
                raise ValueError("E2E identity has conflicting qdrant paths")
            if (
                self.summary.collection
                and self.collection
                and (self.summary.collection != self.collection)
            ):
                raise ValueError("E2E identity has conflicting collections")
        return self


def build_summary_request(web_request: WebSummarizeRequest) -> SummaryRequest:
    """将 Web 摘要请求映射为核心 SummaryRequest，并注入显式风格偏好。"""
    preferences = dict(web_request.personalization_preferences)
    preferences["summary_style"] = web_request.summary_style
    if web_request.meeting_scene != "auto" or "meeting_scene" not in preferences:
        preferences["meeting_scene"] = web_request.meeting_scene
    return SummaryRequest(
        current_material=web_request.current_material,
        source_time=web_request.source_time.strip() or None,
        summary_request=web_request.summary_request,
        target_entities=list(web_request.target_entities),
        personalization_preferences=preferences,
    )


def build_run_config(web_request: WebSummarizeRequest) -> RunConfig:
    """将 Web 运行参数逐项映射为 V2 RunConfig，不改变默认值或协议。"""
    return RunConfig(
        workflow_mode="multi_agent_v2",
        execution_profile=web_request.execution_profile,
        planner_mode=web_request.planner_mode,
        planner_max_tokens=web_request.planner_max_tokens,
        planner_repair_max_tokens=web_request.planner_repair_max_tokens,
        planner_repair_attempts=web_request.planner_repair_attempts,
        planner_timeout_seconds=web_request.planner_timeout_seconds,
        replanner_max_tokens=web_request.replanner_max_tokens,
        replanner_repair_max_tokens=web_request.replanner_repair_max_tokens,
        replanner_repair_attempts=web_request.replanner_repair_attempts,
        replanner_timeout_seconds=web_request.replanner_timeout_seconds,
        worker_mode=web_request.worker_mode,
        worker_max_steps=web_request.worker_max_steps,
        worker_max_model_calls=web_request.worker_max_model_calls,
        worker_action_max_tokens=web_request.worker_action_max_tokens,
        worker_action_repair_attempts=web_request.worker_action_repair_attempts,
        worker_model_timeout_seconds=web_request.worker_model_timeout_seconds,
        retrieval_top_k_per_plan_item=web_request.retrieval_top_k_per_plan_item,
        context_compression_mode=web_request.context_compression_mode,
        context_compression_top_k_per_type=(
            web_request.context_compression_top_k_per_type
        ),
        review_enabled=web_request.enable_review,
        review_shadow_mode=True,
        max_review_rounds=web_request.max_review_rounds,
        max_replan_rounds=web_request.max_replan_rounds,
        max_concurrent_workers=web_request.max_concurrent_workers,
        max_concurrent_segment_maps=web_request.max_concurrent_segment_maps,
        max_concurrent_model_calls=web_request.max_concurrent_model_calls,
        worker_timeout_seconds=web_request.worker_timeout_seconds,
        global_timeout_seconds=web_request.global_timeout_seconds,
        model_context_window_tokens=web_request.model_context_window_tokens,
        model_context_safety_margin_tokens=(
            web_request.model_context_safety_margin_tokens
        ),
        planner_max_tasks=web_request.planner_max_tasks,
        planner_max_depth=web_request.planner_max_depth,
        planner_max_tasks_per_capability=(web_request.planner_max_tasks_per_capability),
        planner_max_total_model_calls=web_request.planner_max_total_model_calls,
        planner_max_total_tool_calls=web_request.planner_max_total_tool_calls,
        planner_max_task_timeout_seconds=(web_request.planner_max_task_timeout_seconds),
        raw_segment_max_chars=web_request.raw_segment_max_chars,
        raw_segment_overlap_ratio=web_request.raw_segment_overlap_ratio,
        current_map_max_tokens=web_request.current_map_max_tokens,
        summary_max_tokens=web_request.summary_max_tokens,
        minutes_output_mode=web_request.minutes_output_mode,
        rich_summary_max_tokens=web_request.rich_summary_max_tokens,
        temporal_max_tool_calls=web_request.temporal_max_tool_calls,
        schema_repair_attempts=web_request.schema_repair_attempts,
        tool_retry_attempts=web_request.tool_retry_attempts,
        review_red_max_tokens=web_request.review_red_max_tokens,
        review_red_max_candidates_per_audit=(
            web_request.review_red_max_candidates_per_audit
        ),
        review_red_adjudication_max_tokens=(
            web_request.review_red_adjudication_max_tokens
        ),
        review_red_adjudication_batch_size=(
            web_request.review_red_adjudication_batch_size
        ),
        review_blue_max_tokens=web_request.review_blue_max_tokens,
        review_judge_max_tokens=web_request.review_judge_max_tokens,
        review_max_concurrent_segment_audits=(
            web_request.review_max_concurrent_segment_audits
        ),
        review_max_issues_per_round=web_request.review_max_issues_per_round,
        review_accept_score=web_request.review_accept_score,
        review_min_improvement=web_request.review_min_improvement,
        review_text_change_convergence_threshold=(
            web_request.review_text_change_convergence_threshold
        ),
        evaluation_enabled=web_request.enable_evaluation,
        evaluation_mode=web_request.evaluation_mode,
        lite_evaluation=web_request.lite_evaluation,
        evaluation_segment_max_chars=web_request.evaluation_segment_max_chars,
        evaluation_source_fact_max_tokens=(
            web_request.evaluation_source_fact_max_tokens
        ),
        evaluation_summary_fact_max_tokens=(
            web_request.evaluation_summary_fact_max_tokens
        ),
        evaluation_match_max_tokens=web_request.evaluation_match_max_tokens,
        evaluation_repair_attempts=web_request.evaluation_repair_attempts,
        evaluation_timeout_seconds=web_request.evaluation_timeout_seconds,
        evaluation_model_timeout_seconds=(web_request.evaluation_model_timeout_seconds),
        evaluation_max_model_calls=web_request.evaluation_max_model_calls,
        evaluation_max_concurrent_segment_maps=(
            web_request.evaluation_max_concurrent_segment_maps
        ),
        evaluation_match_batch_size=web_request.evaluation_match_batch_size,
        evaluation_history_batch_size=web_request.evaluation_history_batch_size,
        evaluation_match_full_scan_threshold=(
            web_request.evaluation_match_full_scan_threshold
        ),
        evaluation_match_candidate_limit_per_fact=(
            web_request.evaluation_match_candidate_limit_per_fact
        ),
        evaluation_match_candidate_union_limit=(
            web_request.evaluation_match_candidate_union_limit
        ),
        evaluation_personalization_batch_size=(
            web_request.evaluation_personalization_batch_size
        ),
        evaluation_personalization_max_tokens=(
            web_request.evaluation_personalization_max_tokens
        ),
        evaluation_personalization_max_model_calls=(
            web_request.evaluation_personalization_max_model_calls
        ),
        evaluation_personalization_timeout_seconds=(
            web_request.evaluation_personalization_timeout_seconds
        ),
    )


# WebRunManager 的运行时/持久化状态快照。
class WebRunSnapshot(WebStrictModel):
    run_id: str
    run_type: str = "multi_agent_summarize"
    status: WebRunStatus
    stage: str = ""
    output_dir: Path
    started_at: str | None = None
    finished_at: str | None = None
    summary: str = ""
    recommended_candidate_index: int | None = None
    evaluation_status: str = "disabled"
    personalization_evaluation_status: str = "disabled"
    warnings: list[str] = Field(default_factory=list)
    error: str = ""
    cancellation_requested: bool = False


# Inspector 使用的脱敏结构化审计快照。
class WebArtifactSnapshot(WebStrictModel):
    run_id: str = ""
    output_dir: Path
    status: WebRunStatus | None = None
    formal_summary: str = ""
    shadow_candidates: list[dict[str, Any]] = Field(default_factory=list)
    planner: dict[str, Any] = Field(default_factory=dict)
    workers: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    compression: dict[str, Any] = Field(default_factory=dict)
    store_probe: dict[str, Any] = Field(default_factory=dict)
    ingest: dict[str, Any] = Field(default_factory=dict)
    replanning: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    minutes_document: dict[str, Any] = Field(default_factory=dict)
    minutes_sources: dict[str, Any] = Field(default_factory=dict)
    minutes_manifest: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


# 单个运行阶段的只读展示状态。
class VisualizationStage(WebStrictModel):
    stage_id: str
    label: str
    status: str = "pending"
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int | None = None
    detail: str = ""
    counters: dict[str, Any] = Field(default_factory=dict)


# 最终有效计划中的 Agent 节点投影。
class AgentGraphNode(WebStrictModel):
    task_id: str
    task_type: str = "unknown"
    description: str = ""
    status: str = "pending"
    critical: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    duration_ms: int | None = None
    model_call_count: int | None = None
    tool_call_count: int | None = None
    evidence_count: int | None = None
    warning_count: int | None = None
    error: str = ""
    fallback_reason: str = ""
    consumed_result_ids: list[str] = Field(default_factory=list)
    preserved: bool = False
    replanned: bool = False


# Planner 声明依赖及其实际消费状态。
class AgentGraphEdge(WebStrictModel):
    source_task_id: str
    destination_task_id: str
    consumed: bool = False


# 执行 trace 中经白名单投影后的时间线项目。
class VisualizationTimelineItem(WebStrictModel):
    item_id: str
    label: str
    category: str
    status: str = "success"
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int | None = None
    task_id: str = ""


# Evidence/Memory 漏斗的单个计数节点；None 表示产物不可用。
class VisualizationFlowMetric(WebStrictModel):
    metric_id: str
    label: str
    value: int | None = None
    detail: str = ""


# 单次运行的统一只读可视化投影。
class WebRunVisualization(WebStrictModel):
    run_id: str = ""
    run_type: str = "unknown"
    output_dir: Path
    overview: dict[str, Any] = Field(default_factory=dict)
    stages: list[VisualizationStage] = Field(default_factory=list)
    graph_nodes: list[AgentGraphNode] = Field(default_factory=list)
    graph_edges: list[AgentGraphEdge] = Field(default_factory=list)
    plan_history: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[VisualizationTimelineItem] = Field(default_factory=list)
    memory_flow: list[VisualizationFlowMetric] = Field(default_factory=list)
    evidence_rows: list[dict[str, Any]] = Field(default_factory=list)
    review: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


# E2E 顶层结果，汇总 Ingest 与可选 Summarize 阶段。
class WebE2ERunResult(WebStrictModel):
    status: WebRunStatus
    final_summary: str = ""
    recommended_candidate_index: int | None = None
    evaluation_status: str = "disabled"
    personalization_evaluation_status: str = "disabled"
    warnings: list[str] = Field(default_factory=list)
    ingest: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


def status_from_runtime(runtime_status: str) -> WebRunStatus:
    """把核心 Pipeline 状态投影为 Web 可展示的运行状态。"""
    mapping = {
        "pending": WebRunStatus.PENDING,
        "planning": WebRunStatus.PLANNING,
        "executing": WebRunStatus.EXECUTING,
        "collecting": WebRunStatus.EXECUTING,
        "replanning": WebRunStatus.PLANNING,
        "synthesizing": WebRunStatus.EXECUTING,
        "reviewing": WebRunStatus.EXECUTING,
        "evaluating": WebRunStatus.EVALUATING,
        "completed": WebRunStatus.COMPLETED,
        "completed_with_warnings": WebRunStatus.COMPLETED_WITH_WARNINGS,
        "failed": WebRunStatus.FAILED,
    }
    return mapping.get(runtime_status, WebRunStatus.RUNNING)
