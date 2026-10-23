"""V2 的请求、计划、Worker、Evidence、Review 和运行结果数据契约。"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id(prefix: str) -> str:
    """生成带业务前缀的运行内唯一 ID。"""
    return f"{prefix}_{uuid4().hex}"


# 基础模型统一禁止未知字段；冻结模型用于跨阶段不可变契约。
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class FrozenStrictModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, protected_namespaces=()
    )


# 下列枚举固定 V2 的工作流、计划来源、Worker 模式和状态协议。
class WorkflowMode(str, Enum):
    BASELINE = "baseline"
    MULTI_AGENT_V2 = "multi_agent_v2"


class TaskType(str, Enum):
    SEMANTIC_MEMORY = "semantic_memory"
    ENTITY_PERSONALIZATION = "entity_personalization"
    TEMPLATE_ADAPTATION = "template_adaptation"
    TEMPORAL_REASONING = "temporal_reasoning"


class PlanSource(str, Enum):
    MODEL = "model"
    MODEL_REPAIRED = "model_repaired"
    RULE_FALLBACK = "rule_fallback"
    REPLAN = "replan"


class PlannerMode(str, Enum):
    MODEL = "model"
    RULE = "rule"


class WorkerMode(str, Enum):
    RULE = "rule"
    MODEL = "model"
    MODEL_WITH_RULE_FALLBACK = "model_with_rule_fallback"


class ContextCompressionMode(str, Enum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"


class MinutesOutputMode(str, Enum):
    CLASSIC = "classic"
    RICH = "rich"


class ContextCompressionAction(str, Enum):
    RETAIN = "retain"
    MERGE = "merge"
    DROP = "drop"


class WorkerActionType(str, Enum):
    CALL_TOOL = "call_tool"
    COMPLETE = "complete"
    COMPLETE_EMPTY = "complete_empty"
    PARTIAL = "partial"
    REQUEST_REPLAN = "request_replan"
    FAIL = "fail"


class WorkerTerminationReason(str, Enum):
    GOAL_SATISFIED = "goal_satisfied"
    LEGAL_EMPTY_RESULT = "legal_empty_result"
    PARTIAL_WITH_EVIDENCE = "partial_with_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REPLAN_REQUESTED = "replan_requested"
    MODEL_BUDGET_EXHAUSTED = "model_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    STEP_LIMIT = "step_limit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SCHEMA_FAILURE = "schema_failure"
    TOOL_FAILURE = "tool_failure"
    CONTEXT_OVERFLOW = "context_overflow"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COLLECTING = "collecting"
    REPLANNING = "replanning"
    SYNTHESIZING = "synthesizing"
    REVIEWING = "reviewing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    DISABLED = "disabled"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationMode(str, Enum):
    STRICT = "strict"
    LITE = "lite"


class SourceKind(str, Enum):
    CURRENT_MATERIAL = "current_material"
    LONG_TERM_MEMORY = "long_term_memory"
    TOOL_OUTPUT = "tool_output"


class ReviewSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JudgeDisposition(str, Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    STOP_WITH_WARNINGS = "stop_with_warnings"


class ReviewAction(str, Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    VERIFY = "verify"
    MARK_UNCERTAIN = "mark_uncertain"
    REGENERATE_SECTION = "regenerate_section"


class ReviewTermination(str, Enum):
    ACCEPTED = "accepted"
    STOP_WITH_WARNINGS = "stop_with_warnings"
    MAX_ROUNDS = "max_rounds"
    CONVERGED = "converged"
    OSCILLATION = "oscillation"
    NO_ACTIONABLE_ISSUES = "no_actionable_issues"
    REVIEW_FAILED = "review_failed"


class CurrentFactCategory(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    TASK = "task"
    RISK = "risk"
    PREFERENCE = "preference"
    QUESTION = "question"


# 单个计划任务在 M4 中的运行状态，不代表最终 AgentResult。
class TaskRuntimeState(StrictModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    status: AgentStatus = AgentStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    last_error: Optional[str] = None


# 请求契约绑定完整当前原文、摘要目标、时间锚点和个性化范围。
class SummaryRequest(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.request.v2"] = (
        "summary_multi_agent.request.v2"
    )
    request_id: str = Field(default_factory=lambda: _new_id("req"), min_length=1)
    current_material: str = Field(min_length=1)
    source_time: Optional[str] = None
    summary_request: str = Field(
        default="生成忠实、可追溯的个性化摘要。", min_length=1
    )
    target_entities: list[str] = Field(default_factory=list)
    personalization_preferences: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_current_material(self) -> "SummaryRequest":
        """确保请求正文和摘要目标都不是空白文本。"""
        if not self.current_material.strip():
            raise ValueError("current_material must contain non-whitespace text")
        if not self.summary_request.strip():
            raise ValueError("summary_request must contain non-whitespace text")
        return self


# 原文分段保留全局字符 offset 和内容 hash，供 Coverage/Evidence 回溯。
class RawSegment(FrozenStrictModel):
    segment_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_offsets_and_hash(self) -> "RawSegment":
        """验证分段长度、边界和内容 hash 与原文切片一致。"""
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("segment offsets must match text length")
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if digest != self.content_sha256:
            raise ValueError("segment content_sha256 does not match text")
        return self


# 一次运行的完整当前原文及其稳定分段 manifest。
class RawMaterialBundle(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.raw_material.v2"] = (
        "summary_multi_agent.raw_material.v2"
    )
    source_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_chars: int = Field(gt=0)
    max_segment_chars: int = Field(gt=0)
    overlap_chars: int = Field(ge=0)
    segments: tuple[RawSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bundle_integrity(self) -> "RawMaterialBundle":
        """验证原文 hash、分段连续性、重叠关系和最终覆盖边界。"""
        if not self.raw_text.strip():
            raise ValueError("raw_text must contain non-whitespace text")
        if self.total_chars != len(self.raw_text):
            raise ValueError("total_chars does not match raw_text")
        digest = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
        if digest != self.content_sha256:
            raise ValueError("bundle content_sha256 does not match raw_text")
        if self.overlap_chars >= self.max_segment_chars:
            raise ValueError("overlap_chars must be smaller than max_segment_chars")

        previous_end = 0
        segment_ids: set[str] = set()
        for expected_ordinal, segment in enumerate(self.segments):
            if segment.ordinal != expected_ordinal:
                raise ValueError("segment ordinals must be contiguous and zero-based")
            if segment.segment_id in segment_ids:
                raise ValueError("segment_id values must be unique")
            if segment.source_id != self.source_id:
                raise ValueError("all segments must reference bundle source_id")
            if expected_ordinal == 0 and segment.start_offset != 0:
                raise ValueError("the first segment must start at offset 0")
            if expected_ordinal > 0 and segment.start_offset > previous_end:
                raise ValueError("raw material segments must not contain gaps")
            if expected_ordinal > 0:
                expected_start = previous_end - self.overlap_chars
                if segment.start_offset != expected_start:
                    raise ValueError("segment offsets do not match configured overlap")
            if len(segment.text) > self.max_segment_chars:
                raise ValueError("segment exceeds max_segment_chars")
            expected_text = self.raw_text[segment.start_offset : segment.end_offset]
            if segment.text != expected_text:
                raise ValueError("segment text does not match raw_text offsets")
            previous_end = segment.end_offset
            segment_ids.add(segment.segment_id)

        if previous_end != self.total_chars:
            raise ValueError("the final segment must reach the end of raw_text")
        return self


class SegmentCoverage(FrozenStrictModel):
    segment_id: str = Field(min_length=1)
    required_stages: tuple[str, ...] = Field(min_length=1)
    completed_stages: tuple[str, ...] = ()
    failed_stages: dict[str, str] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """返回该分段是否完成全部必需阶段。"""
        return set(self.required_stages).issubset(self.completed_stages)


class CoverageLedger(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.coverage_ledger.v2"] = (
        "summary_multi_agent.coverage_ledger.v2"
    )
    source_id: str = Field(min_length=1)
    segments: tuple[SegmentCoverage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_segments(self) -> "CoverageLedger":
        """确保 Coverage Ledger 不含重复分段 ID。"""
        ids = [item.segment_id for item in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("coverage ledger segment_id values must be unique")
        return self

    @property
    def is_complete(self) -> bool:
        """返回所有分段是否都完成必需阶段。"""
        return all(item.is_complete for item in self.segments)

    @property
    def incomplete_segment_ids(self) -> tuple[str, ...]:
        """返回尚未完整覆盖的分段 ID。"""
        return tuple(item.segment_id for item in self.segments if not item.is_complete)


# Planner 输出的最小执行单元，包含依赖、能力工具子集和调用预算。
class SubTask(FrozenStrictModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    description: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    critical: bool = False
    timeout_seconds: int = Field(default=180, ge=1)
    allowed_tools: tuple[str, ...] = ()
    inputs: dict[str, Any] = Field(default_factory=dict)
    max_model_calls: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "SubTask":
        """验证任务不能自依赖，且依赖/工具列表不重复。"""
        if self.task_id in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("task dependencies must be unique")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools must be unique")
        return self


# 完整可执行 DAG 及 Replanning lineage；由 Planner 生成、Orchestrator 执行。
class ExecutionPlan(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.plan.v2"] = (
        "summary_multi_agent.plan.v2"
    )
    plan_id: str = Field(default_factory=lambda: _new_id("plan"), min_length=1)
    source: PlanSource = PlanSource.RULE_FALLBACK
    plan_version: int = Field(default=1, ge=1)
    parent_plan_id: Optional[str] = None
    replan_round: int = Field(default=0, ge=0)
    replan_reason: Optional[str] = None
    preserved_task_ids: tuple[str, ...] = ()
    tasks: tuple[SubTask, ...] = Field(min_length=1, max_length=16)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_dag(self) -> "ExecutionPlan":
        """验证 task identity、依赖存在性、无环和 Replanning 元数据。"""
        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("task_id values must be unique")

        known_ids = set(task_by_id)
        for task in self.tasks:
            missing = set(task.dependencies) - known_ids
            if missing:
                raise ValueError(
                    f"task {task.task_id} has unknown dependencies: {sorted(missing)}"
                )

        indegree = {task_id: 0 for task_id in known_ids}
        dependents: dict[str, list[str]] = {task_id: [] for task_id in known_ids}
        for task in self.tasks:
            indegree[task.task_id] = len(task.dependencies)
            for dependency in task.dependencies:
                dependents[dependency].append(task.task_id)

        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            task_id = ready.pop(0)
            visited += 1
            for dependent in sorted(dependents[task_id]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
            ready.sort()
        if visited != len(self.tasks):
            raise ValueError("execution plan must be acyclic")
        if len(self.preserved_task_ids) != len(set(self.preserved_task_ids)):
            raise ValueError("preserved_task_ids must be unique")
        unknown_preserved = set(self.preserved_task_ids) - known_ids
        if unknown_preserved:
            raise ValueError(
                "preserved_task_ids must exist in the current plan: "
                f"{sorted(unknown_preserved)}"
            )
        if self.source == PlanSource.REPLAN:
            if not self.parent_plan_id:
                raise ValueError("replan requires parent_plan_id")
            if self.replan_round < 1:
                raise ValueError("replan requires replan_round >= 1")
            if not self.replan_reason or not self.replan_reason.strip():
                raise ValueError("replan requires a non-empty replan_reason")
        elif any(
            (
                self.parent_plan_id is not None,
                self.replan_round != 0,
                self.replan_reason is not None,
                bool(self.preserved_task_ids),
            )
        ):
            raise ValueError(
                "only replan plans may set parent/replan/preserved metadata"
            )
        return self


# 模型只输出的初始计划草案；系统随后补充正式 plan identity。
class ModelPlanDraft(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.model_plan_draft.v2"] = (
        "summary_multi_agent.model_plan_draft.v2"
    )
    tasks: tuple[SubTask, ...] = Field(min_length=1, max_length=16)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_as_execution_plan(self) -> "ModelPlanDraft":
        """借助 ExecutionPlan 校验模型草案的基本 DAG 形状。"""
        ExecutionPlan(
            source=PlanSource.MODEL,
            tasks=self.tasks,
            rationale=self.rationale,
        )
        return self


# 模型只输出的 Replanning 草案；父计划、版本和保留任务由系统生成。
class ModelReplanDraft(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.model_replan_draft.v2"] = (
        "summary_multi_agent.model_replan_draft.v2"
    )
    tasks: tuple[SubTask, ...] = Field(min_length=1, max_length=16)
    addressed_failure_task_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_as_execution_plan(self) -> "ModelReplanDraft":
        """验证失败任务声明唯一，并确认草案本身是合法 DAG。"""
        if len(self.addressed_failure_task_ids) != len(
            set(self.addressed_failure_task_ids)
        ):
            raise ValueError("addressed_failure_task_ids must be unique")
        ExecutionPlan(
            source=PlanSource.MODEL,
            tasks=self.tasks,
            rationale=self.rationale,
        )
        return self


# Worker capability 的 Planner 可见权限、执行模式和模型调用边界。
class WorkerCapabilitySpec(FrozenStrictModel):
    capability_id: TaskType
    description: str = Field(min_length=1)
    allowed_tools: tuple[str, ...] = ()
    max_tasks_per_plan: int = Field(default=4, ge=1, le=16)
    execution_mode: WorkerMode = WorkerMode.RULE
    min_model_calls: int = Field(default=0, ge=0, le=16)
    max_model_calls: int = Field(default=0, ge=0, le=16)

    @model_validator(mode="after")
    def validate_allowed_tools(self) -> "WorkerCapabilitySpec":
        """验证工具不重复，且 rule/model 模式的模型预算符合契约。"""
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("worker capability allowed_tools must be unique")
        if self.min_model_calls > self.max_model_calls:
            raise ValueError(
                "worker capability min_model_calls cannot exceed max_model_calls"
            )
        if self.execution_mode == WorkerMode.RULE:
            if self.min_model_calls != 0 or self.max_model_calls != 0:
                raise ValueError("rule worker capability cannot allocate model calls")
        elif self.min_model_calls < 2:
            raise ValueError(
                "model worker capability requires at least two model calls"
            )
        return self


# Tool Registry 暴露给 Planner 的工具名称、描述和允许任务类型。
class ToolCapabilitySpec(FrozenStrictModel):
    tool_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    allowed_task_types: tuple[TaskType, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_allowed_task_types(self) -> "ToolCapabilitySpec":
        """验证一个 Tool 的 allowed task type 列表不重复。"""
        if len(self.allowed_task_types) != len(set(self.allowed_task_types)):
            raise ValueError("tool allowed_task_types must be unique")
        return self


# Planner 可使用的任务数量、DAG 深度、调用和 timeout 上限。
class PlannerBudget(FrozenStrictModel):
    max_tasks: int = Field(default=16, ge=1, le=16)
    max_depth: int = Field(default=8, ge=1, le=16)
    scheduler_concurrency: int = Field(default=4, ge=1, le=32)
    max_total_model_calls: int = Field(default=64, ge=0)
    max_total_tool_calls: int = Field(default=2048, ge=0)
    max_task_timeout_seconds: int = Field(default=600, ge=1)


# 从完整当前材料确定性计算的最低时间任务工作量。
class PlannerWorkloadProfile(FrozenStrictModel):
    temporal_expression_count: int = Field(ge=0)
    temporal_task_required: bool
    temporal_resolution_required: bool
    temporal_required_tool_calls: int = Field(ge=0)
    temporal_tool_call_limit: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_temporal_workload(self) -> "PlannerWorkloadProfile":
        """确保时间表达式数量、必需任务和工具调用下界彼此一致。"""
        if self.temporal_task_required != bool(self.temporal_expression_count):
            raise ValueError(
                "temporal task requirement must match expression presence"
            )
        if self.temporal_resolution_required:
            if not self.temporal_task_required:
                raise ValueError(
                    "required temporal resolution needs a temporal task"
                )
            if self.temporal_required_tool_calls != self.temporal_expression_count:
                raise ValueError(
                    "temporal required tool calls must cover every expression"
                )
        elif self.temporal_required_tool_calls != 0:
            raise ValueError(
                "non-required temporal resolution cannot reserve required calls"
            )
        return self


# Planner/Replanner 读取的单任务执行结果摘要。
class PlannerTaskOutcome(FrozenStrictModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    status: AgentStatus
    is_empty: bool = False
    error: Optional[str] = None
    warning_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    failed_tools: tuple[str, ...] = ()
    termination_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> "PlannerTaskOutcome":
        """确保失败工具名称不重复，便于模型诊断。"""
        if len(self.failed_tools) != len(set(self.failed_tools)):
            raise ValueError("failed_tools must be unique")
        return self


# 传给 Planner/Replanner 的最小 Evidence 投影，不包含完整原始审计。
class PlannerEvidenceSnapshot(FrozenStrictModel):
    evidence_id: str = Field(min_length=1)
    source_kind: SourceKind
    source_id: Optional[str] = None
    segment_id: Optional[str] = None
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    memory_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    source_time: Optional[str] = None
    source_file: Optional[str] = None
    evidence_text: Optional[str] = None

    @model_validator(mode="after")
    def validate_source_locator(self) -> "PlannerEvidenceSnapshot":
        """确保不同 SourceKind 使用对应的 source/offset/Memory 定位方式。"""
        if self.source_kind == SourceKind.CURRENT_MATERIAL:
            required = (
                self.source_id,
                self.segment_id,
                self.start_offset,
                self.end_offset,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "current material evidence snapshot requires source/segment offsets"
                )
            if self.end_offset <= self.start_offset:
                raise ValueError(
                    "evidence snapshot end_offset must be greater than start_offset"
                )
        elif (
            self.source_kind == SourceKind.LONG_TERM_MEMORY
            and not self.memory_id
        ):
            raise ValueError(
                "long-term memory evidence snapshot requires memory_id"
            )
        elif self.source_kind == SourceKind.TOOL_OUTPUT and not self.tool_call_id:
            raise ValueError("tool output evidence snapshot requires tool_call_id")
        return self


# Replanning 可消费的成功 AgentResult 最小投影及其 Evidence Lineage。
class PlannerSuccessfulResult(FrozenStrictModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    is_empty: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[PlannerEvidenceSnapshot, ...] = ()
    warnings: tuple[str, ...] = ()
    consumed_result_ids: tuple[str, ...] = ()
    consumed_evidence_ids: tuple[str, ...] = ()


# Replanning 的已用/剩余 Worker 调用预算分区。
class PlannerRemainingBudget(FrozenStrictModel):
    worker_model_calls_used: int = Field(default=0, ge=0)
    worker_tool_calls_used: int = Field(default=0, ge=0)
    remaining_worker_model_calls: int = Field(ge=0)
    remaining_worker_tool_calls: int = Field(ge=0)
    remaining_replan_rounds: int = Field(ge=0)


# Planner/Replanner 的完整受控输入，包括能力目录、计划历史和失败上下文。
class PlannerInput(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.planner_input.v2"] = (
        "summary_multi_agent.planner_input.v2"
    )
    request: SummaryRequest
    raw_material: RawMaterialBundle
    worker_capabilities: tuple[WorkerCapabilitySpec, ...] = Field(min_length=1)
    tool_capabilities: tuple[ToolCapabilitySpec, ...] = ()
    budget: PlannerBudget = Field(default_factory=PlannerBudget)
    workload_profile: PlannerWorkloadProfile
    previous_plan: Optional[ExecutionPlan] = None
    task_outcomes: tuple[PlannerTaskOutcome, ...] = ()
    successful_results: tuple[PlannerSuccessfulResult, ...] = ()
    plan_history: tuple[ExecutionPlan, ...] = ()
    remaining_budget: Optional[PlannerRemainingBudget] = None
    replan_round: int = Field(default=0, ge=0)
    failure_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_planning_context(self) -> "PlannerInput":
        """验证原文绑定、能力/工具关系、计划 lineage 和预算分区。"""
        if self.request.current_material != self.raw_material.raw_text:
            raise ValueError("planner input must preserve the complete current material")

        capability_ids = [item.capability_id for item in self.worker_capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("worker capability ids must be unique")
        tool_names = [item.tool_name for item in self.tool_capabilities]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool capability names must be unique")

        tool_by_name = {item.tool_name: item for item in self.tool_capabilities}
        for capability in self.worker_capabilities:
            for tool_name in capability.allowed_tools:
                tool = tool_by_name.get(tool_name)
                if tool is None:
                    raise ValueError(
                        f"worker capability references unknown tool: {tool_name}"
                    )
                if capability.capability_id not in tool.allowed_task_types:
                    raise ValueError(
                        f"tool {tool_name} does not permit capability "
                        f"{capability.capability_id.value}"
                    )

        outcome_ids = [item.task_id for item in self.task_outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("planner task outcome ids must be unique")
        if len(self.failure_reasons) != len(set(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique")
        if any(not item.strip() for item in self.failure_reasons):
            raise ValueError("failure_reasons must be non-empty")
        successful_result_ids = [
            item.task_id for item in self.successful_results
        ]
        if len(successful_result_ids) != len(set(successful_result_ids)):
            raise ValueError("successful result task ids must be unique")
        history_ids = [item.plan_id for item in self.plan_history]
        if len(history_ids) != len(set(history_ids)):
            raise ValueError("plan history ids must be unique")
        if self.plan_history:
            if self.plan_history[0].source == PlanSource.REPLAN:
                raise ValueError("plan history must start with an initial plan")
            for parent, child in zip(
                self.plan_history,
                self.plan_history[1:],
            ):
                if child.source != PlanSource.REPLAN:
                    raise ValueError(
                        "every plan after the first history item must be a replan"
                    )
                if child.parent_plan_id != parent.plan_id:
                    raise ValueError("plan history parent lineage is not contiguous")
                if child.plan_version != parent.plan_version + 1:
                    raise ValueError("plan history versions are not contiguous")
                if child.replan_round != parent.replan_round + 1:
                    raise ValueError("plan history rounds are not contiguous")

        if self.replan_round == 0:
            if any(
                (
                    self.previous_plan is not None,
                    bool(self.task_outcomes),
                    bool(self.successful_results),
                    bool(self.plan_history),
                    self.remaining_budget is not None,
                    bool(self.failure_reasons),
                )
            ):
                raise ValueError(
                    "initial planner input cannot include previous execution state"
                )
        else:
            if self.previous_plan is None:
                raise ValueError("replan input requires previous_plan")
            if not self.failure_reasons:
                raise ValueError("replan input requires failure_reasons")
            if self.replan_round != self.previous_plan.replan_round + 1:
                raise ValueError(
                    "replan input round must increment the previous plan round"
                )
            if self.remaining_budget is None:
                raise ValueError("replan input requires remaining_budget")
            if (
                self.remaining_budget.worker_model_calls_used
                + self.remaining_budget.remaining_worker_model_calls
                != self.budget.max_total_model_calls
            ):
                raise ValueError(
                    "replan model-call budget must exactly partition the plan budget"
                )
            if (
                self.remaining_budget.worker_tool_calls_used
                + self.remaining_budget.remaining_worker_tool_calls
                != self.budget.max_total_tool_calls
            ):
                raise ValueError(
                    "replan tool-call budget must exactly partition the plan budget"
                )
            if not self.plan_history or self.plan_history[-1] != self.previous_plan:
                raise ValueError(
                    "replan plan_history must end with previous_plan"
                )
            previous_tasks = {
                task.task_id: task for task in self.previous_plan.tasks
            }
            unknown_outcomes = set(outcome_ids) - set(previous_tasks)
            if unknown_outcomes:
                raise ValueError(
                    f"task outcomes are not in previous plan: {sorted(unknown_outcomes)}"
                )
            for outcome in self.task_outcomes:
                if previous_tasks[outcome.task_id].task_type != outcome.task_type:
                    raise ValueError(
                        f"task outcome type mismatch: {outcome.task_id}"
                    )
            successful_outcome_ids = {
                item.task_id
                for item in self.task_outcomes
                if item.status == AgentStatus.SUCCESS
            }
            if set(successful_result_ids) != successful_outcome_ids:
                raise ValueError(
                    "successful result snapshots must match successful outcomes"
                )
            for result in self.successful_results:
                if previous_tasks[result.task_id].task_type != result.task_type:
                    raise ValueError(
                        f"successful result type mismatch: {result.task_id}"
                    )
        return self


# 模型或估算器记录的 token 使用量。
class TokenUsage(FrozenStrictModel):
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    estimated: bool = False


# Worker 请求 Tool Registry 执行的一次带 timeout 调用。
class ToolCall(FrozenStrictModel):
    call_id: str = Field(default_factory=lambda: _new_id("tool"), min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=60, ge=1)


# Tool Registry 返回的成功、失败或超时结构化 Observation 来源。
class ToolResult(FrozenStrictModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: ToolStatus
    payload: Any = None
    error: Optional[str] = None
    attempts: int = Field(default=1, ge=1)
    duration_ms: int = Field(default=0, ge=0)


# 模型 action 中尚未分配 call_id 的工具调用草案。
class WorkerToolCallDraft(FrozenStrictModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=60, ge=1)


# Worker 模型每一步允许返回的受控动作和依赖/Evidence 引用。
class WorkerAction(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.worker_action.v2"] = (
        "summary_multi_agent.worker_action.v2"
    )
    action: WorkerActionType
    action_rationale: str = Field(min_length=1, max_length=500)
    tool_calls: tuple[WorkerToolCallDraft, ...] = Field(default=(), max_length=128)
    selected_observation_ids: tuple[str, ...] = ()
    selected_memory_ids: Optional[tuple[str, ...]] = None
    referenced_prior_result_ids: tuple[str, ...] = ()
    referenced_prior_evidence_ids: tuple[str, ...] = ()
    result_summary: Optional[str] = Field(default=None, max_length=1000)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_action_shape(self) -> "WorkerAction":
        """确保 terminal/call_tool 动作形状、观察和依赖 ID 不冲突。"""
        if self.action == WorkerActionType.CALL_TOOL:
            if not self.tool_calls:
                raise ValueError("call_tool action requires at least one tool call")
        elif self.tool_calls:
            raise ValueError("terminal worker actions cannot include tool calls")
        if self.action == WorkerActionType.CALL_TOOL and self.selected_memory_ids is not None:
            raise ValueError("call_tool actions cannot select memory ids")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("worker action warnings must be unique")
        if len(self.selected_observation_ids) != len(
            set(self.selected_observation_ids)
        ):
            raise ValueError("selected observation ids must be unique")
        if self.selected_memory_ids is not None and len(
            self.selected_memory_ids
        ) != len(set(self.selected_memory_ids)):
            raise ValueError("selected memory ids must be unique")
        if len(self.referenced_prior_result_ids) != len(
            set(self.referenced_prior_result_ids)
        ):
            raise ValueError("referenced prior result ids must be unique")
        if len(self.referenced_prior_evidence_ids) != len(
            set(self.referenced_prior_evidence_ids)
        ):
            raise ValueError("referenced prior evidence ids must be unique")
        return self


# Worker Action Loop 某一步之后的不可超越预算快照。
class WorkerBudgetSnapshot(FrozenStrictModel):
    step_limit: int = Field(ge=1)
    model_call_limit: int = Field(ge=0)
    tool_call_limit: int = Field(ge=0)
    steps_used: int = Field(ge=0)
    model_calls_used: int = Field(ge=0)
    tool_calls_used: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "WorkerBudgetSnapshot":
        """确保步骤、模型调用和工具调用没有超过各自上限。"""
        if self.steps_used > self.step_limit:
            raise ValueError("worker steps exceed the step limit")
        if self.model_calls_used > self.model_call_limit:
            raise ValueError("worker model calls exceed the model-call limit")
        if self.tool_calls_used > self.tool_call_limit:
            raise ValueError("worker tool calls exceed the tool-call limit")
        return self


# 将 ToolResult 绑定到动作步骤的结构化 Observation。
class WorkerObservation(FrozenStrictModel):
    observation_id: str = Field(
        default_factory=lambda: _new_id("observation"), min_length=1
    )
    step_index: int = Field(ge=1)
    tool_result: ToolResult


# Worker action/repair 模型尝试的审计摘要。
class WorkerModelAttempt(FrozenStrictModel):
    attempt_index: int = Field(ge=1)
    step_index: int = Field(ge=1)
    stage: Literal["action", "repair"]
    max_tokens: int = Field(ge=1)
    raw_output: Optional[str] = None
    failure_kind: Optional[str] = None
    error: Optional[str] = None


# Worker 单步动作、工具结果、Observation 引用和预算记录。
class WorkerStepRecord(FrozenStrictModel):
    step_id: str = Field(default_factory=lambda: _new_id("worker_step"), min_length=1)
    step_index: int = Field(ge=1)
    action: WorkerAction
    tool_results: tuple[ToolResult, ...] = ()
    observation_ids: tuple[str, ...] = ()
    budget_after: WorkerBudgetSnapshot
    duration_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_step_links(self) -> "WorkerStepRecord":
        """确保 call_tool 的每个请求都有对应结果和 Observation。"""
        if len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("worker step observation ids must be unique")
        if self.action.action == WorkerActionType.CALL_TOOL:
            if len(self.tool_results) != len(self.action.tool_calls):
                raise ValueError("tool action must record every requested tool result")
            if len(self.observation_ids) != len(self.tool_results):
                raise ValueError("tool action must expose every result as an observation")
        elif self.tool_results or self.observation_ids:
            raise ValueError("terminal worker step cannot contain tool observations")
        return self


# 自主 Worker 的完整步骤、Observation、预算、依赖消费和 fallback 审计。
class WorkerExecutionAudit(FrozenStrictModel):
    mode: WorkerMode
    termination_reason: Optional[WorkerTerminationReason] = None
    steps: tuple[WorkerStepRecord, ...] = ()
    observations: tuple[WorkerObservation, ...] = ()
    model_attempts: tuple[WorkerModelAttempt, ...] = ()
    final_budget: Optional[WorkerBudgetSnapshot] = None
    selected_observation_ids: tuple[str, ...] = ()
    available_prior_result_ids: tuple[str, ...] = ()
    consumed_prior_result_ids: tuple[str, ...] = ()
    consumed_prior_evidence_ids: tuple[str, ...] = ()
    fallback_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_execution_audit(self) -> "WorkerExecutionAudit":
        """验证 Observation/依赖 lineage，并区分 model 与 rule audit 形态。"""
        if len(self.selected_observation_ids) != len(
            set(self.selected_observation_ids)
        ):
            raise ValueError("selected observation ids must be unique")
        known_observations = {
            item.observation_id for item in self.observations
        }
        unknown = set(self.selected_observation_ids) - known_observations
        if unknown:
            raise ValueError(
                "worker execution audit references unknown observations: "
                f"{sorted(unknown)}"
            )
        if len(self.available_prior_result_ids) != len(
            set(self.available_prior_result_ids)
        ):
            raise ValueError("available prior result ids must be unique")
        if len(self.consumed_prior_result_ids) != len(
            set(self.consumed_prior_result_ids)
        ):
            raise ValueError("consumed prior result ids must be unique")
        unknown_prior_results = set(self.consumed_prior_result_ids) - set(
            self.available_prior_result_ids
        )
        if unknown_prior_results:
            raise ValueError(
                "worker execution audit consumed unavailable prior results: "
                f"{sorted(unknown_prior_results)}"
            )
        if len(self.consumed_prior_evidence_ids) != len(
            set(self.consumed_prior_evidence_ids)
        ):
            raise ValueError("consumed prior evidence ids must be unique")
        if self.mode == WorkerMode.RULE:
            if any(
                (
                    self.termination_reason is not None,
                    bool(self.steps),
                    bool(self.observations),
                    bool(self.model_attempts),
                    self.final_budget is not None,
                    bool(self.selected_observation_ids),
                    bool(self.available_prior_result_ids),
                    bool(self.consumed_prior_result_ids),
                    bool(self.consumed_prior_evidence_ids),
                    self.fallback_reason is not None,
                )
            ):
                raise ValueError("rule worker execution audit cannot contain model trace")
        elif self.termination_reason is None or self.final_budget is None:
            raise ValueError(
                "model worker execution audit requires termination and budget"
            )
        if self.mode == WorkerMode.MODEL_WITH_RULE_FALLBACK:
            if not self.fallback_reason:
                raise ValueError("rule fallback audit requires fallback_reason")
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason is only valid for rule fallback")
        return self


# 统一引用当前原文、Long-term Memory 或 Tool 输出的证据定位。
class EvidenceRef(FrozenStrictModel):
    evidence_id: str = Field(default_factory=lambda: _new_id("evidence"), min_length=1)
    source_kind: SourceKind
    source_id: Optional[str] = None
    segment_id: Optional[str] = None
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    memory_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    source_time: Optional[str] = None
    source_file: Optional[str] = None
    evidence_text: Optional[str] = None

    @model_validator(mode="after")
    def validate_source_locator(self) -> "EvidenceRef":
        """根据 SourceKind 强制 source offset、Memory ID 或 Tool ID 定位。"""
        if self.source_kind == SourceKind.CURRENT_MATERIAL:
            required = (self.source_id, self.segment_id, self.start_offset, self.end_offset)
            if any(value is None for value in required):
                raise ValueError("current material evidence requires source/segment offsets")
            if self.end_offset <= self.start_offset:
                raise ValueError("evidence end_offset must be greater than start_offset")
        elif self.source_kind == SourceKind.LONG_TERM_MEMORY and not self.memory_id:
            raise ValueError("long-term memory evidence requires memory_id")
        elif self.source_kind == SourceKind.TOOL_OUTPUT and not self.tool_call_id:
            raise ValueError("tool output evidence requires tool_call_id")
        return self


# Worker 的终态结果、领域 payload、ToolResult 和 Evidence Lineage。
class AgentResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.agent_result.v2"] = (
        "summary_multi_agent.agent_result.v2"
    )
    task_id: str = Field(min_length=1)
    task_type: TaskType
    status: AgentStatus
    is_empty: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tool_results: tuple[ToolResult, ...] = ()
    warnings: tuple[str, ...] = ()
    error: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    duration_ms: int = Field(default=0, ge=0)
    worker_execution: Optional[WorkerExecutionAudit] = None
    consumed_result_ids: tuple[str, ...] = ()
    consumed_evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result_state(self) -> "AgentResult":
        """确保终态结果的 status、error、空结果和依赖 ID 组合合法。"""
        if self.status in {AgentStatus.PENDING, AgentStatus.RUNNING}:
            raise ValueError("AgentResult must describe a terminal task state")
        if self.status == AgentStatus.SUCCESS and self.error:
            raise ValueError("successful AgentResult cannot contain an error")
        if self.status in {AgentStatus.FAILED, AgentStatus.TIMEOUT} and not self.error:
            raise ValueError("failed or timed-out AgentResult requires an error")
        if self.is_empty and self.status != AgentStatus.SUCCESS:
            raise ValueError("is_empty is only valid for successful results")
        if len(self.consumed_result_ids) != len(set(self.consumed_result_ids)):
            raise ValueError("consumed result ids must be unique")
        if len(self.consumed_evidence_ids) != len(
            set(self.consumed_evidence_ids)
        ):
            raise ValueError("consumed evidence ids must be unique")
        return self


# 传给自主 Worker 的最小任务、成功依赖、工具 Schema、Observation 和预算。
class WorkerAgentInput(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.worker_agent_input.v2"] = (
        "summary_multi_agent.worker_agent_input.v2"
    )
    task: SubTask
    request: SummaryRequest
    raw_material: RawMaterialBundle
    prior_results: tuple[AgentResult, ...] = ()
    tool_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    observations: tuple[WorkerObservation, ...] = ()
    budget: WorkerBudgetSnapshot

    @model_validator(mode="after")
    def validate_worker_scope(self) -> "WorkerAgentInput":
        """确保当前原文不变、工具不越权且 prior_results 精确匹配声明依赖。"""
        if self.request.current_material != self.raw_material.raw_text:
            raise ValueError("worker input must preserve the complete current material")
        unknown_tools = set(self.tool_schemas) - set(self.task.allowed_tools)
        if unknown_tools:
            raise ValueError(
                f"worker input exposes tools outside the task allowance: {sorted(unknown_tools)}"
            )
        prior_ids = [item.task_id for item in self.prior_results]
        if len(prior_ids) != len(set(prior_ids)):
            raise ValueError("worker prior result ids must be unique")
        expected_prior_ids = set(self.task.dependencies)
        actual_prior_ids = set(prior_ids)
        if actual_prior_ids != expected_prior_ids:
            raise ValueError(
                "worker prior results must exactly match declared dependencies; "
                f"missing={sorted(expected_prior_ids - actual_prior_ids)}, "
                f"unexpected={sorted(actual_prior_ids - expected_prior_ids)}"
            )
        unsuccessful = [
            item.task_id
            for item in self.prior_results
            if item.status != AgentStatus.SUCCESS
        ]
        if unsuccessful:
            raise ValueError(
                "worker prior results must be successful: "
                f"{sorted(unsuccessful)}"
            )
        return self


# 自主 Worker Action Loop 的终态、步骤、Observation、预算和依赖消费结果。
class WorkerLoopOutcome(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.worker_loop_outcome.v2"] = (
        "summary_multi_agent.worker_loop_outcome.v2"
    )
    task_id: str = Field(min_length=1)
    task_type: TaskType
    termination_reason: WorkerTerminationReason
    final_action: Optional[WorkerAction] = None
    steps: tuple[WorkerStepRecord, ...] = ()
    observations: tuple[WorkerObservation, ...] = ()
    model_attempts: tuple[WorkerModelAttempt, ...] = ()
    final_budget: WorkerBudgetSnapshot
    consumed_prior_result_ids: tuple[str, ...] = ()
    consumed_prior_evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "WorkerLoopOutcome":
        """确保最终预算不倒退，步骤数与使用量及消费 ID 一致。"""
        if self.final_budget.steps_used != len(self.steps):
            raise ValueError("worker outcome step usage must match recorded steps")
        if self.steps:
            previous = self.steps[-1].budget_after
            if self.final_budget.model_calls_used < previous.model_calls_used:
                raise ValueError("worker outcome cannot lose model-call usage")
            if self.final_budget.tool_calls_used < previous.tool_calls_used:
                raise ValueError("worker outcome cannot lose tool-call usage")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("worker outcome warnings must be unique")
        if len(self.consumed_prior_result_ids) != len(
            set(self.consumed_prior_result_ids)
        ):
            raise ValueError("consumed prior result ids must be unique")
        if len(self.consumed_prior_evidence_ids) != len(
            set(self.consumed_prior_evidence_ids)
        ):
            raise ValueError("consumed prior evidence ids must be unique")
        return self


# 精简评测的共享预算和上下文契约；在线评测与独立横评使用同一组参数。
class LiteEvaluationConfig(FrozenStrictModel):
    reference_chunking: Literal["fixed", "auto"] = "fixed"
    segment_max_chars: int = Field(default=3000, ge=100)
    segment_overlap_chars: int = Field(default=100, ge=0)
    reference_max_tokens: int = Field(default=4096, ge=128)
    claim_max_tokens: int = Field(default=4096, ge=128)
    match_max_tokens: int = Field(default=4096, ge=128)
    verification_max_tokens: int = Field(default=4096, ge=128)
    repair_attempts: int = Field(default=1, ge=0, le=1)
    model_timeout_seconds: int = Field(default=180, ge=1)
    timeout_seconds: int = Field(default=1200, ge=1)
    max_model_calls: int = Field(default=64, ge=1)
    max_concurrent_model_calls: int = Field(default=1, ge=1, le=4)
    model_context_window_tokens: Optional[int] = Field(default=None, ge=1024)
    context_safety_margin_tokens: int = Field(default=1024, ge=0)

    @classmethod
    def cloud_fast(cls) -> "LiteEvaluationConfig":
        return cls(
            reference_chunking="auto",
            segment_max_chars=12000,
            segment_overlap_chars=100,
            reference_max_tokens=8192,
            claim_max_tokens=8192,
            max_concurrent_model_calls=4,
        )

    @model_validator(mode="after")
    def validate_segment_overlap(self) -> "LiteEvaluationConfig":
        if self.segment_overlap_chars >= self.segment_max_chars:
            raise ValueError(
                "lite evaluation segment overlap must be smaller than segment size"
            )
        return self


# V2 Planner、Worker、Review、Compression 和 Evaluation 的运行参数契约。
class RunConfig(FrozenStrictModel):
    execution_profile: Literal["legacy", "cloud_fast"] = "legacy"
    workflow_mode: WorkflowMode = WorkflowMode.BASELINE
    planner_mode: PlannerMode = PlannerMode.MODEL
    planner_max_tokens: int = Field(default=1600, ge=128)
    planner_repair_max_tokens: int = Field(default=1600, ge=128)
    planner_repair_attempts: int = Field(default=2, ge=0, le=5)
    planner_timeout_seconds: int = Field(default=180, ge=1)
    replanner_max_tokens: int = Field(default=1800, ge=128)
    replanner_repair_max_tokens: int = Field(default=1800, ge=128)
    replanner_repair_attempts: int = Field(default=2, ge=0, le=5)
    replanner_timeout_seconds: int = Field(default=180, ge=1)
    worker_mode: WorkerMode = WorkerMode.RULE
    worker_max_steps: int = Field(default=6, ge=1, le=32)
    worker_max_model_calls: int = Field(default=4, ge=2, le=16)
    worker_action_max_tokens: int = Field(default=3000, ge=128)
    worker_action_repair_attempts: int = Field(default=1, ge=0, le=5)
    worker_model_timeout_seconds: int = Field(default=180, ge=1)
    max_concurrent_model_calls: int = Field(default=1, ge=1, le=16)
    max_concurrent_workers: int = Field(default=4, ge=1, le=32)
    worker_timeout_seconds: int = Field(default=180, ge=1)
    global_timeout_seconds: int = Field(default=600, ge=1)
    model_context_window_tokens: Optional[int] = Field(default=None, ge=1024)
    model_context_safety_margin_tokens: int = Field(default=1024, ge=0)
    max_replan_rounds: int = Field(default=2, ge=0, le=10)
    planner_max_tasks: int = Field(default=16, ge=1, le=16)
    planner_max_depth: int = Field(default=8, ge=1, le=16)
    planner_max_tasks_per_capability: int = Field(default=4, ge=1, le=16)
    planner_max_total_model_calls: int = Field(default=64, ge=0)
    planner_max_total_tool_calls: int = Field(default=2048, ge=0)
    planner_max_task_timeout_seconds: int = Field(default=600, ge=1)
    max_review_rounds: int = Field(default=3, ge=0, le=10)
    tool_retry_attempts: int = Field(default=2, ge=1, le=10)
    schema_repair_attempts: int = Field(default=2, ge=0, le=10)
    retrieval_top_k_per_plan_item: int = Field(default=10, ge=1, le=100)
    raw_segment_max_chars: int = Field(default=3000, ge=100)
    raw_segment_overlap_ratio: float = Field(default=0.05, ge=0.0, lt=0.5)
    max_concurrent_segment_maps: int = Field(default=2, ge=1, le=16)
    current_map_max_tokens: int = Field(default=2400, ge=128)
    summary_max_tokens: int = Field(default=2400, ge=128)
    minutes_output_mode: MinutesOutputMode = MinutesOutputMode.CLASSIC
    rich_summary_max_tokens: int = Field(default=6000, ge=256)
    context_compression_mode: ContextCompressionMode = (
        ContextCompressionMode.DETERMINISTIC
    )
    context_compression_top_k_per_type: int = Field(default=10, ge=1, le=100)
    temporal_max_tool_calls: int = Field(default=128, ge=1, le=1024)
    review_enabled: bool = False
    review_shadow_mode: bool = True
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
    evaluation_enabled: bool = False
    evaluation_mode: EvaluationMode = EvaluationMode.STRICT
    lite_evaluation: LiteEvaluationConfig = Field(
        default_factory=LiteEvaluationConfig
    )
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
    evaluation_adaptive_split_enabled: bool = True
    evaluation_adaptive_split_min_chars: int = Field(default=500, ge=100)
    evaluation_adaptive_split_max_depth: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def validate_runtime_modes(self) -> "RunConfig":
        """强制 Review Shadow 边界和自主 Worker 的最小模型调用预算。"""
        if self.review_enabled and not self.review_shadow_mode:
            raise ValueError(
                "P5 review only supports shadow mode; active replacement is not enabled"
            )
        if self.worker_mode != WorkerMode.RULE:
            required_model_calls = 2 * (1 + self.worker_action_repair_attempts)
            if self.worker_max_model_calls < required_model_calls:
                raise ValueError(
                    "model Worker budget must cover one action decision and one "
                    "completion decision, including configured repair attempts: "
                    f"worker_max_model_calls must be >= {required_model_calls}"
                )
        return self


# 当前原文映射出的单条带 Evidence 的事实。
class CurrentFactItem(FrozenStrictModel):
    item_id: str = Field(default_factory=lambda: _new_id("current_fact"), min_length=1)
    category: CurrentFactCategory
    statement: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    entities: tuple[str, ...] = ()
    time_expression: Optional[str] = None


# 单个原文分段的当前事实分析结果和 warning。
class CurrentSegmentAnalysis(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.current_segment_analysis.v2"] = (
        "summary_multi_agent.current_segment_analysis.v2"
    )
    segment_id: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    items: tuple[CurrentFactItem, ...] = ()
    warnings: tuple[str, ...] = ()


# 辅助上下文中某项被保留、归并或丢弃的可审计决定。
class ContextCompressionDecision(FrozenStrictModel):
    item_kind: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    action: ContextCompressionAction
    reason: str = Field(min_length=1)
    retained_id: Optional[str] = None
    memory_type: Optional[str] = None

    @model_validator(mode="after")
    def validate_decision_ids(self) -> "ContextCompressionDecision":
        """确保决策来源/Evidence 不重复，归并项有 retained_id。"""
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("compression decision source ids must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("compression decision evidence ids must be unique")
        if self.action == ContextCompressionAction.MERGE and not self.retained_id:
            raise ValueError("merged compression decision requires retained_id")
        return self


# 辅助上下文压缩规模、hash 不变量、逐项决策和 warning 报告。
class ContextCompressionReport(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.context_compression.v2"] = (
        "summary_multi_agent.context_compression.v2"
    )
    mode: ContextCompressionMode
    input_chars: int = Field(ge=0)
    output_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0)
    input_record_count: int = Field(ge=0)
    unique_record_count: int = Field(ge=0)
    retained_record_count: int = Field(ge=0)
    merged_record_count: int = Field(ge=0)
    dropped_record_count: int = Field(ge=0)
    protected_record_count: int = Field(ge=0)
    current_material_compressed: Literal[False] = False
    current_material_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_material_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_manifest_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_manifest_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_ledger_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_ledger_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_evidence_offsets_sha256_before: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    current_evidence_offsets_sha256_after: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    invariants_preserved: Literal[True] = True
    decisions: tuple[ContextCompressionDecision, ...] = ()
    warnings: tuple[str, ...] = ()


# Summarizer 消费的当前事实、历史 canonical 记录、模板和 Evidence 索引。
class SynthesisContext(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.synthesis_context.v2"] = (
        "summary_multi_agent.synthesis_context.v2"
    )
    request_id: str = Field(min_length=1)
    source_time: Optional[str] = None
    summary_request: str = Field(min_length=1)
    target_entities: tuple[str, ...] = ()
    personalization_preferences: dict[str, Any] = Field(default_factory=dict)
    segment_analyses: tuple[CurrentSegmentAnalysis, ...] = ()
    merged_current_items: tuple[CurrentFactItem, ...] = ()
    worker_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    historical_records: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence_index: dict[str, EvidenceRef] = Field(default_factory=dict)
    template: dict[str, Any] = Field(default_factory=dict)
    auxiliary_context_info: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# Red 发现、Blue 修复、Judge 裁决所共享的摘要问题描述。
class ReviewIssue(FrozenStrictModel):
    issue_id: str = Field(default_factory=lambda: _new_id("issue"), min_length=1)
    issue_type: str = Field(default="other", min_length=1)
    severity: ReviewSeverity
    section: str = "unknown"
    description: str = Field(min_length=1)
    summary_quote: str = ""
    recommended_action: ReviewAction = ReviewAction.MODIFY
    repair_instruction: str = ""
    route_to: str = Field(default="summary_writer", min_length=1)
    evidence_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    audit_origin: Literal["global", "segment"] = "global"
    target_entities: tuple[str, ...] = ()
    related_task_ids: tuple[str, ...] = ()
    needs_verification: bool = False
    verification_query: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ReviewIssueDisposition(FrozenStrictModel):
    issue_id: str = Field(min_length=1)
    disposition: Literal["kept", "dismissed", "deferred"]
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


# 一轮 Red 全局/逐段审核的 Issue、延迟项和裁决说明。
class RedReviewReport(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.red_report.v2"] = (
        "summary_multi_agent.red_report.v2"
    )
    round_index: int = Field(ge=1)
    candidate_index: int = Field(ge=0)
    passed: bool
    candidate_issue_count: int = Field(default=0, ge=0)
    issues: tuple[ReviewIssue, ...] = ()
    deferred_issues: tuple[ReviewIssue, ...] = ()
    issue_dispositions: tuple[ReviewIssueDisposition, ...] = ()
    checked_segment_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


# Blue 针对单个 Issue 的局部修复记录。
class BlueRepairRecord(FrozenStrictModel):
    issue_id: str = Field(min_length=1)
    action: ReviewAction
    applied: bool
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


# 一轮 Blue 候选正文、实际修复和未解决 Issue。
class BlueReviewResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.blue_result.v2"] = (
        "summary_multi_agent.blue_result.v2"
    )
    round_index: int = Field(ge=1)
    source_candidate_index: int = Field(ge=0)
    candidate_summary: str = Field(min_length=1)
    repairs: tuple[BlueRepairRecord, ...] = ()
    unresolved_issue_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


# Review Issue 经 ad-hoc Worker 验证后的状态和新增 Evidence。
class ReviewVerificationResult(FrozenStrictModel):
    issue_id: str = Field(min_length=1)
    route_to: str = Field(min_length=1)
    status: Literal["verified", "not_found", "failed", "not_required"]
    evidence_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None


# Judge 对候选的独立质量分数、处置和剩余 Issue。
class JudgeDecision(FrozenStrictModel):
    round_index: int = Field(ge=1)
    candidate_index: int = Field(default=0, ge=0)
    disposition: JudgeDisposition
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    remaining_issue_ids: tuple[str, ...] = ()
    improvement: float = Field(default=0.0, ge=-1.0, le=1.0)
    candidate_hash: str = ""
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_dimension_scores(self) -> "JudgeDecision":
        """确保各质量维度分数位于 0 到 1。"""
        invalid = {
            key: value
            for key, value in self.dimension_scores.items()
            if value < 0.0 or value > 1.0
        }
        if invalid:
            raise ValueError(f"judge dimension scores must be in [0, 1]: {invalid}")
        return self


# 记录已解决 Issue 签名在后续 Review 轮次重新出现。
class ReviewIssueReappearance(FrozenStrictModel):
    signature: str = Field(min_length=1)
    previous_round_index: int = Field(ge=1)
    previous_issue_id: str = Field(min_length=1)
    current_issue_id: str = Field(min_length=1)


# 一轮 Review 的输入/输出候选、Red、Verify、Blue 和 Judge 结果。
class ReviewRoundRecord(FrozenStrictModel):
    round_index: int = Field(ge=1)
    input_candidate_index: int = Field(ge=0)
    output_candidate_index: int = Field(ge=0)
    text_change_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    issue_reappearances: tuple[ReviewIssueReappearance, ...] = ()
    red_report: RedReviewReport
    verification_results: tuple[ReviewVerificationResult, ...] = ()
    blue_result: Optional[BlueReviewResult] = None
    judge_decision: JudgeDecision


# 整个 Shadow Review 的终止原因、推荐候选和轮次历史。
class ReviewRunResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.review_run.v2"] = (
        "summary_multi_agent.review_run.v2"
    )
    enabled: bool = True
    shadow_mode: bool = True
    termination: ReviewTermination
    original_candidate_index: int = Field(default=0, ge=0)
    recommended_candidate_index: int = Field(default=0, ge=0)
    rounds: tuple[ReviewRoundRecord, ...] = ()
    warnings: tuple[str, ...] = ()


# 单次运行的 M4 Shared Working Memory，可变但只能由编排阶段更新。
class SummaryWorkingState(StrictModel):
    schema_version: Literal["summary_multi_agent.working_state.v2"] = (
        "summary_multi_agent.working_state.v2"
    )
    run_id: str = Field(default_factory=lambda: _new_id("run"), min_length=1)
    status: RunStatus = RunStatus.PENDING
    request: SummaryRequest
    raw_material: RawMaterialBundle
    coverage_ledger: CoverageLedger
    plan: Optional[ExecutionPlan] = None
    plan_history: list[ExecutionPlan] = Field(default_factory=list)
    task_states: dict[str, TaskRuntimeState] = Field(default_factory=dict)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    evidence_index: dict[str, EvidenceRef] = Field(default_factory=dict)
    historical_evidence_board: Optional[dict[str, Any]] = None
    current_segment_analyses: list[CurrentSegmentAnalysis] = Field(default_factory=list)
    context_compression_report: Optional[ContextCompressionReport] = None
    synthesis_context: Optional[SynthesisContext] = None
    candidate_summaries: list[str] = Field(default_factory=list)
    review_history: list[JudgeDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    replan_round: int = Field(default=0, ge=0)
    worker_model_calls_used: int = Field(default=0, ge=0)
    worker_tool_calls_used: int = Field(default=0, ge=0)


# Pipeline 对外返回的摘要、候选、Review/Evaluation 状态和审计路径。
class SummaryRunResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.run_result.v2"] = (
        "summary_multi_agent.run_result.v2"
    )
    run_id: str = Field(min_length=1)
    status: RunStatus
    final_summary: Optional[str] = None
    selected_candidate_index: Optional[int] = Field(default=None, ge=0)
    review_enabled: bool = False
    review_shadow_mode: bool = True
    review_termination: Optional[ReviewTermination] = None
    recommended_candidate_index: Optional[int] = Field(default=None, ge=0)
    evaluation_enabled: bool = False
    evaluation_mode: EvaluationMode = EvaluationMode.STRICT
    evaluation_status: EvaluationStatus = EvaluationStatus.DISABLED
    evaluation_metrics_path: Optional[str] = None
    personalization_evaluation_status: EvaluationStatus = EvaluationStatus.DISABLED
    personalization_evaluation_metrics_path: Optional[str] = None
    warnings: tuple[str, ...] = ()
    coverage_complete: bool
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_completion(self) -> "SummaryRunResult":
        """确保完成状态必须有正式摘要且 Coverage 已完整。"""
        if self.status == RunStatus.COMPLETED and not self.coverage_complete:
            raise ValueError("completed result requires complete raw material coverage")
        if self.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}:
            if not self.final_summary:
                raise ValueError("completed result requires final_summary")
        return self
