"""定义 Worker 共享上下文、工具访问和结果装配的基础接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import AgentResult, RawMaterialBundle, SubTask, SummaryRequest, TaskType
from ..tools.registry import ToolRegistry


@dataclass(frozen=True)
class WorkerContext:
    """提供单个 Worker 的请求、完整原文、任务、工具和依赖结果边界。"""

    run_id: str
    request: SummaryRequest
    raw_material: RawMaterialBundle
    task: SubTask
    tool_registry: ToolRegistry
    tool_retry_attempts: int = 2
    prior_results: tuple[AgentResult, ...] = ()


class BaseWorker(ABC):
    """定义所有规则或自主 Worker 必须实现的任务类型和执行接口。"""

    task_type: TaskType

    @abstractmethod
    async def execute(self, context: WorkerContext) -> AgentResult:
        """在给定子任务范围内执行并返回终态 ``AgentResult``。"""
        raise NotImplementedError

    def validate_context(self, context: WorkerContext) -> None:
        """拒绝把错误 TaskType 的任务交给当前 Worker。"""
        if context.task.task_type != self.task_type:
            raise ValueError(
                f"worker {self.task_type.value} cannot execute "
                f"task {context.task.task_type.value}"
            )
