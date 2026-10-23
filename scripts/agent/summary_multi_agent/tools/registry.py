"""管理白名单工具、参数校验、权限边界、超时和有限重试。"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Type

from pydantic import BaseModel, ValidationError

from ..schemas import (
    TaskType,
    ToolCall,
    ToolCapabilitySpec,
    ToolResult,
    ToolStatus,
)


class ToolRegistryError(RuntimeError):
    """Tool Registry 或工具执行边界相关的基础异常。"""


class ToolPermissionError(ToolRegistryError):
    """表示任务试图调用未授权的工具或参数范围。"""


class RetryableToolError(ToolRegistryError):
    """表示工具失败可由 Registry 按配置重试。"""


@dataclass(frozen=True)
class ToolContext:
    """向 Tool 提供本次运行、任务和 TaskType 的最小调用上下文。"""

    run_id: str
    task_id: str
    task_type: TaskType


class BaseTool(ABC):
    """定义白名单 Tool 的名称、输入/输出 Schema 和异步执行接口。"""

    name: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    allowed_task_types: frozenset[TaskType]

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolContext) -> BaseModel:
        """执行一次已通过 Registry 权限和参数校验的工具调用。"""
        raise NotImplementedError


class ToolRegistry:
    """集中管理 Tool 白名单、能力描述、权限、超时和有限重试。"""

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        """注册初始工具；重复名称或空名称会立即抛出 ``ValueError``。"""
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """返回已注册工具名的稳定排序视图。"""
        return tuple(sorted(self._tools))

    @property
    def planner_capabilities(self) -> tuple[ToolCapabilitySpec, ...]:
        """生成 Planner 可见的工具能力、权限任务类型和描述。"""
        return tuple(
            ToolCapabilitySpec(
                tool_name=tool.name,
                description=(
                    (type(tool).__doc__ or "").strip()
                    or f"Registered tool {tool.name}."
                ),
                allowed_task_types=tuple(
                    sorted(tool.allowed_task_types, key=lambda item: item.value)
                ),
            )
            for tool in (self._tools[name] for name in sorted(self._tools))
        )

    def register(self, tool: BaseTool) -> None:
        """将一个 Tool 加入白名单，并拒绝非法或重复名称。"""
        if not tool.name.strip():
            raise ValueError("tool name must be non-empty")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """按名称取得已注册 Tool，未知名称转为带上下文的 ``KeyError``。"""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def describe_for_worker(
        self, names: Iterable[str], *, task_type: TaskType
    ) -> dict[str, dict[str, Any]]:
        """为 Worker 投影指定 Tool 的描述和输入/输出 JSON Schema。"""
        descriptions: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(names):
            tool = self.get(name)
            if task_type not in tool.allowed_task_types:
                raise ToolPermissionError(
                    f"tool {name} does not allow task type {task_type.value}"
                )
            descriptions[name] = {
                "description": (
                    (type(tool).__doc__ or "").strip()
                    or f"Registered tool {name}."
                ),
                "input_schema": tool.input_model.model_json_schema(),
                "output_schema": tool.output_model.model_json_schema(),
            }
        return descriptions

    async def execute(
        self,
        call: ToolCall,
        *,
        context: ToolContext,
        allowed_tools: Iterable[str],
        retry_attempts: int = 2,
    ) -> ToolResult:
        """校验并限时执行 Tool，将权限、超时和异常编码为 ``ToolResult``。

        可重试异常只在剩余次数内重试；取消不会被吞掉，调用方仍可终止 Worker
        或整个 Pipeline。
        """
        started = time.perf_counter()
        allowed = set(allowed_tools)
        try:
            tool = self.get(call.tool_name)
            if call.tool_name not in allowed:
                raise ToolPermissionError(
                    f"task {context.task_id} is not allowed to call {call.tool_name}"
                )
            if context.task_type not in tool.allowed_task_types:
                raise ToolPermissionError(
                    f"tool {call.tool_name} does not allow task type {context.task_type.value}"
                )
            arguments = tool.input_model.model_validate(call.arguments)
        except (KeyError, ToolPermissionError, ValidationError, ValueError) as exc:
            return self._failure_result(call, exc, started, attempts=1)

        attempts = max(1, retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                raw_result = await asyncio.wait_for(
                    tool.execute(arguments, context),
                    timeout=call.timeout_seconds,
                )
                result = tool.output_model.model_validate(raw_result)
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status=ToolStatus.SUCCESS,
                    payload=result.model_dump(mode="json"),
                    attempts=attempt,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status=ToolStatus.TIMEOUT,
                    error=f"tool timed out after {call.timeout_seconds} seconds",
                    attempts=attempt,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except RetryableToolError as exc:
                if attempt == attempts:
                    return self._failure_result(call, exc, started, attempts=attempt)
            except Exception as exc:
                return self._failure_result(call, exc, started, attempts=attempt)

        raise AssertionError("tool retry loop exited unexpectedly")

    @staticmethod
    def _failure_result(
        call: ToolCall,
        error: Exception,
        started: float,
        *,
        attempts: int,
    ) -> ToolResult:
        """把一次工具失败转换为带耗时和尝试次数的结构化结果。"""
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            attempts=attempts,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
