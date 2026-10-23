"""根据摘要请求和偏好适配输出模板的规则 Worker。"""

from __future__ import annotations

import time

from ..schemas import AgentResult, AgentStatus, TaskType, ToolCall, ToolStatus
from ..tools.registry import ToolContext
from .base import BaseWorker, WorkerContext


class TemplateAdaptationWorker(BaseWorker):
    """调用模板 Tool，将摘要请求和显式偏好转换为受控格式建议。"""

    task_type = TaskType.TEMPLATE_ADAPTATION

    async def execute(self, context: WorkerContext) -> AgentResult:
        """执行一次模板适配，并把 Tool 输出作为 Worker payload 返回。"""
        self.validate_context(context)
        started = time.perf_counter()
        requested_style = context.task.inputs.get("requested_style")
        call = ToolCall(
            tool_name="template_adaptation",
            arguments={
                "summary_request": context.request.summary_request,
                "requested_style": requested_style,
                "requested_scene": context.task.inputs.get("requested_scene"),
                "preferences": context.request.personalization_preferences,
            },
            timeout_seconds=min(context.task.timeout_seconds, 30),
        )
        tool_result = await context.tool_registry.execute(
            call,
            context=ToolContext(
                run_id=context.run_id,
                task_id=context.task.task_id,
                task_type=self.task_type,
            ),
            allowed_tools=context.task.allowed_tools,
            retry_attempts=context.tool_retry_attempts,
        )
        if tool_result.status != ToolStatus.SUCCESS:
            return AgentResult(
                task_id=context.task.task_id,
                task_type=self.task_type,
                status=(
                    AgentStatus.TIMEOUT
                    if tool_result.status == ToolStatus.TIMEOUT
                    else AgentStatus.FAILED
                ),
                tool_results=(tool_result,),
                error=tool_result.error or "template adaptation failed",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
        warnings = tuple(str(item) for item in payload.get("warnings", []))
        return AgentResult(
            task_id=context.task.task_id,
            task_type=self.task_type,
            status=AgentStatus.SUCCESS,
            payload=payload,
            tool_results=(tool_result,),
            warnings=warnings,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
