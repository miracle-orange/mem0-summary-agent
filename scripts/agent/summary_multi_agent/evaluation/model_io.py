"""评测模型调用、JSON 提取、有限修复与审计。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ..adapters.model import ContextOverflowError, ModelClient


ModelT = TypeVar("ModelT", bound=BaseModel)


class SharedEvaluationModelClient:
    """在多个评测阶段之间共享模型调用硬上限和串行锁。"""

    def __init__(self, model_client: ModelClient, *, max_calls: int) -> None:
        """保存底层客户端并初始化原子调用计数。"""
        self._model_client = model_client
        self._max_calls = max_calls
        self._lock = asyncio.Lock()
        self.model_call_count = 0

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """在共享预算内串行执行一次模型调用。"""
        async with self._lock:
            if self.model_call_count >= self._max_calls:
                raise RuntimeError(
                    "evaluation model call budget exhausted: "
                    f"limit={self._max_calls}"
                )
            self.model_call_count += 1
            return await getattr(self._model_client, "complete_json", self._model_client.complete)(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )


def extract_json_object(raw_output: str) -> dict[str, Any]:
    """从评测模型输出提取 JSON object，拒绝空结果和其他顶层类型。"""
    text = (raw_output or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output JSON must be an object")
    return payload


def normalize_known_payload_variants(value: Any) -> tuple[Any, list[str]]:
    """在严格 Pydantic 校验前归一化已知且无害的线格式变体。"""

    variants: list[str] = []

    def visit(item: Any) -> Any:
        """递归复制 payload，仅转换已知的整数 turn_id。"""
        if isinstance(item, list):
            return [visit(value) for value in item]
        if not isinstance(item, dict):
            return item
        normalized: dict[str, Any] = {}
        for key, child in item.items():
            if key == "turn_id" and isinstance(child, int) and not isinstance(child, bool):
                normalized[key] = str(child)
                variants.append("integer_turn_id_to_string")
            else:
                normalized[key] = visit(child)
        return normalized

    return visit(value), list(dict.fromkeys(variants))


class EvaluationModelIO:
    """管理 Evaluation Sidecar 的模型调用预算、修复和 trace。"""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        max_model_calls: int | None = None,
    ) -> None:
        """设置全局模型调用上限；该上限只约束评测，不影响生成主链。"""
        self._model_client = model_client
        self._max_model_calls = max_model_calls
        self.model_call_count = 0
        self.repair_call_count = 0
        self.traces: list[dict[str, Any]] = []

    def last_failure_kind(self, stage: str) -> str | None:
        """返回指定阶段最后一次已落入 trace 的失败类型。"""
        for trace in reversed(self.traces):
            if trace.get("stage") != stage:
                continue
            attempts = trace.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                return None
            value = attempts[-1].get("failure_kind")
            return str(value) if value else None
        return None

    async def complete_json(
        self,
        *,
        stage: str,
        prompt: str,
        response_model: type[ModelT],
        max_tokens: int,
        repair_attempts: int,
        timeout_seconds: int,
        validator: Callable[[ModelT], None] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> ModelT:
        """在单次 timeout 和全局调用预算内完成并校验一份评测 JSON。"""
        attempts: list[dict[str, Any]] = []
        next_prompt = prompt
        for attempt_index in range(repair_attempts + 1):
            raw_output = ""
            try:
                if (
                    self._max_model_calls is not None
                    and self.model_call_count >= self._max_model_calls
                ):
                    raise RuntimeError(
                        "evaluation model call budget exhausted: "
                        f"limit={self._max_model_calls}"
                    )
                self.model_call_count += 1
                if attempt_index:
                    self.repair_call_count += 1
                raw_output = await asyncio.wait_for(
                    getattr(self._model_client, "complete_json", self._model_client.complete)(
                        next_prompt,
                        temperature=0.0,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout_seconds,
                )
                payload, normalized_variants = normalize_known_payload_variants(
                    extract_json_object(raw_output)
                )
                parsed = response_model.model_validate(payload)
                if validator is not None:
                    validator(parsed)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_output": raw_output,
                        "error": None,
                        "normalized_variants": normalized_variants,
                    }
                )
                self.traces.append(
                    {
                        "stage": stage,
                        "prompt": prompt,
                        "metadata": {
                            **dict(trace_metadata or {}),
                            **(
                                {"normalized_variants": normalized_variants}
                                if normalized_variants
                                else {}
                            ),
                        },
                        "attempts": attempts,
                    }
                )
                return parsed
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failure_kind = _failure_kind(exc, raw_output)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_output": raw_output,
                        "error": error,
                        "failure_kind": failure_kind,
                    }
                )
                if isinstance(exc, ContextOverflowError):
                    self.traces.append(
                        {
                            "stage": stage,
                            "prompt": prompt,
                            "metadata": dict(trace_metadata or {}),
                            "attempts": attempts,
                        }
                    )
                    raise
                if attempt_index >= repair_attempts:
                    self.traces.append(
                        {
                            "stage": stage,
                            "prompt": prompt,
                            "metadata": dict(trace_metadata or {}),
                            "attempts": attempts,
                        }
                    )
                    raise
                repair_instruction = (
                    "上一次输出疑似被截断。请减少 reason/classification_reason 的长度，"
                    "不要遗漏任何原子事实，并只返回一个完整 JSON。"
                    if failure_kind == "likely_output_truncation"
                    else "上一次输出未通过 JSON Schema 或业务契约校验。"
                )
                next_prompt = (
                    prompt
                    + "\n\n"
                    + repair_instruction
                    + "错误："
                    + error
                    + "\n不要回显上一次输出，不要解释。"
                )
        raise AssertionError("evaluation repair loop exited unexpectedly")


def _failure_kind(exc: Exception, raw_output: str) -> str:
    """区分疑似截断、空输出和一般校验失败，避免误判模型异常。"""

    if isinstance(exc, ContextOverflowError):
        return "context_overflow"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    message = str(exc).lower()
    if "model call budget exhausted" in message:
        return "model_budget_exhausted"
    if "gateway returned an error" in message:
        return "gateway_error"
    if not raw_output.strip():
        return "empty_output"
    if "unterminated" in message or "does not contain a json object" in message:
        return "likely_output_truncation"
    if "expecting ',' delimiter" in message and not raw_output.rstrip().endswith("}"):
        return "likely_output_truncation"
    return "validation_error"
