"""提供 Review 阶段的结构化模型调用和有限 JSON 修复。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, TypeVar

from ..adapters.model import ContextOverflowError
from ..adapters.model import ModelClient
from ..model_io import (
    extract_json_object,
    extract_json_value,
    looks_like_repetitive_output,
    looks_like_truncated_json,
)
from ..model_input_audit import ModelInputAuditRecorder


T = TypeVar("T")

_AUDIT_SECTION_TAGS = (
    "current_material",
    "current_segment",
    "previous_candidate",
    "candidate_summary",
    "source_candidate",
)


def _review_audit_sections(prompt: str) -> dict[str, str]:
    """Split large Review source/candidate regions from instructions for audit."""

    sections: dict[str, str] = {}
    spans: list[tuple[int, int, str]] = []
    for tag in _AUDIT_SECTION_TAGS:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        start = prompt.find(open_tag)
        if start < 0:
            continue
        content_start = start + len(open_tag)
        end = prompt.find(close_tag, content_start)
        if end < 0:
            continue
        content = prompt[content_start:end].strip("\n")
        sections[tag] = content
        spans.append((content_start, end, tag))
    if not spans:
        return {"review_input": prompt}
    instructions = prompt
    for start, end, tag in sorted(spans, reverse=True):
        instructions = (
            instructions[:start]
            + f"\n[{tag} measured separately]\n"
            + instructions[end:]
        )
    return {"instructions_and_structured_context": instructions, **sections}


async def complete_validated_json(
    *,
    model_client: ModelClient,
    prompt: str,
    validator: Callable[[Any], T],
    max_tokens: int,
    timeout_seconds: int,
    repair_attempts: int,
    trace: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    compact_retry_hint: str = "",
    model_input_audit: ModelInputAuditRecorder | None = None,
) -> T:
    """执行 Review 阶段结构化模型调用，并在有限次数内修复 JSON/Schema。

    每次尝试都会追加到传入的 trace；修复耗尽后重新抛出最后一次异常，Review
    Loop 再决定将其转成 ``REVIEW_FAILED``，不修改正式摘要。
    """
    next_prompt = prompt
    attempts: list[dict[str, Any]] = []
    for attempt in range(repair_attempts + 1):
        raw_output: str | None = None
        call_id: str | None = None
        started = time.perf_counter()
        if model_input_audit is not None:
            call_id = model_input_audit.start_call(
                stage=f"review_{trace_metadata.get('stage', 'model')}",
                prompt_version="summary_multi_agent.review_prompt.v3",
                prompt=next_prompt,
                sections=_review_audit_sections(next_prompt),
                max_tokens=max_tokens,
                attempt_index=attempt + 1,
            )
        try:
            raw_output = await asyncio.wait_for(
                getattr(model_client, "complete_json", model_client.complete)(
                    next_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_seconds,
            )
            value = validator(extract_json_value(raw_output))
            if call_id is not None and model_input_audit is not None:
                model_input_audit.finish_call(
                    call_id,
                    status="completed",
                    model_call_duration_ms=int(
                        (time.perf_counter() - started) * 1000
                    ),
                )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "prompt": next_prompt,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "raw_output": raw_output,
                    "error": None,
                }
            )
            trace.append({**trace_metadata, "attempts": attempts})
            return value
        except Exception as exc:
            if call_id is not None and model_input_audit is not None:
                model_input_audit.finish_call(
                    call_id,
                    status="failed",
                    model_call_duration_ms=int(
                        (time.perf_counter() - started) * 1000
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ContextOverflowError):
                failure_kind = "context_overflow"
            elif looks_like_repetitive_output(raw_output):
                failure_kind = "repetitive_output"
            elif looks_like_truncated_json(raw_output, exc):
                failure_kind = "likely_output_truncation"
            else:
                failure_kind = "validation_error"
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "prompt": next_prompt,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "raw_output": raw_output,
                    "error": error,
                    "failure_kind": failure_kind,
                }
            )
            if isinstance(exc, ContextOverflowError):
                trace.append({**trace_metadata, "attempts": attempts})
                raise
            if attempt >= repair_attempts:
                trace.append({**trace_metadata, "attempts": attempts})
                raise
            if failure_kind in {"likely_output_truncation", "repetitive_output"}:
                failure_note = (
                    "上一次输出出现了连续重复，未形成完整 JSON。"
                    if failure_kind == "repetitive_output"
                    else "上一次输出未形成完整 JSON，可能因长度被截断。"
                )
                next_prompt = (
                    prompt
                    + "\n\n"
                    + failure_note
                    + "请丢弃而不是续写上一次输出，从头重新生成更紧凑的完整 JSON。"
                    + "每个文本字段只写一个短句，同一短语不得连续重复；"
                    + "不重复证据原文，不输出 Markdown 或额外解释。"
                    + "不要通过删除必须保留的事实规避长度限制。"
                    + compact_retry_hint
                    + "\n解析错误："
                    + error
                )
            else:
                next_prompt = (
                    prompt
                    + "\n\n上一次输出未通过 JSON Schema 校验。请修复后只返回完整 JSON。"
                    + "不得删除事实来规避校验。错误："
                    + error
                    + "\n上一次输出：\n"
                    + str(raw_output or "<no output>")[:6000]
                )
    raise AssertionError("review model validation loop exited unexpectedly")
