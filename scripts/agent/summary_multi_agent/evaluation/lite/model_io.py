"""精简评测的共享调用预算、有限修复、缓存和输入审计。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ...adapters.model import (
    ContextOverflowError,
    ContextWindowModelClient,
    ModelClient,
)
from ...model_input_audit import _TokenCounter
from ...schemas import LiteEvaluationConfig
from ..model_io import extract_json_object
from .prompts import PROMPT_VERSION

ModelT = TypeVar("ModelT", bound=BaseModel)


class LiteEvaluationModelIO:
    """在一轮 lite 评测内共享预算，并保留每次实际请求的可审计记录。"""

    def __init__(
        self,
        model_client: ModelClient,
        config: LiteEvaluationConfig,
        *,
        trace_path: Path | None = None,
    ) -> None:
        self._model_client = ContextWindowModelClient(
            model_client,
            context_window_tokens=config.model_context_window_tokens,
            safety_margin_tokens=config.context_safety_margin_tokens,
        )
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_model_calls)
        self._budget_lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._counter = _TokenCounter()
        self._trace_path = trace_path
        self.model_call_count = 0
        self.repair_call_count = 0
        self.cache_hit_count = 0
        self.traces: list[dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        stage: str,
        prompt: str,
        sections: dict[str, str],
        response_model: type[ModelT],
        max_tokens: int,
        validator: Callable[[ModelT], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelT:
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "stage": stage,
                    "prompt": prompt,
                    "response_model": response_model.__name__,
                    "max_tokens": max_tokens,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            parsed = response_model.model_validate(cached)
            if validator is not None:
                validator(parsed)
            self.cache_hit_count += 1
            self.traces.append(
                {
                    "stage": stage,
                    "prompt_version": PROMPT_VERSION,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "status": "completed",
                    "metadata": dict(metadata or {}),
                    "input": self._input_metrics(prompt, sections, max_tokens),
                    "attempts": [],
                    "prompt": prompt,
                }
            )
            self._flush_traces()
            return parsed

        attempts: list[dict[str, Any]] = []
        trace_record = {
            "stage": stage,
            "prompt_version": PROMPT_VERSION,
            "cache_key": cache_key,
            "cache_hit": False,
            "status": "started",
            "metadata": dict(metadata or {}),
            "input": self._input_metrics(prompt, sections, max_tokens),
            "attempts": attempts,
            "prompt": prompt,
        }
        self.traces.append(trace_record)
        self._flush_traces()
        next_prompt = prompt
        for attempt_index in range(self._config.repair_attempts + 1):
            call_started = time.monotonic()
            raw_output = ""
            wait_started = time.monotonic()
            try:
                async with self._semaphore:
                    lock_wait_ms = int((time.monotonic() - wait_started) * 1000)
                    await self._reserve_call()
                    if attempt_index:
                        self.repair_call_count += 1
                    raw_output = await asyncio.wait_for(
                        getattr(self._model_client, "complete_json", self._model_client.complete)(
                            next_prompt,
                            temperature=0.0,
                            max_tokens=max_tokens,
                        ),
                        timeout=self._config.model_timeout_seconds,
                    )
                parsed = response_model.model_validate(extract_json_object(raw_output))
                if validator is not None:
                    validator(parsed)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "completed",
                        "lock_wait_ms": lock_wait_ms,
                        "duration_ms": int((time.monotonic() - call_started) * 1000),
                        "raw_output": raw_output,
                        "error": None,
                    }
                )
                self._cache[cache_key] = parsed.model_dump(mode="json")
                trace_record["status"] = "completed"
                self._flush_traces()
                return parsed
            except asyncio.CancelledError:
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "cancelled",
                        "duration_ms": int((time.monotonic() - call_started) * 1000),
                        "raw_output": raw_output,
                        "error": "CancelledError",
                    }
                )
                trace_record["status"] = "cancelled"
                self._flush_traces()
                raise
            except Exception as exc:
                failure_kind = self._failure_kind(exc, raw_output)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "failed",
                        "duration_ms": int((time.monotonic() - call_started) * 1000),
                        "raw_output": raw_output,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failure_kind": failure_kind,
                    }
                )
                repairable = failure_kind in {
                    "invalid_json",
                    "validation_error",
                    "likely_output_truncation",
                }
                if not repairable or attempt_index >= self._config.repair_attempts:
                    trace_record["status"] = "failed"
                    self._flush_traces()
                    raise
                trace_record["status"] = "repairing"
                self._flush_traces()
                next_prompt = (
                    prompt
                    + "\n\n上一次返回未通过 JSON 或编号契约校验。"
                    + "请缩短文本，只返回完整 JSON，不要解释。错误："
                    + f"{type(exc).__name__}: {exc}"
                )
        raise AssertionError("lite evaluation repair loop exited unexpectedly")

    async def _reserve_call(self) -> None:
        async with self._budget_lock:
            if self.model_call_count >= self._config.max_model_calls:
                raise RuntimeError(
                    "lite evaluation model call budget exhausted: "
                    f"limit={self._config.max_model_calls}"
                )
            self.model_call_count += 1

    def _input_metrics(
        self, prompt: str, sections: dict[str, str], max_tokens: int
    ) -> dict[str, Any]:
        total = self._counter.measure(prompt)
        projected_total = (
            total["tokens"] + max_tokens + self._config.context_safety_margin_tokens
        )
        if self._config.model_context_window_tokens is None:
            context_status = "window_unconfigured"
        elif total["token_count_measured"]:
            context_status = (
                "measured_over_window"
                if projected_total > self._config.model_context_window_tokens
                else "measured_within_window"
            )
        else:
            context_status = (
                "estimated_over_window_risk"
                if projected_total > self._config.model_context_window_tokens
                else "estimated_within_window"
            )
        return {
            "sections": {
                key: self._counter.measure(value) for key, value in sections.items()
            },
            "totals": total,
            "max_output_tokens": max_tokens,
            "context_budget": {
                "configured_window_tokens": (self._config.model_context_window_tokens),
                "safety_margin_tokens": (self._config.context_safety_margin_tokens),
                "projected_total_tokens": projected_total,
                "status": context_status,
            },
        }

    def _flush_traces(self) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_path.write_text(
            json.dumps(self.traces, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _failure_kind(exc: Exception, raw_output: str) -> str:
        if isinstance(exc, ContextOverflowError):
            return "context_overflow"
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        message = str(exc).lower()
        if "model call budget exhausted" in message:
            return "model_budget_exhausted"
        if not raw_output.strip():
            return "empty_output"
        if "unterminated" in message or "does not contain a json object" in message:
            return "likely_output_truncation"
        if isinstance(exc, json.JSONDecodeError):
            return "invalid_json"
        return "validation_error"
