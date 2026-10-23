"""Incrementally persist model-facing input sizes without storing duplicate prompts."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODEL_CALL_INPUT_REPORT_VERSION = "summary_multi_agent.model_call_input_report.v2"


def _local_tokenizer_candidates() -> tuple[str, ...]:
    """Return explicitly configured local tokenizer candidates without network lookup."""

    values: list[str] = []
    for name in (
        "SMW_TOKENIZER_NAME_OR_PATH",
        "SMW_MODEL_TOKENIZER",
        "SMW_LOCAL_MODEL_PATH",
        "VLLM_MODEL",
    ):
        value = os.getenv(name, "").strip()
        if value and Path(value).expanduser().exists():
            values.append(str(Path(value).expanduser().resolve()))
    return tuple(dict.fromkeys(values))


class _TokenCounter:
    """Use a matching local tokenizer when configured; otherwise report an estimate."""

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._estimator = "utf8_bytes_div_4_estimate"
        for candidate in _local_tokenizer_candidates():
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self._estimator = f"tokenizer:{candidate}"
                break
            except Exception:
                continue

    def measure(self, text: str) -> dict[str, Any]:
        encoded = text.encode("utf-8")
        if self._tokenizer is not None:
            tokens = len(
                self._tokenizer.encode(text, add_special_tokens=False)
            )
            measured = True
        else:
            tokens = math.ceil(len(encoded) / 4) if encoded else 0
            measured = False
        return {
            "chars": len(text),
            "utf8_bytes": len(encoded),
            "tokens": tokens,
            "token_estimator": self._estimator,
            "token_count_measured": measured,
            "token_count_kind": (
                "local_tokenizer_prompt_text" if measured else "estimated_prompt_text"
            ),
        }


class ModelInputAuditRecorder:
    """Persist one append-safe report for Planner, Worker and Summarizer calls."""

    def __init__(
        self,
        output_dir: Path,
        *,
        context_window_tokens: int | None = None,
        safety_margin_tokens: int = 1024,
    ) -> None:
        self._path = output_dir / "model_call_input_report.json"
        self._lock = threading.RLock()
        self._counter = _TokenCounter()
        self._context_window_tokens = context_window_tokens
        self._safety_margin_tokens = safety_margin_tokens
        self._calls: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._worker_progress: list[dict[str, Any]] = []

    @property
    def path(self) -> Path:
        return self._path

    def start_call(
        self,
        *,
        stage: str,
        prompt_version: str,
        prompt: str,
        sections: dict[str, str],
        max_tokens: int,
        task_id: str | None = None,
        step_index: int | None = None,
        attempt_index: int | None = None,
        input_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Write the input report before the model call begins."""

        call_id = f"model_call_{uuid.uuid4().hex}"
        total_measurement = self._counter.measure(prompt)
        projected_total = (
            total_measurement["tokens"] + max_tokens + self._safety_margin_tokens
        )
        if self._context_window_tokens is None:
            context_status = "window_unconfigured"
        elif not total_measurement["token_count_measured"]:
            context_status = (
                "estimated_over_window_risk"
                if projected_total > self._context_window_tokens
                else "estimated_within_window"
            )
        else:
            context_status = (
                "measured_over_window"
                if projected_total > self._context_window_tokens
                else "measured_within_window"
            )
        record = {
            "call_id": call_id,
            "stage": stage,
            "task_id": task_id,
            "step_index": step_index,
            "attempt_index": attempt_index,
            "prompt_version": prompt_version,
            "max_output_tokens": max_tokens,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sections": {
                name: self._counter.measure(value)
                for name, value in sections.items()
            },
            "totals": total_measurement,
            "context_budget": {
                "configured_window_tokens": self._context_window_tokens,
                "max_output_tokens": max_tokens,
                "safety_margin_tokens": self._safety_margin_tokens,
                "projected_total_tokens": projected_total,
                "status": context_status,
            },
            "input_metrics": deepcopy(input_metrics) if input_metrics else {},
            "message_tokens": None,
            "server_usage_tokens": None,
            "model_lock_wait_ms": None,
            "model_call_duration_ms": None,
            "error": None,
        }
        with self._lock:
            self._calls[call_id] = record
            self._order.append(call_id)
            self._flush_locked()
        return call_id

    def finish_call(
        self,
        call_id: str,
        *,
        status: str,
        model_lock_wait_ms: int = 0,
        model_call_duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update a previously written call while retaining started calls on cancellation."""

        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                return
            record.update(
                {
                    "status": status,
                    "model_lock_wait_ms": model_lock_wait_ms,
                    "model_call_duration_ms": model_call_duration_ms,
                    "error": error,
                }
            )
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._flush_locked()

    def record_worker_progress(
        self,
        *,
        task_id: str,
        step_index: int,
        event: str,
        observation_ids: Iterable[str] = (),
        model_calls_used: int,
        tool_calls_used: int,
    ) -> None:
        """Checkpoint completed Worker progress independently of the final AgentResult."""

        with self._lock:
            self._worker_progress.append(
                {
                    "task_id": task_id,
                    "step_index": step_index,
                    "event": event,
                    "observation_ids": list(observation_ids),
                    "model_calls_used": model_calls_used,
                    "tool_calls_used": tool_calls_used,
                }
            )
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MODEL_CALL_INPUT_REPORT_VERSION,
            "calls": [self._calls[call_id] for call_id in self._order],
            "worker_progress": list(self._worker_progress),
        }
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self._path)
