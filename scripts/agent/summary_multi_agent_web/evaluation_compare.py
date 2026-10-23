"""Web V2 独立多纪要横评 Runner，不访问长期 Memory 或生成主链。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from summary_memory_llm import (
    OPENAI_COMPATIBLE_BACKEND, call_summary_llm, get_llm_backend, get_llm_runtime_info,
)
from summary_multi_agent.adapters.model import (
    AsyncOpenAICompatibleModelClient, SyncCallableModelClient,
)
from summary_multi_agent.evaluation.comparison import EvaluationComparisonRunner
from summary_multi_agent.evaluation.comparison_artifacts import (
    write_comparison_artifacts,
    write_comparison_failure_artifacts,
)
from summary_multi_agent.evaluation.schemas import EvaluationCompareRequest
from summary_multi_agent.evaluation.lite import (
    LiteEvaluationComparisonRunner,
    write_lite_comparison_artifacts,
    write_lite_comparison_failure_artifacts,
)
from summary_multi_agent.schemas import EvaluationMode


def _write_json(path: Path, value: object) -> None:
    """稳定写入 Web 横评元数据。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_compare_web_run_info(
    output_dir: Path,
    request: EvaluationCompareRequest,
    *,
    status: str,
    stage: str,
    error: str = "",
) -> None:
    """写入不包含原文和纪要全文的横评运行身份。"""
    path = output_dir / "web_run_info.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        previous = {}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": "summary_multi_agent_web.run_info.v1",
        "run_type": "evaluation_compare",
        "run_id": previous.get("run_id", ""),
        "output_dir": previous.get("output_dir", str(output_dir)),
        "project": request.project,
        "status": status,
        "stage": stage,
        "created_at": previous.get("created_at", now),
        "started_at": previous.get("started_at", now),
        "updated_at": now,
        "current_source_id": request.current_source.source_id,
        "current_source_time": request.current_source.source_time,
        "historical_source_count": len(request.historical_sources),
        "summary_count": len(request.summaries),
        "summary_ids": [item.summary_id for item in request.summaries],
        "target_entities": list(request.target_entities),
        "config": request.evaluation_config.model_dump(mode="json"),
        "llm_runtime": get_llm_runtime_info(),
        "memory_access": "disabled",
        "error": error,
    }
    if status in {"completed", "completed_with_warnings", "failed"}:
        payload["finished_at"] = now
    _write_json(path, payload)


async def run_web_evaluation_compare(
    request: EvaluationCompareRequest,
    output_dir: Path,
    *,
    llm_call: Callable[..., str] = call_summary_llm,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """执行只读横评并持久化结果；失败不触碰 Memory 或摘要运行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "evaluation_compare_request.json",
        request.model_dump(mode="json"),
    )
    write_compare_web_run_info(
        output_dir,
        request,
        status="running",
        stage="validating",
    )

    def update(stage: str) -> None:
        """同步运行阶段到 Web 元数据和 manager 回调。"""
        write_compare_web_run_info(
            output_dir,
            request,
            status="running",
            stage=stage,
        )
        if status_callback is not None:
            status_callback(stage)

    external_client = (
        AsyncOpenAICompatibleModelClient(trace_path=output_dir / "external_model_calls.jsonl")
        if llm_call is call_summary_llm and get_llm_backend() == OPENAI_COMPATIBLE_BACKEND
        else None
    )
    model_client = external_client or SyncCallableModelClient(llm_call)
    if request.evaluation_config.evaluation_mode == EvaluationMode.LITE:
        lite_runner = LiteEvaluationComparisonRunner(
            model_client=model_client, output_dir=output_dir
        )
        try:
            result = await lite_runner.run(request, status_callback=update)
            write_lite_comparison_artifacts(
                output_dir,
                request,
                result,
                model_trace=lite_runner.model_trace,
            )
            write_compare_web_run_info(
                output_dir,
                request,
                status=result.status,
                stage=result.status,
            )
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            write_lite_comparison_failure_artifacts(
                output_dir,
                request,
                stage=lite_runner.last_stage,
                error=error,
                reference=lite_runner.reference,
                model_trace=lite_runner.model_trace,
                execution_stats=lite_runner.execution_stats,
            )
            write_compare_web_run_info(
                output_dir,
                request,
                status="failed",
                stage=lite_runner.last_stage,
                error=error,
            )
            raise
        finally:
            if external_client is not None:
                await external_client.aclose()

    runner = EvaluationComparisonRunner(model_client=model_client)
    try:
        result = await runner.run(request, status_callback=update)
        write_comparison_artifacts(output_dir, request, result)
        write_compare_web_run_info(
            output_dir,
            request,
            status=result.status,
            stage=result.status,
        )
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_comparison_failure_artifacts(
            output_dir,
            request,
            stage=runner.last_stage,
            error=error,
            source_model_trace=runner.last_source_model_trace,
            source_personalization_model_trace=(
                runner.last_source_personalization_model_trace
            ),
            source_execution_stats=runner.last_source_execution_stats,
        )
        write_compare_web_run_info(
            output_dir,
            request,
            status="failed",
            stage=runner.last_stage,
            error=error,
        )
        raise
    finally:
        if external_client is not None:
            await external_client.aclose()
