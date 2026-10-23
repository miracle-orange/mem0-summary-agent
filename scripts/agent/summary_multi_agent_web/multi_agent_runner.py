"""把 Web 请求映射到现有 V2 Pipeline，并管理 Memory 连接生命周期。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from summary_memory_case_demo import (
    PROJECT_ROOT,
    build_summary_memory,
    build_summary_memory_config,
    close_summary_memory,
    patch_huggingface_embedding,
)
from summary_memory_llm import call_summary_llm, get_llm_runtime_info
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline
from summary_multi_agent.schemas import SummaryRunResult
from summary_multi_agent.schemas import RunStatus

from .schemas import (
    WebSummarizeRequest,
    build_run_config,
    build_summary_request,
    status_from_runtime,
)
from .store_probe import probe_memory_store


MemoryFactory = Callable[[], Any]
MemoryCloser = Callable[[Any], None]
PipelineBuilder = Callable[..., Any]


def memory_store_info(
    qdrant_path: str = "",
    collection: str = "",
) -> dict[str, str]:
    """返回本次有效 Store identity，不暴露凭据。"""
    project_root = os.getenv("MEM0_SUMMARY_ROOT", str(PROJECT_ROOT))
    return {
        "qdrant_path": qdrant_path.strip() or os.getenv(
            "MEM0_QDRANT_SUMMARY_PATH",
            str(Path(project_root) / "qdrant_summary"),
        ),
        "collection": collection.strip()
        or os.getenv("MEM0_SUMMARY_COLLECTION", "mem0_qwen_summary"),
    }


def build_web_memory(qdrant_path: str = "", collection: str = "") -> Any:
    """为 Web 运行构造 Memory 客户端，不改变 Baseline 全局配置。"""
    if not qdrant_path.strip() and not collection.strip():
        return build_summary_memory()
    from mem0 import Memory

    patch_huggingface_embedding()
    config = build_summary_memory_config()
    identity = memory_store_info(qdrant_path, collection)
    config["vector_store"]["config"]["path"] = identity["qdrant_path"]
    config["vector_store"]["config"]["collection_name"] = identity["collection"]
    return Memory.from_config(config)


def _write_json(path: Path, payload: Any) -> None:
    """将 Web 运行元数据写为可读 JSON。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _existing_started_at(path: Path, now: str) -> str:
    """从已有运行信息保留首次启动时间，损坏文件则使用当前时间。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return now
    return str(payload.get("started_at") or payload.get("created_at") or now)


def write_web_run_info(
    output_dir: Path,
    web_request: WebSummarizeRequest,
    *,
    status: str = "running",
    error: str = "",
) -> None:
    """写入 Web 运行 identity、Store/过滤条件、配置和状态元数据。"""
    now = datetime.now(timezone.utc).isoformat()
    info_path = output_dir / "web_run_info.json"
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        previous = {}
    payload = {
        "schema_version": "summary_multi_agent_web.run_info.v1",
        "run_type": "multi_agent_summarize",
        "status": status,
        "created_at": now,
        "started_at": _existing_started_at(info_path, now),
        "run_id": previous.get("run_id", ""),
        "output_dir": previous.get("output_dir", str(output_dir)),
        "updated_at": now,
        "user_id": web_request.user_id,
        "project": web_request.project,
        "source_time": web_request.source_time.strip() or None,
        "summary_style": web_request.summary_style,
        "meeting_scene": build_summary_request(web_request).personalization_preferences.get(
            "meeting_scene", "auto"
        ),
        "store": memory_store_info(web_request.qdrant_path, web_request.collection),
        "filters": {
            "user_id": web_request.user_id,
            "project": web_request.project,
            "memory_schema_version": "summary_typed_memory.v1",
        },
        "llm_runtime": get_llm_runtime_info(),
        "config": build_run_config(web_request).model_dump(mode="json"),
        "error": error,
    }
    if status in {"completed", "completed_with_warnings", "failed", "cancelling"}:
        payload["finished_at"] = now
    _write_json(output_dir / "web_run_info.json", payload)


async def run_multi_agent_summary(
    web_request: WebSummarizeRequest,
    output_dir: Path,
    *,
    memory_factory: MemoryFactory = build_summary_memory,
    memory_closer: MemoryCloser = close_summary_memory,
    llm_call: Callable[..., str] = call_summary_llm,
    pipeline_builder: PipelineBuilder = build_multi_agent_v2_pipeline,
    status_callback: Callable[[str], None] | None = None,
) -> SummaryRunResult:
    """在调用方后台事件循环中直接运行 V2，并管理 Memory 的 Probe 与关闭。

    Store Probe 为空时允许继续但追加历史增强 warning；默认生产 Memory 的 Probe
    出错会在 Pipeline 前失败。下游异常会写入 Web 运行信息后继续向调用方抛出。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    request = build_summary_request(web_request)
    config = build_run_config(web_request)
    write_web_run_info(output_dir, web_request)
    if status_callback is not None:
        status_callback("planning")

    effective_memory_factory = memory_factory
    if memory_factory is build_summary_memory and (
        web_request.qdrant_path.strip() or web_request.collection.strip()
    ):
        effective_memory_factory = lambda: build_web_memory(
            web_request.qdrant_path,
            web_request.collection,
        )
    memory = effective_memory_factory()
    try:
        probe = probe_memory_store(
            memory,
            user_id=web_request.user_id,
            project=web_request.project,
        ) if callable(getattr(memory, "get_all", None)) else {
            "schema_version": "summary_multi_agent_web.store_probe.v1",
            "status": "unavailable",
            "identity": {
                "user_id": web_request.user_id,
                "project": web_request.project,
            },
            "error": "injected Memory adapter does not provide get_all",
        }
        _write_json(output_dir / "memory_store_probe.json", probe)
        if probe.get("status") == "error" and memory_factory is build_summary_memory:
            raise RuntimeError(f"Summarize store 预检失败：{probe.get('error', 'unknown error')}")
        preflight_warning = ""
        if probe.get("status") == "empty":
            preflight_warning = "Summarize 预检发现当前身份范围内没有 Typed Memory，本次将不使用历史增强。"

        pipeline = pipeline_builder(
            memory=memory,
            llm_call=llm_call,
            user_id=web_request.user_id,
            project=web_request.project,
            output_dir=output_dir,
            config=config,
        )
        if status_callback is not None:
            status_callback("executing")
        result = await pipeline.run(request)
        if preflight_warning:
            result = result.model_copy(
                update={
                    "status": (
                        RunStatus.COMPLETED_WITH_WARNINGS
                        if result.status == RunStatus.COMPLETED
                        else result.status
                    ),
                    "warnings": tuple(
                        dict.fromkeys((*result.warnings, preflight_warning))
                    ),
                }
            )
    except Exception as exc:
        write_web_run_info(
            output_dir,
            web_request,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        memory_closer(memory)

    write_web_run_info(
        output_dir,
        web_request,
        status=result.status.value,
    )
    if status_callback is not None:
        status_callback(status_from_runtime(result.status.value).value)
    return result
