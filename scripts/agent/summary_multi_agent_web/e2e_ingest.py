"""编排 E2E 历史入库：raw chunk、Typed Memory、去重和 infer=False 写入。"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from summary_memory_case_demo import (
    build_summary_memory,
    build_summary_memory_config,
    close_summary_memory,
    deduplicate_batch_typed_memory_records,
    patch_huggingface_embedding,
    write_typed_summary_memory_records,
)
from summary_memory_e2e_preprocess import (
    RawTextSource,
    get_preprocess_config,
    save_e2e_history_preprocess_outputs,
    split_text_into_chunks,
    summarize_raw_history_sources,
)
from summary_memory_llm import (
    OPENAI_COMPATIBLE_BACKEND, call_summary_llm, get_llm_backend,
    get_llm_runtime_info, make_audited_llm_call,
)
from summary_memory_typed_memory import extract_typed_memories_from_summary
from summary_memory_agent_schemas import stable_typed_memory_id

from .multi_agent_runner import (
    build_web_memory,
    memory_store_info,
    run_multi_agent_summary,
)
from .ingest_config import resolve_ingest_config
from .schemas import (
    WebIngestRequest,
    WebE2ERequest,
    WebE2ERunResult,
    WebIngestResult,
    WebHistorySource,
    WebRunStatus,
    WebSummarizeRequest,
    status_from_runtime,
)
from .store_probe import (
    operation_ids_from_add_results,
    probe_memory_store,
    reopen_and_verify,
)


MemoryFactory = Callable[[], Any]
MemoryCloser = Callable[[Any], None]


def _write_json(path: Path, payload: Any) -> None:
    """创建父目录并写入 JSON 产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    """创建父目录并写入带统一末尾换行的文本产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_e2e_web_run_info(
    output_dir: Path,
    request: WebE2ERequest,
    *,
    status: WebRunStatus,
    stage: str,
    error: str = "",
) -> None:
    """更新 E2E 顶层运行状态、阶段、Store identity 和失败信息。"""
    now = datetime.now(timezone.utc).isoformat()
    info_path = output_dir / "web_run_info.json"
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        previous = {}
    _write_json(
        info_path,
        {
            "schema_version": "summary_multi_agent_web.run_info.v1",
            "run_type": getattr(request, "run_type", "e2e_ingest"),
            "status": status.value,
            "stage": stage,
            "created_at": now,
            "started_at": previous.get("started_at", now),
            "run_id": previous.get("run_id", ""),
            "output_dir": previous.get("output_dir", str(output_dir)),
            "updated_at": now,
            "user_id": request.user_id,
            "project": request.project,
            "source_time": request.source_time.strip() or None,
            "source_type": request.source_type,
            "store": memory_store_info(
                request.qdrant_path
                or (request.summary.qdrant_path if request.summary else ""),
                request.collection
                or (request.summary.collection if request.summary else ""),
            ),
            "filters": {
                "user_id": request.user_id,
                "project": request.project,
                "memory_schema_version": "summary_typed_memory.v1",
            },
            "e2e_ingest_dir": str(output_dir / "e2e_ingest"),
            "multi_agent_summarize_dir": str(output_dir / "multi_agent_summarize"),
            "error": error,
            **(
                {"finished_at": now}
                if status
                in {
                    WebRunStatus.COMPLETED,
                    WebRunStatus.COMPLETED_WITH_WARNINGS,
                    WebRunStatus.FAILED,
                    WebRunStatus.CANCELLING,
                }
                else {}
            ),
        },
    )


def _write_e2e_result(output_dir: Path, result: WebE2ERunResult) -> None:
    """保存 E2E 对外结果快照。"""
    _write_json(output_dir / "e2e_run_result.json", result.model_dump(mode="json"))


def _summary_result_payload(result: Any) -> dict[str, Any]:
    """将核心 SummaryRunResult 或测试替身转换为 E2E JSON payload。"""
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {
        "status": getattr(getattr(result, "status", None), "value", "failed"),
        "final_summary": getattr(result, "final_summary", None),
        "recommended_candidate_index": getattr(
            result, "recommended_candidate_index", None
        ),
        "warnings": list(getattr(result, "warnings", ()) or ()),
    }


def history_sources_from_request(
    sources: list[WebHistorySource],
) -> list[RawTextSource]:
    """把 Web 历史输入转换为既有预处理器使用的 RawTextSource。"""
    return [
        RawTextSource(
            label=source.label,
            text=source.text,
            source_path=Path(source.source_path) if source.source_path else None,
        )
        for source in sources
    ]


def build_raw_chunk_records(
    sources: list[RawTextSource],
    *,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    chunk_char_limit: int | None = None,
    source_times: list[str] | None = None,
    overlap_chars: int = 0,
) -> list[dict[str, Any]]:
    """按原始 source label 分块，准备 Typed Memory 抽取输入记录。"""
    # 这里复用既有 E2E 分块规则；history_source_id 保持为原始 source label，
    # 这是后续跨 chunk 去重限定在同一份历史材料内的关键锚点。
    chunk_limit = chunk_char_limit or get_preprocess_config()["chunk_char_limit"]
    overlap_chars = min(overlap_chars, max(0, chunk_limit - 100))
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        base_chunks = split_text_into_chunks(source.text, chunk_limit - overlap_chars)
        chunks = [
            (base_chunks[index - 1][-overlap_chars:] + chunk
             if overlap_chars and index else chunk)
            for index, chunk in enumerate(base_chunks)
        ]
        if not chunks:
            raise ValueError(f"历史原始资料为空：{source.label}")
        for chunk_index, raw_chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "raw_chunk": raw_chunk,
                    "user_id": user_id,
                    "project": project,
                    "source_type": source_type,
                    "source_time": (source_times[source_index] if source_times else source_time),
                }
            )
    return records


def _save_chunk_extraction_traces(output_dir: Path, results: list[dict[str, Any]]) -> None:
    """保存每个 chunk 的有限请求、结果计数和错误 trace。"""
    trace_dir = output_dir / "typed_memory_extraction_traces"
    for index, item in enumerate(results, start=1):
        trace = item.get("extraction_trace", {})
        stem = f"{index:03d}_{item['history_source_id']}_typed_memory_extraction"
        prompt = str(trace.get("prompt", ""))
        raw_output = str(trace.get("raw_output", ""))
        if prompt:
            _write_text(trace_dir / f"{stem}_prompt.txt", prompt)
        if raw_output:
            _write_text(trace_dir / f"{stem}_raw_output.txt", raw_output)
        _write_json(
            trace_dir / f"{stem}_trace.json",
            {
                "status": trace.get("status", "completed"),
                "chunk_index": item.get("chunk_index"),
                "history_source_id": item.get("history_source_id", ""),
                "max_tokens": trace.get("max_tokens"),
                "model": trace.get("model"),
                "error": trace.get("error", ""),
                "raw_output_chars": len(raw_output),
                "raw_record_count": trace.get("raw_record_count"),
                "accepted_record_count": trace.get("accepted_record_count"),
                "rejected_record_count": trace.get("rejected_record_count"),
            },
        )


def _save_ingest_outputs(
    output_dir: Path,
    request: WebIngestRequest | WebE2ERequest,
    source_records: list[dict[str, Any]],
    dedup_report: dict[str, Any],
    *,
    resolved_config: dict[str, int],
    probe: dict[str, Any] | None = None,
    readback: dict[str, Any] | None = None,
) -> None:
    """保存 Typed Memory、去重、Store Probe 和 Readback 汇总产物。"""
    typed_payload = [
        {
            "source_file": item.get("source_file"),
            "history_source_id": item.get("history_source_id", ""),
            "typed_memories": item.get("typed_memories", []),
        }
        for item in source_records
    ]
    add_payload = [
        {
            "source_file": item.get("source_file"),
            "history_source_id": item.get("history_source_id", ""),
            "add_results": item.get("add_results", []),
        }
        for item in source_records
    ]
    _write_json(output_dir / "typed_memories.json", typed_payload)
    _write_json(output_dir / "add_results.json", add_payload)
    _write_json(output_dir / "typed_memory_dedup_report.json", dedup_report)
    _write_json(
        output_dir / "web_ingest_info.json",
        {
            "schema_version": "summary_multi_agent_web.ingest.v1",
            "run_type": getattr(request, "run_type", "ingest"),
            "user_id": request.user_id,
            "project": request.project,
            "source_type": request.source_type,
            "source_time": request.source_time,
            "store": memory_store_info(
                request.qdrant_path
                or (request.summary.qdrant_path if getattr(request, "summary", None) else ""),
                request.collection
                or (request.summary.collection if getattr(request, "summary", None) else ""),
            ),
            "llm_runtime": get_llm_runtime_info(),
            "preprocess_config": resolved_config,
            "typed_memory_source": "raw_history_chunks",
            "memory_type_contract": [
                "time",
                "decision",
                "constraint",
                "task",
                "risk",
                "preference",
            ],
            "add_call_count": _add_call_count(source_records),
            "operation_count": _operation_count(source_records),
            "event_counts": _event_counts(source_records),
            "store_probe": probe or {},
            "readback": readback or {},
        },
    )


def _write_ingest_stage_report(
    output_dir: Path,
    *,
    stage: str,
    status: str,
    request: WebIngestRequest,
    config: dict[str, int],
    error: str = "",
    chunk_index: int | None = None,
    add_call_count: int = 0,
    operation_count: int = 0,
) -> None:
    """写入当前 Ingest 阶段的状态、计数、配置和错误报告。"""
    _write_json(
        output_dir / "ingest_stage_report.json",
        {
            "schema_version": "summary_multi_agent_web.ingest_stage.v1",
            "status": status,
            "stage": stage,
            "user_id": request.user_id,
            "project": request.project,
            "source_time": request.source_time,
            "source_type": request.source_type,
            "store": memory_store_info(request.qdrant_path, request.collection),
            "config": config,
            "chunk_index": chunk_index,
            "add_call_count": add_call_count,
            "operation_count": operation_count,
            "error": error,
        },
    )


def _write_add_journal(output_dir: Path, batch_results: list[dict[str, Any]]) -> None:
    """追加已完成 ADD/UPDATE 批次，保留失败恢复所需的写入边界。"""
    _write_json(
        output_dir / "ingest_write_journal.json",
        [
            {
                "history_source_id": item.get("history_source_id", ""),
                "source_file": item.get("source_file"),
                "add_results": item.get("add_results", []),
            }
            for item in batch_results
            if item.get("add_results")
        ],
    )


def _event_counts(batch_results: list[dict[str, Any]]) -> dict[str, int]:
    """统计 Mem0 写入响应中的 ADD/UPDATE 等操作事件。"""
    counts: dict[str, int] = {}
    for item in batch_results:
        for result in item.get("add_results", []):
            events = result.get("results", []) if isinstance(result, dict) else []
            if not isinstance(events, list) or not events:
                events = [result]
            for event_result in events:
                if not isinstance(event_result, dict):
                    continue
                event = str(
                    event_result.get("event", event_result.get("operation", "unknown"))
                )
                counts[event] = counts.get(event, 0) + 1
    return counts


def _add_call_count(batch_results: list[dict[str, Any]]) -> int:
    """统计实际发出的 Memory add 调用数量。"""
    return sum(len(item.get("add_results", [])) for item in batch_results)


def _operation_count(batch_results: list[dict[str, Any]]) -> int:
    """统计嵌套响应中的实际 Memory 操作数量。"""
    total = 0
    for item in batch_results:
        for result in item.get("add_results", []):
            nested = result.get("results", []) if isinstance(result, dict) else []
            total += len(nested) if isinstance(nested, list) and nested else 1
    return total


def _coerce_ingest_request(request: WebIngestRequest | WebE2ERequest) -> WebIngestRequest:
    """将 E2E 请求投影为独立 Ingest 请求，保留身份和 Store 配置。"""
    if isinstance(request, WebIngestRequest):
        return request
    return WebIngestRequest(
        execution_profile=request.execution_profile,
        history_sources=request.history_sources,
        user_id=request.user_id,
        project=request.project,
        source_time=request.source_time,
        source_type=request.source_type,
        qdrant_path=request.qdrant_path,
        collection=request.collection,
    )


def ingest_history_sources(
    request: WebIngestRequest | WebE2ERequest,
    output_dir: Path,
    *,
    memory_factory: MemoryFactory = build_summary_memory,
    memory_closer: MemoryCloser = close_summary_memory,
    llm_call: Callable[..., str] = call_summary_llm,
    verify_readback: bool = False,
) -> dict[str, Any]:
    """完成历史资料 Ingest：抽取、全局去重、写入、journal 和可选回读。

    所有 chunk 必须先抽取完成，随后才统一去重和写入；Memory 写入发生在该
    函数的唯一写入阶段，写后可通过关闭重开客户端验证持久性。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_request = _coerce_ingest_request(request)
    cloud_fast = ingest_request.execution_profile == "cloud_fast"
    external_json_mode = (
        cloud_fast and llm_call is call_summary_llm
        and get_llm_backend() == OPENAI_COMPATIBLE_BACKEND
    )
    if llm_call is call_summary_llm and get_llm_backend() == OPENAI_COMPATIBLE_BACKEND:
        llm_call = make_audited_llm_call(
            output_dir / "external_model_calls.jsonl", json_output=external_json_mode
        )
    resolved_config = resolve_ingest_config(
        ingest_request.config, execution_profile=ingest_request.execution_profile
    )
    _write_ingest_stage_report(
        output_dir,
        stage="validation",
        status="running",
        request=ingest_request,
        config=resolved_config,
    )
    sources = history_sources_from_request(ingest_request.history_sources)
    if cloud_fast:
        _write_ingest_stage_report(
            output_dir, stage="preprocessing", status="skipped",
            request=ingest_request, config=resolved_config,
        )
    else:
        try:
            _write_ingest_stage_report(
                output_dir, stage="preprocessing", status="running",
                request=ingest_request, config=resolved_config,
            )
            summaries = summarize_raw_history_sources(
                type("WorkflowAdapter", (), {"call_summary_llm": staticmethod(llm_call)})(),
                sources, config=resolved_config,
            )
        except Exception as exc:
            _write_ingest_stage_report(
                output_dir, stage="preprocessing", status="failed",
                request=ingest_request, config=resolved_config,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        save_e2e_history_preprocess_outputs(
            output_dir, summaries,
            metadata={
                "user_id": ingest_request.user_id,
                "project": ingest_request.project,
                "source_type": ingest_request.source_type,
                "source_time": ingest_request.source_time,
                "typed_memory_source": "raw_history_chunks",
            }, config=resolved_config,
        )
        _write_ingest_stage_report(
            output_dir, stage="preprocessing", status="completed",
            request=ingest_request, config=resolved_config,
        )

    chunk_inputs = build_raw_chunk_records(
        sources,
        user_id=ingest_request.user_id,
        project=ingest_request.project,
        source_type=ingest_request.source_type,
        source_time=ingest_request.source_time,
        chunk_char_limit=resolved_config["chunk_char_limit"],
        source_times=[source.source_time or ingest_request.source_time
                      for source in ingest_request.history_sources],
        overlap_chars=100 if cloud_fast else 0,
    )
    qdrant_path = ingest_request.qdrant_path or (
        request.summary.qdrant_path if getattr(request, "summary", None) else ""
    )
    collection = ingest_request.collection or (
        request.summary.collection if getattr(request, "summary", None) else ""
    )
    effective_memory_factory = memory_factory
    if memory_factory is build_summary_memory and (qdrant_path or collection):
        effective_memory_factory = lambda: build_web_memory(qdrant_path, collection)
    try:
        memory = effective_memory_factory()
    except Exception as exc:
        _write_ingest_stage_report(
            output_dir,
            stage="extracting",
            status="failed",
            request=ingest_request,
            config=resolved_config,
            error=f"Memory open failed: {type(exc).__name__}: {exc}",
        )
        raise
    _write_ingest_stage_report(
        output_dir,
        stage="extracting",
        status="running",
        request=ingest_request,
        config=resolved_config,
        chunk_index=0,
    )
    try:
        batch_results: list[dict[str, Any]] = []
        try:
            extraction_runtime = get_llm_runtime_info()
            extraction_model = extraction_runtime.get("llm_model", "")
        except Exception:
            extraction_model = ""
        try:
            def extract_one(item: dict[str, Any]) -> dict[str, Any]:
                source = item["source"]
                extraction_trace: dict[str, Any] = {
                    "max_tokens": resolved_config["typed_extraction_max_tokens"],
                    "model": extraction_model,
                }
                typed_records = extract_typed_memories_from_summary(
                    summary_text=item["raw_chunk"],
                    llm_call=llm_call,
                    source_time=item["source_time"],
                    source_file=source.source_path,
                    max_tokens=resolved_config["typed_extraction_max_tokens"],
                    trace=extraction_trace,
                    json_object_output=external_json_mode,
                )
                if cloud_fast:
                    typed_records = [
                        replace(
                            record,
                            source={**record.source, "source_time": item["source_time"]},
                            memory_id=stable_typed_memory_id(
                                record.memory_type, record.content, item["source_time"],
                                record.source.get("evidence_span", ""),
                            ),
                        )
                        for record in typed_records
                    ]
                return {
                        "source_file": source.source_path,
                        "history_source_id": source.label,
                        "chunk_index": item["chunk_index"],
                        "source_time": item["source_time"],
                        "typed_records": typed_records,
                        "typed_memories": [],
                        "extraction_trace": {
                            **extraction_trace,
                            "status": "completed",
                        },
                    }

            if cloud_fast:
                ordered: list[dict[str, Any] | None] = [None] * len(chunk_inputs)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(extract_one, item): index
                        for index, item in enumerate(chunk_inputs)
                    }
                    for future in as_completed(futures):
                        try:
                            ordered[futures[future]] = future.result()
                        except Exception:
                            for pending in futures:
                                pending.cancel()
                            batch_results = [item for item in ordered if item is not None]
                            raise
                batch_results = [item for item in ordered if item is not None]
                _save_chunk_extraction_traces(output_dir, batch_results)
                _write_ingest_stage_report(
                    output_dir,
                    stage="extracting",
                    status="running",
                    request=ingest_request,
                    config=resolved_config,
                    chunk_index=len(batch_results),
                )
            else:
                for item in chunk_inputs:
                    batch_results.append(extract_one(item))
                    _save_chunk_extraction_traces(output_dir, batch_results)
                    _write_ingest_stage_report(
                        output_dir, stage="extracting", status="running",
                        request=ingest_request, config=resolved_config,
                        chunk_index=item["chunk_index"],
                    )
        except Exception as exc:
            # 模型可能在返回 raw output 前就被网关拒绝。保留这个失败 chunk
            # 的有限请求元数据，不写入任何 Typed Memory，也不保存隐藏思维链。
            failed_item = locals().get("item")
            failed_source = failed_item.get("source") if isinstance(failed_item, dict) else None
            failed_trace = locals().get("extraction_trace")
            if isinstance(failed_item, dict) and failed_source is not None:
                if not isinstance(failed_trace, dict):
                    failed_trace = {}
                failed_trace.update(
                    {
                        "status": "failed",
                        "max_tokens": resolved_config["typed_extraction_max_tokens"],
                        "model": extraction_model,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                batch_results.append(
                    {
                        "source_file": failed_source.source_path,
                        "history_source_id": failed_source.label,
                        "chunk_index": failed_item["chunk_index"],
                        "source_time": failed_item["source_time"],
                        "typed_records": [],
                        "typed_memories": [],
                        "extraction_trace": failed_trace,
                    }
                )
            _save_chunk_extraction_traces(output_dir, batch_results)
            _write_ingest_stage_report(
                output_dir,
                stage="extracting",
                status="failed",
                request=ingest_request,
                config=resolved_config,
                chunk_index=len(batch_results) + 1,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        _write_ingest_stage_report(
            output_dir,
            stage="extracting",
            status="completed",
            request=ingest_request,
            config=resolved_config,
            chunk_index=len(batch_results),
        )

        # 必须先完成所有 chunk 的抽取，再统一去重和写入；否则同源记录可能
        # 因为处理顺序不同而产生不一致的入库结果。
        _write_ingest_stage_report(
            output_dir,
            stage="deduplicating",
            status="running",
            request=ingest_request,
            config=resolved_config,
        )
        try:
            dedup_report = deduplicate_batch_typed_memory_records(
                batch_results,
                judge_llm_call=llm_call,
                judge_max_tokens=resolved_config["dedup_judge_max_tokens"],
                judge_max_candidates=resolved_config["dedup_judge_max_candidates"],
            )
        except Exception as exc:
            _write_ingest_stage_report(
                output_dir,
                stage="deduplicating",
                status="failed",
                request=ingest_request,
                config=resolved_config,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_ingest_stage_report(
            output_dir,
            stage="deduplicating",
            status="completed",
            request=ingest_request,
            config=resolved_config,
        )
        _write_ingest_stage_report(
            output_dir,
            stage="writing",
            status="running",
            request=ingest_request,
            config=resolved_config,
        )
        try:
            for item in batch_results:
                typed_memories, memory_texts, add_results = write_typed_summary_memory_records(
                    memory=memory,
                    typed_records=item["typed_records"],
                    user_id=ingest_request.user_id,
                    project=ingest_request.project,
                    source_type=ingest_request.source_type,
                    source_time=item["source_time"],
                    source_file=item.get("source_file"),
                )
                item["typed_memories"] = typed_memories
                item["memories"] = memory_texts
                item["add_results"] = add_results
                _write_add_journal(output_dir, batch_results)
        except Exception as exc:
            _write_add_journal(output_dir, batch_results)
            _write_ingest_stage_report(
                output_dir,
                stage="writing",
                status="failed",
                request=ingest_request,
                config=resolved_config,
                add_call_count=_add_call_count(batch_results),
                operation_count=_operation_count(batch_results),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_ingest_stage_report(
            output_dir,
            stage="writing",
            status="completed",
            request=ingest_request,
            config=resolved_config,
            add_call_count=_add_call_count(batch_results),
            operation_count=_operation_count(batch_results),
        )
    finally:
        memory_closer(memory)

    operation_ids = operation_ids_from_add_results(
        [result for item in batch_results for result in item.get("add_results", [])]
    )
    probe: dict[str, Any] = {}
    readback: dict[str, Any] = {}
    warnings: list[str] = []
    if verify_readback:
        _write_ingest_stage_report(
            output_dir,
            stage="verifying",
            status="running",
            request=ingest_request,
            config=resolved_config,
            add_call_count=_add_call_count(batch_results),
            operation_count=len(operation_ids),
        )
        try:
            probe, readback = reopen_and_verify(
                effective_memory_factory,
                memory_closer,
                operation_ids,
                user_id=ingest_request.user_id,
                project=ingest_request.project,
            )
            _write_json(output_dir / "memory_store_probe.json", probe)
            _write_json(output_dir / "ingest_readback_report.json", readback)
        except Exception as exc:
            _write_ingest_stage_report(
                output_dir,
                stage="verifying",
                status="failed",
                request=ingest_request,
                config=resolved_config,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        verified = int(readback.get("recovered_operation_count", 0))
        try:
            if probe.get("status") == "error":
                raise RuntimeError(f"写入后 store 预检失败：{probe.get('error', 'unknown error')}")
            if operation_ids and verified == 0:
                raise RuntimeError("写入操作全部无法通过关闭重开后的 store 回读")
            if operation_ids and verified < len(operation_ids):
                warnings.append(
                    f"写入后回读不完整：{verified}/{len(operation_ids)} 个操作可回读。"
                )
            if not operation_ids and probe.get("status") == "ready":
                warnings.append("本次没有产生新写入操作，但身份范围内已有 Typed Memory，可能是重复导入。")
            if not operation_ids and probe.get("status") == "empty":
                raise RuntimeError("本次没有产生写入操作，且 store 身份范围内没有 Typed Memory")
        except Exception as exc:
            _write_ingest_stage_report(
                output_dir,
                stage="verifying",
                status="failed",
                request=ingest_request,
                config=resolved_config,
                add_call_count=_add_call_count(batch_results),
                operation_count=len(operation_ids),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_ingest_stage_report(
            output_dir,
            stage="verifying",
            status="completed",
            request=ingest_request,
            config=resolved_config,
            add_call_count=_add_call_count(batch_results),
            operation_count=len(operation_ids),
        )

    _save_ingest_outputs(
        output_dir,
        request,
        batch_results,
        dedup_report,
        resolved_config=resolved_config,
        probe=probe,
        readback=readback,
    )
    _write_add_journal(output_dir, batch_results)
    _save_chunk_extraction_traces(output_dir, batch_results)
    return {
        "typed_memory_count": sum(len(item.get("typed_memories", [])) for item in batch_results),
        "add_count": sum(len(item.get("add_results", [])) for item in batch_results),
        "add_call_count": _add_call_count(batch_results),
        "operation_count": _operation_count(batch_results),
        "event_counts": _event_counts(batch_results),
        "empty_add_result_count": sum(
            1 for item in batch_results if not item.get("add_results")
        ),
        "chunk_count": len(batch_results),
        "dedup_report": dedup_report,
        "store_probe": probe,
        "readback": readback,
        "warnings": warnings,
        "sources": batch_results,
    }


def _write_ingest_web_run_info(
    output_dir: Path,
    request: WebIngestRequest,
    config: dict[str, int],
    *,
    status: WebRunStatus,
    stage: str,
    error: str = "",
) -> None:
    """写入独立 Ingest 的 Web 状态、阶段和运行 identity。"""
    now = datetime.now(timezone.utc).isoformat()
    info_path = output_dir / "web_run_info.json"
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        previous = {}
    payload = {
        "schema_version": "summary_multi_agent_web.run_info.v1",
        "run_type": "ingest",
        "status": status.value,
        "stage": stage,
        "created_at": previous.get("created_at", now),
        "started_at": previous.get("started_at", now),
        "run_id": previous.get("run_id", ""),
        "output_dir": previous.get("output_dir", str(output_dir)),
        "updated_at": now,
        "user_id": request.user_id,
        "project": request.project,
        "source_time": request.source_time.strip() or None,
        "source_type": request.source_type,
        "store": memory_store_info(request.qdrant_path, request.collection),
        "filters": {
            "user_id": request.user_id,
            "project": request.project,
            "memory_schema_version": "summary_typed_memory.v1",
        },
        "llm_runtime": get_llm_runtime_info(),
        "config": config,
        "error": error,
    }
    if status in {
        WebRunStatus.COMPLETED,
        WebRunStatus.COMPLETED_WITH_WARNINGS,
        WebRunStatus.FAILED,
        WebRunStatus.CANCELLING,
    }:
        payload["finished_at"] = now
    _write_json(info_path, payload)


async def run_web_ingest(
    request: WebIngestRequest,
    output_dir: Path,
    *,
    memory_factory: MemoryFactory = build_summary_memory,
    memory_closer: MemoryCloser = close_summary_memory,
    llm_call: Callable[..., str] = call_summary_llm,
    status_callback: Callable[[str], None] | None = None,
) -> WebIngestResult:
    """在线程中运行独立 Web Ingest，并在成功写入后重开 Store 做只读回读。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_ingest_config(request.config)
    _write_ingest_web_run_info(
        output_dir,
        request,
        config,
        status=WebRunStatus.RUNNING,
        stage="validation",
    )
    if status_callback is not None:
        status_callback("validation")
    try:
        result = await asyncio.to_thread(
            ingest_history_sources,
            request,
            output_dir,
            memory_factory=memory_factory,
            memory_closer=memory_closer,
            llm_call=llm_call,
            verify_readback=True,
        )
    except Exception as exc:
        message = f"Web Ingest 失败：{type(exc).__name__}: {exc}"
        failure = WebIngestResult(status=WebRunStatus.FAILED, warnings=[message])
        _write_json(output_dir / "ingest_run_result.json", failure.model_dump(mode="json"))
        _write_ingest_web_run_info(
            output_dir,
            request,
            config,
            status=WebRunStatus.FAILED,
            stage="failed",
            error=message,
        )
        raise

    warnings = list(result.get("warnings", []))
    result_status = (
        WebRunStatus.COMPLETED_WITH_WARNINGS
        if warnings
        else WebRunStatus.COMPLETED
    )
    final = WebIngestResult(
        status=result_status,
        typed_memory_count=result["typed_memory_count"],
        add_call_count=result["add_call_count"],
        operation_count=result["operation_count"],
        chunk_count=result["chunk_count"],
        warnings=warnings,
        store_probe=result.get("store_probe", {}),
        readback=result.get("readback", {}),
    )
    _write_json(output_dir / "ingest_run_result.json", final.model_dump(mode="json"))
    _write_ingest_web_run_info(
        output_dir,
        request,
        config,
        status=final.status,
        stage="completed",
    )
    if status_callback is not None:
        status_callback(final.status.value)
    return final


async def run_e2e_ingest(
    request: WebE2ERequest,
    output_dir: Path,
    *,
    memory_factory: MemoryFactory = build_summary_memory,
    memory_closer: MemoryCloser = close_summary_memory,
    llm_call: Callable[..., str] = call_summary_llm,
    status_callback: Callable[[str], None] | None = None,
) -> WebE2ERunResult:
    """在线程中执行 E2E Ingest，并保存顶层 E2E 结果和阶段状态。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir = output_dir / "e2e_ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    _write_e2e_web_run_info(
        output_dir,
        request,
        status=WebRunStatus.RUNNING,
        stage="ingest",
    )
    if status_callback is not None:
        status_callback("ingest")
    try:
        ingest_result = await asyncio.to_thread(
            ingest_history_sources,
            request,
            ingest_dir,
            memory_factory=memory_factory,
            memory_closer=memory_closer,
            llm_call=llm_call,
        )
    except Exception as exc:
        result = WebE2ERunResult(
            status=WebRunStatus.FAILED,
            warnings=[f"E2E Ingest 失败：{type(exc).__name__}: {exc}"],
        )
        _write_e2e_result(output_dir, result)
        _write_e2e_web_run_info(
            output_dir,
            request,
            status=WebRunStatus.FAILED,
            stage="ingest",
            error=result.warnings[0],
        )
        raise

    result = WebE2ERunResult(
        status=WebRunStatus.COMPLETED,
        ingest={
            "chunk_count": ingest_result["chunk_count"],
            "typed_memory_count": ingest_result["typed_memory_count"],
            "add_count": ingest_result["add_count"],
            "dedup_report": ingest_result["dedup_report"],
        },
    )
    _write_e2e_result(output_dir, result)
    _write_e2e_web_run_info(
        output_dir,
        request,
        status=result.status,
        stage="completed",
    )
    return result


async def run_e2e_run(
    request: WebE2ERequest,
    output_dir: Path,
    *,
    ingest_memory_factory: MemoryFactory = build_summary_memory,
    ingest_memory_closer: MemoryCloser = close_summary_memory,
    summary_memory_factory: MemoryFactory = build_summary_memory,
    summary_memory_closer: MemoryCloser = close_summary_memory,
    llm_call: Callable[..., str] = call_summary_llm,
    pipeline_builder: Callable[..., Any] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> WebE2ERunResult:
    """先完成 E2E Ingest，再使用同一身份/Store 启动独立 Summarize。"""
    if request.summary is None:
        raise ValueError("e2e_run requires a summarize request")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_e2e_web_run_info(
        output_dir,
        request,
        status=WebRunStatus.RUNNING,
        stage="ingest",
    )
    ingest_request = request.model_copy(
        update={"run_type": "e2e_ingest", "summary": None}
    )
    try:
        ingest_result = await asyncio.to_thread(
            ingest_history_sources,
            ingest_request,
            output_dir / "e2e_ingest",
            memory_factory=ingest_memory_factory,
            memory_closer=ingest_memory_closer,
            llm_call=llm_call,
        )
    except Exception as exc:
        result = WebE2ERunResult(
            status=WebRunStatus.FAILED,
            warnings=[f"E2E Ingest 失败，未启动 Summarize：{type(exc).__name__}: {exc}"],
            ingest={"failed": True},
        )
        _write_e2e_result(output_dir, result)
        _write_e2e_web_run_info(
            output_dir,
            request,
            status=WebRunStatus.FAILED,
            stage="ingest",
            error=result.warnings[0],
        )
        raise

    if status_callback is not None:
        status_callback("summarize")
    summary_request = request.summary.model_copy(
        update={
            "qdrant_path": request.qdrant_path or request.summary.qdrant_path,
            "collection": request.collection or request.summary.collection,
        }
    )
    summary_dir = output_dir / "multi_agent_summarize"
    try:
        summary_result = await run_multi_agent_summary(
            summary_request,
            summary_dir,
            memory_factory=summary_memory_factory,
            memory_closer=summary_memory_closer,
            llm_call=llm_call,
            pipeline_builder=pipeline_builder
            if pipeline_builder is not None
            else __import__(
                "summary_multi_agent.runtime",
                fromlist=["build_multi_agent_v2_pipeline"],
            ).build_multi_agent_v2_pipeline,
            status_callback=status_callback,
        )
    except Exception as exc:
        result = WebE2ERunResult(
            status=WebRunStatus.FAILED,
            warnings=[f"E2E Summarize 失败：{type(exc).__name__}: {exc}"],
            ingest={
                "chunk_count": ingest_result["chunk_count"],
                "typed_memory_count": ingest_result["typed_memory_count"],
                "add_count": ingest_result["add_count"],
            },
            summary={"failed": True},
        )
        _write_e2e_result(output_dir, result)
        _write_e2e_web_run_info(
            output_dir,
            request,
            status=WebRunStatus.FAILED,
            stage="summarize",
            error=result.warnings[0],
        )
        raise

    result = WebE2ERunResult(
        status=status_from_runtime(summary_result.status.value),
        final_summary=summary_result.final_summary or "",
        recommended_candidate_index=summary_result.recommended_candidate_index,
        evaluation_status=getattr(
            getattr(summary_result, "evaluation_status", "disabled"),
            "value",
            getattr(summary_result, "evaluation_status", "disabled"),
        ),
        personalization_evaluation_status=getattr(
            getattr(
                summary_result,
                "personalization_evaluation_status",
                "disabled",
            ),
            "value",
            getattr(
                summary_result,
                "personalization_evaluation_status",
                "disabled",
            ),
        ),
        warnings=list(summary_result.warnings),
        ingest={
            "chunk_count": ingest_result["chunk_count"],
            "typed_memory_count": ingest_result["typed_memory_count"],
            "add_count": ingest_result["add_count"],
            "dedup_report": ingest_result["dedup_report"],
        },
        summary=_summary_result_payload(summary_result),
    )
    _write_e2e_result(output_dir, result)
    _write_e2e_web_run_info(
        output_dir,
        request,
        status=result.status,
        stage="completed" if result.status != WebRunStatus.FAILED else "summarize",
    )
    return result
