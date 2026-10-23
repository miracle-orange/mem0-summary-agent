"""Summary Multi-Agent Web V2 的 Gradio 入口和用户输入绑定。"""

from __future__ import annotations

import argparse
import html
import inspect
import json
import os
import site
import sys
import time
from pathlib import Path
from typing import Any

from summary_memory_llm import (
    get_llm_runtime_info, load_runtime_env, validate_llm_configuration,
)

load_runtime_env()

from summary_multi_agent.evaluation.schemas import (
    EvaluationCompareConfig,
    EvaluationCompareRequest,
    EvaluationSourceDocument,
    EvaluationSummaryDocument,
)
from summary_multi_agent.minutes_report import render_web_report_fragment
from summary_multi_agent.minutes_report.models import MinutesDocument, MinutesSources
from summary_multi_agent.schemas import LiteEvaluationConfig
from summary_multi_agent.profiles import CLOUD_FAST_DEFAULTS
from summary_multi_agent_web import (
    WebIngestConfig,
    WebIngestRequest,
    WebRunManager,
    WebSummarizeRequest,
)
from summary_multi_agent_web.multi_agent_runner import memory_store_info
from summary_multi_agent_web.schemas import (
    WebE2ERequest,
    WebHistorySource,
    WebRunStatus,
)
from summary_multi_agent_web.visualization_render import (
    WEB_VISUALIZATION_CSS,
    render_agent_graph_html,
    render_evidence_memory_html,
    render_overview_html,
    render_review_evaluation_html,
    render_task_detail_html,
    render_timeline_html,
    task_choices,
)
from summary_multi_agent_web.theme import CLOUD_UI_CSS
from summary_multi_agent_web.ui_presentation import (
    combine_source_dates,
    file_date_rows,
    filtered_runs,
    ready_message,
    run_heading_html,
    run_label,
    run_list_html,
)


def parse_args() -> argparse.Namespace:
    """解析 Web 服务地址、根路径、输出目录和调试开关。"""
    parser = argparse.ArgumentParser(
        description="Summary Multi-Agent Web V2: direct V2 summarize UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--root-path", default=os.getenv("GRADIO_ROOT_PATH", ""))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/agent_multi_web")
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def normalize_root_path(value: str) -> str:
    """将 Gradio root path 规范为带单个前导斜杠的路径。"""
    value = (value or "").strip()
    if not value:
        return ""
    return "/" + value.strip("/")


def require_gradio() -> Any:
    """加载并检查 Web 依赖是否来自当前 Conda 环境。"""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Gradio，请安装 scripts/agent/requirements-summary-web.txt。"
        ) from exc
    try:
        import fastapi
        import pydantic
    except ImportError as exc:
        raise RuntimeError(
            "Gradio 运行时缺少 FastAPI/Pydantic；请在当前 Conda 环境补齐 Web 依赖。"
        ) from exc

    prefix = Path(sys.prefix).resolve()
    raw_user_site = site.getusersitepackages()
    user_sites = [Path(raw_user_site).resolve()] if raw_user_site else []
    loaded = {
        "gradio": getattr(gr, "__file__", ""),
        "fastapi": getattr(fastapi, "__file__", ""),
        "pydantic": getattr(pydantic, "__file__", ""),
    }
    contaminated = []
    for name, raw_path in loaded.items():
        if not raw_path:
            continue
        module_path = Path(raw_path).resolve()
        in_user_site = any(
            module_path == user_site or user_site in module_path.parents
            for user_site in user_sites
        )
        in_env = module_path == prefix or prefix in module_path.parents
        if not in_env:
            source = "~/.local" if in_user_site else "当前 Conda 环境之外"
            contaminated.append(f"{name}={module_path}（来源：{source}）")
    if contaminated:
        raise RuntimeError(
            "检测到 Web 依赖没有全部来自当前 Conda 环境，可能造成 Gradio/FastAPI/Pydantic 不兼容："
            + "; ".join(contaminated)
            + "。请在目标环境执行：export PYTHONNOUSERSITE=1; unset PYTHONPATH，然后重启 Web。"
        )
    return gr


def parse_target_entities(value: str) -> list[str]:
    """把逗号/换行分隔的目标实体转换为去空白列表。"""
    return [
        item.strip()
        for item in (value or "").replace("\n", ",").split(",")
        if item.strip()
    ]


def _path_values(value: Any) -> list[Path]:
    """兼容 Gradio 文件对象和字符串路径，统一转换为 Path 列表。"""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    paths: list[Path] = []
    for item in values:
        raw_path = getattr(item, "name", item)
        if raw_path:
            paths.append(Path(str(raw_path)).expanduser())
    return paths


def _read_text_path(path: Path) -> str:
    """读取并校验 UTF-8 的 txt/md 材料文件。"""
    if not path.exists() or not path.is_file():
        raise ValueError(f"材料文件不存在或不是文件：{path}")
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError(f"只支持 .txt / .md 材料文件：{path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"材料文件不是有效 UTF-8 文本：{path}") from exc


def build_history_sources(
    history_text: str,
    history_files: Any,
    history_paths: str,
) -> list[WebHistorySource]:
    """合并粘贴文本、上传文件和目标机路径为统一历史来源列表。"""
    # 文本、上传文件和目标机路径最终统一为 WebHistorySource，后续 E2E
    # 流程不再区分输入来源，避免不同入口产生不同的分块与去重语义。
    sources: list[WebHistorySource] = []
    pasted = history_text or ""
    if pasted:
        sources.append(WebHistorySource(label="pasted_history", text=pasted))

    file_paths = _path_values(history_files)
    path_lines = [
        line.strip() for line in (history_paths or "").splitlines() if line.strip()
    ]
    file_paths.extend(Path(line).expanduser() for line in path_lines)
    for index, path in enumerate(file_paths, start=1):
        text = _read_text_path(path)
        if not text.strip():
            raise ValueError(f"历史材料文件为空：{path}")
        sources.append(
            WebHistorySource(
                label=path.name or f"history_{index}", text=text, source_path=str(path)
            )
        )
    if not sources:
        raise ValueError("请提供历史文本、上传历史文件或填写目标机历史路径。")
    return sources


def build_current_material(
    current_text: str,
    current_file: Any,
    current_path: str,
) -> str:
    """选择唯一当前材料来源，并完整保留原文供 V2 offset/coverage 使用。"""
    # 当前原始材料只能选择一种来源，并且保留原文内容交给 V2 RawMaterial；
    # 这里不做摘要、首尾截断或清洗，避免破坏 Coverage/Evidence offset。
    material = current_text or ""
    paths = _path_values(current_file)
    path_value = (current_path or "").strip()
    if path_value:
        paths.append(Path(path_value).expanduser())
    if len(paths) > 1:
        raise ValueError("当前材料只允许一个上传文件或目标机路径。")
    if paths:
        material = _read_text_path(paths[0])
    if not material.strip():
        raise ValueError("请提供当前材料文本、上传文件或填写目标机路径。")
    return material


def _exclusive_text_input(
    pasted_text: str,
    uploaded_file: Any,
    *,
    label: str,
    required: bool,
) -> str | None:
    """读取一个粘贴或文件输入槽位，并禁止同槽位双重输入。"""
    pasted = pasted_text or ""
    paths = _path_values(uploaded_file)
    if pasted.strip() and paths:
        raise ValueError(f"{label} 只能选择粘贴文本或上传文件之一。")
    if len(paths) > 1:
        raise ValueError(f"{label} 只允许一个文件。")
    value = _read_text_path(paths[0]) if paths else pasted
    if not value.strip():
        if required:
            raise ValueError(f"请提供{label}。")
        return None
    return value


def build_evaluation_compare_request(
    *,
    project: str,
    summary_request: str,
    target_entities: str,
    current_values: tuple[Any, ...],
    history_values: tuple[tuple[Any, ...], ...],
    summary_values: tuple[tuple[Any, ...], ...],
    evaluation_config_values: tuple[Any, ...] = (),
    evaluation_mode: str = "strict",
    lite_config_values: tuple[Any, ...] = (),
    execution_profile: str = "legacy",
) -> EvaluationCompareRequest:
    """将固定 Web 输入槽位转换为共享 Snapshot 横评请求。"""
    current_id, current_label, current_time, current_text, current_file = current_values
    current_material = _exclusive_text_input(
        current_text,
        current_file,
        label="当前原始材料",
        required=True,
    )
    historical_sources = []
    for index, values in enumerate(history_values, start=1):
        source_id, label, source_time, text, uploaded = values
        material = _exclusive_text_input(
            text,
            uploaded,
            label=f"历史材料 {index}",
            required=False,
        )
        metadata_present = any(
            str(item or "").strip() for item in (label, source_time)
        )
        if material is None:
            if metadata_present:
                raise ValueError(f"历史材料 {index} 已填写元数据但没有正文。")
            continue
        historical_sources.append(
            EvaluationSourceDocument(
                source_id=str(source_id or f"history_{index}"),
                label=str(label or f"历史材料 {index}"),
                source_time=str(source_time),
                text=material,
            )
        )
    summaries = []
    for index, values in enumerate(summary_values, start=1):
        summary_id, display_name, text, uploaded = values
        material = _exclusive_text_input(
            text,
            uploaded,
            label=f"纪要 {index}",
            required=index <= 2,
        )
        metadata_present = bool(str(display_name or "").strip())
        if material is None:
            if index > 2 and metadata_present:
                raise ValueError(f"纪要 {index} 已填写元数据但没有正文。")
            continue
        summaries.append(
            EvaluationSummaryDocument(
                summary_id=str(summary_id or f"summary_{index}"),
                display_name=str(display_name or f"纪要 {index}"),
                text=material,
            )
        )
    assert current_material is not None
    lite_config = (
        LiteEvaluationConfig(
            reference_chunking="auto" if execution_profile == "cloud_fast" else "fixed",
            segment_max_chars=int(lite_config_values[0]),
            segment_overlap_chars=int(lite_config_values[1]),
            reference_max_tokens=int(lite_config_values[2]),
            claim_max_tokens=int(lite_config_values[3]),
            match_max_tokens=int(lite_config_values[4]),
            verification_max_tokens=int(lite_config_values[5]),
            repair_attempts=int(lite_config_values[6]),
            model_timeout_seconds=int(lite_config_values[7]),
            timeout_seconds=int(lite_config_values[8]),
            max_model_calls=int(lite_config_values[9]),
            max_concurrent_model_calls=int(lite_config_values[10]),
            model_context_window_tokens=(
                int(lite_config_values[11])
                if len(lite_config_values) > 11
                and lite_config_values[11] not in {None, ""}
                else None
            ),
            context_safety_margin_tokens=(
                int(lite_config_values[12]) if len(lite_config_values) > 12 else 1024
            ),
        )
        if lite_config_values
        else LiteEvaluationConfig.cloud_fast() if execution_profile == "cloud_fast" else LiteEvaluationConfig()
    )
    config = (
        EvaluationCompareConfig(
            evaluation_mode=evaluation_mode,
            lite_evaluation=lite_config,
            segment_max_chars=int(evaluation_config_values[0]),
            source_fact_max_tokens=int(evaluation_config_values[1]),
            summary_fact_max_tokens=int(evaluation_config_values[2]),
            match_max_tokens=int(evaluation_config_values[3]),
            personalization_max_tokens=int(evaluation_config_values[4]),
            repair_attempts=int(evaluation_config_values[5]),
            adaptive_split_enabled=bool(evaluation_config_values[6]),
            adaptive_split_min_chars=int(evaluation_config_values[7]),
            adaptive_split_max_depth=int(evaluation_config_values[8]),
            comparison_interpretation_enabled=(
                bool(evaluation_config_values[9])
                if len(evaluation_config_values) > 9
                else True
            ),
        )
        if evaluation_config_values
        else EvaluationCompareConfig(
            evaluation_mode=evaluation_mode,
            lite_evaluation=lite_config,
            comparison_interpretation_enabled=True,
        )
    )
    return EvaluationCompareRequest(
        project=project,
        summary_request=summary_request,
        current_source=EvaluationSourceDocument(
            source_id=str(current_id),
            label=str(current_label),
            source_time=str(current_time),
            text=current_material,
        ),
        historical_sources=tuple(historical_sources),
        summaries=tuple(summaries),
        target_entities=tuple(parse_target_entities(target_entities)),
        evaluation_config=config,
    )


def build_web_request(
    current_material: str,
    summary_request: str,
    user_id: str,
    project: str,
    source_time: str,
    target_entities: str,
    summary_style: str,
    planner_mode: str,
    worker_mode: str,
    compression_mode: str,
    enable_review: bool,
    enable_evaluation: bool,
    max_replan_rounds: int,
    max_review_rounds: int,
    max_concurrent_workers: int,
    max_concurrent_segment_maps: int,
    max_concurrent_model_calls: int,
    worker_timeout_seconds: int,
    global_timeout_seconds: int,
    worker_model_timeout_seconds: int = 180,
    qdrant_path: str = "",
    collection: str = "",
    raw_segment_max_chars: int = 3000,
    model_context_window_tokens: int | None = None,
    model_context_safety_margin_tokens: int = 1024,
    meeting_scene: str = "auto",
    minutes_output_mode: str = "classic",
    rich_summary_max_tokens: int = 6000,
    evaluation_mode: str = "strict",
    lite_evaluation_values: tuple[Any, ...] = (),
    execution_profile: str = "legacy",
) -> WebSummarizeRequest:
    """将 UI 字段转换为 WebSummarizeRequest，并保持参数类型稳定。"""
    lite_config = (
        LiteEvaluationConfig(
            reference_chunking="auto" if execution_profile == "cloud_fast" else "fixed",
            segment_max_chars=int(lite_evaluation_values[0]),
            segment_overlap_chars=int(lite_evaluation_values[1]),
            reference_max_tokens=int(lite_evaluation_values[2]),
            claim_max_tokens=int(lite_evaluation_values[3]),
            match_max_tokens=int(lite_evaluation_values[4]),
            verification_max_tokens=int(lite_evaluation_values[5]),
            repair_attempts=int(lite_evaluation_values[6]),
            model_timeout_seconds=int(lite_evaluation_values[7]),
            timeout_seconds=int(lite_evaluation_values[8]),
            max_model_calls=int(lite_evaluation_values[9]),
            max_concurrent_model_calls=int(lite_evaluation_values[10]),
        )
        if lite_evaluation_values
        else LiteEvaluationConfig.cloud_fast() if execution_profile == "cloud_fast" else LiteEvaluationConfig()
    )
    return WebSummarizeRequest(
        execution_profile=execution_profile,
        current_material=current_material,
        summary_request=summary_request,
        user_id=user_id,
        project=project,
        source_time=source_time,
        target_entities=parse_target_entities(target_entities),
        summary_style=summary_style,
        meeting_scene=meeting_scene,
        minutes_output_mode=minutes_output_mode,
        rich_summary_max_tokens=int(rich_summary_max_tokens),
        planner_mode=planner_mode,
        worker_mode=worker_mode,
        context_compression_mode=compression_mode,
        enable_review=enable_review,
        enable_evaluation=enable_evaluation,
        evaluation_mode=evaluation_mode,
        lite_evaluation=lite_config,
        max_replan_rounds=int(max_replan_rounds),
        max_review_rounds=int(max_review_rounds),
        max_concurrent_workers=int(max_concurrent_workers),
        max_concurrent_segment_maps=int(max_concurrent_segment_maps),
        max_concurrent_model_calls=int(max_concurrent_model_calls),
        worker_model_timeout_seconds=int(worker_model_timeout_seconds),
        worker_timeout_seconds=int(worker_timeout_seconds),
        global_timeout_seconds=int(global_timeout_seconds),
        raw_segment_max_chars=int(raw_segment_max_chars),
        model_context_window_tokens=(
            int(model_context_window_tokens)
            if model_context_window_tokens not in {None, ""}
            else None
        ),
        model_context_safety_margin_tokens=int(model_context_safety_margin_tokens),
        qdrant_path=qdrant_path.strip(),
        collection=collection.strip(),
    )


def build_ingest_request(
    history_text: str,
    history_files: Any,
    history_paths: str,
    user_id: str,
    project: str,
    source_time: str,
    source_type: str,
    qdrant_path: str,
    collection: str,
    chunk_char_limit: int | None = None,
    chunk_max_tokens: int | None = None,
    merge_max_tokens: int | None = None,
    typed_extraction_max_tokens: int | None = None,
    dedup_judge_max_tokens: int | None = None,
    dedup_judge_max_candidates: int | None = None,
    execution_profile: str = "legacy",
    source_dates: str = "",
) -> WebIngestRequest:
    """将历史输入和可选 Ingest 配置转换为 WebIngestRequest。"""
    sources = build_history_sources(history_text, history_files, history_paths)
    dates = (source_dates or "").splitlines()
    if len(dates) > len(sources):
        raise ValueError("材料日期行数不能超过材料份数")
    sources = [
        source.model_copy(update={"source_time": dates[index].strip() or None})
        if index < len(dates) else source
        for index, source in enumerate(sources)
    ]
    return WebIngestRequest(
        execution_profile=execution_profile,
        history_sources=sources,
        user_id=user_id,
        project=project,
        source_time=source_time,
        source_type=source_type,
        qdrant_path=(qdrant_path or "").strip(),
        collection=(collection or "").strip(),
        config=WebIngestConfig(
            chunk_char_limit=chunk_char_limit,
            chunk_max_tokens=chunk_max_tokens,
            merge_max_tokens=merge_max_tokens,
            typed_extraction_max_tokens=typed_extraction_max_tokens,
            dedup_judge_max_tokens=dedup_judge_max_tokens,
            dedup_judge_max_candidates=dedup_judge_max_candidates,
        ),
    )


def build_e2e_request(
    run_type: str,
    history_text: str,
    history_files: Any,
    history_paths: str,
    current_text: str,
    current_file: Any,
    current_path: str,
    source_type: str,
    qdrant_path: str,
    collection: str,
    summary_args: tuple[Any, ...],
    meeting_scene: str = "auto",
    execution_profile: str = "legacy",
) -> WebE2ERequest:
    """构造 E2E Ingest/Run 请求；只有 e2e_run 才要求当前材料。"""
    # E2E Ingest 可以没有当前材料，E2E Run 才要求同时构造 Summarize 请求。
    # 这样单独测试历史入库时不会被迫伪造一份当前材料。
    sources = build_history_sources(history_text, history_files, history_paths)
    current_material = ""
    if run_type == "e2e_run":
        current_material = build_current_material(
            current_text, current_file, current_path
        )
    summary = None
    if run_type == "e2e_run":
        summary = build_web_request(
            current_material,
            *summary_args,
            qdrant_path=qdrant_path,
            collection=collection,
            meeting_scene=meeting_scene,
            execution_profile=execution_profile,
        )
    return WebE2ERequest(
        execution_profile=execution_profile,
        run_type=run_type,
        history_sources=sources,
        user_id=str(summary_args[1]),
        project=str(summary_args[2]),
        source_time=str(summary_args[3]),
        source_type=source_type,
        qdrant_path=qdrant_path.strip(),
        collection=collection.strip(),
        summary=summary,
    )


def _snapshot_text(snapshot: Any) -> str:
    """将运行快照渲染为 UI 状态文本，不读取隐藏模型轨迹。"""
    if snapshot is None:
        return "暂无运行"
    lines = [
        f"状态：{snapshot.status.value}",
        f"阶段：{snapshot.stage or '--'}",
        f"Run ID：{snapshot.run_id}",
        f"输出目录：{snapshot.output_dir}",
    ]
    if snapshot.recommended_candidate_index is not None:
        lines.append(
            f"Shadow 推荐 candidate：{snapshot.recommended_candidate_index}（正式输出仍为 candidate 0）"
        )
    if snapshot.evaluation_status != "disabled":
        lines.append(f"事实级评测：{snapshot.evaluation_status}")
    if snapshot.personalization_evaluation_status != "disabled":
        lines.append(
            f"目标实体个性化评测：{snapshot.personalization_evaluation_status}"
        )
    if snapshot.warnings:
        lines.append("Warnings：\n- " + "\n- ".join(snapshot.warnings))
    if snapshot.error:
        lines.append(f"错误：{snapshot.error}")
    return "\n".join(lines)


def _optional_int(value: Any) -> int | None:
    """把可选数值控件的空值转换为 None。"""
    # Gradio 3 的 Number 控件可能把 value=None 提交为 0；这些覆盖值均须为正数。
    if value is None or value == "" or value == 0:
        return None
    return int(value)


def build_app(
    manager: WebRunManager | None = None, *, output_dir: Path | None = None
) -> Any:
    """构造独立 Gradio Blocks、输入绑定和运行/审计回调。"""
    gr = require_gradio()
    compact_upload = {"height": 150} if "height" in inspect.signature(gr.File).parameters else {}
    run_manager = manager or WebRunManager()
    default_output_dir = output_dir or Path("outputs/agent_multi_web")

    def start_ingest(
        history_text: str,
        history_files: Any,
        history_paths: str,
        user_id: str,
        project: str,
        source_time: str,
        source_type: str,
        qdrant_path: str,
        collection: str,
        chunk_char_limit: int | None,
        chunk_max_tokens: int | None,
        merge_max_tokens: int | None,
        typed_extraction_max_tokens: int | None,
        dedup_judge_max_tokens: int | None,
        dedup_judge_max_candidates: int | None,
        source_dates: str = "",
        execution_profile: str = "cloud_fast",
        file_dates: Any = None,
    ) -> tuple[str, str, str, str]:
        """校验并排队历史 Ingest，返回状态、run_id、目录和 Store identity。"""
        try:
            ordered_dates = combine_source_dates(
                pasted=history_text,
                uploaded=history_files,
                server_paths=history_paths,
                manual_dates=source_dates,
                table=file_dates,
            )
            request = build_ingest_request(
                history_text,
                history_files,
                history_paths,
                user_id,
                project,
                source_time,
                source_type,
                qdrant_path,
                collection,
                _optional_int(chunk_char_limit),
                _optional_int(chunk_max_tokens),
                _optional_int(merge_max_tokens),
                _optional_int(typed_extraction_max_tokens),
                _optional_int(dedup_judge_max_tokens),
                _optional_int(dedup_judge_max_candidates),
                execution_profile=execution_profile,
                source_dates=ordered_dates,
            )
            snapshot = run_manager.start_ingest(request, default_output_dir)
            store_text = json.dumps(
                memory_store_info(request.qdrant_path, request.collection),
                ensure_ascii=False,
                indent=2,
            )
            return (
                _snapshot_text(snapshot),
                snapshot.run_id,
                str(snapshot.output_dir),
                store_text,
            )
        except Exception as exc:
            return f"启动失败：{type(exc).__name__}: {exc}", "", "", ""

    def start_summarize(
        current_text: str,
        current_file: Any,
        current_path: str,
        summary_request: str,
        user_id: str,
        project: str,
        source_time: str,
        qdrant_path: str,
        collection: str,
        target_entities: str,
        summary_style: str,
        planner_mode: str,
        worker_mode: str,
        compression_mode: str,
        enable_review: bool,
        enable_evaluation: bool,
        max_replan_rounds: int,
        max_review_rounds: int,
        max_concurrent_workers: int,
        max_concurrent_segment_maps: int,
        max_concurrent_model_calls: int,
        worker_model_timeout_seconds: int,
        worker_timeout_seconds: int,
        global_timeout_seconds: int,
        raw_segment_max_chars: int,
        model_context_window_tokens: int | None,
        model_context_safety_margin_tokens: int,
        meeting_scene: str = "auto",
        minutes_output_mode: str = "classic",
        rich_summary_max_tokens: int = 6000,
        evaluation_mode: str = "strict",
        lite_segment_max_chars: int = 3000,
        lite_segment_overlap_chars: int = 100,
        lite_reference_max_tokens: int = 4096,
        lite_claim_max_tokens: int = 4096,
        lite_match_max_tokens: int = 4096,
        lite_verification_max_tokens: int = 4096,
        lite_repair_attempts: int = 1,
        lite_model_timeout_seconds: int = 180,
        lite_timeout_seconds: int = 1200,
        lite_max_model_calls: int = 64,
        lite_max_concurrent_model_calls: int = 1,
        execution_profile: str = "cloud_fast",
    ) -> tuple[str, str, str, str]:
        """校验并排队 Multi-Agent Summarize，返回初始运行信息。"""
        try:
            material = build_current_material(current_text, current_file, current_path)
            request = build_web_request(
                material,
                summary_request,
                user_id,
                project,
                source_time,
                target_entities,
                summary_style,
                planner_mode,
                worker_mode,
                compression_mode,
                enable_review,
                enable_evaluation,
                int(max_replan_rounds),
                int(max_review_rounds),
                int(max_concurrent_workers),
                int(max_concurrent_segment_maps),
                int(max_concurrent_model_calls),
                int(worker_timeout_seconds),
                int(global_timeout_seconds),
                worker_model_timeout_seconds=int(worker_model_timeout_seconds),
                qdrant_path=qdrant_path,
                collection=collection,
                raw_segment_max_chars=int(raw_segment_max_chars),
                model_context_window_tokens=model_context_window_tokens,
                model_context_safety_margin_tokens=int(
                    model_context_safety_margin_tokens
                ),
                meeting_scene=meeting_scene,
                minutes_output_mode=minutes_output_mode,
                rich_summary_max_tokens=int(rich_summary_max_tokens),
                evaluation_mode=evaluation_mode,
                lite_evaluation_values=(
                    lite_segment_max_chars,
                    lite_segment_overlap_chars,
                    lite_reference_max_tokens,
                    lite_claim_max_tokens,
                    lite_match_max_tokens,
                    lite_verification_max_tokens,
                    lite_repair_attempts,
                    lite_model_timeout_seconds,
                    lite_timeout_seconds,
                    lite_max_model_calls,
                    lite_max_concurrent_model_calls,
                ),
                execution_profile=execution_profile,
            )
            snapshot = run_manager.start_summarize(request, default_output_dir)
            store_text = json.dumps(
                memory_store_info(request.qdrant_path, request.collection),
                ensure_ascii=False,
                indent=2,
            )
            return (
                _snapshot_text(snapshot),
                snapshot.run_id,
                str(snapshot.output_dir),
                store_text,
            )
        except Exception as exc:
            return f"启动失败：{type(exc).__name__}: {exc}", "", "", ""

    def start_evaluation_compare(
        project: str,
        summary_request: str,
        target_entities: str,
        segment_max_chars: int,
        source_fact_max_tokens: int,
        summary_fact_max_tokens: int,
        match_max_tokens: int,
        personalization_max_tokens: int,
        repair_attempts: int,
        adaptive_split_enabled: bool,
        adaptive_split_min_chars: int,
        adaptive_split_max_depth: int,
        interpretation_enabled: bool,
        evaluation_mode: str,
        lite_segment_max_chars: int,
        lite_segment_overlap_chars: int,
        lite_reference_max_tokens: int,
        lite_claim_max_tokens: int,
        lite_match_max_tokens: int,
        lite_verification_max_tokens: int,
        lite_repair_attempts: int,
        lite_model_timeout_seconds: int,
        lite_timeout_seconds: int,
        lite_max_model_calls: int,
        lite_max_concurrent_model_calls: int,
        lite_model_context_window_tokens: int | None,
        lite_context_safety_margin_tokens: int,
        execution_profile: str,
        *values: Any,
    ) -> tuple[str, str, str, str]:
        """校验并排队独立横评，返回运行状态和只读边界说明。"""
        try:
            if len(values) != 50:
                raise ValueError("横评页面输入契约不完整")
            current_values = tuple(values[:5])
            history_flat = values[5:30]
            summary_flat = values[30:50]
            history_values = tuple(
                tuple(history_flat[index : index + 5]) for index in range(0, 25, 5)
            )
            summary_values = tuple(
                tuple(summary_flat[index : index + 4]) for index in range(0, 20, 4)
            )
            request = build_evaluation_compare_request(
                project=project,
                summary_request=summary_request,
                target_entities=target_entities,
                current_values=current_values,
                history_values=history_values,
                summary_values=summary_values,
                evaluation_config_values=(
                    segment_max_chars,
                    source_fact_max_tokens,
                    summary_fact_max_tokens,
                    match_max_tokens,
                    personalization_max_tokens,
                    repair_attempts,
                    adaptive_split_enabled,
                    adaptive_split_min_chars,
                    adaptive_split_max_depth,
                    interpretation_enabled,
                ),
                evaluation_mode=evaluation_mode,
                lite_config_values=(
                    lite_segment_max_chars,
                    lite_segment_overlap_chars,
                    lite_reference_max_tokens,
                    lite_claim_max_tokens,
                    lite_match_max_tokens,
                    lite_verification_max_tokens,
                    lite_repair_attempts,
                    lite_model_timeout_seconds,
                    lite_timeout_seconds,
                    lite_max_model_calls,
                    lite_max_concurrent_model_calls,
                    lite_model_context_window_tokens,
                    lite_context_safety_margin_tokens,
                ),
                execution_profile=execution_profile,
            )
            snapshot = run_manager.start_evaluation_compare(request, default_output_dir)
            boundary = json.dumps(
                {
                    "memory_access": "disabled",
                    "planner_worker_summarizer_review": "disabled",
                    "source_snapshot_mode": "fresh_once_per_comparison",
                    "summary_count": len(request.summaries),
                    "evaluation_config": request.evaluation_config.model_dump(
                        mode="json"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            return (
                _snapshot_text(snapshot),
                snapshot.run_id,
                str(snapshot.output_dir),
                boundary,
            )
        except Exception as exc:
            return f"启动失败：{type(exc).__name__}: {exc}", "", "", ""

    def refresh(
        run_id: str,
    ) -> tuple[Any, ...]:
        """一次读取状态、可视化、脱敏 Inspector 和 Evaluation 报告。"""
        empty_visual = '<div class="smw-panel">暂无运行可视化。</div>'
        empty_report = '<div class="smw-panel">该运行尚未生成丰富纪要成果。</div>'
        if not run_id.strip():
            return (
                "暂无运行",
                "",
                "{}",
                {},
                "",
                {},
                "",
                empty_visual,
                empty_visual,
                empty_visual,
                empty_visual,
                empty_visual,
                gr.update(choices=[], value=None),
                empty_visual,
                empty_report,
                "",
                "",
                '<div class="smw-result-heading"><h2>选择一个运行查看成果</h2></div>',
                '<div class="smw-list-empty">尚未开始评测</div>',
            )
        try:
            snapshot = run_manager.get_snapshot(run_id.strip())
        except KeyError:
            return (
                "找不到该运行",
                "",
                "{}",
                {},
                "",
                {},
                "",
                empty_visual,
                empty_visual,
                empty_visual,
                empty_visual,
                empty_visual,
                gr.update(choices=[], value=None),
                empty_visual,
                empty_report,
                "",
                "",
                '<div class="smw-result-heading"><h2>找不到该运行</h2></div>',
                '<div class="smw-list-empty">尚未开始评测</div>',
            )
        evaluation_metrics: dict[str, Any] = {}
        evaluation_report = ""
        personalization_metrics: dict[str, Any] = {}
        personalization_report = ""
        overview_html = empty_visual
        graph_html = empty_visual
        timeline_html = empty_visual
        evidence_html = empty_visual
        review_evaluation_html = empty_visual
        task_update = gr.update(choices=[], value=None)
        task_detail_html = empty_visual
        minutes_report_html = empty_report
        refreshed_output_path = str(snapshot.output_dir)
        refreshed_store_identity = ""
        document_title = ""
        try:
            inspector = run_manager.get_artifacts(run_id.strip())
            evaluation_metrics = dict(
                inspector.evaluation.get("metrics", {})
                if isinstance(inspector.evaluation, dict)
                else {}
            )
            evaluation_report = str(
                inspector.evaluation.get("report_markdown", "")
                if isinstance(inspector.evaluation, dict)
                else ""
            )
            personalization = (
                inspector.evaluation.get("personalization", {})
                if isinstance(inspector.evaluation, dict)
                else {}
            )
            personalization_metrics = dict(
                personalization.get("metrics", {})
                if isinstance(personalization, dict)
                else {}
            )
            personalization_report = str(
                personalization.get("report_markdown", "")
                if isinstance(personalization, dict)
                else ""
            )
            inspector_text = json.dumps(
                inspector.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            refreshed_store_identity = json.dumps(
                inspector.store_probe.get("identity", {})
                if isinstance(inspector.store_probe, dict)
                else {},
                ensure_ascii=False,
                indent=2,
            )
            if inspector.minutes_document and inspector.minutes_sources:
                try:
                    document = MinutesDocument.model_validate(inspector.minutes_document)
                    document_title = document.title
                    minutes_report_html = render_web_report_fragment(
                        document,
                        MinutesSources.model_validate(inspector.minutes_sources),
                    )
                except Exception as exc:
                    minutes_report_html = (
                        '<div class="smw-error">纪要成果读取失败：'
                        + html.escape(type(exc).__name__ + ": " + str(exc), quote=True)
                        + "</div>"
                    )
        except Exception as exc:
            inspector_text = json.dumps(
                {"error": f"Inspector 读取失败：{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        try:
            visualization = run_manager.get_visualization(run_id.strip())
            overview_html = render_overview_html(visualization)
            graph_html = render_agent_graph_html(visualization)
            timeline_html = render_timeline_html(visualization)
            evidence_html = render_evidence_memory_html(visualization)
            review_evaluation_html = render_review_evaluation_html(visualization)
            choices = task_choices(visualization)
            selected_task = choices[0][1] if choices else None
            task_update = gr.update(choices=choices, value=selected_task)
            task_detail_html = render_task_detail_html(
                visualization, selected_task or ""
            )
            store_payload = {
                "store": visualization.overview.get("store", {}),
                "filters": visualization.overview.get("filters", {}),
                "memory_access": visualization.overview.get("memory_access", ""),
            }
            if any(store_payload.values()):
                refreshed_store_identity = json.dumps(
                    store_payload,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            message = (
                '<div class="smw-error">可视化读取失败：'
                f"{html.escape(type(exc).__name__ + ': ' + str(exc), quote=True)}</div>"
            )
            overview_html = message
            graph_html = message
            timeline_html = message
            evidence_html = message
            review_evaluation_html = message
        if minutes_report_html == empty_report and snapshot.summary:
            minutes_report_html = (
                '<div class="smw-plain-minutes"><pre>'
                + html.escape(snapshot.summary)
                + '</pre></div>'
            )
        try:
            info = json.loads((snapshot.output_dir / "web_run_info.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            info = {}
        heading = run_heading_html(
            title=document_title or {
                "ingest": "资料导入",
                "evaluation_compare": "纪要横评",
            }.get(snapshot.run_type, "会议纪要"),
            project=str(info.get("project") or ""),
            source_time=str(info.get("source_time") or ""),
            status=snapshot.status.value,
        )
        compare_preview = (
            review_evaluation_html if snapshot.run_type == "evaluation_compare"
            else '<div class="smw-list-empty">尚未开始评测</div>'
        )
        return (
            _snapshot_text(snapshot),
            snapshot.summary,
            inspector_text,
            evaluation_metrics,
            evaluation_report,
            personalization_metrics,
            personalization_report,
            overview_html,
            graph_html,
            timeline_html,
            evidence_html,
            review_evaluation_html,
            task_update,
            task_detail_html,
            minutes_report_html,
            refreshed_output_path,
            refreshed_store_identity,
            heading,
            compare_preview,
        )

    def watch_run(run_id: str, initial_status: str = ""):
        """Stream persisted progress without starting another model call."""
        if not (run_id or "").strip():
            yield (
                initial_status, "", gr.update(), gr.update(),
                '<div class="smw-progress smw-progress-failed"><strong>未能启动</strong>'
                f'<span>{html.escape(initial_status[:220])}</span></div>',
            )
            return
        last_summary = None
        for _ in range(3600):
            try:
                snapshot = run_manager.get_snapshot(run_id.strip())
            except KeyError:
                return
            if snapshot.summary != last_summary or snapshot.status in {
                WebRunStatus.COMPLETED, WebRunStatus.COMPLETED_WITH_WARNINGS,
                WebRunStatus.FAILED,
            }:
                view = refresh(run_id)
                last_summary = snapshot.summary
                yield view[0], view[1], view[14], view[18], progress_html(snapshot)
            else:
                yield (
                    _snapshot_text(snapshot), snapshot.summary,
                    gr.update(), gr.update(), progress_html(snapshot),
                )
            if snapshot.status in {
                WebRunStatus.COMPLETED, WebRunStatus.COMPLETED_WITH_WARNINGS,
                WebRunStatus.FAILED,
            }:
                return
            time.sleep(2)

    def progress_html(snapshot: Any) -> str:
        status_value = snapshot.status.value
        label = {
            "queued": "已加入队列", "running": "正在处理",
            "completed": "已完成", "completed_with_warnings": "已完成，有提示",
            "failed": "处理失败", "cancelling": "正在停止",
        }.get(status_value, status_value)
        stage = str(snapshot.stage or "")
        detail = snapshot.error or (f"当前阶段：{stage}" if stage else "")
        return (
            f'<div class="smw-progress smw-progress-{html.escape(status_value)}">'
            f'<strong>{html.escape(label)}</strong>'
            f'<span>{html.escape(detail[:220])}</span></div>'
        )

    def refresh_task_detail(run_id: str, task_id: str) -> str:
        """按下拉框选择读取 Agent 节点详情，不依赖 SVG JavaScript。"""
        if not run_id.strip() or not task_id:
            return '<div class="smw-panel">请选择一个 Agent 任务查看详情。</div>'
        try:
            visualization = run_manager.get_visualization(run_id.strip())
            return render_task_detail_html(visualization, task_id)
        except Exception as exc:
            return (
                '<div class="smw-error">任务详情读取失败：'
                f"{html.escape(type(exc).__name__ + ': ' + str(exc), quote=True)}</div>"
            )

    def refresh_runs(query: str = "") -> tuple[Any, str]:
        """刷新可从磁盘恢复的运行选择和最近导入。"""
        all_entries = run_manager.list_persisted_runs(default_output_dir)
        entries = filtered_runs(all_entries, query)[:30]
        choices = [
            (run_label(item), item["run_id"])
            for item in entries
        ]
        ingests = [item for item in all_entries if item["run_type"] in {"ingest", "e2e_ingest"}]
        return (
            gr.update(choices=choices, value=choices[0][1] if choices else None),
            run_list_html(ingests[:5], empty="还没有导入记录"),
        )

    def use_selected_run(selected_run_id: str) -> str:
        """将下拉框选择的 run_id 写入当前 Inspector 输入。"""
        return selected_run_id or ""

    def stop(run_id: str) -> str:
        """提交协作式停止请求，并返回最新状态文本。"""
        if not run_id.strip():
            return "没有可停止的运行"
        try:
            return _snapshot_text(run_manager.request_stop(run_id.strip()))
        except KeyError:
            return "找不到该运行"

    def download(run_id: str) -> str | None:
        """创建并返回安全审计归档路径，失败时返回空值。"""
        if not run_id.strip():
            return None
        try:
            return str(run_manager.create_archive(run_id.strip()))
        except (KeyError, ValueError, OSError):
            return None

    def download_report_artifact(run_id: str, name: str) -> str | None:
        """Return one public rich-report artifact without rebuilding a run."""

        if not run_id.strip():
            return None
        try:
            return str(run_manager.get_report_artifact(run_id.strip(), name))
        except (KeyError, ValueError, OSError):
            return None

    def prepared_report_download(run_id: str, name: str) -> Any:
        path = download_report_artifact(run_id, name)
        return gr.update(value=path, visible=bool(path))

    with gr.Blocks(
        title="会议纪要工作台",
        css=WEB_VISUALIZATION_CSS + CLOUD_UI_CSS,
    ) as demo:
        try:
            model_info = get_llm_runtime_info()
            model_label = str(model_info.get("llm_model") or "未配置")
        except RuntimeError:
            model_label = "尚未配置模型"
        with gr.Row(elem_classes=["smw-appbar"]):
            gr.HTML(
                '<div class="smw-brand"><span class="smw-brand-mark" aria-hidden="true">会</span>'
                '<strong>会议工作台</strong></div>'
            )
            gr.HTML(
                '<div class="smw-model-chip"><span aria-hidden="true"></span>模型配置：'
                + html.escape(model_label) + '</div>'
            )
            ingest_project = gr.Textbox(label="所属项目", placeholder="输入项目名称")
            ingest_user_id = gr.Textbox(label="记忆身份", placeholder="同一项目请保持一致")
        with gr.Accordion("处理设置", open=False):
            configured_profile = os.getenv("SMW_EXECUTION_PROFILE", "cloud_fast").strip().lower()
            execution_profile_input = gr.Dropdown(
                choices=[("快速流程", "cloud_fast"), ("兼容流程", "legacy")],
                value=configured_profile if configured_profile in {"cloud_fast", "legacy"} else "cloud_fast",
                label="处理方式",
            )
        with gr.Tabs(elem_classes=["smw-primary-tabs"]):
            with gr.Tab("资料导入"):
                gr.Markdown("## 资料导入", elem_classes=["smw-page-title"])
                with gr.Row(elem_classes=["smw-page-columns"]):
                    with gr.Column(scale=3, min_width=340):
                        with gr.Tabs(elem_classes=["smw-mode-tabs"]):
                            with gr.Tab("上传文件"):
                                ingest_history_files = gr.File(
                                    label="上传会议材料（.txt / .md）", file_count="multiple",
                                    file_types=[".txt", ".md"],
                                    elem_classes=["smw-upload"],
                                    **compact_upload,
                                )
                                ingest_file_dates = gr.Dataframe(
                                    headers=["已选文件", "资料日期（YYYY-MM-DD，可留空）"],
                                    datatype=["str", "str"],
                                    row_count=(0, "dynamic"),
                                    col_count=(2, "fixed"),
                                    type="array",
                                    interactive=True,
                                    visible=False,
                                    label="逐文件日期",
                                    elem_classes=["smw-file-dates"],
                                )
                            with gr.Tab("粘贴文本"):
                                ingest_history_text = gr.Textbox(label="历史材料", lines=12)
                        with gr.Accordion("使用服务器文件路径", open=False):
                            ingest_history_paths = gr.Textbox(label="每行一个路径", lines=3)
                    with gr.Column(scale=2, min_width=260, elem_classes=["smw-side-form"]):
                        gr.HTML('<div class="smw-field-caption">导入范围</div><div class="smw-scope-chip">历史会议</div>')
                        ingest_source_time = gr.Textbox(
                            label="默认资料日期", placeholder="例如 2026-07-21"
                        )
                        ingest_ready = gr.HTML(value=ready_message(None))
                        ingest_start_button = gr.Button("提取并保存", variant="primary")
                        ingest_progress = gr.HTML(value="")
                        with gr.Accordion("高级设置", open=False):
                            ingest_source_dates = gr.Textbox(
                                label="粘贴/服务器路径的逐份日期（可选，每行一个）", lines=3
                            )
                            ingest_source_type = gr.Textbox(label="来源类型", value="history")
                            ingest_qdrant_path = gr.Textbox(label="存储路径覆盖")
                            ingest_collection = gr.Textbox(label="集合名覆盖")
                            ingest_chunk_chars = gr.Number(
                                value=None,
                                precision=0,
                                label="历史分块字符数 chunk_char_limit（越小越稳、调用越多）",
                            )
                            ingest_typed_tokens = gr.Number(
                                value=None,
                                precision=0,
                                label="单块记忆抽取上限 typed_extraction_max_tokens",
                            )
                            ingest_chunk_tokens = gr.Number(
                                value=None,
                                precision=0,
                                label="单块基础摘要上限 chunk_max_tokens",
                            )
                            ingest_merge_tokens = gr.Number(
                                value=None,
                                precision=0,
                                label="合并基础摘要上限 merge_max_tokens",
                            )
                            ingest_dedup_tokens = gr.Number(
                                value=None,
                                precision=0,
                                label="跨块去重判断上限 dedup_judge_max_tokens",
                            )
                            ingest_dedup_candidates = gr.Number(
                                value=None,
                                precision=0,
                                label="单次去重候选数 dedup_judge_max_candidates",
                            )
                gr.HTML('<div class="smw-section-heading">最近导入</div>')
                recent_imports = gr.HTML(value=run_list_html([], empty="还没有导入记录"))

            with gr.Tab("生成纪要"):
                gr.Markdown("## 生成纪要", elem_classes=["smw-page-title"])
                with gr.Row(elem_classes=["smw-page-columns"]):
                    with gr.Column(scale=3, min_width=360):
                        with gr.Tabs(elem_classes=["smw-mode-tabs"]):
                            with gr.Tab("粘贴原文"):
                                summary_current_material = gr.Textbox(label="本次会议原文", lines=16)
                            with gr.Tab("上传文件"):
                                summary_current_file = gr.File(
                                    label="选择 .txt 或 .md", file_count="single",
                                    file_types=[".txt", ".md"],
                                )
                        with gr.Accordion("使用服务器文件路径", open=False):
                            summary_current_path = gr.Textbox(label="当前材料路径")
                        summary_request_input = gr.Textbox(
                            label="这次纪要重点关注什么？",
                            lines=4,
                        )
                    with gr.Column(scale=2, min_width=260, elem_classes=["smw-side-form"]):
                        summary_user_id = ingest_user_id
                        summary_project = ingest_project
                        summary_source_time = gr.Textbox(
                            label="本次会议日期", placeholder="例如 2026-07-28"
                        )
                        summary_scene_input = gr.Dropdown(
                            choices=[
                                ("自动识别", "auto"),
                                ("通用会议", "general"),
                                ("项目例会", "project_sync"),
                                ("决策评审", "decision_review"),
                                ("专项讨论", "focused_discussion"),
                            ],
                            value="auto",
                            label="会议场景",
                        )
                        summary_style_input = gr.Radio(
                            choices=[("结构化", "structured"), ("自然叙述", "natural")],
                            value="structured", label="表达方式",
                            elem_classes=["smw-segmented"],
                        )
                        summary_output_mode = gr.Radio(
                            choices=[
                                ("普通纪要", "classic"),
                                ("完整成果", "rich"),
                            ],
                            value="rich",
                            label="产出模式",
                            elem_classes=["smw-segmented"],
                        )
                        gr.HTML('<div class="smw-history-available">可使用所选项目的记忆 <span>实际检索以运行结果为准</span></div>')
                        with gr.Accordion("高级设置", open=False):
                            summary_targets = gr.Textbox(
                                label="关注的人或团队（可选，多个用逗号分隔）"
                            )
                            summary_enable_review = gr.Checkbox(
                                label="深度审核（会增加处理时间）", value=False,
                            )
                            summary_qdrant_path = gr.Textbox(label="存储路径覆盖")
                            summary_collection = gr.Textbox(label="集合名覆盖")
                            summary_planner_mode = gr.Dropdown(
                                choices=[("模型规划", "model"), ("规则规划", "rule")],
                                value="model",
                                label="规划方式",
                            )
                            summary_worker_mode = gr.Dropdown(
                                choices=[("模型执行", "model"), ("模型执行并允许规则回退", "model_with_rule_fallback"), ("规则执行", "rule")],
                                value="model_with_rule_fallback",
                                label="任务执行方式",
                            )
                            summary_compression_mode = gr.Dropdown(
                                choices=[("整理重复历史", "deterministic"), ("不整理", "disabled")],
                                value="deterministic",
                                label="历史整理",
                            )
                        summary_enable_evaluation = gr.Checkbox(
                            label="生成后评测",
                            value=False,
                        )
                        summary_evaluation_mode = gr.Dropdown(
                            choices=[("严格评测", "strict"), ("精简评测", "lite")],
                            value="lite",
                            label="生成后评测模式",
                            visible=False,
                        )
                        with gr.Accordion("纪要高级参数（通常保持默认）", open=False):
                            summary_raw_segment = gr.Number(
                                value=12000,
                                precision=0,
                                label="当前材料分块字符数 raw_segment_max_chars",
                            )
                            summary_max_replan = gr.Number(
                                value=1,
                                precision=0,
                                label="最大重规划轮数 max_replan_rounds",
                            )
                            summary_max_review = gr.Number(
                                value=3,
                                precision=0,
                                label="最大审核轮数 max_review_rounds",
                            )
                            summary_max_workers = gr.Number(
                                value=4,
                                precision=0,
                                label="最大并发 Worker 数 max_concurrent_workers",
                            )
                            summary_max_segments = gr.Number(
                                value=2,
                                precision=0,
                                label="最大并发材料分块数 max_concurrent_segment_maps",
                            )
                            summary_max_calls = gr.Number(
                                value=4,
                                precision=0,
                                label="最大并发模型调用数 max_concurrent_model_calls",
                            )
                            summary_worker_model_timeout = gr.Number(
                                value=180,
                                precision=0,
                                label="单次 Worker 模型调用超时（秒）worker_model_timeout_seconds",
                            )
                            summary_worker_timeout = gr.Number(
                                value=600,
                                precision=0,
                                label="单个 Worker 任务总超时（秒）worker_timeout_seconds",
                            )
                            summary_global_timeout = gr.Number(
                                value=1200,
                                precision=0,
                                label="整轮 Pipeline 超时（秒）global_timeout_seconds",
                            )
                            summary_context_window = gr.Number(
                                value=None,
                                precision=0,
                                label="已知模型上下文窗口（token，可留空）model_context_window_tokens",
                            )
                            summary_context_margin = gr.Number(
                                value=1024,
                                precision=0,
                                label="上下文安全余量（token）model_context_safety_margin_tokens",
                            )
                            summary_rich_tokens = gr.Number(
                                value=8192,
                                precision=0,
                                label="丰富纪要输出上限 rich_summary_max_tokens",
                            )
                            gr.Markdown(
                                "以下参数只控制精简评测；严格评测继续使用原有独立配置。"
                            )
                            with gr.Row():
                                summary_lite_segment_chars = gr.Number(
                                    value=12000,
                                    precision=0,
                                    label="精简评测参考事实分块字符数",
                                )
                                summary_lite_overlap_chars = gr.Number(
                                    value=100,
                                    precision=0,
                                    label="精简评测分块重叠字符数",
                                )
                                summary_lite_concurrency = gr.Number(
                                    value=4, precision=0, label="精简评测模型并发数"
                                )
                            with gr.Row():
                                summary_lite_reference_tokens = gr.Number(
                                    value=8192,
                                    precision=0,
                                    label="参考事实抽取输出上限",
                                )
                                summary_lite_claim_tokens = gr.Number(
                                    value=8192,
                                    precision=0,
                                    label="纪要事实拆分输出上限",
                                )
                                summary_lite_match_tokens = gr.Number(
                                    value=4096, precision=0, label="覆盖匹配输出上限"
                                )
                                summary_lite_verify_tokens = gr.Number(
                                    value=4096, precision=0, label="支持判断输出上限"
                                )
                            with gr.Row():
                                summary_lite_repairs = gr.Number(
                                    value=1, precision=0, label="精简评测 JSON 修复次数"
                                )
                                summary_lite_model_timeout = gr.Number(
                                    value=180,
                                    precision=0,
                                    label="精简评测单次调用超时（秒）",
                                )
                                summary_lite_timeout = gr.Number(
                                    value=1200,
                                    precision=0,
                                    label="精简评测整轮超时（秒）",
                                )
                                summary_lite_max_calls = gr.Number(
                                    value=64,
                                    precision=0,
                                    label="精简评测最大模型调用数",
                                )
                        summarize_start_button = gr.Button("生成纪要", variant="primary")
                        summary_progress = gr.HTML(value="")

            with gr.Tab("纪要横评"):
                gr.Markdown("## 纪要横评", elem_classes=["smw-page-title"])
                with gr.Row(elem_classes=["smw-compare-top"]):
                    compare_project = ingest_project
                    compare_targets = gr.Textbox(
                        label="关注的人或团队（严格评测可选）", visible=False,
                    )
                    compare_mode = gr.Radio(
                        choices=[("精简评测", "lite"), ("严格评测", "strict")],
                        value="lite", label="评测模式",
                        elem_classes=["smw-segmented"],
                    )
                with gr.Accordion("评测高级设置", open=False):
                    compare_strict_note = gr.Markdown(
                        "严格模式使用下列事实快照、双范围匹配、个性化与解释参数；"
                        "精简模式不会调用这些旁路。"
                        "分段越小，结构化抽取通常越稳定，但模型调用次数会增加。"
                        "自适应拆分只在单个源分段的 JSON 修复耗尽后触发，不修改原文和全局 offset。"
                        "数字指标完成后会自动生成全材料/当前材料直接对比，并二次核验最多 12 条/纪要对/范围的机械覆盖差异；"
                        "解释层最多使用 32 次模型调用，失败不改变已有指标。",
                        visible=False,
                    )
                    with gr.Row():
                        compare_segment_chars = gr.Number(
                            value=500,
                            precision=0,
                            label="源材料分块字符数 segment_max_chars",
                            visible=False,
                        )
                        compare_source_tokens = gr.Number(
                            value=3000,
                            precision=0,
                            label="源事实抽取输出上限 source_fact_max_tokens",
                            visible=False,
                        )
                        compare_summary_tokens = gr.Number(
                            value=3000,
                            precision=0,
                            label="纪要事实抽取输出上限 summary_fact_max_tokens",
                            visible=False,
                        )
                    with gr.Row():
                        compare_match_tokens = gr.Number(
                            value=3000,
                            precision=0,
                            label="事实匹配输出上限 match_max_tokens",
                            visible=False,
                        )
                        compare_personalization_tokens = gr.Number(
                            value=2400,
                            precision=0,
                            label="个性化标注输出上限 personalization_max_tokens",
                            visible=False,
                        )
                        compare_repair_attempts = gr.Number(
                            value=2,
                            precision=0,
                            label="JSON 协议修复次数 repair_attempts",
                            visible=False,
                        )
                    with gr.Row():
                        compare_adaptive_split = gr.Checkbox(
                            value=True,
                            label="启用失败分段自适应拆分 adaptive_split_enabled",
                            visible=False,
                        )
                        compare_adaptive_min_chars = gr.Number(
                            value=500,
                            precision=0,
                            label="自适应拆分最小字符数 adaptive_split_min_chars",
                            visible=False,
                        )
                        compare_adaptive_max_depth = gr.Number(
                            value=1,
                            precision=0,
                            label="自适应拆分最大层数 adaptive_split_max_depth",
                            visible=False,
                        )
                    compare_interpretation = gr.Checkbox(
                        value=True,
                        label=(
                            "启用横评解释 comparison_interpretation "
                            "（直接定性对比并二次核验事实增减；可选，会增加耗时）"
                        ),
                        visible=False,
                    )
                    compare_lite_note = gr.Markdown(
                        "精简模式只执行参考事实抽取、纪要事实拆分、一次批量覆盖匹配和一次批量支持判断。"
                    )
                    with gr.Row():
                        compare_lite_segment_chars = gr.Number(
                            value=12000, precision=0, label="精简参考事实分块字符数"
                        )
                        compare_lite_overlap_chars = gr.Number(
                            value=100, precision=0, label="精简分块重叠字符数"
                        )
                        compare_lite_concurrency = gr.Number(
                            value=4, precision=0, label="精简模型并发数"
                        )
                    with gr.Row():
                        compare_lite_reference_tokens = gr.Number(
                            value=8192, precision=0, label="精简参考事实输出上限"
                        )
                        compare_lite_claim_tokens = gr.Number(
                            value=8192, precision=0, label="精简纪要事实输出上限"
                        )
                        compare_lite_match_tokens = gr.Number(
                            value=4096, precision=0, label="精简覆盖匹配输出上限"
                        )
                        compare_lite_verify_tokens = gr.Number(
                            value=4096, precision=0, label="精简支持判断输出上限"
                        )
                    with gr.Row():
                        compare_lite_repairs = gr.Number(
                            value=1, precision=0, label="精简 JSON 修复次数"
                        )
                        compare_lite_model_timeout = gr.Number(
                            value=180, precision=0, label="精简单次调用超时（秒）"
                        )
                        compare_lite_timeout = gr.Number(
                            value=1200, precision=0, label="精简整轮超时（秒）"
                        )
                        compare_lite_max_calls = gr.Number(
                            value=64, precision=0, label="精简最大模型调用数"
                        )
                    with gr.Row():
                        compare_lite_context_window = gr.Number(
                            value=None,
                            precision=0,
                            label="精简评测模型上下文窗口（可选）",
                        )
                        compare_lite_context_margin = gr.Number(
                            value=1024,
                            precision=0,
                            label="精简评测上下文安全余量（token）",
                        )
                with gr.Row(elem_classes=["smw-compare-layout"]):
                    with gr.Column(scale=1, min_width=245, elem_classes=["smw-compare-source"]):
                        gr.HTML('<div class="smw-field-caption">共同材料</div>')
                        compare_request = gr.Textbox(label="评测关注点（可选）", lines=3)
                        compare_current_id = gr.Textbox(value="current", visible=False)
                        compare_current_label = gr.Textbox(value="当前材料", visible=False)
                        compare_current_time = gr.Textbox(label="本次会议日期")
                        compare_current_text = gr.Textbox(
                            label="本次会议原文（与上传二选一）", lines=5
                        )
                        with gr.Accordion("上传本次会议文件", open=False):
                            compare_current_file = gr.File(
                                label="上传本次会议（.txt / .md）", file_count="single",
                                file_types=[".txt", ".md"],
                            )
                        compare_history_components: list[Any] = []
                        for index in range(1, 6):
                            with gr.Accordion(f"历史材料 {index}（可选）", open=False):
                                history_id = gr.Textbox(value=f"history_{index}", visible=False)
                                history_label = gr.Textbox(label="资料名称")
                                history_time = gr.Textbox(label="材料日期")
                                history_text = gr.Textbox(
                                    label="历史原文（与上传二选一）", lines=5
                                )
                                with gr.Accordion("上传历史原文文件", open=False):
                                    history_file = gr.File(
                                        label="上传历史原文（.txt / .md）",
                                        file_count="single", file_types=[".txt", ".md"],
                                    )
                            compare_history_components.extend([
                                history_id, history_label, history_time,
                                history_text, history_file,
                            ])
                    with gr.Column(scale=3, min_width=360):
                        gr.HTML('<div class="smw-field-caption">待比较纪要</div>')
                        compare_summary_components: list[Any] = []
                        extra_candidate_panels: list[Any] = []
                        for pair_start in (1, 3, 5):
                            with gr.Row(elem_classes=["smw-candidate-row"]):
                                for index in range(pair_start, min(pair_start + 2, 6)):
                                    with gr.Column():
                                        with gr.Accordion(
                                            "基线纪要" if index == 1 else f"纪要 {index}",
                                            open=index <= 2, visible=index <= 2,
                                            elem_classes=["smw-candidate"],
                                        ) as candidate_panel:
                                            summary_id = gr.Textbox(
                                                value=f"summary_{index}", visible=False,
                                            )
                                            display_name = gr.Textbox(
                                                label="显示名称",
                                                value=("直接生成（基线）" if index == 1 else
                                                       "多 Agent 纪要" if index == 2 else ""),
                                            )
                                            compare_summary_text = gr.Textbox(
                                                label="纪要正文", lines=10,
                                            )
                                            with gr.Accordion("上传纪要文件", open=False):
                                                compare_summary_file = gr.File(
                                                    label="上传 .txt 或 .md", file_count="single",
                                                    file_types=[".txt", ".md"],
                                                )
                                        if index > 2:
                                            extra_candidate_panels.append(candidate_panel)
                                        compare_summary_components.extend([
                                            summary_id, display_name,
                                            compare_summary_text, compare_summary_file,
                                        ])
                        candidate_count = gr.State(2)
                        add_candidate_button = gr.Button("＋ 添加纪要")
                        compare_start_button = gr.Button("开始对比评测", variant="primary")
                        compare_progress = gr.HTML(value="")
                gr.HTML('<div class="smw-section-heading">评测结果</div>')
                compare_result_preview = gr.HTML(
                    value='<div class="smw-list-empty">尚未开始评测</div>'
                )

            with gr.Tab("结果中心"):
                gr.Markdown("## 结果中心", elem_classes=["smw-page-title"])
                with gr.Row(elem_classes=["smw-results-layout"]):
                    with gr.Column(scale=1, min_width=260):
                        run_search = gr.Textbox(label="搜索运行", placeholder="项目、日期或状态")
                        persisted_runs = gr.Radio(
                            choices=[], label="运行记录", interactive=True,
                            elem_classes=["smw-run-selector"],
                        )
                        list_runs_button = gr.Button("刷新列表")
                        with gr.Accordion("处理状态与运行编号", open=False):
                            run_id = gr.Textbox(label="运行编号", interactive=True)
                            status = gr.Textbox(label="处理状态", lines=7)
                            with gr.Row():
                                refresh_button = gr.Button("刷新状态")
                                stop_button = gr.Button("停止")
                    with gr.Column(scale=3, min_width=260):
                        run_heading = gr.HTML(
                            value='<div class="smw-result-heading"><h2>选择一个运行查看成果</h2></div>'
                        )
                        with gr.Row(elem_classes=["smw-result-actions"]):
                            copy_button = gr.Button("复制正文")
                            report_html_button = gr.Button("下载 HTML")
                            report_bundle_button = gr.Button("导出成果")
                        with gr.Row(elem_classes=["smw-download-links"]):
                            report_html_file = gr.File(
                                label="HTML", interactive=False, visible=False,
                            )
                            report_bundle_file = gr.File(
                                label="成果 ZIP", interactive=False, visible=False,
                            )
                        with gr.Tabs():
                            with gr.Tab("完整纪要"):
                                minutes_report_visual = gr.HTML(
                                    value='<div class="smw-panel">尚未选择运行。</div>'
                                )
                            with gr.Tab("处理详情"):
                                with gr.Tabs():
                                    with gr.Tab("运行概况"):
                                        overview_visual = gr.HTML(
                                            value='<div class="smw-panel">选择运行后点击“刷新状态”。</div>'
                                        )
                                        output_path = gr.Textbox(
                                            label="运行产物目录", interactive=False
                                        )
                                        store_identity = gr.Textbox(
                                            label="记忆库信息", lines=4, interactive=False,
                                        )
                                        summary = gr.Markdown(label="正式纪要正文")
                                    with gr.Tab("任务图"):
                                        agent_graph_visual = gr.HTML(
                                            value='<div class="smw-panel">暂无 Agent 运行图。</div>'
                                        )
                                        agent_task_selector = gr.Dropdown(
                                            choices=[], label="任务详情", interactive=True,
                                        )
                                        agent_task_detail = gr.HTML(
                                            value='<div class="smw-panel">请选择一个 Agent 任务。</div>'
                                        )
                                    with gr.Tab("时间线"):
                                        timeline_visual = gr.HTML(
                                            value='<div class="smw-panel">暂无执行时间线。</div>'
                                        )
                                    with gr.Tab("证据与记忆"):
                                        evidence_memory_visual = gr.HTML(
                                            value='<div class="smw-panel">暂无 Evidence/Memory 数据。</div>'
                                        )
                                    with gr.Tab("审核与评测"):
                                        review_evaluation_visual = gr.HTML(
                                            value='<div class="smw-panel">暂无 Review/Evaluation 数据。</div>'
                                        )
                                        with gr.Accordion("整体评测指标与报告", open=False):
                                            evaluation_metrics = gr.JSON(label="整体评测指标")
                                            evaluation_report = gr.Markdown(label="整体评测报告")
                                        with gr.Accordion("个性化评测", open=False):
                                            personalization_metrics = gr.JSON(
                                                label="目标实体评测指标"
                                            )
                                            personalization_report = gr.Markdown(
                                                label="目标实体评测报告"
                                            )
                                    with gr.Tab("原始审计"):
                                        inspector = gr.Textbox(
                                            label="运行审计", lines=28, interactive=False,
                                        )
                                        download_file = gr.File(
                                            label="审计产物", interactive=False,
                                        )
                                        download_button = gr.Button("下载审计产物")

        ingest_inputs = [
            ingest_history_text,
            ingest_history_files,
            ingest_history_paths,
            ingest_user_id,
            ingest_project,
            ingest_source_time,
            ingest_source_type,
            ingest_qdrant_path,
            ingest_collection,
            ingest_chunk_chars,
            ingest_chunk_tokens,
            ingest_merge_tokens,
            ingest_typed_tokens,
            ingest_dedup_tokens,
            ingest_dedup_candidates,
            ingest_source_dates,
            execution_profile_input,
            ingest_file_dates,
        ]
        ingest_event = ingest_start_button.click(
            start_ingest,
            inputs=ingest_inputs,
            outputs=[status, run_id, output_path, store_identity],
        )
        ingest_history_files.change(
            lambda files, previous, pasted: (
                gr.update(
                    value=file_date_rows(files, previous),
                    visible=bool(files),
                ),
                ready_message(files, pasted),
            ),
            inputs=[ingest_history_files, ingest_file_dates, ingest_history_text],
            outputs=[ingest_file_dates, ingest_ready],
        )
        ingest_history_text.change(
            ready_message,
            inputs=[ingest_history_files, ingest_history_text],
            outputs=[ingest_ready],
        )
        summarize_inputs = [
            summary_current_material,
            summary_current_file,
            summary_current_path,
            summary_request_input,
            summary_user_id,
            summary_project,
            summary_source_time,
            summary_qdrant_path,
            summary_collection,
            summary_targets,
            summary_style_input,
            summary_planner_mode,
            summary_worker_mode,
            summary_compression_mode,
            summary_enable_review,
            summary_enable_evaluation,
            summary_max_replan,
            summary_max_review,
            summary_max_workers,
            summary_max_segments,
            summary_max_calls,
            summary_worker_model_timeout,
            summary_worker_timeout,
            summary_global_timeout,
            summary_raw_segment,
            summary_context_window,
            summary_context_margin,
            summary_scene_input,
            summary_output_mode,
            summary_rich_tokens,
            summary_evaluation_mode,
            summary_lite_segment_chars,
            summary_lite_overlap_chars,
            summary_lite_reference_tokens,
            summary_lite_claim_tokens,
            summary_lite_match_tokens,
            summary_lite_verify_tokens,
            summary_lite_repairs,
            summary_lite_model_timeout,
            summary_lite_timeout,
            summary_lite_max_calls,
            summary_lite_concurrency,
            execution_profile_input,
        ]
        summarize_event = summarize_start_button.click(
            start_summarize,
            inputs=summarize_inputs,
            outputs=[status, run_id, output_path, store_identity],
        )
        summary_enable_evaluation.change(
            lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[summary_enable_evaluation], outputs=[summary_evaluation_mode],
        )
        compare_inputs = [
            compare_project,
            compare_request,
            compare_targets,
            compare_segment_chars,
            compare_source_tokens,
            compare_summary_tokens,
            compare_match_tokens,
            compare_personalization_tokens,
            compare_repair_attempts,
            compare_adaptive_split,
            compare_adaptive_min_chars,
            compare_adaptive_max_depth,
            compare_interpretation,
            compare_mode,
            compare_lite_segment_chars,
            compare_lite_overlap_chars,
            compare_lite_reference_tokens,
            compare_lite_claim_tokens,
            compare_lite_match_tokens,
            compare_lite_verify_tokens,
            compare_lite_repairs,
            compare_lite_model_timeout,
            compare_lite_timeout,
            compare_lite_max_calls,
            compare_lite_concurrency,
            compare_lite_context_window,
            compare_lite_context_margin,
            execution_profile_input,
            compare_current_id,
            compare_current_label,
            compare_current_time,
            compare_current_text,
            compare_current_file,
            *compare_history_components,
            *compare_summary_components,
        ]
        compare_event = compare_start_button.click(
            start_evaluation_compare,
            inputs=compare_inputs,
            outputs=[status, run_id, output_path, store_identity],
        )
        def reveal_candidate(count: int) -> tuple[Any, ...]:
            next_count = min(max(int(count or 2), 2) + 1, 5)
            return (
                next_count,
                *[
                    gr.update(visible=index + 3 <= next_count, open=index + 3 <= next_count)
                    for index in range(3)
                ],
                gr.update(interactive=next_count < 5),
            )

        add_candidate_button.click(
            reveal_candidate,
            inputs=[candidate_count],
            outputs=[candidate_count, *extra_candidate_panels, add_candidate_button],
        )
        strict_controls = [
            compare_targets, compare_strict_note, compare_segment_chars,
            compare_source_tokens, compare_summary_tokens, compare_match_tokens,
            compare_personalization_tokens, compare_repair_attempts,
            compare_adaptive_split, compare_adaptive_min_chars,
            compare_adaptive_max_depth, compare_interpretation,
        ]
        lite_controls = [
            compare_lite_note, compare_lite_segment_chars,
            compare_lite_overlap_chars, compare_lite_concurrency,
            compare_lite_reference_tokens, compare_lite_claim_tokens,
            compare_lite_match_tokens, compare_lite_verify_tokens,
            compare_lite_repairs, compare_lite_model_timeout,
            compare_lite_timeout, compare_lite_max_calls,
            compare_lite_context_window, compare_lite_context_margin,
        ]
        compare_mode.change(
            lambda mode: [gr.update(visible=mode == "strict") for _ in strict_controls]
            + [gr.update(visible=mode == "lite") for _ in lite_controls],
            inputs=[compare_mode], outputs=strict_controls + lite_controls,
        )
        for event, local_progress in (
            (ingest_event, ingest_progress),
            (summarize_event, summary_progress),
            (compare_event, compare_progress),
        ):
            event.then(
                watch_run,
                inputs=[run_id, status],
                outputs=[
                    status, summary, minutes_report_visual,
                    compare_result_preview, local_progress,
                ],
            ).then(
                refresh,
                inputs=[run_id],
                outputs=[
                    status, summary, inspector, evaluation_metrics, evaluation_report,
                    personalization_metrics, personalization_report, overview_visual,
                    agent_graph_visual, timeline_visual, evidence_memory_visual,
                    review_evaluation_visual, agent_task_selector, agent_task_detail,
                    minutes_report_visual, output_path, store_identity,
                    run_heading, compare_result_preview,
                ],
            )
        demo.load(refresh_runs, inputs=[run_search],
                  outputs=[persisted_runs, recent_imports])
        list_runs_button.click(
            refresh_runs,
            inputs=[run_search],
            outputs=[persisted_runs, recent_imports],
        )
        run_search.change(
            refresh_runs, inputs=[run_search],
            outputs=[persisted_runs, recent_imports],
        )
        refresh_button.click(
            refresh,
            inputs=[run_id],
            outputs=[
                status,
                summary,
                inspector,
                evaluation_metrics,
                evaluation_report,
                personalization_metrics,
                personalization_report,
                overview_visual,
                agent_graph_visual,
                timeline_visual,
                evidence_memory_visual,
                review_evaluation_visual,
                agent_task_selector,
                agent_task_detail,
                minutes_report_visual,
                output_path,
                store_identity,
                run_heading,
                compare_result_preview,
            ],
        )
        persisted_runs.change(
            use_selected_run, inputs=[persisted_runs], outputs=[run_id],
        ).then(
            refresh, inputs=[run_id], outputs=[
                status, summary, inspector, evaluation_metrics, evaluation_report,
                personalization_metrics, personalization_report, overview_visual,
                agent_graph_visual, timeline_visual, evidence_memory_visual,
                review_evaluation_visual, agent_task_selector, agent_task_detail,
                minutes_report_visual, output_path, store_identity,
                run_heading, compare_result_preview,
            ],
        )
        agent_task_selector.change(
            refresh_task_detail,
            inputs=[run_id, agent_task_selector],
            outputs=[agent_task_detail],
        )
        stop_button.click(stop, inputs=[run_id], outputs=[status])
        download_button.click(download, inputs=[run_id], outputs=[download_file])
        report_html_button.click(
            lambda value: prepared_report_download(value, "minutes_report.html"),
            inputs=[run_id],
            outputs=[report_html_file],
        )
        report_bundle_button.click(
            lambda value: prepared_report_download(value, "minutes_report_bundle.zip"),
            inputs=[run_id],
            outputs=[report_bundle_file],
        )
        copy_js = "(value) => { if (value) navigator.clipboard.writeText(value); return []; }"
        click_parameters = inspect.signature(copy_button.click).parameters
        if "js" in click_parameters:
            copy_button.click(fn=None, inputs=[summary], outputs=[], js=copy_js)
        elif "_js" in click_parameters:
            copy_button.click(fn=None, inputs=[summary], outputs=[], _js=copy_js)
    return demo


def launch_demo(demo: Any, args: argparse.Namespace) -> None:
    """按当前 Gradio 版本兼容配置队列并启动 Web 服务。"""
    root_path = normalize_root_path(args.root_path)
    if root_path:
        os.environ["GRADIO_ROOT_PATH"] = root_path
    app = demo
    try:
        # Keep a slot available for stop/refresh while a progress stream is active.
        app = demo.queue(default_concurrency_limit=2)
    except TypeError:
        app = demo.queue(concurrency_count=2)
    launch_kwargs: dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
    }
    signature = inspect.signature(app.launch)
    if root_path:
        if "root_path" in signature.parameters:
            launch_kwargs["root_path"] = root_path
        elif "app_kwargs" in signature.parameters:
            launch_kwargs["app_kwargs"] = {"root_path": root_path}
    if "debug" in signature.parameters:
        launch_kwargs["debug"] = args.debug
    app.launch(**launch_kwargs)


def main() -> None:
    """创建 Web 应用并把进程生命周期交给 Gradio。"""
    os.environ.setdefault("SMW_LLM_BACKEND", "openai_compatible")
    args = parse_args()
    validate_llm_configuration()
    demo = build_app(output_dir=args.output_dir)
    try:
        launch_demo(demo, args)
    finally:
        # Gradio owns the process lifetime; this is best-effort for tests and shutdown paths.
        pass


if __name__ == "__main__":
    main()
