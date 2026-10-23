"""Gradio LAN web wrapper for the Summary Memory workflow demo.

Target machines should provide scripts/summary_memory_workflow_score_demo.py.
This local repo currently has scripts/summary_memory_case_demo.py, so the import
falls back to that name for compatibility checks.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Optional

import summary_memory_e2e_preprocess as e2e_preprocess
import summary_memory_enhancement_analysis as enhancement_analysis


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web"
WORKFLOW_MODULE_CANDIDATES = ("summary_memory_workflow_score_demo", "summary_memory_case_demo")
ALLOWED_SUFFIXES = {".txt", ".md"}
ALLOWED_HISTORY_SUFFIXES = {".txt", ".md", ".zip"}
LOCAL_NO_PROXY_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
HISTORY_TEXT_SPLIT_PATTERN = re.compile(r"(?m)^\s*(?:-{3,}|={3,})\s*$")
RESULT_JSON_ITEM_LIMIT = 20
RESULT_JSON_CHAR_LIMIT = 12000
RESULT_TEXT_CHAR_LIMIT = 12000
RESULT_LOG_CHAR_LIMIT = 12000
BACKGROUND_JOBS: dict[str, dict[str, Any]] = {}
BACKGROUND_JOBS_LOCK = threading.Lock()
APP_CSS = """
:root {
  --smw-bg: #f7f9fc;
  --smw-ink: #111827;
  --smw-muted: #6b7280;
  --smw-line: #e5e7eb;
  --smw-accent: #2563eb;
  --smw-accent-dark: #1d4ed8;
  --smw-success: #10b981;
  --smw-warning: #f59e0b;
  --smw-panel: #ffffff;
  --smw-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
  background: var(--smw-bg) !important;
  color: var(--smw-ink) !important;
  padding: 18px 20px 28px !important;
}
.smw-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.smw-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}
.smw-logo {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border: 2px solid #bfdbfe;
  border-radius: 14px;
  color: var(--smw-accent);
  background: #eff6ff;
  font-size: 28px;
  font-weight: 800;
}
.smw-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.smw-title-row h1 {
  margin: 0;
  font-size: 25px;
  line-height: 1.2;
  font-weight: 800;
}
.smw-demo-badge {
  padding: 3px 9px;
  border-radius: 999px;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 700;
}
.smw-subtitle {
  margin: 5px 0 0;
  color: var(--smw-muted);
  font-size: 14px;
}
.smw-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.smw-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--smw-line);
  border-radius: 8px;
  background: #fff;
  color: #1f2937;
  font-size: 13px;
  font-weight: 700;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.smw-panel {
  border: 1px solid var(--smw-line);
  border-radius: 14px;
  background: var(--smw-panel);
  box-shadow: var(--smw-shadow);
  padding: 18px;
}
.smw-sidebar-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 16px;
  font-size: 17px;
  font-weight: 800;
}
.smw-side-divider {
  height: 1px;
  background: var(--smw-line);
  margin: 16px 0;
}
.smw-status-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}
.smw-status-line {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: #475467;
  font-size: 13px;
}
.smw-status-line strong {
  color: #059669;
}
.smw-stepper {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
  align-items: center;
  gap: 14px;
  padding: 18px 22px;
  border: 1px solid var(--smw-line);
  border-radius: 14px;
  background: #fff;
  box-shadow: var(--smw-shadow);
  margin-bottom: 14px;
}
.smw-step {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}
.smw-step-num {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  border: 1px solid #cbd5e1;
  color: #64748b;
  font-weight: 800;
  background: #fff;
}
.smw-step.active .smw-step-num {
  border-color: var(--smw-accent);
  color: #fff;
  background: var(--smw-accent);
}
.smw-step-title {
  color: #111827;
  font-size: 14px;
  font-weight: 800;
}
.smw-step-desc {
  margin-top: 2px;
  color: var(--smw-muted);
  font-size: 12px;
}
.smw-arrow {
  color: #94a3b8;
  font-size: 24px;
}
.smw-card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
  font-size: 18px;
  font-weight: 800;
}
.smw-card-subtitle {
  margin: 0 0 12px;
  color: var(--smw-muted);
  font-size: 13px;
}
.smw-upload-note {
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  padding: 14px;
  color: var(--smw-muted);
  background: #f8fafc;
  text-align: center;
  font-size: 13px;
}
.smw-stat-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(140px, 1fr));
  gap: 14px;
  margin-bottom: 14px;
}
.smw-stat-card {
  border: 1px solid var(--smw-line);
  border-radius: 12px;
  background: #fff;
  padding: 15px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.smw-stat-card.memory { border-color: #ddd6fe; background: linear-gradient(180deg, #fff, #faf5ff); }
.smw-stat-card.search { border-color: #99f6e4; background: linear-gradient(180deg, #fff, #f0fdfa); }
.smw-stat-card.structured { border-color: #bfdbfe; background: linear-gradient(180deg, #fff, #eff6ff); }
.smw-stat-card.natural { border-color: #fed7aa; background: linear-gradient(180deg, #fff, #fff7ed); }
.smw-stat-card.score { border-color: #a7f3d0; background: linear-gradient(180deg, #fff, #ecfdf5); }
.smw-stat-head {
  color: #374151;
  font-size: 13px;
  font-weight: 800;
}
.smw-stat-value {
  margin-top: 10px;
  color: #111827;
  font-size: 28px;
  line-height: 1;
  font-weight: 900;
}
.smw-stat-value small {
  color: var(--smw-muted);
  font-size: 13px;
  font-weight: 600;
}
.smw-stat-foot {
  margin-top: 8px;
  color: var(--smw-muted);
  font-size: 13px;
}
.smw-report-bar {
  border: 1px solid var(--smw-line);
  border-radius: 12px;
  background: #fff;
  padding: 14px 16px;
  margin-bottom: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.smw-report-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  font-weight: 800;
}
.smw-report-meta {
  color: var(--smw-muted);
  font-size: 13px;
}
.smw-section-title {
  margin: 16px 0 8px;
  color: #344054;
  font-weight: 700;
}
.smw-compact textarea {
  min-height: 210px !important;
}
.smw-short textarea {
  min-height: 78px !important;
}
.smw-results textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 13px !important;
}
.smw-hint {
  color: var(--smw-muted);
  font-size: 13px;
  line-height: 1.45;
  margin: 4px 0 10px;
}
button.primary {
  min-height: 44px !important;
  font-weight: 700 !important;
  border-radius: 9px !important;
}
.smw-blue button {
  background: var(--smw-accent) !important;
  border-color: var(--smw-accent) !important;
}
.smw-green button {
  background: var(--smw-success) !important;
  border-color: var(--smw-success) !important;
}
@media (max-width: 1100px) {
  .smw-stepper {
    grid-template-columns: 1fr;
  }
  .smw-arrow {
    display: none;
  }
  .smw-stat-grid {
    grid-template-columns: repeat(2, minmax(140px, 1fr));
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a LAN Gradio UI for the Summary Memory workflow demo.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for Gradio. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for per-run output folders.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link. Default is LAN/proxy only.",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Enable Gradio queue. Default is disabled because some notebook/proxy paths block queue requests.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed Gradio errors in the terminal.",
    )
    parser.add_argument("--auth-user", default="", help="Optional Gradio basic-auth username for shared demos.")
    parser.add_argument("--auth-password", default="", help="Optional Gradio basic-auth password for shared demos.")
    parser.add_argument(
        "--return-zip",
        action="store_true",
        help="Return a zip file through Gradio. Disabled by default for notebook/proxy stability.",
    )
    parser.add_argument(
        "--root-path",
        default=os.getenv("GRADIO_ROOT_PATH", ""),
        help="URL path prefix for reverse proxies, for example /proxy/7860.",
    )
    return parser.parse_args()


def ensure_local_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        parts = [part.strip() for part in existing.split(",") if part.strip()]
        lower_parts = {part.lower() for part in parts}
        for host in LOCAL_NO_PROXY_HOSTS:
            if host.lower() not in lower_parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def normalize_root_path(root_path: str) -> str:
    value = (root_path or "").strip()
    if not value:
        return ""
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/")


def launch_demo(demo: Any, args: argparse.Namespace) -> None:
    root_path = normalize_root_path(args.root_path)
    app = demo
    if args.queue:
        try:
            app = demo.queue(default_concurrency_limit=1)
        except TypeError:
            app = demo.queue(concurrency_count=1)

    launch_kwargs: dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
    }
    launch_signature = inspect.signature(app.launch)
    if root_path:
        os.environ["GRADIO_ROOT_PATH"] = root_path
        if "root_path" in launch_signature.parameters:
            launch_kwargs["root_path"] = root_path
        elif "app_kwargs" in launch_signature.parameters:
            launch_kwargs["app_kwargs"] = {"root_path": root_path}
    if "show_api" in launch_signature.parameters:
        launch_kwargs["show_api"] = True
    if "api_open" in launch_signature.parameters:
        launch_kwargs["api_open"] = True
    if args.auth_user and args.auth_password and "auth" in launch_signature.parameters:
        launch_kwargs["auth"] = (args.auth_user, args.auth_password)
    if "debug" in launch_signature.parameters:
        launch_kwargs["debug"] = args.debug
    app.launch(**launch_kwargs)


def import_workflow_module() -> Any:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    missing: list[str] = []
    for module_name in WORKFLOW_MODULE_CANDIDATES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
            continue
        return importlib.import_module(module_name)

    missing_text = ", ".join(missing)
    raise RuntimeError(f"Cannot find workflow module. Tried: {missing_text}")


def require_gradio() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install it on the target machine with:\n"
            "python -m pip install -r scripts/requirements-summary-web.txt"
        ) from exc
    return gr


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "目标机器局域网IP"


def get_default_input(workflow: Any) -> Any:
    try:
        return workflow.build_builtin_demo_input()
    except Exception:
        return workflow.SummaryDemoInput(
            user_id="team_research_ops",
            project="paper_summary_assistant",
            source_type="project_weekly_summary",
            source_time="2026-04-03",
            history_summary="",
            current_content="",
            source_path=None,
            source_file=None,
        )


def normalize_files(files: Any, allowed_suffixes: set[str] = ALLOWED_SUFFIXES) -> list[Path]:
    if not files:
        return []
    if isinstance(files, (str, Path, dict)) or hasattr(files, "name") or hasattr(files, "path"):
        raw_files = [files]
    else:
        try:
            raw_files = list(files)
        except TypeError:
            raw_files = [files]

    paths: list[Path] = []
    for item in raw_files:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            path = Path(item)
        else:
            path_value = getattr(item, "name", None) or getattr(item, "path", None)
            if not path_value and isinstance(item, dict):
                path_value = item.get("name") or item.get("path")
            if not path_value:
                raise ValueError(f"Cannot read uploaded file path from: {item!r}")
            path = Path(path_value)

        validate_file_suffix(path, allowed_suffixes)
        paths.append(path)
    return paths


def normalize_path_text(path_text: str, allowed_suffixes: set[str] = ALLOWED_SUFFIXES) -> list[Path]:
    paths: list[Path] = []
    for raw_line in (path_text or "").splitlines():
        raw_path = raw_line.strip().strip('"').strip("'")
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        validate_file_suffix(path, allowed_suffixes)
        if not path.exists():
            raise ValueError(f"文件不存在：{path}")
        paths.append(path)
    return paths


def validate_file_suffix(path: Path, allowed_suffixes: set[str]) -> None:
    if path.suffix.lower() not in allowed_suffixes:
        suffix_text = " / ".join(sorted(allowed_suffixes))
        raise ValueError(f"Only {suffix_text} files are supported: {path.name}")


def validate_text_file(path: Path) -> None:
    validate_file_suffix(path, ALLOWED_SUFFIXES)


def read_text_path(path: Path) -> str:
    validate_text_file(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")


def decode_text_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8-sig", errors="replace")


def split_history_text(history_text: str) -> list[str]:
    clean_text = (history_text or "").strip()
    if not clean_text:
        return []
    parts = [part.strip() for part in HISTORY_TEXT_SPLIT_PATTERN.split(clean_text)]
    return [part for part in parts if part]


def iter_history_sources(paths: list[Path]) -> list[tuple[str, str, Path]]:
    # ZIP 条目只作为文本输入和展示标签处理，不能把内部名称当成本机写入路径。
    sources: list[tuple[str, str, Path]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = sorted(
                    name
                    for name in archive.namelist()
                    if not name.endswith("/") and Path(name).suffix.lower() in ALLOWED_SUFFIXES
                )
                if not names:
                    raise ValueError(f"历史材料压缩包中没有 .txt 或 .md 文件：{path}")
                for name in names:
                    content = decode_text_bytes(archive.read(name)).strip()
                    if content:
                        sources.append((Path(name).name, content, path))
            continue

        sources.append((path.name, read_text_path(path).strip(), path))
    return sources


def require_text_or_files(
    text: str,
    files: Any,
    path_text: str,
    label: str,
    allowed_suffixes: set[str] = ALLOWED_SUFFIXES,
) -> tuple[str, list[Path]]:
    clean_text = (text or "").strip()
    uploaded_paths = normalize_files(files, allowed_suffixes)
    typed_paths = normalize_path_text(path_text, allowed_suffixes)
    input_count = sum(bool(item) for item in (clean_text, uploaded_paths, typed_paths))
    if input_count > 1:
        raise ValueError(f"{label}不能同时使用粘贴文本和上传文件，请保留一种输入。")
    paths = typed_paths or uploaded_paths
    if not clean_text and not paths:
        suffix_text = " / ".join(sorted(allowed_suffixes))
        raise ValueError(f"请提供{label}文本，或上传 {suffix_text} 文件。")
    return clean_text, paths


def parse_rewrite_from_critic(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_score_summary(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "开启", "启用"}


def resolve_source_type(source_type: str, custom_source_type: str = "") -> str:
    custom_value = (custom_source_type or "").strip()
    if custom_value:
        return custom_value
    value = (source_type or "").strip()
    if value and value != "custom":
        return value
    return "project_history_summary"


def build_history_inputs(
    workflow: Any,
    history_text: str,
    history_files: Any,
    history_path_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
) -> list[Any]:
    clean_text, paths = require_text_or_files(
        history_text, history_files, history_path_text, "历史材料", ALLOWED_HISTORY_SUFFIXES
    )
    if clean_text:
        history_parts = split_history_text(clean_text)
        if not history_parts:
            raise ValueError("历史材料文本为空，请粘贴实际内容。")
        return [
            workflow.SummaryDemoInput(
                user_id=user_id,
                project=project,
                source_type=source_type,
                source_time=source_time,
                history_summary=part,
                current_content="",
                source_path=None,
                source_file=None,
                history_source_id=f"pasted_history_{index}",
            )
            for index, part in enumerate(history_parts, start=1)
        ]

    return [
        workflow.SummaryDemoInput(
            user_id=user_id,
            project=project,
            source_type=source_type,
            source_time=source_time,
            history_summary=content,
            current_content="",
            source_path=source_path,
            source_file=source_path,
            history_source_id=str(source_path),
        )
        for _label, content, source_path in iter_history_sources(paths)
    ]


def build_current_input(
    workflow: Any,
    current_text: str,
    current_file: Any,
    current_path_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
) -> Any:
    clean_text, paths = require_text_or_files(current_text, current_file, current_path_text, "当前材料")
    if len(paths) > 1:
        raise ValueError("当前材料只支持上传一个 .txt 或 .md 文件。")

    source_file: Optional[Path] = None
    if paths:
        source_file = paths[0]
        clean_text = read_text_path(source_file).strip()

    return workflow.SummaryDemoInput(
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
        history_summary="",
        current_content=clean_text,
        source_path=source_file,
        source_file=source_file,
    )


def build_raw_history_sources(
    history_text: str,
    history_files: Any,
    history_path_text: str,
) -> list[e2e_preprocess.RawTextSource]:
    clean_text, paths = require_text_or_files(
        history_text, history_files, history_path_text, "历史原始资料", ALLOWED_HISTORY_SUFFIXES
    )
    if clean_text:
        history_parts = split_history_text(clean_text)
        if not history_parts:
            raise ValueError("历史原始资料文本为空，请粘贴实际内容。")
        return [
            e2e_preprocess.RawTextSource(label=f"pasted_history_{index}", text=part)
            for index, part in enumerate(history_parts, start=1)
        ]

    return [
        e2e_preprocess.RawTextSource(label=label, text=content, source_path=source_path)
        for label, content, source_path in iter_history_sources(paths)
    ]


def build_raw_current_source(
    current_text: str,
    current_file: Any,
    current_path_text: str,
) -> e2e_preprocess.RawTextSource:
    clean_text, paths = require_text_or_files(current_text, current_file, current_path_text, "当前原始资料")
    if len(paths) > 1:
        raise ValueError("当前原始资料只支持上传一个 .txt 或 .md 文件。")
    if paths:
        path = paths[0]
        return e2e_preprocess.RawTextSource(label=path.name, text=read_text_path(path).strip(), source_path=path)
    return e2e_preprocess.RawTextSource(label="pasted_current", text=clean_text)


def build_e2e_raw_history_ingest_inputs(
    workflow: Any,
    raw_history_sources: list[e2e_preprocess.RawTextSource],
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
) -> list[Any]:
    chunk_char_limit = e2e_preprocess.get_preprocess_config()["chunk_char_limit"]
    history_inputs: list[Any] = []
    for source in raw_history_sources:
        chunks = e2e_preprocess.split_text_into_chunks(source.text, chunk_char_limit)
        if not chunks:
            raise ValueError(f"历史原始资料为空：{source.label}")
        for chunk in chunks:
            history_inputs.append(
                workflow.SummaryDemoInput(
                    user_id=user_id,
                    project=project,
                    source_type=source_type,
                    source_time=source_time,
                    history_summary=chunk,
                    current_content="",
                    source_path=source.source_path,
                    source_file=source.source_path,
                    history_source_id=source.label,
                )
            )
    return history_inputs


def build_run_input(
    workflow: Any,
    history_text: str,
    history_files: Any,
    history_path_text: str,
    current_text: str,
    current_file: Any,
    current_path_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
) -> Any:
    clean_history, history_paths = require_text_or_files(
        history_text, history_files, history_path_text, "历史材料", ALLOWED_HISTORY_SUFFIXES
    )
    current_input = build_current_input(
        workflow=workflow,
        current_text=current_text,
        current_file=current_file,
        current_path_text=current_path_text,
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
    )

    source_file: Optional[Path] = None
    history_parts: list[str] = []
    if history_paths:
        source_file = history_paths[0] if len(history_paths) == 1 else None
        for label, content, _source_path in iter_history_sources(history_paths):
            history_parts.append(f"# {label}\n\n{content}")
    else:
        split_parts = split_history_text(clean_history)
        if len(split_parts) > 1:
            history_parts = [f"# pasted_history_{index}\n\n{part}" for index, part in enumerate(split_parts, start=1)]
        else:
            history_parts = split_parts

    clean_history = "\n\n---\n\n".join(history_parts)
    if not clean_history.strip():
        raise ValueError("历史材料文本为空，请提供实际内容。")

    return workflow.SummaryDemoInput(
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
        history_summary=clean_history.strip(),
        current_content=current_input.current_content,
        source_path=source_file,
        source_file=source_file,
    )


def capture_run(fn: Any, *args: Any, **kwargs: Any) -> tuple[bool, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            fn(*args, **kwargs)
    except Exception:
        log = stdout.getvalue()
        err = stderr.getvalue()
        tb = traceback.format_exc()
        return False, "\n".join(part for part in (log, err, tb) if part)

    log = stdout.getvalue()
    err = stderr.getvalue()
    return True, "\n".join(part for part in (log, err) if part)


def run_enhancement_analysis(
    workflow: Any,
    output_run_dir: Optional[Path],
    baseline_summary: str,
    source_time: str,
    summary_style: str,
    baseline_source: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    if not output_run_dir:
        return ""
    try:
        info = enhancement_analysis.save_enhancement_analysis_outputs(
            workflow=workflow,
            output_run_dir=output_run_dir,
            baseline_summary=baseline_summary,
            source_time=source_time,
            summary_style=summary_style,
            baseline_source=baseline_source,
            metadata=metadata,
        )
    except Exception:
        return "[增强对比与时间分析失败]\n" + traceback.format_exc()

    styles = ", ".join(info.get("generated_styles") or [])
    return f"[增强对比与时间分析完成] styles={styles or 'none'}"


def capture_full_chain(
    workflow: Any,
    history_text: str,
    history_files: Any,
    history_path_text: str,
    current_text: str,
    current_file: Any,
    current_path_text: str,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> tuple[bool, str]:
    history_inputs = build_history_inputs(
        workflow,
        history_text,
        history_files,
        history_path_text,
        user_id,
        project,
        source_type,
        source_time,
    )
    current_input = build_current_input(
        workflow,
        current_text,
        current_file,
        current_path_text,
        user_id,
        project,
        source_type,
        source_time,
    )

    ingest_ok, ingest_log = capture_run(workflow.run_ingest, history_inputs, output_run_dir=output_run_dir)
    if not ingest_ok:
        return False, "\n\n".join(["[完整链路 run / ingest 阶段失败]", ingest_log])

    summarize_ok, summarize_log = capture_run(
        workflow.run_summarize,
        current_input,
        top_k=int(top_k),
        summary_style=summary_style,
        score_summary=score_summary,
        score_type=score_type,
        rewrite_from_critic=rewrite_from_critic,
        output_run_dir=output_run_dir,
    )
    log_parts = [
        "[完整链路 run / ingest 阶段]",
        ingest_log,
        "[完整链路 run / summarize 阶段]",
        summarize_log,
    ]
    if summarize_ok:
        analysis_log = run_enhancement_analysis(
            workflow=workflow,
            output_run_dir=output_run_dir,
            baseline_summary=current_input.current_content,
            source_time=source_time,
            summary_style=summary_style,
            baseline_source="run_current_input",
            metadata={"command": "run"},
        )
        if analysis_log:
            log_parts.extend(["[完整链路 run / 增强对比与时间分析]", analysis_log])
    if output_run_dir:
        web_run_info = {
            "command": "run",
            "web_run_mode": "ingest_then_summarize",
            "history_input_count": len(history_inputs),
            "top_k": int(top_k),
            "summary_style": summary_style,
            "score_summary": score_summary,
            "score_type": score_type,
        }
        (output_run_dir / "web_run_info.json").write_text(
            json.dumps(web_run_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summarize_ok, "\n\n".join(part for part in log_parts if part)


def capture_full_chain_prepared(
    workflow: Any,
    history_inputs: list[Any],
    current_input: Any,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> tuple[bool, str]:
    ingest_ok, ingest_log = capture_run(workflow.run_ingest, history_inputs, output_run_dir=output_run_dir)
    if not ingest_ok:
        return False, "\n\n".join(["[完整链路 run / ingest 阶段失败]", ingest_log])

    summarize_ok, summarize_log = capture_run(
        workflow.run_summarize,
        current_input,
        top_k=int(top_k),
        summary_style=summary_style,
        score_summary=score_summary,
        score_type=score_type,
        rewrite_from_critic=rewrite_from_critic,
        output_run_dir=output_run_dir,
    )
    log_parts = [
        "[完整链路 run / ingest 阶段]",
        ingest_log,
        "[完整链路 run / summarize 阶段]",
        summarize_log,
    ]
    if summarize_ok:
        analysis_log = run_enhancement_analysis(
            workflow=workflow,
            output_run_dir=output_run_dir,
            baseline_summary=current_input.current_content,
            source_time=getattr(current_input, "source_time", ""),
            summary_style=summary_style,
            baseline_source="run_current_input",
            metadata={"command": "run"},
        )
        if analysis_log:
            log_parts.extend(["[完整链路 run / 增强对比与时间分析]", analysis_log])
    if output_run_dir:
        web_run_info = {
            "command": "run",
            "web_run_mode": "background_ingest_then_summarize",
            "history_input_count": len(history_inputs),
            "top_k": int(top_k),
            "summary_style": summary_style,
            "score_summary": score_summary,
            "score_type": score_type,
        }
        (output_run_dir / "web_run_info.json").write_text(
            json.dumps(web_run_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summarize_ok, "\n\n".join(part for part in log_parts if part)


def capture_e2e_full_chain_prepared(
    workflow: Any,
    raw_history_sources: list[e2e_preprocess.RawTextSource],
    raw_current_source: e2e_preprocess.RawTextSource,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> tuple[bool, str]:
    log_parts: list[str] = ["[端到端 E2E / 原始资料基础摘要阶段]"]
    history_summaries = e2e_preprocess.summarize_raw_history_sources(workflow, raw_history_sources)
    current_summary = e2e_preprocess.summarize_raw_current_source(
        workflow,
        raw_current_source,
        source_time=source_time,
    )
    e2e_preprocess.save_e2e_preprocess_outputs(
        output_run_dir,
        history_summaries,
        current_summary,
        metadata={
            "user_id": user_id,
            "project": project,
            "source_type": source_type,
            "source_time": source_time,
            "top_k": int(top_k),
            "summary_style": summary_style,
            "score_summary": score_summary,
            "score_type": score_type,
        },
    )
    log_parts.append(f"历史原始资料基础摘要生成完成：{len(history_summaries)} 份（仅用于展示与对比）。")
    log_parts.append("当前原始资料基础摘要生成完成。")

    history_inputs = build_e2e_raw_history_ingest_inputs(
        workflow,
        raw_history_sources,
        user_id,
        project,
        source_type,
        source_time,
    )
    current_input = workflow.SummaryDemoInput(
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
        history_summary="",
        current_content=raw_current_source.text,
        source_path=raw_current_source.source_path,
        source_file=raw_current_source.source_path,
    )

    log_parts.append(f"Typed Memory 将直接从历史原始资料分块抽取：{len(history_inputs)} 块。")
    ingest_ok, ingest_log = capture_run(workflow.run_ingest, history_inputs, output_run_dir=output_run_dir)
    if not ingest_ok:
        return False, "\n\n".join(log_parts + ["[端到端 E2E / ingest 阶段失败]", ingest_log])

    summarize_ok, summarize_log = capture_run(
        workflow.run_summarize,
        current_input,
        top_k=int(top_k),
        summary_style=summary_style,
        score_summary=score_summary,
        score_type=score_type,
        rewrite_from_critic=rewrite_from_critic,
        output_run_dir=output_run_dir,
    )
    log_parts.extend(
        [
            "[端到端 E2E / ingest 阶段]",
            ingest_log,
            "[端到端 E2E / summarize 阶段]",
            summarize_log,
        ]
    )
    if summarize_ok:
        analysis_log = run_enhancement_analysis(
            workflow=workflow,
            output_run_dir=output_run_dir,
            baseline_summary=current_summary.summary_text,
            source_time=source_time,
            summary_style=summary_style,
            baseline_source="e2e_generated_current_summary",
            metadata={"command": "e2e_run", "current_source_label": current_summary.label},
        )
        if analysis_log:
            log_parts.extend(["[端到端 E2E / 增强对比与时间分析]", analysis_log])
    web_run_info = {
        "command": "e2e_run",
        "web_run_mode": "background_e2e_preprocess_ingest_then_summarize",
        "history_source_count": len(raw_history_sources),
        "history_input_count": len(history_inputs),
        "history_ingest_source": "raw_history_chunks",
        "top_k": int(top_k),
        "summary_style": summary_style,
        "score_summary": score_summary,
        "score_type": score_type,
    }
    (output_run_dir / "web_run_info.json").write_text(
        json.dumps(web_run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summarize_ok, "\n\n".join(part for part in log_parts if part)


def capture_e2e_ingest_prepared(
    workflow: Any,
    raw_history_sources: list[e2e_preprocess.RawTextSource],
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    output_run_dir: Path,
) -> tuple[bool, str]:
    log_parts: list[str] = ["[端到端 E2E Ingest / 历史原始资料基础摘要阶段]"]
    history_summaries = e2e_preprocess.summarize_raw_history_sources(workflow, raw_history_sources)
    e2e_preprocess.save_e2e_history_preprocess_outputs(
        output_run_dir,
        history_summaries,
        metadata={
            "user_id": user_id,
            "project": project,
            "source_type": source_type,
            "source_time": source_time,
        },
    )
    log_parts.append(f"历史原始资料基础摘要生成完成：{len(history_summaries)} 份（仅用于展示与对比）。")

    history_inputs = build_e2e_raw_history_ingest_inputs(
        workflow,
        raw_history_sources,
        user_id,
        project,
        source_type,
        source_time,
    )
    log_parts.append(f"Typed Memory 将直接从历史原始资料分块抽取：{len(history_inputs)} 块。")
    ingest_ok, ingest_log = capture_run(workflow.run_ingest, history_inputs, output_run_dir=output_run_dir)
    log_parts.extend(["[端到端 E2E Ingest / ingest 阶段]", ingest_log])
    web_run_info = {
        "command": "e2e_ingest",
        "web_run_mode": "background_e2e_preprocess_then_ingest",
        "history_source_count": len(raw_history_sources),
        "history_input_count": len(history_inputs),
        "history_ingest_source": "raw_history_chunks",
    }
    (output_run_dir / "web_run_info.json").write_text(
        json.dumps(web_run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ingest_ok, "\n\n".join(part for part in log_parts if part)


def capture_e2e_summarize_prepared(
    workflow: Any,
    raw_current_source: e2e_preprocess.RawTextSource,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> tuple[bool, str]:
    log_parts: list[str] = ["[端到端 E2E Summarize / 当前原始资料基础摘要阶段]"]
    current_summary = e2e_preprocess.summarize_raw_current_source(
        workflow,
        raw_current_source,
        source_time=source_time,
    )
    e2e_preprocess.save_e2e_current_preprocess_outputs(
        output_run_dir,
        current_summary,
        metadata={
            "user_id": user_id,
            "project": project,
            "source_type": source_type,
            "source_time": source_time,
            "top_k": int(top_k),
            "summary_style": summary_style,
            "score_summary": score_summary,
            "score_type": score_type,
        },
    )
    log_parts.append("当前原始资料基础摘要生成完成。")

    current_input = workflow.SummaryDemoInput(
        user_id=user_id,
        project=project,
        source_type=source_type,
        source_time=source_time,
        history_summary="",
        current_content=raw_current_source.text,
        source_path=raw_current_source.source_path,
        source_file=raw_current_source.source_path,
    )
    summarize_ok, summarize_log = capture_run(
        workflow.run_summarize,
        current_input,
        top_k=int(top_k),
        summary_style=summary_style,
        score_summary=score_summary,
        score_type=score_type,
        rewrite_from_critic=rewrite_from_critic,
        output_run_dir=output_run_dir,
    )
    log_parts.extend(["[端到端 E2E Summarize / summarize 阶段]", summarize_log])
    if summarize_ok:
        analysis_log = run_enhancement_analysis(
            workflow=workflow,
            output_run_dir=output_run_dir,
            baseline_summary=current_summary.summary_text,
            source_time=source_time,
            summary_style=summary_style,
            baseline_source="e2e_generated_current_summary",
            metadata={"command": "e2e_summarize", "current_source_label": current_summary.label},
        )
        if analysis_log:
            log_parts.extend(["[端到端 E2E Summarize / 增强对比与时间分析]", analysis_log])
    web_run_info = {
        "command": "e2e_summarize",
        "web_run_mode": "background_e2e_preprocess_then_summarize",
        "top_k": int(top_k),
        "summary_style": summary_style,
        "score_summary": score_summary,
        "score_type": score_type,
    }
    (output_run_dir / "web_run_info.json").write_text(
        json.dumps(web_run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summarize_ok, "\n\n".join(part for part in log_parts if part)


def set_background_job(job_key: str, **updates: Any) -> None:
    with BACKGROUND_JOBS_LOCK:
        job = BACKGROUND_JOBS.setdefault(job_key, {})
        job.update(updates)


def get_background_job(job_key: str) -> dict[str, Any]:
    with BACKGROUND_JOBS_LOCK:
        return dict(BACKGROUND_JOBS.get(job_key, {}))


def start_full_chain_background(
    workflow: Any,
    history_inputs: list[Any],
    current_input: Any,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> None:
    job_key = str(output_run_dir)
    set_background_job(
        job_key,
        command="run",
        status="running",
        ok=None,
        log="后台完整链路任务已启动。",
        started_at=time.time(),
        elapsed=None,
    )

    def worker() -> None:
        start_time = time.perf_counter()
        try:
            ok, log = capture_full_chain_prepared(
                workflow,
                history_inputs,
                current_input,
                top_k,
                summary_style,
                score_summary,
                score_type,
                rewrite_from_critic,
                output_run_dir,
            )
        except Exception:
            ok = False
            log = traceback.format_exc()
        set_background_job(
            job_key,
            status="finished",
            ok=ok,
            log=log,
            elapsed=time.perf_counter() - start_time,
        )

    threading.Thread(target=worker, daemon=True).start()


def start_e2e_full_chain_background(
    workflow: Any,
    raw_history_sources: list[e2e_preprocess.RawTextSource],
    raw_current_source: e2e_preprocess.RawTextSource,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> None:
    job_key = str(output_run_dir)
    set_background_job(
        job_key,
        command="e2e_run",
        status="running",
        ok=None,
        log="后台端到端 E2E 任务已启动，正在从原始资料生成基础摘要。",
        started_at=time.time(),
        elapsed=None,
    )

    def worker() -> None:
        start_time = time.perf_counter()
        try:
            ok, log = capture_e2e_full_chain_prepared(
                workflow,
                raw_history_sources,
                raw_current_source,
                user_id,
                project,
                source_type,
                source_time,
                top_k,
                summary_style,
                score_summary,
                score_type,
                rewrite_from_critic,
                output_run_dir,
            )
        except Exception:
            ok = False
            log = traceback.format_exc()
        set_background_job(
            job_key,
            status="finished",
            ok=ok,
            log=log,
            elapsed=time.perf_counter() - start_time,
        )

    threading.Thread(target=worker, daemon=True).start()


def start_e2e_ingest_background(
    workflow: Any,
    raw_history_sources: list[e2e_preprocess.RawTextSource],
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    output_run_dir: Path,
) -> None:
    job_key = str(output_run_dir)
    set_background_job(
        job_key,
        command="e2e_ingest",
        status="running",
        ok=None,
        log="后台 E2E Ingest 任务已启动，正在从历史原始资料生成基础摘要。",
        started_at=time.time(),
        elapsed=None,
    )

    def worker() -> None:
        start_time = time.perf_counter()
        try:
            ok, log = capture_e2e_ingest_prepared(
                workflow,
                raw_history_sources,
                user_id,
                project,
                source_type,
                source_time,
                output_run_dir,
            )
        except Exception:
            ok = False
            log = traceback.format_exc()
        set_background_job(
            job_key,
            status="finished",
            ok=ok,
            log=log,
            elapsed=time.perf_counter() - start_time,
        )

    threading.Thread(target=worker, daemon=True).start()


def start_e2e_summarize_background(
    workflow: Any,
    raw_current_source: e2e_preprocess.RawTextSource,
    user_id: str,
    project: str,
    source_type: str,
    source_time: str,
    top_k: int,
    summary_style: str,
    score_summary: bool,
    score_type: str,
    rewrite_from_critic,
    output_run_dir: Path,
) -> None:
    job_key = str(output_run_dir)
    set_background_job(
        job_key,
        command="e2e_summarize",
        status="running",
        ok=None,
        log="后台 E2E Summarize 任务已启动，正在从当前原始资料生成基础摘要。",
        started_at=time.time(),
        elapsed=None,
    )

    def worker() -> None:
        start_time = time.perf_counter()
        try:
            ok, log = capture_e2e_summarize_prepared(
                workflow,
                raw_current_source,
                user_id,
                project,
                source_type,
                source_time,
                top_k,
                summary_style,
                score_summary,
                score_type,
                rewrite_from_critic,
                output_run_dir,
            )
        except Exception:
            ok = False
            log = traceback.format_exc()
        set_background_job(
            job_key,
            status="finished",
            ok=ok,
            log=log,
            elapsed=time.perf_counter() - start_time,
        )

    threading.Thread(target=worker, daemon=True).start()


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def truncate_preview(text: str, max_chars: int, path: Optional[Path] = None) -> str:
    if len(text) <= max_chars:
        return text
    location = f"\n\n完整文件：{path}" if path else ""
    return text[:max_chars] + f"\n\n... 内容过长，页面仅展示前 {max_chars} 字符。{location}"


def read_text_preview_if_exists(path: Path, max_chars: int = RESULT_TEXT_CHAR_LIMIT) -> str:
    if not path.exists():
        return ""
    return truncate_preview(path.read_text(encoding="utf-8", errors="replace"), max_chars, path)


def read_json_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def read_json_preview_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= RESULT_JSON_CHAR_LIMIT:
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return json.dumps(
            {
                "preview_note": f"内容过长，页面仅展示前 {RESULT_JSON_CHAR_LIMIT} 字符。",
                "file": str(path),
                "preview_text": text[:RESULT_JSON_CHAR_LIMIT],
            },
            ensure_ascii=False,
            indent=2,
        )

    total_count: Optional[int] = None
    preview_value: Any = value
    if isinstance(value, list):
        total_count = len(value)
        preview_value = value[:RESULT_JSON_ITEM_LIMIT]
    elif isinstance(value, dict):
        preview_value = {}
        for key, item in value.items():
            if isinstance(item, list):
                preview_value[key] = item[:RESULT_JSON_ITEM_LIMIT]
                preview_value[f"{key}_total_count"] = len(item)
            else:
                preview_value[key] = item

    return json.dumps(
        {
            "preview_note": (
                f"内容过长，页面仅展示最多 {RESULT_JSON_ITEM_LIMIT} 条或前 {RESULT_JSON_CHAR_LIMIT} 字符。"
            ),
            "file": str(path),
            "total_count": total_count,
            "preview": preview_value,
        },
        ensure_ascii=False,
        indent=2,
    )


def load_json_value(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def count_memory_items(value: Any) -> int:
    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("typed_memories"), list):
                total += len(item["typed_memories"])
            elif isinstance(item, dict) and isinstance(item.get("memories"), list):
                total += len(item["memories"])
            else:
                total += 1
        return total
    if isinstance(value, dict):
        for key in ("typed_memories", "results", "memories", "retrieved_memories"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return 0


def find_score(value: Any, style: str) -> Optional[float]:
    if not isinstance(value, dict):
        return None
    candidates: list[Any] = []
    if style in value:
        candidates.append(value[style])
    candidates.append(value)
    for item in candidates:
        if isinstance(item, dict):
            for key in ("final_score", "score", "overall_score"):
                if key in item:
                    try:
                        return float(item[key])
                    except (TypeError, ValueError):
                        return None
    return None


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def build_stats_html(output_run_dir: Optional[Path]) -> str:
    extracted_count = typed_count = retrieved_count = retrieved_typed_count = 0
    structured_score = natural_score = overall_score = None
    if output_run_dir:
        extracted_count = count_memory_items(load_json_value(output_run_dir / "extracted_memories.json"))
        typed_count = count_memory_items(load_json_value(output_run_dir / "typed_memories.json"))
        retrieved_count = count_memory_items(load_json_value(output_run_dir / "retrieved_memories.json"))
        retrieved_typed_count = count_memory_items(load_json_value(output_run_dir / "retrieved_typed_memories.json"))
        score_data = load_json_value(output_run_dir / "summary_scores.json")
        structured_score = find_score(score_data, "structured")
        natural_score = find_score(score_data, "natural")
        scores = [score for score in (structured_score, natural_score) if score is not None]
        overall_score = sum(scores) / len(scores) if scores else find_score(score_data, "overall")

    memory_count = typed_count or extracted_count
    memory_label = "typed memories" if typed_count else "长期记忆"
    search_count = retrieved_typed_count or retrieved_count
    search_label = "typed retrieval" if retrieved_typed_count else "相关记忆"

    return f"""
    <div class="smw-stat-grid">
      <div class="smw-stat-card memory">
        <div class="smw-stat-head">抽取到的记忆</div>
        <div class="smw-stat-value">{memory_count or "--"} <small>条</small></div>
        <div class="smw-stat-foot">{memory_label}</div>
      </div>
      <div class="smw-stat-card search">
        <div class="smw-stat-head">检索到的记忆</div>
        <div class="smw-stat-value">{search_count or "--"} <small>条</small></div>
        <div class="smw-stat-foot">{search_label}</div>
      </div>
      <div class="smw-stat-card structured">
        <div class="smw-stat-head">结构化摘要</div>
        <div class="smw-stat-value">{format_score(structured_score)} <small>/100</small></div>
        <div class="smw-stat-foot">评分</div>
      </div>
      <div class="smw-stat-card natural">
        <div class="smw-stat-head">自然化摘要</div>
        <div class="smw-stat-value">{format_score(natural_score)} <small>/100</small></div>
        <div class="smw-stat-foot">评分</div>
      </div>
      <div class="smw-stat-card score">
        <div class="smw-stat-head">自动评分结果</div>
        <div class="smw-stat-value">{format_score(overall_score)} <small>/100</small></div>
        <div class="smw-stat-foot">综合评分</div>
      </div>
    </div>
    """


def build_sidebar_status_html(state: str = "空闲", output_dir: str = "--", mode: str = "--", elapsed: str = "--") -> str:
    return f"""
    <div class="smw-status-grid">
      <div class="smw-status-line"><span>当前状态</span><strong>{state}</strong></div>
      <div class="smw-status-line"><span>输出目录</span><span>{output_dir or "--"}</span></div>
      <div class="smw-status-line"><span>最近运行</span><span>{mode or "--"}</span></div>
      <div class="smw-status-line"><span>运行耗时</span><span>{elapsed or "--"}</span></div>
    </div>
    """


def build_report_summary_html(
    command: str = "--",
    output_dir: str = "--",
    message: str = "等待运行。完成后这里会显示本次运行概览。",
) -> str:
    return f"""
    <div class="smw-report-bar">
      <div class="smw-report-title">
        <span>本次运行概览（Report）</span>
        <span class="smw-report-meta">运行模式：{command or "--"}</span>
      </div>
      <div class="smw-report-meta">输出目录：{output_dir or "--"}</div>
      <div class="smw-report-meta" style="margin-top:8px;">{message}</div>
    </div>
    """


def create_zip(output_run_dir: Optional[Path]) -> Optional[str]:
    if not output_run_dir or not output_run_dir.exists():
        return None

    zip_path = output_run_dir.parent / f"{output_run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_run_dir))
    return str(zip_path)


def load_result_payload(
    ok: bool,
    command: str,
    output_run_dir: Optional[Path],
    log: str,
    elapsed_seconds: Optional[float] = None,
    include_zip: bool = False,
    state_label: Optional[str] = None,
) -> tuple[Any, ...]:
    output_dir_text = str(output_run_dir) if output_run_dir else ""
    status = state_label or ("完成" if ok else "失败")
    if output_dir_text:
        status = f"{status}：{command} 输出目录 {output_dir_text}"
    else:
        status = f"{status}：{command}"

    if not ok and state_label != "运行中":
        status += "\n请查看运行日志中的错误信息。"
    elif output_dir_text and state_label != "运行中":
        status += "\n页面仅回传轻量预览；完整结果请查看输出目录。"
    elif output_dir_text:
        status += "\n后台任务仍在运行，请稍后点击刷新结果。"

    elapsed_text = f"{elapsed_seconds:.1f}s" if elapsed_seconds is not None else "--"
    typed_memories = read_json_preview_if_exists(output_run_dir / "typed_memories.json") if output_run_dir else ""
    typed_memory_dedup_report = (
        read_json_preview_if_exists(output_run_dir / "typed_memory_dedup_report.json") if output_run_dir else ""
    )
    typed_extraction_raw_output = (
        read_text_preview_if_exists(output_run_dir / "typed_memory_extraction_raw_output.txt") if output_run_dir else ""
    )
    typed_retrieval_plan = (
        read_json_preview_if_exists(output_run_dir / "typed_retrieval_plan.json") if output_run_dir else ""
    )
    retrieved_typed_memories = (
        read_json_preview_if_exists(output_run_dir / "retrieved_typed_memories.json") if output_run_dir else ""
    )
    artifact_board = read_json_preview_if_exists(output_run_dir / "artifact_board.json") if output_run_dir else ""
    augmented_summary_prompt = (
        read_text_preview_if_exists(output_run_dir / "augmented_summary_prompt.txt") if output_run_dir else ""
    )
    extracted = read_json_preview_if_exists(output_run_dir / "extracted_memories.json") if output_run_dir else ""
    add_results = read_json_preview_if_exists(output_run_dir / "add_results.json") if output_run_dir else ""
    retrieved = read_json_preview_if_exists(output_run_dir / "retrieved_memories.json") if output_run_dir else ""
    baseline_summary = read_text_preview_if_exists(output_run_dir / "baseline_summary.md") if output_run_dir else ""
    structured = read_text_preview_if_exists(output_run_dir / "summary_structured.md") if output_run_dir else ""
    natural = read_text_preview_if_exists(output_run_dir / "summary_natural.md") if output_run_dir else ""
    enhancement_comparison = (
        read_text_preview_if_exists(output_run_dir / "enhancement_comparison.md") if output_run_dir else ""
    )
    temporal_enrichment = (
        read_text_preview_if_exists(output_run_dir / "temporal_enrichment.md") if output_run_dir else ""
    )
    scores = read_json_preview_if_exists(output_run_dir / "summary_scores.json") if output_run_dir else ""
    summary_critic_prompt = (
        read_text_preview_if_exists(output_run_dir / "summary_critic_prompt.txt") if output_run_dir else ""
    )
    summary_critic = read_text_preview_if_exists(output_run_dir / "summary_critic_report.md") if output_run_dir else ""
    summary_critic_json = (
        read_json_preview_if_exists(output_run_dir / "summary_critic_report.json") if output_run_dir else ""
    )
    summary_rewrite_prompt = (
        read_text_preview_if_exists(output_run_dir / "summary_rewrite_prompt.txt") if output_run_dir else ""
    )
    summary_rewritten_structured = (
        read_text_preview_if_exists(output_run_dir / "summary_rewritten_structured.md") if output_run_dir else ""
    )
    summary_rewritten_natural = (
        read_text_preview_if_exists(output_run_dir / "summary_rewritten_natural.md") if output_run_dir else ""
    )
    summary_rewrite_report = (
        read_json_preview_if_exists(output_run_dir / "summary_rewrite_report.json") if output_run_dir else ""
    )
    report = read_text_preview_if_exists(output_run_dir / "report.md") if output_run_dir else ""
    run_info = read_json_preview_if_exists(output_run_dir / "run_info.json") if output_run_dir else ""
    e2e_history_summaries = (
        read_text_preview_if_exists(output_run_dir / "generated_history_summaries.md") if output_run_dir else ""
    )
    e2e_current_summary = (
        read_text_preview_if_exists(output_run_dir / "generated_current_summary.md") if output_run_dir else ""
    )
    e2e_preprocess_info = (
        read_json_preview_if_exists(output_run_dir / "e2e_preprocess_info.json") if output_run_dir else ""
    )
    if output_run_dir and (output_run_dir / "web_run_info.json").exists():
        combined_run_info = {
            "web_run_info": load_json_value(output_run_dir / "web_run_info.json"),
            "workflow_run_info": load_json_value(output_run_dir / "run_info.json"),
        }
        run_info = json.dumps(combined_run_info, ensure_ascii=False, indent=2)
    zip_file = create_zip(output_run_dir) if include_zip else None
    log = truncate_preview(log, RESULT_LOG_CHAR_LIMIT)
    stats_html = build_stats_html(output_run_dir)
    sidebar_state = state_label or ("完成" if ok else "出错")
    sidebar_status = build_sidebar_status_html(sidebar_state, output_dir_text, command, elapsed_text)
    report_message = (
        "运行完成。页面仅展示轻量预览，完整 JSON、Markdown、日志和下载产物请查看输出目录。"
        if ok
        else "运行失败，请查看运行日志。"
    )
    if state_label == "运行中":
        report_message = "后台完整链路任务正在运行。完成后点击刷新结果读取 outputs/web 中的产物。"
    report_summary = build_report_summary_html(command, output_dir_text, report_message)
    return (
        status,
        output_dir_text,
        stats_html,
        report_summary,
        sidebar_status,
        extracted,
        typed_memories,
        typed_memory_dedup_report,
        typed_extraction_raw_output,
        add_results,
        typed_retrieval_plan,
        retrieved_typed_memories,
        artifact_board,
        augmented_summary_prompt,
        retrieved,
        baseline_summary,
        structured,
        natural,
        enhancement_comparison,
        temporal_enrichment,
        scores,
        summary_critic_prompt,
        summary_critic,
        summary_critic_json,
        summary_rewrite_prompt,
        summary_rewritten_structured,
        summary_rewritten_natural,
        summary_rewrite_report,
        report,
        run_info,
        e2e_history_summaries,
        e2e_current_summary,
        e2e_preprocess_info,
        log,
        zip_file,
    )


def load_output_dir_payload(output_dir_text: str, include_zip: bool = False):
    clean_path = (output_dir_text or "").strip()
    if not clean_path:
        return load_result_payload(False, "refresh", None, "请先运行任务，或填写一个输出目录。")

    output_run_dir = Path(clean_path).expanduser()
    if not output_run_dir.exists():
        return load_result_payload(False, "refresh", None, f"输出目录不存在：{output_run_dir}")

    job = get_background_job(str(output_run_dir))
    command = str(job.get("command") or "refresh")
    if job.get("status") == "running":
        return load_result_payload(
            True,
            command,
            output_run_dir,
            str(job.get("log") or "后台任务运行中。"),
            include_zip=include_zip,
            state_label="运行中",
        )

    if job:
        ok = bool(job.get("ok"))
        return load_result_payload(
            ok,
            command,
            output_run_dir,
            str(job.get("log") or ""),
            elapsed_seconds=job.get("elapsed"),
            include_zip=include_zip,
        )

    has_outputs = any(
        (output_run_dir / name).exists()
        for name in (
            "report.md",
            "summary_structured.md",
            "summary_natural.md",
            "summary_scores.json",
            "summary_critic_report.md",
            "summary_critic_report.json",
            "typed_memories.json",
            "typed_memory_dedup_report.json",
            "typed_memory_extraction_raw_output.txt",
            "typed_retrieval_plan.json",
            "retrieved_typed_memories.json",
            "artifact_board.json",
            "augmented_summary_prompt.txt",
            "summary_critic_prompt.txt",
            "summary_rewrite_prompt.txt",
            "summary_rewritten_structured.md",
            "summary_rewritten_natural.md",
            "summary_rewrite_report.json",
            "retrieved_memories.json",
            "baseline_summary.md",
            "generated_history_summaries.md",
            "generated_current_summary.md",
            "e2e_preprocess_info.json",
            "enhancement_comparison.md",
            "temporal_enrichment.md",
            "enhancement_analysis_info.json",
        )
    )
    return load_result_payload(
        has_outputs,
        "refresh",
        output_run_dir,
        "从输出目录加载已有产物。" if has_outputs else "该目录下还没有可展示的摘要产物。",
        include_zip=include_zip,
    )


def make_output_dir(workflow: Any, output_root: str, command: str, project: str) -> Path:
    root = Path((output_root or "").strip() or DEFAULT_OUTPUT_DIR)
    return workflow.create_output_run_dir(root, command, project)


def build_app(workflow: Any, args: argparse.Namespace) -> Any:
    gr = require_gradio()
    defaults = get_default_input(workflow)
    module_name = getattr(workflow, "__name__", "unknown")
    lan_ip = get_lan_ip()
    root_path = normalize_root_path(args.root_path) or "none"

    def run_ingest_click(
        user_id: str,
        project: str,
        source_type: str,
        source_time: str,
        output_root: str,
        history_text: str,
        history_files: Any,
        history_path_text: str,
    ) -> tuple[Any, ...]:
        start_time = time.perf_counter()
        try:
            demo_inputs = build_history_inputs(
                workflow, history_text, history_files, history_path_text, user_id, project, source_type, source_time
            )
            output_run_dir = make_output_dir(workflow, output_root, "ingest", project)
            ok, log = capture_run(workflow.run_ingest, demo_inputs, output_run_dir=output_run_dir)
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok, "ingest", output_run_dir, log, time.perf_counter() - start_time, include_zip=args.return_zip
        )

    def run_summarize_click(
        user_id: str,
        project: str,
        source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: bool,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        current_text: str,
        current_file: Any,
        current_path_text: str,
    ) -> tuple[Any, ...]:
        start_time = time.perf_counter()
        try:
            demo_input = build_current_input(
                workflow, current_text, current_file, current_path_text, user_id, project, source_type, source_time
            )
            output_run_dir = make_output_dir(workflow, output_root, "summarize", project)
            ok, log = capture_run(
                workflow.run_summarize,
                demo_input,
                top_k=int(top_k),
                summary_style=summary_style,
                score_summary=parse_score_summary(score_summary),
                score_type=score_type,
                rewrite_from_critic=parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir=output_run_dir,
            )
            if ok:
                analysis_log = run_enhancement_analysis(
                    workflow=workflow,
                    output_run_dir=output_run_dir,
                    baseline_summary=demo_input.current_content,
                    source_time=source_time,
                    summary_style=summary_style,
                    baseline_source="summarize_current_input",
                    metadata={"command": "summarize"},
                )
                if analysis_log:
                    log = "\n\n".join(part for part in (log, "[summarize / 增强对比与时间分析]", analysis_log) if part)
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok, "summarize", output_run_dir, log, time.perf_counter() - start_time, include_zip=args.return_zip
        )

    def run_full_click(
        user_id: str,
        project: str,
        source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: bool,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        history_text: str,
        history_files: Any,
        history_path_text: str,
        current_text: str,
        current_file: Any,
        current_path_text: str,
    ) -> tuple[Any, ...]:
        start_time = time.perf_counter()
        try:
            output_run_dir = make_output_dir(workflow, output_root, "run", project)
            ok, log = capture_full_chain(
                workflow,
                history_text,
                history_files,
                history_path_text,
                current_text,
                current_file,
                current_path_text,
                user_id,
                project,
                source_type,
                source_time,
                top_k=int(top_k),
                summary_style=summary_style,
                score_summary=parse_score_summary(score_summary),
                score_type=score_type,
                rewrite_from_critic=parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir=output_run_dir,
            )
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok, "run", output_run_dir, log, time.perf_counter() - start_time, include_zip=args.return_zip
        )

    share_text = "enabled" if args.share else "disabled"
    with gr.Blocks(title="Summary Memory Workflow Demo", css=APP_CSS) as demo:
        empty_path_text = gr.State(value="")
        gr.HTML(
            f"""
            <section class="smw-hero">
              <h1 class="smw-title">Summary Memory Workflow</h1>
              <p class="smw-subtitle">Mem0 + Qdrant + local Qwen summary-memory demo.</p>
              <div class="smw-pill-row">
                <span class="smw-pill"><strong>Module</strong>{module_name}</span>
                <span class="smw-pill"><strong>LAN</strong>http://{lan_ip}:{args.port}</span>
                <span class="smw-pill"><strong>Share</strong>{share_text}</span>
                <span class="smw-pill"><strong>Root</strong>{root_path}</span>
              </div>
            </section>
            """
        )

        with gr.Accordion("1. Project Config", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    user_id = gr.Textbox(label="user_id", value=defaults.user_id)
                with gr.Column(scale=2):
                    project = gr.Textbox(label="project", value=defaults.project)
                with gr.Column(scale=2):
                    source_type = gr.Textbox(label="source_type", value=defaults.source_type)
                with gr.Column(scale=2):
                    source_time = gr.Textbox(label="source_time", value=defaults.source_time)
            with gr.Row():
                with gr.Column(scale=1):
                    top_k = gr.Number(label="top_k", value=8, precision=0)
                with gr.Column(scale=3):
                    summary_style = gr.Dropdown(
                        label="summary_style",
                        choices=["structured", "natural", "both"],
                        value="structured",
                    )
                with gr.Column(scale=3):
                    score_type = gr.Dropdown(
                        label="score_type",
                        choices=["general", "meeting", "interview"],
                        value="general",
                    )
                with gr.Column(scale=1):
                    rewrite_from_critic = gr.Dropdown(
                        label="rewrite_from_critic",
                        choices=["false", "true"],
                        value="false",
                    )
                    score_summary = gr.Dropdown(
                        label="score_summary",
                        choices=["false", "true"],
                        value="false",
                    )
            output_root = gr.Textbox(label="output_dir", value=str(args.output_dir), elem_classes=["smw-short"])

        with gr.Tabs():
            with gr.Tab("2A. Ingest"):
                with gr.Row():
                    with gr.Column(scale=3):
                        ingest_history_text = gr.Textbox(
                            label="历史材料文本",
                            lines=12,
                            placeholder="粘贴历史摘要、会议纪要、项目周报或学习记录。多份材料可用单独一行 --- 分隔。",
                            elem_classes=["smw-compact"],
                        )
                    with gr.Column(scale=2):
                        gr.HTML(
                            '<div class="smw-hint">支持上传 .txt / .md；历史材料也可以上传一个 .zip。</div>'
                        )
                        ingest_history_files = gr.File(
                            label="历史材料文件（.txt / .md 可多选，或一个 .zip）",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".zip"],
                        )
                        ingest_button = gr.Button("导入历史记忆 / Ingest", variant="primary")

            with gr.Tab("2B. Summarize"):
                with gr.Row():
                    with gr.Column(scale=3):
                        summarize_current_text = gr.Textbox(
                            label="当前材料文本",
                            lines=12,
                            placeholder="粘贴当前需要增强摘要的材料。",
                            elem_classes=["smw-compact"],
                        )
                    with gr.Column(scale=2):
                        gr.HTML(
                            '<div class="smw-hint">当前材料只支持单个 .txt / .md 文件，或直接粘贴文本。</div>'
                        )
                        summarize_current_file = gr.File(
                            label="当前材料文件（.txt / .md，单文件）",
                            file_count="single",
                            file_types=[".txt", ".md"],
                        )
                        summarize_button = gr.Button("生成增强摘要 / Summarize", variant="primary")

            with gr.Tab("2C. Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        run_history_text = gr.Textbox(
                            label="历史材料文本",
                            lines=10,
                            placeholder="粘贴历史材料。多份材料可用单独一行 --- 分隔。",
                            elem_classes=["smw-compact"],
                        )
                        run_history_files = gr.File(
                            label="历史材料文件（.txt / .md 可多选，或一个 .zip）",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".zip"],
                        )
                    with gr.Column(scale=1):
                        run_current_text = gr.Textbox(
                            label="当前材料文本",
                            lines=10,
                            placeholder="粘贴当前材料。",
                            elem_classes=["smw-compact"],
                        )
                        run_current_file = gr.File(
                            label="当前材料文件（.txt / .md，单文件）",
                            file_count="single",
                            file_types=[".txt", ".md"],
                        )
                run_button = gr.Button("执行完整链路 / Run", variant="primary")

        gr.HTML('<div class="smw-section-title">3. Results</div>')
        stats_html = gr.HTML(build_stats_html(None), visible=False)
        report_summary = gr.HTML(build_report_summary_html(), visible=False)
        sidebar_status = gr.HTML(build_sidebar_status_html(), visible=False)
        with gr.Row():
            with gr.Column(scale=3):
                status = gr.Textbox(label="状态", lines=3, elem_classes=["smw-results"])
            with gr.Column(scale=4):
                output_dir = gr.Textbox(label="本次输出目录", lines=3, elem_classes=["smw-results"])
            with gr.Column(scale=2):
                download_zip = gr.File(label="下载产物（代理模式默认不自动回传）")
        with gr.Tabs(elem_classes=["smw-results"]):
            with gr.Tab("Extracted Memories"):
                extracted = gr.Code(label="extracted_memories.json", language="json")
            with gr.Tab("Typed Memories"):
                typed_memories = gr.Code(label="typed_memories.json", language="json")
            with gr.Tab("Typed Memory Dedup"):
                typed_memory_dedup_report = gr.Code(label="typed_memory_dedup_report.json", language="json")
            with gr.Tab("Typed Extraction Raw Output"):
                typed_extraction_raw_output = gr.Textbox(
                    label="typed_memory_extraction_raw_output.txt", lines=18
                )
            with gr.Tab("Add Results"):
                add_results = gr.Code(label="add_results.json", language="json")
            with gr.Tab("Typed Retrieval Plan"):
                typed_retrieval_plan = gr.Code(label="typed_retrieval_plan.json", language="json")
            with gr.Tab("Retrieved Typed Memories"):
                retrieved_typed_memories = gr.Code(label="retrieved_typed_memories.json", language="json")
            with gr.Tab("Artifact Board"):
                artifact_board = gr.Code(label="artifact_board.json", language="json")
            with gr.Tab("Augmented Summary Prompt"):
                augmented_summary_prompt = gr.Textbox(label="augmented_summary_prompt.txt", lines=18)
            with gr.Tab("Retrieved Memories"):
                retrieved = gr.Code(label="retrieved_memories.json", language="json")
            with gr.Tab("Baseline Summary"):
                baseline_summary = gr.Markdown(label="baseline_summary.md")
            with gr.Tab("Structured Summary"):
                structured = gr.Markdown(label="summary_structured.md")
            with gr.Tab("Natural Summary"):
                natural = gr.Markdown(label="summary_natural.md")
            with gr.Tab("Enhancement Comparison"):
                enhancement_comparison = gr.Markdown(label="enhancement_comparison.md")
            with gr.Tab("Temporal Enrichment"):
                temporal_enrichment = gr.Markdown(label="temporal_enrichment.md")
            with gr.Tab("Scores"):
                scores = gr.Code(label="summary_scores.json", language="json")
            with gr.Tab("Summary Critic Prompt"):
                summary_critic_prompt = gr.Textbox(label="summary_critic_prompt.txt", lines=18)
            with gr.Tab("Summary Critic"):
                summary_critic = gr.Markdown(label="summary_critic_report.md")
            with gr.Tab("Summary Critic JSON"):
                summary_critic_json = gr.Code(label="summary_critic_report.json", language="json")
            with gr.Tab("Summary Rewrite Prompt"):
                summary_rewrite_prompt = gr.Textbox(label="summary_rewrite_prompt.txt", lines=18)
            with gr.Tab("Rewritten Structured"):
                summary_rewritten_structured = gr.Markdown(label="summary_rewritten_structured.md")
            with gr.Tab("Rewritten Natural"):
                summary_rewritten_natural = gr.Markdown(label="summary_rewritten_natural.md")
            with gr.Tab("Summary Rewrite Report"):
                summary_rewrite_report = gr.Code(label="summary_rewrite_report.json", language="json")
            with gr.Tab("Report"):
                report = gr.Markdown(label="report.md")
            with gr.Tab("Run Info"):
                run_info = gr.Code(label="run_info.json", language="json")
            with gr.Tab("Generated History Summaries"):
                e2e_history_summaries = gr.Markdown(label="generated_history_summaries.md")
            with gr.Tab("Generated Current Summary"):
                e2e_current_summary = gr.Markdown(label="generated_current_summary.md")
            with gr.Tab("E2E Preprocess Info"):
                e2e_preprocess_info = gr.Code(label="e2e_preprocess_info.json", language="json")
            with gr.Tab("Logs"):
                log = gr.Textbox(label="运行日志", lines=18)

        common_config_inputs = [user_id, project, source_type, source_time]
        summarize_config_inputs = [top_k, summary_style, score_summary, score_type, rewrite_from_critic]
        outputs = [
            status,
            output_dir,
            stats_html,
            report_summary,
            sidebar_status,
            extracted,
            typed_memories,
            typed_memory_dedup_report,
            typed_extraction_raw_output,
            add_results,
            typed_retrieval_plan,
            retrieved_typed_memories,
            artifact_board,
            augmented_summary_prompt,
            retrieved,
            baseline_summary,
            structured,
            natural,
            enhancement_comparison,
            temporal_enrichment,
            scores,
            summary_critic_prompt,
            summary_critic,
            summary_critic_json,
            summary_rewrite_prompt,
            summary_rewritten_structured,
            summary_rewritten_natural,
            summary_rewrite_report,
            report,
            run_info,
            e2e_history_summaries,
            e2e_current_summary,
            e2e_preprocess_info,
            log,
            download_zip,
        ]

        ingest_button.click(
            fn=run_ingest_click,
            inputs=common_config_inputs + [output_root, ingest_history_text, ingest_history_files, empty_path_text],
            outputs=outputs,
        )
        summarize_button.click(
            fn=run_summarize_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [output_root, summarize_current_text, summarize_current_file, empty_path_text],
            outputs=outputs,
        )
        run_button.click(
            fn=run_full_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [output_root, run_history_text, run_history_files, empty_path_text, run_current_text, run_current_file, empty_path_text],
            outputs=outputs,
        )

    return demo


def build_dashboard_app(workflow: Any, args: argparse.Namespace) -> Any:
    gr = require_gradio()
    module_name = getattr(workflow, "__name__", "unknown")
    lan_ip = get_lan_ip()
    root_path = normalize_root_path(args.root_path) or "none"
    share_text = "enabled" if args.share else "disabled"
    queue_text = "enabled" if args.queue else "disabled"
    auth_text = "enabled" if args.auth_user and args.auth_password else "disabled"

    def run_ingest_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        output_root: str,
        history_text: str,
        history_files: Any,
        history_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            demo_inputs = build_history_inputs(
                workflow,
                history_text,
                history_files,
                history_path_text,
                user_id,
                project,
                resolved_source_type,
                source_time,
            )
            output_run_dir = make_output_dir(workflow, output_root, "ingest", project)
            ok, log = capture_run(workflow.run_ingest, demo_inputs, output_run_dir=output_run_dir)
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok, "ingest", output_run_dir, log, time.perf_counter() - start_time, include_zip=args.return_zip
        )

    def run_summarize_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: str,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        current_text: str,
        current_file: Any,
        current_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            demo_input = build_current_input(
                workflow,
                current_text,
                current_file,
                current_path_text,
                user_id,
                project,
                resolved_source_type,
                source_time,
            )
            output_run_dir = make_output_dir(workflow, output_root, "summarize", project)
            ok, log = capture_run(
                workflow.run_summarize,
                demo_input,
                top_k=int(top_k),
                summary_style=summary_style,
                score_summary=parse_score_summary(score_summary),
                score_type=score_type,
                rewrite_from_critic=parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir=output_run_dir,
            )
            if ok:
                analysis_log = run_enhancement_analysis(
                    workflow=workflow,
                    output_run_dir=output_run_dir,
                    baseline_summary=demo_input.current_content,
                    source_time=source_time,
                    summary_style=summary_style,
                    baseline_source="summarize_current_input",
                    metadata={"command": "summarize"},
                )
                if analysis_log:
                    log = "\n\n".join(part for part in (log, "[summarize / 增强对比与时间分析]", analysis_log) if part)
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok, "summarize", output_run_dir, log, time.perf_counter() - start_time, include_zip=args.return_zip
        )

    def run_full_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: str,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        history_text: str,
        history_files: Any,
        history_path_text: str,
        current_text: str,
        current_file: Any,
        current_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            output_run_dir = make_output_dir(workflow, output_root, "run", project)
            history_inputs = build_history_inputs(
                workflow,
                history_text,
                history_files,
                history_path_text,
                user_id,
                project,
                resolved_source_type,
                source_time,
            )
            current_input = build_current_input(
                workflow,
                current_text,
                current_file,
                current_path_text,
                user_id,
                project,
                resolved_source_type,
                source_time,
            )
            start_full_chain_background(
                workflow,
                history_inputs,
                current_input,
                int(top_k),
                summary_style,
                parse_score_summary(score_summary),
                score_type,
                parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir,
            )
            ok = True
            log = "后台完整链路任务已启动。请稍后点击“刷新结果”查看 outputs/web 中的产物。"
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok,
            "run",
            output_run_dir,
            log,
            time.perf_counter() - start_time,
            include_zip=args.return_zip,
            state_label="运行中" if ok else None,
        )

    def run_e2e_ingest_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        output_root: str,
        raw_history_text: str,
        raw_history_files: Any,
        history_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            output_run_dir = make_output_dir(workflow, output_root, "e2e_ingest", project)
            raw_history_sources = build_raw_history_sources(
                raw_history_text,
                raw_history_files,
                history_path_text,
            )
            start_e2e_ingest_background(
                workflow,
                raw_history_sources,
                user_id,
                project,
                resolved_source_type,
                source_time,
                output_run_dir,
            )
            ok = True
            log = "后台 E2E Ingest 任务已启动。请稍后点击“刷新结果”查看历史基础摘要和记忆写入产物。"
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok,
            "e2e_ingest",
            output_run_dir,
            log,
            time.perf_counter() - start_time,
            include_zip=args.return_zip,
            state_label="运行中" if ok else None,
        )

    def run_e2e_summarize_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: str,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        raw_current_text: str,
        raw_current_file: Any,
        current_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            output_run_dir = make_output_dir(workflow, output_root, "e2e_summarize", project)
            raw_current_source = build_raw_current_source(
                raw_current_text,
                raw_current_file,
                current_path_text,
            )
            start_e2e_summarize_background(
                workflow,
                raw_current_source,
                user_id,
                project,
                resolved_source_type,
                source_time,
                int(top_k),
                summary_style,
                parse_score_summary(score_summary),
                score_type,
                parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir,
            )
            ok = True
            log = "后台 E2E Summarize 任务已启动。请稍后点击“刷新结果”查看当前基础摘要和增强摘要产物。"
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok,
            "e2e_summarize",
            output_run_dir,
            log,
            time.perf_counter() - start_time,
            include_zip=args.return_zip,
            state_label="运行中" if ok else None,
        )

    def run_e2e_click(
        user_id: str,
        project: str,
        source_type: str,
        custom_source_type: str,
        source_time: str,
        top_k: int,
        summary_style: str,
        score_summary: str,
        score_type: str,
        rewrite_from_critic,
        output_root: str,
        raw_history_text: str,
        raw_history_files: Any,
        history_path_text: str,
        raw_current_text: str,
        raw_current_file: Any,
        current_path_text: str,
    ):
        start_time = time.perf_counter()
        try:
            resolved_source_type = resolve_source_type(source_type, custom_source_type)
            output_run_dir = make_output_dir(workflow, output_root, "e2e_run", project)
            raw_history_sources = build_raw_history_sources(
                raw_history_text,
                raw_history_files,
                history_path_text,
            )
            raw_current_source = build_raw_current_source(
                raw_current_text,
                raw_current_file,
                current_path_text,
            )
            start_e2e_full_chain_background(
                workflow,
                raw_history_sources,
                raw_current_source,
                user_id,
                project,
                resolved_source_type,
                source_time,
                int(top_k),
                summary_style,
                parse_score_summary(score_summary),
                score_type,
                parse_rewrite_from_critic(rewrite_from_critic),
                output_run_dir,
            )
            ok = True
            log = "后台端到端 E2E 任务已启动。请稍后点击“刷新结果”查看基础摘要和后续流程产物。"
        except Exception:
            output_run_dir = None
            ok = False
            log = traceback.format_exc()
        return load_result_payload(
            ok,
            "e2e_run",
            output_run_dir,
            log,
            time.perf_counter() - start_time,
            include_zip=args.return_zip,
            state_label="运行中" if ok else None,
        )

    def refresh_click(output_dir_text: str):
        return load_output_dir_payload(output_dir_text, include_zip=args.return_zip)

    with gr.Blocks(title="Summary Memory Workflow Demo", css=APP_CSS) as demo:
        empty_path_text = gr.State(value="")
        gr.HTML(
            """
            <header class="smw-topbar">
              <div class="smw-brand">
                <div class="smw-logo">∿</div>
                <div>
                  <div class="smw-title-row">
                    <h1>Summary Memory Workflow</h1>
                    <span class="smw-demo-badge">Demo</span>
                  </div>
                  <p class="smw-subtitle">基于 mem0 + Qdrant + 本地 Qwen 的摘要增强工作流</p>
                </div>
              </div>
              <div class="smw-actions">
                <span class="smw-action">☼ 深色模式</span>
                <span class="smw-action">? 使用说明</span>
                <span class="smw-action">GitHub</span>
              </div>
            </header>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Group(elem_classes=["smw-panel"]):
                    gr.HTML('<div class="smw-sidebar-title">▣ 项目配置</div>')
                    user_id = gr.Textbox(label="User ID", value="paper_summary_assistant")
                    project = gr.Textbox(label="Project", value="paper_summary_assistant")
                    source_type = gr.Dropdown(
                        label="Source Type",
                        choices=[
                            "project_history_summary",
                            "meeting_minutes",
                            "project_weekly_summary",
                            "learning_summary",
                            "custom",
                        ],
                        value="project_history_summary",
                    )
                    custom_source_type = gr.Textbox(
                        label="Custom Source Type（可选，填写后覆盖上方选择）",
                        placeholder="例如：paper_review, experiment_log, customer_feedback",
                    )
                    source_time = gr.Textbox(label="Source Time", value="2026-06-04")
                    gr.HTML('<div class="smw-side-divider"></div>')
                    gr.HTML('<div class="smw-sidebar-title">▤ 摘要生成设置</div>')
                    summary_style = gr.Dropdown(
                        label="Summary Style",
                        choices=["structured", "natural", "both"],
                        value="both",
                    )
                    score_type = gr.Dropdown(
                        label="Score Type",
                        choices=["general", "meeting", "interview"],
                        value="meeting",
                    )
                    score_summary = gr.Checkbox(label="启用自动评分", value=True)
                    rewrite_from_critic = gr.Checkbox(label="启用 Critic Rewrite Draft", value=False)
                    top_k = gr.Number(label="Top K", value=8, precision=0)
                    output_root = gr.Textbox(label="Output Dir", value=str(args.output_dir), elem_classes=["smw-short"])

                with gr.Group(elem_classes=["smw-panel"]):
                    gr.HTML('<div class="smw-sidebar-title">◴ 运行状态</div>')
                    sidebar_status = gr.HTML(build_sidebar_status_html())
                    gr.HTML(
                        f"""
                        <div class="smw-side-divider"></div>
                        <div class="smw-status-line"><span>模块</span><span>{module_name}</span></div>
                        <div class="smw-status-line"><span>LAN</span><span>{lan_ip}:{args.port}</span></div>
                        <div class="smw-status-line"><span>Root</span><span>{root_path}</span></div>
                        <div class="smw-status-line"><span>Share</span><span>{share_text}</span></div>
                        <div class="smw-status-line"><span>Queue</span><span>{queue_text}</span></div>
                        <div class="smw-status-line"><span>Auth</span><span>{auth_text}</span></div>
                        """
                    )

            with gr.Column(scale=4):
                gr.HTML(
                    """
                    <div class="smw-stepper">
                      <div class="smw-step active">
                        <div class="smw-step-num">1</div>
                        <div><div class="smw-step-title">导入历史材料</div><div class="smw-step-desc">抽取并写入长期记忆</div></div>
                      </div>
                      <div class="smw-arrow">→</div>
                      <div class="smw-step">
                        <div class="smw-step-num">2</div>
                        <div><div class="smw-step-title">生成增强摘要</div><div class="smw-step-desc">检索记忆并生成摘要</div></div>
                      </div>
                      <div class="smw-arrow">→</div>
                      <div class="smw-step">
                        <div class="smw-step-num">3</div>
                        <div><div class="smw-step-title">查看结果</div><div class="smw-step-desc">摘要、评分与记忆</div></div>
                      </div>
                      <div class="smw-arrow">→</div>
                      <div class="smw-step">
                        <div class="smw-step-num">4</div>
                        <div><div class="smw-step-title">下载产物</div><div class="smw-step-desc">下载完整运行结果</div></div>
                      </div>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("历史材料导入（Ingest）", open=True):
                            gr.HTML(
                                '<div class="smw-card-title">▧ 历史材料导入（Ingest）</div>'
                                '<p class="smw-card-subtitle">支持多个 .txt / .md、一个 .zip，或粘贴文本分段</p>'
                            )
                            with gr.Tabs():
                                with gr.Tab("上传文件"):
                                    gr.HTML(
                                        '<div class="smw-upload-note">点击上传或拖拽文件到此处<br>支持 .txt / .md 文件</div>'
                                    )
                                    ingest_history_files = gr.File(
                                        label="历史材料文件（.txt / .md 可多选，或一个 .zip）",
                                        file_count="multiple",
                                        file_types=[".txt", ".md", ".zip"],
                                    )
                                with gr.Tab("粘贴文本"):
                                    ingest_history_text = gr.Textbox(
                                        label="历史材料文本",
                                        lines=10,
                                        placeholder="粘贴历史摘要、会议纪要、项目周报或学习记录。多份材料可用单独一行 --- 分隔。",
                                        elem_classes=["smw-compact"],
                                    )
                            ingest_button = gr.Button(
                                "开始导入记忆（Ingest）", variant="primary", elem_classes=["smw-blue"]
                            )

                    with gr.Column(scale=1):
                        with gr.Accordion("当前材料摘要生成（Summarize）", open=True):
                            gr.HTML(
                                '<div class="smw-card-title">✧ 当前材料摘要生成（Summarize）</div>'
                                '<p class="smw-card-subtitle">上传或粘贴当前材料，基于记忆生成增强摘要</p>'
                            )
                            with gr.Tabs():
                                with gr.Tab("上传文件"):
                                    gr.HTML(
                                        '<div class="smw-upload-note">点击上传或拖拽文件到此处<br>支持 .txt / .md 文件</div>'
                                    )
                                    summarize_current_file = gr.File(
                                        label="当前材料文件",
                                        file_count="single",
                                        file_types=[".txt", ".md"],
                                    )
                                with gr.Tab("粘贴文本"):
                                    summarize_current_text = gr.Textbox(
                                        label="当前材料文本",
                                        lines=10,
                                        placeholder="粘贴当前需要增强摘要的材料。",
                                        elem_classes=["smw-compact"],
                                    )
                            summarize_button = gr.Button(
                                "生成增强摘要（Summarize）", variant="primary", elem_classes=["smw-green"]
                            )

                with gr.Accordion("完整链路 Run（可选）", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            run_history_text = gr.Textbox(
                                label="历史材料文本",
                                lines=7,
                                placeholder="粘贴历史材料。多份材料可用单独一行 --- 分隔。",
                                elem_classes=["smw-compact"],
                            )
                            run_history_files = gr.File(
                                label="历史材料文件（.txt / .md 可多选，或一个 .zip）",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".zip"],
                            )
                        with gr.Column(scale=1):
                            run_current_text = gr.Textbox(
                                label="当前材料文本", lines=7, placeholder="粘贴当前材料。", elem_classes=["smw-compact"]
                            )
                            run_current_file = gr.File(
                                label="当前材料文件（.txt / .md，单文件）",
                                file_count="single",
                                file_types=[".txt", ".md"],
                            )
                    run_button = gr.Button("执行完整链路（Run）", variant="primary", elem_classes=["smw-blue"])

                with gr.Accordion("端到端 E2E：原始资料 → 基础摘要 → 增强摘要", open=False):
                    gr.HTML(
                        '<p class="smw-card-subtitle">'
                        '输入原始会议、项目、学习或访谈资料；系统先生成基础摘要，再复用现有 Ingest + Summarize 流程。'
                        '</p>'
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            e2e_history_text = gr.Textbox(
                                label="历史原始资料文本",
                                lines=8,
                                placeholder="粘贴历史原始会议记录、转写稿、项目资料等。多份材料可用单独一行 --- 或 === 分隔。",
                                elem_classes=["smw-compact"],
                            )
                            e2e_history_files = gr.File(
                                label="历史原始资料文件（.txt / .md 可多选，或一个 .zip）",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".zip"],
                            )
                        with gr.Column(scale=1):
                            e2e_current_text = gr.Textbox(
                                label="当前原始资料文本",
                                lines=8,
                                placeholder="粘贴当前需要生成增强摘要的原始资料。",
                                elem_classes=["smw-compact"],
                            )
                            e2e_current_file = gr.File(
                                label="当前原始资料文件（.txt / .md，单文件）",
                                file_count="single",
                                file_types=[".txt", ".md"],
                            )
                    with gr.Row():
                        e2e_ingest_button = gr.Button(
                            "E2E 导入历史记忆（Ingest）", variant="primary", elem_classes=["smw-blue"]
                        )
                        e2e_summarize_button = gr.Button(
                            "E2E 生成增强摘要（Summarize）", variant="primary", elem_classes=["smw-green"]
                        )
                        e2e_button = gr.Button("执行端到端链路（E2E Run）", variant="primary", elem_classes=["smw-green"])

                gr.HTML('<div class="smw-section-title">运行结果</div>')
                stats_html = gr.HTML(build_stats_html(None))
                report_summary = gr.HTML(build_report_summary_html())

                with gr.Row():
                    with gr.Column(scale=3):
                        status = gr.Textbox(label="当前状态", lines=3, elem_classes=["smw-results"])
                    with gr.Column(scale=4):
                        output_dir = gr.Textbox(label="本次输出目录", lines=3, elem_classes=["smw-results"])
                    with gr.Column(scale=2):
                        refresh_button = gr.Button("刷新结果", variant="secondary")
                        download_zip = gr.File(label="下载产物（代理模式默认不自动回传）")

                with gr.Tabs(elem_classes=["smw-results"]):
                    with gr.Tab("Report"):
                        report = gr.Markdown(label="report.md")
                    with gr.Tab("Extracted Memories"):
                        extracted = gr.Code(label="extracted_memories.json", language="json")
                    with gr.Tab("Typed Memories"):
                        typed_memories = gr.Code(label="typed_memories.json", language="json")
                    with gr.Tab("Typed Memory Dedup"):
                        typed_memory_dedup_report = gr.Code(label="typed_memory_dedup_report.json", language="json")
                    with gr.Tab("Typed Extraction Raw Output"):
                        typed_extraction_raw_output = gr.Textbox(
                            label="typed_memory_extraction_raw_output.txt", lines=18
                        )
                    with gr.Tab("Add Results"):
                        add_results = gr.Code(label="add_results.json", language="json")
                    with gr.Tab("Typed Retrieval Plan"):
                        typed_retrieval_plan = gr.Code(label="typed_retrieval_plan.json", language="json")
                    with gr.Tab("Retrieved Typed Memories"):
                        retrieved_typed_memories = gr.Code(label="retrieved_typed_memories.json", language="json")
                    with gr.Tab("Artifact Board"):
                        artifact_board = gr.Code(label="artifact_board.json", language="json")
                    with gr.Tab("Augmented Summary Prompt"):
                        augmented_summary_prompt = gr.Textbox(label="augmented_summary_prompt.txt", lines=18)
                    with gr.Tab("Retrieved Memories"):
                        retrieved = gr.Code(label="retrieved_memories.json", language="json")
                    with gr.Tab("Baseline Summary"):
                        baseline_summary = gr.Markdown(label="baseline_summary.md")
                    with gr.Tab("Structured Summary"):
                        structured = gr.Markdown(label="summary_structured.md")
                    with gr.Tab("Natural Summary"):
                        natural = gr.Markdown(label="summary_natural.md")
                    with gr.Tab("Enhancement Comparison"):
                        enhancement_comparison = gr.Markdown(label="enhancement_comparison.md")
                    with gr.Tab("Temporal Enrichment"):
                        temporal_enrichment = gr.Markdown(label="temporal_enrichment.md")
                    with gr.Tab("Scores"):
                        scores = gr.Code(label="summary_scores.json", language="json")
                    with gr.Tab("Summary Critic Prompt"):
                        summary_critic_prompt = gr.Textbox(label="summary_critic_prompt.txt", lines=18)
                    with gr.Tab("Summary Critic"):
                        summary_critic = gr.Markdown(label="summary_critic_report.md")
                    with gr.Tab("Summary Critic JSON"):
                        summary_critic_json = gr.Code(label="summary_critic_report.json", language="json")
                    with gr.Tab("Summary Rewrite Prompt"):
                        summary_rewrite_prompt = gr.Textbox(label="summary_rewrite_prompt.txt", lines=18)
                    with gr.Tab("Rewritten Structured"):
                        summary_rewritten_structured = gr.Markdown(label="summary_rewritten_structured.md")
                    with gr.Tab("Rewritten Natural"):
                        summary_rewritten_natural = gr.Markdown(label="summary_rewritten_natural.md")
                    with gr.Tab("Summary Rewrite Report"):
                        summary_rewrite_report = gr.Code(label="summary_rewrite_report.json", language="json")
                    with gr.Tab("Run Info"):
                        run_info = gr.Code(label="run_info.json", language="json")
                    with gr.Tab("Generated History Summaries"):
                        e2e_history_summaries = gr.Markdown(label="generated_history_summaries.md")
                    with gr.Tab("Generated Current Summary"):
                        e2e_current_summary = gr.Markdown(label="generated_current_summary.md")
                    with gr.Tab("E2E Preprocess Info"):
                        e2e_preprocess_info = gr.Code(label="e2e_preprocess_info.json", language="json")
                    with gr.Tab("Logs"):
                        log = gr.Textbox(label="完整运行日志", lines=18)

        common_config_inputs = [user_id, project, source_type, custom_source_type, source_time]
        summarize_config_inputs = [top_k, summary_style, score_summary, score_type, rewrite_from_critic]
        outputs = [
            status,
            output_dir,
            stats_html,
            report_summary,
            sidebar_status,
            extracted,
            typed_memories,
            typed_memory_dedup_report,
            typed_extraction_raw_output,
            add_results,
            typed_retrieval_plan,
            retrieved_typed_memories,
            artifact_board,
            augmented_summary_prompt,
            retrieved,
            baseline_summary,
            structured,
            natural,
            enhancement_comparison,
            temporal_enrichment,
            scores,
            summary_critic_prompt,
            summary_critic,
            summary_critic_json,
            summary_rewrite_prompt,
            summary_rewritten_structured,
            summary_rewritten_natural,
            summary_rewrite_report,
            report,
            run_info,
            e2e_history_summaries,
            e2e_current_summary,
            e2e_preprocess_info,
            log,
            download_zip,
        ]

        ingest_button.click(
            fn=run_ingest_click,
            inputs=common_config_inputs + [output_root, ingest_history_text, ingest_history_files, empty_path_text],
            outputs=outputs,
        )
        summarize_button.click(
            fn=run_summarize_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [output_root, summarize_current_text, summarize_current_file, empty_path_text],
            outputs=outputs,
        )
        run_button.click(
            fn=run_full_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [
                output_root,
                run_history_text,
                run_history_files,
                empty_path_text,
                run_current_text,
                run_current_file,
                empty_path_text,
            ],
            outputs=outputs,
        )
        e2e_ingest_button.click(
            fn=run_e2e_ingest_click,
            inputs=common_config_inputs + [output_root, e2e_history_text, e2e_history_files, empty_path_text],
            outputs=outputs,
        )
        e2e_summarize_button.click(
            fn=run_e2e_summarize_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [output_root, e2e_current_text, e2e_current_file, empty_path_text],
            outputs=outputs,
        )
        e2e_button.click(
            fn=run_e2e_click,
            inputs=common_config_inputs
            + summarize_config_inputs
            + [
                output_root,
                e2e_history_text,
                e2e_history_files,
                empty_path_text,
                e2e_current_text,
                e2e_current_file,
                empty_path_text,
            ],
            outputs=outputs,
        )
        refresh_button.click(
            fn=refresh_click,
            inputs=[output_dir],
            outputs=outputs,
        )

    return demo


def main() -> None:
    args = parse_args()
    ensure_local_no_proxy()
    root_path = normalize_root_path(args.root_path)
    if root_path:
        os.environ["GRADIO_ROOT_PATH"] = root_path
    workflow = import_workflow_module()
    demo = build_dashboard_app(workflow, args)
    print(f"Loaded workflow module: {workflow.__name__}")
    print(f"LAN access: http://{get_lan_ip()}:{args.port}")
    print(f"Queue mode: {'enabled' if args.queue else 'disabled'}")
    print(f"Auth mode: {'enabled' if args.auth_user and args.auth_password else 'disabled'}")
    if root_path:
        print(f"Proxy root path: {root_path}")
    launch_demo(demo, args)


if __name__ == "__main__":
    main()
