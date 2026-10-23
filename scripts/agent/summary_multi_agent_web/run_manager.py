"""管理 Web 单活跃后台运行、状态快照、停止请求和安全产物下载。"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .e2e_ingest import run_e2e_ingest, run_e2e_run, run_web_ingest
from .evaluation_compare import run_web_evaluation_compare
from .multi_agent_runner import run_multi_agent_summary
from .artifact_reader import create_safe_artifact_archive, read_artifact_snapshot
from .visualization import build_run_visualization
from .schemas import (
    WebArtifactSnapshot,
    WebE2ERequest,
    WebIngestRequest,
    WebRunSnapshot,
    WebRunStatus,
    WebRunVisualization,
    WebSummarizeRequest,
    status_from_runtime,
)
from summary_multi_agent.evaluation.schemas import EvaluationCompareRequest
from summary_multi_agent.schemas import EvaluationMode


AsyncRunner = Callable[..., Awaitable[Any]]


def _timestamp() -> str:
    """返回 Web 状态快照使用的 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _safe_segment(value: str) -> str:
    """将项目或运行类型转换为安全、有限长度的目录名片段。"""
    cleaned = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in value.strip()
    )
    return cleaned.strip("._-")[:80] or "project"


def create_run_directory(
    output_root: Path,
    project: str,
    *,
    run_kind: str = "summarize",
) -> Path:
    """为一次运行创建唯一目录，保留同一秒内的重试审计产物。"""
    # 每次重试都创建新目录，保留旧运行的审计产物，避免页面刷新时读到混合结果。
    root = output_root.expanduser() / _safe_segment(run_kind)
    root.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{_safe_segment(run_kind)}_{_safe_segment(project)}"
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _persist_run_identity(output_dir: Path, run_id: str) -> None:
    """把进程内 run_id 写入运行目录，支持服务重启后恢复 Inspector。"""
    path = output_dir / "web_run_info.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    payload.update({"run_id": run_id, "output_dir": str(output_dir)})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class WebRunManager:
    """管理单活跃后台 asyncio Pipeline 及其可观察的 Web 状态。

    每次运行在线程池的独立事件循环中执行；停止请求是协作式的，不能强杀已
    开始执行的线程。状态、Future 和运行目录由线程锁保护。
    """

    def __init__(
        self,
        *,
        max_active_runs: int = 1,
        runner: AsyncRunner = run_multi_agent_summary,
        ingest_runner: AsyncRunner = run_web_ingest,
        comparison_runner: AsyncRunner = run_web_evaluation_compare,
    ) -> None:
        """创建单线程执行器和内存状态表；当前 Web V2 只允许一个活跃运行。"""
        if max_active_runs != 1:
            raise ValueError("Web V2 currently supports exactly one active run")
        self._runner = runner
        self._ingest_runner = ingest_runner
        self._comparison_runner = comparison_runner
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="summary-web"
        )
        self._lock = threading.RLock()
        self._active_run_id: str | None = None
        self._snapshots: dict[str, WebRunSnapshot] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._output_roots: set[Path] = set()

    def close(self) -> None:
        """关闭后台执行器，并取消尚未开始的 Future。"""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _has_active_run(self) -> bool:
        """判断当前活跃 run 是否仍有未完成 Future。"""
        if self._active_run_id is None:
            return False
        future = self._futures.get(self._active_run_id)
        return future is not None and not future.done()

    def start_summarize(
        self,
        web_request: WebSummarizeRequest,
        output_root: Path,
    ) -> WebRunSnapshot:
        """创建并排队一次独立 Summarize 运行，返回 queued 快照。"""
        with self._lock:
            if self._has_active_run():
                raise RuntimeError("已有运行正在执行；第一版 Web 只允许一个活跃运行。")

            output_dir = create_run_directory(
                output_root, web_request.project, run_kind="summarize"
            )
            self._output_roots.add(output_root.expanduser())
            run_id = f"web_{uuid.uuid4().hex}"
            snapshot = WebRunSnapshot(
                run_id=run_id,
                run_type="multi_agent_summarize",
                status=WebRunStatus.QUEUED,
                stage="queued",
                output_dir=output_dir,
                started_at=_timestamp(),
            )
            self._snapshots[run_id] = snapshot
            self._active_run_id = run_id
            _persist_run_identity(output_dir, run_id)
            future = self._executor.submit(
                self._run_in_background, run_id, web_request, output_dir
            )
            self._futures[run_id] = future
            return snapshot

    def start_e2e(
        self,
        web_request: WebE2ERequest,
        output_root: Path,
    ) -> WebRunSnapshot:
        """创建并排队一次 E2E Ingest 或 E2E Run，保持单活跃约束。"""
        with self._lock:
            if self._has_active_run():
                raise RuntimeError("已有运行正在执行；第一版 Web 只允许一个活跃运行。")
            output_dir = create_run_directory(
                output_root, web_request.project, run_kind=web_request.run_type
            )
            self._output_roots.add(output_root.expanduser())
            run_id = f"web_{uuid.uuid4().hex}"
            snapshot = WebRunSnapshot(
                run_id=run_id,
                run_type=web_request.run_type,
                status=WebRunStatus.QUEUED,
                stage="queued",
                output_dir=output_dir,
                started_at=_timestamp(),
            )
            self._snapshots[run_id] = snapshot
            self._active_run_id = run_id
            _persist_run_identity(output_dir, run_id)
            future = self._executor.submit(
                self._run_e2e_in_background,
                run_id,
                web_request,
                output_dir,
            )
            self._futures[run_id] = future
            return snapshot

    def start_ingest(
        self,
        web_request: WebIngestRequest,
        output_root: Path,
    ) -> WebRunSnapshot:
        """创建并排队一次独立历史 Ingest 运行。"""
        with self._lock:
            if self._has_active_run():
                raise RuntimeError("已有运行正在执行；第一版 Web 只允许一个活跃运行。")
            output_dir = create_run_directory(
                output_root, web_request.project, run_kind="ingest"
            )
            self._output_roots.add(output_root.expanduser())
            run_id = f"web_{uuid.uuid4().hex}"
            snapshot = WebRunSnapshot(
                run_id=run_id,
                run_type="ingest",
                status=WebRunStatus.QUEUED,
                stage="queued",
                output_dir=output_dir,
                started_at=_timestamp(),
            )
            self._snapshots[run_id] = snapshot
            self._active_run_id = run_id
            _persist_run_identity(output_dir, run_id)
            future = self._executor.submit(
                self._run_ingest_in_background,
                run_id,
                web_request,
                output_dir,
            )
            self._futures[run_id] = future
            return snapshot

    def start_evaluation_compare(
        self,
        request: EvaluationCompareRequest,
        output_root: Path,
    ) -> WebRunSnapshot:
        """创建并排队一次独立多纪要横评。"""
        with self._lock:
            if self._has_active_run():
                raise RuntimeError("已有运行正在执行；Web V2 只允许一个活跃运行。")
            output_dir = create_run_directory(
                output_root, request.project, run_kind="evaluate"
            )
            self._output_roots.add(output_root.expanduser())
            run_id = f"web_{uuid.uuid4().hex}"
            snapshot = WebRunSnapshot(
                run_id=run_id,
                run_type="evaluation_compare",
                status=WebRunStatus.QUEUED,
                stage="queued",
                output_dir=output_dir,
                started_at=_timestamp(),
                evaluation_status="pending",
                personalization_evaluation_status=(
                    "pending"
                    if (
                        request.target_entities
                        and request.evaluation_config.evaluation_mode
                        == EvaluationMode.STRICT
                    )
                    else "disabled"
                ),
            )
            self._snapshots[run_id] = snapshot
            self._active_run_id = run_id
            _persist_run_identity(output_dir, run_id)
            future = self._executor.submit(
                self._run_comparison_in_background,
                run_id,
                request,
                output_dir,
            )
            self._futures[run_id] = future
            return snapshot

    def _update(self, run_id: str, **updates: Any) -> None:
        """在锁内合并并严格校验一个运行状态快照。"""
        with self._lock:
            current = self._snapshots[run_id]
            payload = current.model_dump()
            payload.update(updates)
            self._snapshots[run_id] = WebRunSnapshot.model_validate(payload)

    def _run_in_background(
        self,
        run_id: str,
        web_request: WebSummarizeRequest,
        output_dir: Path,
    ) -> None:
        """在线程中运行普通 Summarize，并把核心状态映射为 Web 快照。"""
        # 每个运行在线程内创建独立事件循环；不要在 Gradio 主事件循环中
        # 嵌套 asyncio.run，也不要跨运行共享带有 Local Qdrant 状态的 Pipeline。
        self._update(run_id, status=WebRunStatus.RUNNING, stage="running")

        def on_status(value: str) -> None:
            """将核心 Pipeline 状态回调投影到当前 Web 运行。"""
            self._update(
                run_id,
                status=status_from_runtime(value),
                stage=value,
            )

        try:
            result = asyncio.run(
                self._runner(
                    web_request,
                    output_dir,
                    status_callback=on_status,
                )
            )
            runtime_status = status_from_runtime(result.status.value)
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            warnings = list(result.warnings)
            if requested and runtime_status in {
                WebRunStatus.COMPLETED,
                WebRunStatus.COMPLETED_WITH_WARNINGS,
            }:
                warnings.append(
                    "已请求停止，但底层 Pipeline 在停止生效前完成；未宣称已取消。"
                )
                runtime_status = WebRunStatus.COMPLETED_WITH_WARNINGS
            self._update(
                run_id,
                status=runtime_status,
                stage=runtime_status.value,
                summary=result.final_summary or "",
                recommended_candidate_index=result.recommended_candidate_index,
                evaluation_status=getattr(
                    getattr(result, "evaluation_status", "disabled"),
                    "value",
                    getattr(result, "evaluation_status", "disabled"),
                ),
                personalization_evaluation_status=getattr(
                    getattr(
                        result,
                        "personalization_evaluation_status",
                        "disabled",
                    ),
                    "value",
                    getattr(
                        result,
                        "personalization_evaluation_status",
                        "disabled",
                    ),
                ),
                warnings=list(dict.fromkeys(warnings)),
                finished_at=_timestamp(),
            )
        except Exception as exc:
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            self._update(
                run_id,
                status=WebRunStatus.CANCELLING if requested else WebRunStatus.FAILED,
                stage="cancelling" if requested else "failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_timestamp(),
            )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _run_e2e_in_background(
        self,
        run_id: str,
        web_request: WebE2ERequest,
        output_dir: Path,
    ) -> None:
        """在线程中运行 E2E 流程，并保持 Ingest 到 Summarize 的阶段边界。"""
        # E2E 采用与普通 Summarize 相同的单活跃运行约束，但由 E2E runner
        # 保证 Ingest 成功后才进入 Summarize，不能在 Web 层并行启动两个阶段。
        self._update(run_id, status=WebRunStatus.RUNNING, stage="running")

        def on_status(value: str) -> None:
            """将 E2E 阶段回调写入 Web 快照。"""
            stage = value if value in {"ingest", "summarize"} else value
            self._update(run_id, status=WebRunStatus.EXECUTING, stage=stage)

        try:
            runner = (
                run_e2e_run if web_request.run_type == "e2e_run" else run_e2e_ingest
            )
            result = asyncio.run(
                runner(web_request, output_dir, status_callback=on_status)
            )
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            warnings = list(result.warnings)
            result_status = result.status
            if requested and result_status in {
                WebRunStatus.COMPLETED,
                WebRunStatus.COMPLETED_WITH_WARNINGS,
            }:
                warnings.append(
                    "已请求停止，但底层 E2E 阶段在停止生效前完成；未宣称已取消。"
                )
                result_status = WebRunStatus.COMPLETED_WITH_WARNINGS
            self._update(
                run_id,
                status=result_status,
                stage=result_status.value,
                summary=result.final_summary,
                recommended_candidate_index=result.recommended_candidate_index,
                evaluation_status=getattr(result, "evaluation_status", "disabled"),
                personalization_evaluation_status=getattr(
                    result,
                    "personalization_evaluation_status",
                    "disabled",
                ),
                warnings=list(dict.fromkeys(warnings)),
                finished_at=_timestamp(),
            )
        except Exception as exc:
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            self._update(
                run_id,
                status=WebRunStatus.CANCELLING if requested else WebRunStatus.FAILED,
                stage="cancelling" if requested else "failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_timestamp(),
            )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _run_ingest_in_background(
        self,
        run_id: str,
        web_request: WebIngestRequest,
        output_dir: Path,
    ) -> None:
        """在线程中运行独立 Ingest，并在完成或失败时释放活跃运行槽位。"""
        self._update(run_id, status=WebRunStatus.RUNNING, stage="validation")

        def on_status(value: str) -> None:
            """将 Ingest 阶段回调写入 Web 快照。"""
            self._update(run_id, status=WebRunStatus.EXECUTING, stage=value)

        try:
            result = asyncio.run(
                self._ingest_runner(
                    web_request,
                    output_dir,
                    status_callback=on_status,
                )
            )
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            warnings = list(result.warnings)
            result_status = result.status
            if requested and result_status in {
                WebRunStatus.COMPLETED,
                WebRunStatus.COMPLETED_WITH_WARNINGS,
            }:
                warnings.append(
                    "已请求停止，但底层 Ingest 在停止生效前完成；未宣称已取消。"
                )
                result_status = WebRunStatus.COMPLETED_WITH_WARNINGS
            self._update(
                run_id,
                status=result_status,
                stage=result_status.value,
                warnings=list(dict.fromkeys(warnings)),
                finished_at=_timestamp(),
            )
        except Exception as exc:
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            self._update(
                run_id,
                status=WebRunStatus.CANCELLING if requested else WebRunStatus.FAILED,
                stage="cancelling" if requested else "failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_timestamp(),
            )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _run_comparison_in_background(
        self,
        run_id: str,
        request: EvaluationCompareRequest,
        output_dir: Path,
    ) -> None:
        """在线程中串行横评多份纪要并保持单活跃约束。"""
        self._update(run_id, status=WebRunStatus.RUNNING, stage="validating")

        def on_status(value: str) -> None:
            """把 Snapshot/逐份评分/聚合阶段投影到页面。"""
            self._update(
                run_id,
                status=(
                    WebRunStatus.EVALUATING
                    if (value.startswith("scoring_summary_") or value == "lite_scoring")
                    else WebRunStatus.EXECUTING
                ),
                stage=value,
            )

        try:
            result = asyncio.run(
                self._comparison_runner(
                    request,
                    output_dir,
                    status_callback=on_status,
                )
            )
            status = WebRunStatus(result.status)
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            warnings = list(result.warnings)
            if requested and status in {
                WebRunStatus.COMPLETED,
                WebRunStatus.COMPLETED_WITH_WARNINGS,
            }:
                warnings.append("已请求停止，但横评在停止生效前完成；未宣称已取消。")
                status = WebRunStatus.COMPLETED_WITH_WARNINGS
            if request.evaluation_config.evaluation_mode == EvaluationMode.LITE:
                personalization_status = "disabled"
            else:
                personalization_status = (
                    "disabled"
                    if not request.target_entities
                    else (
                        "failed"
                        if all(
                            item.artifacts is None
                            or item.artifacts.personalization is None
                            or item.artifacts.personalization.metrics.status.value
                            == "failed"
                            for item in result.summaries
                        )
                        else (
                            "completed_with_warnings"
                            if any(
                                item.artifacts is None
                                or item.artifacts.personalization is None
                                or item.artifacts.personalization.metrics.status.value
                                == "failed"
                                for item in result.summaries
                            )
                            else "completed"
                        )
                    )
                )
            self._update(
                run_id,
                status=status,
                stage=status.value,
                evaluation_status=(
                    "failed" if status == WebRunStatus.FAILED else "completed"
                ),
                personalization_evaluation_status=personalization_status,
                warnings=list(dict.fromkeys(warnings)),
                finished_at=_timestamp(),
            )
        except Exception as exc:
            with self._lock:
                requested = self._snapshots[run_id].cancellation_requested
            self._update(
                run_id,
                status=(WebRunStatus.CANCELLING if requested else WebRunStatus.FAILED),
                stage="cancelling" if requested else "failed",
                evaluation_status="failed",
                personalization_evaluation_status=(
                    "failed"
                    if (
                        request.target_entities
                        and request.evaluation_config.evaluation_mode
                        == EvaluationMode.STRICT
                    )
                    else "disabled"
                ),
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_timestamp(),
            )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def request_stop(self, run_id: str) -> WebRunSnapshot:
        """记录协作式停止请求；只有尚未启动的 Future 才能被直接取消。"""
        with self._lock:
            snapshot = self._snapshots[run_id]
            if snapshot.status in {
                WebRunStatus.COMPLETED,
                WebRunStatus.COMPLETED_WITH_WARNINGS,
                WebRunStatus.FAILED,
                WebRunStatus.CANCELLING,
            }:
                return snapshot
            updated = snapshot.model_copy(
                update={
                    "status": WebRunStatus.CANCELLATION_REQUESTED,
                    "stage": "cancellation_requested",
                    "cancellation_requested": True,
                }
            )
            # Future.cancel() 无法中断已经运行的线程；因此请求停止后先显示
            # cancelling，只有确认底层任务实际结束后才能给出终态。
            self._snapshots[run_id] = updated
            future = self._futures.get(run_id)
            if future is not None and future.cancel():
                self._snapshots[run_id] = updated.model_copy(
                    update={
                        "status": WebRunStatus.CANCELLING,
                        "stage": "cancelling",
                        "finished_at": _timestamp(),
                    }
                )
                if self._active_run_id == run_id:
                    self._active_run_id = None
            elif future is not None:
                self._snapshots[run_id] = updated.model_copy(
                    update={"status": WebRunStatus.CANCELLING, "stage": "cancelling"}
                )
            return self._snapshots[run_id]

    def get_snapshot(self, run_id: str) -> WebRunSnapshot:
        """读取内存快照，必要时从已持久化的 Web 运行信息恢复快照。"""
        with self._lock:
            snapshot = self._snapshots.get(run_id)
            if snapshot is not None:
                return snapshot
            roots = tuple(self._output_roots)
        for root in roots:
            for info_path in root.rglob("web_run_info.json"):
                try:
                    payload = json.loads(info_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if str(payload.get("run_id") or info_path.parent.name) != run_id:
                    continue
                output_dir = info_path.parent
                status_raw = str(payload.get("status") or "failed")
                try:
                    status = WebRunStatus(status_raw)
                except ValueError:
                    status = WebRunStatus.FAILED
                summary_path = output_dir / "summary_multi_agent.md"
                summary = ""
                if summary_path.is_file():
                    try:
                        summary = summary_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        pass
                result_payload: dict[str, Any] = {}
                result_path = (
                    output_dir / "evaluation_compare_run_result.json"
                    if payload.get("run_type") == "evaluation_compare"
                    else output_dir / "summary_run_result.json"
                )
                if result_path.is_file():
                    try:
                        loaded = json.loads(result_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            result_payload = loaded
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        pass
                return WebRunSnapshot(
                    run_id=run_id,
                    run_type=str(payload.get("run_type") or "summarize"),
                    status=status,
                    stage=str(payload.get("stage") or status.value),
                    output_dir=output_dir,
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    summary=summary,
                    recommended_candidate_index=result_payload.get(
                        "recommended_candidate_index"
                    ),
                    evaluation_status=str(
                        result_payload.get("evaluation_status")
                        or (
                            "completed"
                            if payload.get("run_type") == "evaluation_compare"
                            and result_payload.get("status")
                            in {"completed", "completed_with_warnings"}
                            else "disabled"
                        )
                    ),
                    personalization_evaluation_status=str(
                        result_payload.get("personalization_evaluation_status")
                        or "disabled"
                    ),
                    warnings=[],
                    error=str(payload.get("error") or ""),
                )
        raise KeyError(run_id)

    def list_persisted_runs(self, output_root: Path) -> list[dict[str, str]]:
        """返回审计页可选择的持久化运行 identity，按目录时间倒序排列。"""

        root = output_root.expanduser()
        with self._lock:
            self._output_roots.add(root)
        entries: list[dict[str, str]] = []
        for info_path in root.rglob("web_run_info.json"):
            try:
                payload = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            run_id = str(payload.get("run_id") or info_path.parent.name)
            entries.append(
                {
                    "run_id": run_id,
                    "run_type": str(payload.get("run_type") or "unknown"),
                    "status": str(payload.get("status") or "unknown"),
                    "output_dir": str(info_path.parent),
                    "project": str(payload.get("project") or ""),
                    "source_time": str(payload.get("source_time") or ""),
                    "started_at": str(payload.get("started_at") or ""),
                }
            )
        return sorted(entries, key=lambda item: item["output_dir"], reverse=True)

    def latest_snapshot(self) -> WebRunSnapshot | None:
        """返回当前进程最近创建的快照，没有运行时返回 ``None``。"""
        with self._lock:
            if not self._snapshots:
                return None
            return next(reversed(self._snapshots.values()))

    def get_artifacts(self, run_id: str) -> WebArtifactSnapshot:
        """按运行类型选择阶段目录，并读取脱敏后的 Inspector 快照。"""
        # 页面重启后运行只存在于磁盘；统一通过 get_snapshot 读取，确保
        # 审计列表中的历史运行也能打开 Inspector，而不只支持当前进程快照。
        snapshot = self.get_snapshot(run_id)
        # E2E 的顶层目录同时包含 Ingest 和 Summarize，Inspector 必须读取
        # 当前阶段目录，避免把两个阶段同名产物混在一个视图里。
        artifact_dir = self._artifact_dir(snapshot)
        return read_artifact_snapshot(artifact_dir)

    def get_visualization(self, run_id: str) -> WebRunVisualization:
        """读取指定运行的现有产物并构造只读可视化投影。"""
        snapshot = self.get_snapshot(run_id)
        artifact_dir = self._artifact_dir(snapshot)
        return build_run_visualization(artifact_dir)

    @staticmethod
    def _artifact_dir(snapshot: WebRunSnapshot) -> Path:
        """Resolve the stage directory containing summary artifacts."""

        artifact_dir = snapshot.output_dir
        if snapshot.run_type in {"e2e_ingest", "e2e_run"}:
            artifact_dir = artifact_dir / "e2e_ingest"
        if snapshot.run_type == "e2e_run":
            artifact_dir = snapshot.output_dir / "multi_agent_summarize"
        return artifact_dir if artifact_dir.exists() else snapshot.output_dir

    def get_report_artifact(self, run_id: str, name: str) -> Path:
        """Return one explicitly public report artifact for direct download."""

        allowed = {
            "summary_multi_agent.md",
            "minutes_document.json",
            "minutes_sources.json",
            "minutes_report.html",
            "minutes_report_bundle.zip",
        }
        if name not in allowed:
            raise ValueError("report artifact is not public")
        snapshot = self.get_snapshot(run_id)
        root = self._artifact_dir(snapshot).expanduser().resolve()
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("report artifact escapes the run directory") from exc
        if not path.is_file():
            raise ValueError(f"report artifact does not exist: {name}")
        return path

    def create_archive(self, run_id: str) -> Path:
        """为指定运行创建只包含白名单审计文件的安全归档包。"""
        snapshot = self.get_snapshot(run_id)
        return create_safe_artifact_archive(self._artifact_dir(snapshot))
