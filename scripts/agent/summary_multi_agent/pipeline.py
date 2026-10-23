"""Multi-Agent V2 摘要主流水线：编排阶段执行、生成摘要并落盘审计产物。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

from .context_builder import build_synthesis_context
from .current_material import CurrentMaterialAnalyzer
from .orchestrator import SummaryOrchestrator
from .raw_material import assert_complete_coverage, build_raw_material_bundle
from .schemas import (
    EvaluationMode,
    EvaluationStatus,
    ReviewRunResult,
    ReviewTermination,
    RunConfig,
    RunStatus,
    SummaryRequest,
    SummaryRunResult,
    SummaryWorkingState,
)
from .evaluation import EvaluationArtifacts, SummaryEvaluator
from .evaluation.lite import (
    LiteSummaryEvaluator,
    write_lite_evaluation_artifacts,
    write_lite_failure_artifacts,
)
from .evaluation.metrics import persisted_summary_sha256
from .evaluation.personalization import (
    disabled_personalization_artifacts,
    failed_personalization_artifacts,
)
from .history_selection import build_history_selection_report
from .minutes_report import MinutesReportBundle, persist_report_bundle
from .minutes_report.models import MinutesReportManifest
from .evaluation.schemas import EvaluationMetrics, PersonalizationEvaluationArtifacts
from .review import ReviewLoop
from .synthesis import SummarySynthesizer
from .tracing import TraceRecorder
from .working_memory import initialize_working_state, save_working_state_checkpoint


ArtifactBoardBuilder = Callable[[SummaryWorkingState], dict[str, Any]]


class MultiAgentSummaryPipeline:
    """执行 V2 从原文分段到摘要、Review、Evaluation 和审计落盘的总控。

    Pipeline 保留正式 candidate 0；Review 只在 Shadow 模式运行，Evaluation 是
    摘要生成后的只读 Sidecar。生成失败会返回失败结果，评测失败只降级为带
    warning 的完成结果，不影响已经生成的正式摘要。
    """

    def __init__(
        self,
        *,
        orchestrator: SummaryOrchestrator,
        current_material_analyzer: CurrentMaterialAnalyzer,
        synthesizer: SummarySynthesizer,
        config: RunConfig,
        output_dir: Path,
        trace_recorder: TraceRecorder,
        artifact_board_builder: Optional[ArtifactBoardBuilder] = None,
        reviewer: Optional[ReviewLoop] = None,
        evaluator: Optional[SummaryEvaluator | LiteSummaryEvaluator] = None,
        model_closer: Any = None,
    ) -> None:
        """注入编排器、当前事实分析器、Summarizer、配置和可选后处理器。"""
        self._orchestrator = orchestrator
        self._analyzer = current_material_analyzer
        self._synthesizer = synthesizer
        self._config = config
        self._output_dir = output_dir
        self._trace = trace_recorder
        self._artifact_board_builder = artifact_board_builder
        self._reviewer = reviewer
        self._evaluator = evaluator
        self._model_closer = model_closer

    async def run(self, request: SummaryRequest) -> SummaryRunResult:
        try:
            return await self._run_impl(request)
        finally:
            if self._model_closer is not None:
                await self._model_closer()

    async def _run_impl(self, request: SummaryRequest) -> SummaryRunResult:
        """运行一次完整摘要流程，并按全局 timeout 返回结构化终态结果。"""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        raw_material = build_raw_material_bundle(
            request.current_material,
            source_id=f"current_{request.request_id}",
            max_segment_chars=self._config.raw_segment_max_chars,
            overlap_ratio=self._config.raw_segment_overlap_ratio,
        )
        current_map_required = (
            self._config.execution_profile == "legacy" or self._config.review_enabled
        )
        state = initialize_working_state(
            request, raw_material, current_map_required=current_map_required
        )
        await self._trace.emit(
            "pipeline.started",
            run_id=state.run_id,
            payload={
                "request_id": request.request_id,
                "segment_count": len(raw_material.segments),
            },
        )

        try:
            result = await asyncio.wait_for(
                self._run_stages(state),
                timeout=self._config.global_timeout_seconds,
            )
        except asyncio.TimeoutError:
            state.errors.append(
                f"pipeline exceeded global timeout of "
                f"{self._config.global_timeout_seconds} seconds"
            )
            return await self._finish_failed(state, "pipeline global timeout")
        if result.status == RunStatus.FAILED:
            return result
        if self._config.evaluation_enabled:
            result = await self._run_evaluation_sidecar(state, result)
        await self._trace.emit(
            "pipeline.finished",
            run_id=state.run_id,
            payload={"status": result.status.value},
        )
        return result

    async def _run_stages(self, state: SummaryWorkingState) -> SummaryRunResult:
        """依次执行 Worker、当前事实、上下文、摘要和可选 Shadow Review。"""
        state = await self._orchestrator.run_workers(state)
        if state.status == RunStatus.FAILED:
            return await self._finish_failed(state, "worker orchestration failed")

        if self._artifact_board_builder is not None:
            try:
                state.historical_evidence_board = self._artifact_board_builder(state)
            except Exception as exc:
                state.warnings.append(
                    f"artifact board compatibility view failed: {type(exc).__name__}: {exc}"
                )

        if any(
            "current_fact_map" in item.required_stages
            for item in state.coverage_ledger.segments
        ):
            await self._set_status(state, RunStatus.EXECUTING, stage="current_fact_map")
            state = await self._analyzer.analyze(state)
            current_map_incomplete = any(
                "current_fact_map" not in segment.completed_stages
                for segment in state.coverage_ledger.segments
            )
            if current_map_incomplete:
                return await self._finish_failed(
                    state, "one or more current raw segments failed current_fact_map"
                )

        try:
            context = build_synthesis_context(
                state,
                compression_mode=self._config.context_compression_mode,
                compression_top_k_per_type=(
                    self._config.context_compression_top_k_per_type
                ),
            )
            if state.context_compression_report is not None:
                report = state.context_compression_report
                await self._trace.emit(
                    "context.compressed",
                    run_id=state.run_id,
                    payload={
                        "mode": report.mode.value,
                        "input_chars": report.input_chars,
                        "output_chars": report.output_chars,
                        "estimated_input_tokens": report.estimated_input_tokens,
                        "estimated_output_tokens": report.estimated_output_tokens,
                        "compression_ratio": report.compression_ratio,
                        "input_record_count": report.input_record_count,
                        "retained_record_count": report.retained_record_count,
                        "merged_record_count": report.merged_record_count,
                        "dropped_record_count": report.dropped_record_count,
                        "current_material_compressed": False,
                        "invariants_preserved": report.invariants_preserved,
                    },
                )
            assert_complete_coverage(state.coverage_ledger)
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            return await self._finish_failed(state, "synthesis context build failed")

        await self._set_status(state, RunStatus.SYNTHESIZING)
        try:
            synthesis_outcome = await self._synthesizer.generate(
                context,
                current_material=state.request.current_material,
            )
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            return await self._finish_failed(
                state,
                "summary synthesis failed",
                summary_prompt=self._synthesizer.last_prompt,
                model_input_report=self._synthesizer.last_model_input_report,
            )

        summary = synthesis_outcome.summary
        summary_prompt = synthesis_outcome.prompt
        state.warnings.extend(synthesis_outcome.warnings)
        state.candidate_summaries.append(summary)
        review_result: Optional[ReviewRunResult] = None
        if self._config.review_enabled:
            await self._set_status(state, RunStatus.REVIEWING)
            if self._reviewer is None:
                review_result = ReviewRunResult(
                    shadow_mode=True,
                    termination=ReviewTermination.REVIEW_FAILED,
                    warnings=("review was enabled but no ReviewLoop was configured",),
                )
            else:
                review_result = await self._reviewer.run(state, summary)
            state.warnings.extend(review_result.warnings)
            if review_result.termination != ReviewTermination.ACCEPTED:
                state.warnings.append(
                    "shadow review did not reach acceptance: "
                    f"{review_result.termination.value}"
                )
            elif review_result.recommended_candidate_index != 0:
                state.warnings.append(
                    "shadow review recommends candidate "
                    f"{review_result.recommended_candidate_index}; "
                    "the original summary remains the formal output"
                )

        recommended_candidate_index = (
            review_result.recommended_candidate_index
            if review_result is not None
            else None
        )
        has_reviewed_candidate = bool(
            recommended_candidate_index is not None
            and recommended_candidate_index != 0
            and recommended_candidate_index < len(state.candidate_summaries)
        )
        final_status = (
            RunStatus.COMPLETED_WITH_WARNINGS if state.warnings else RunStatus.COMPLETED
        )
        await self._set_status(state, final_status)
        artifact_paths = self._artifact_paths(
            include_summary=True,
            include_review=review_result is not None,
            include_reviewed_candidate=has_reviewed_candidate,
            include_rich=synthesis_outcome.report_bundle is not None,
        )
        result = SummaryRunResult(
            run_id=state.run_id,
            status=final_status,
            final_summary=summary,
            selected_candidate_index=0,
            review_enabled=self._config.review_enabled,
            review_shadow_mode=self._config.review_shadow_mode,
            review_termination=(
                review_result.termination if review_result is not None else None
            ),
            recommended_candidate_index=recommended_candidate_index,
            evaluation_enabled=self._config.evaluation_enabled,
            evaluation_mode=self._config.evaluation_mode,
            evaluation_status=(
                EvaluationStatus.PENDING
                if self._config.evaluation_enabled
                else EvaluationStatus.DISABLED
            ),
            personalization_evaluation_status=(
                EvaluationStatus.PENDING
                if (
                    self._config.evaluation_enabled
                    and self._config.evaluation_mode == EvaluationMode.STRICT
                    and state.request.target_entities
                )
                else EvaluationStatus.DISABLED
            ),
            warnings=tuple(dict.fromkeys(state.warnings)),
            coverage_complete=state.coverage_ledger.is_complete,
            artifact_paths=artifact_paths,
        )
        report_manifest = self._save_artifacts(
            state,
            result,
            summary_prompt=summary_prompt,
            model_input_report=synthesis_outcome.model_input_report,
            review_result=review_result,
            report_bundle=synthesis_outcome.report_bundle,
        )
        if report_manifest is not None:
            failed_artifacts = sorted(
                name
                for name, status in report_manifest.artifact_status.items()
                if status == "failed"
            )
            if failed_artifacts:
                for name in failed_artifacts:
                    artifact_paths.pop(name, None)
                    state.warnings.append(f"rich minutes artifact failed: {name}")
                result = result.model_copy(
                    update={
                        "status": RunStatus.COMPLETED_WITH_WARNINGS,
                        "warnings": tuple(dict.fromkeys(state.warnings)),
                        "artifact_paths": artifact_paths,
                    }
                )
                if state.status != RunStatus.COMPLETED_WITH_WARNINGS:
                    await self._set_status(state, RunStatus.COMPLETED_WITH_WARNINGS)
                save_working_state_checkpoint(state, self._output_dir)
                self._write_json(
                    "summary_run_result.json", result.model_dump(mode="json")
                )
        return result

    async def _run_evaluation_sidecar(
        self,
        state: SummaryWorkingState,
        result: SummaryRunResult,
    ) -> SummaryRunResult:
        """对正式 candidate 0 执行选定的只读 Evaluation。"""
        if self._config.evaluation_mode == EvaluationMode.LITE:
            return await self._run_lite_evaluation_sidecar(state, result)
        await self._set_status(state, RunStatus.EVALUATING)
        running_result = result.model_copy(
            update={
                "evaluation_status": EvaluationStatus.RUNNING,
                "personalization_evaluation_status": (
                    EvaluationStatus.RUNNING
                    if state.request.target_entities
                    else EvaluationStatus.DISABLED
                ),
            }
        )
        self._write_json(
            "summary_run_result.json", running_result.model_dump(mode="json")
        )
        await self._trace.emit(
            "evaluation.started",
            run_id=state.run_id,
            payload={"evaluated_candidate_index": 0},
        )
        evaluation_warnings: list[str] = []
        evaluation_status = EvaluationStatus.COMPLETED
        personalization_status = (
            EvaluationStatus.RUNNING
            if state.request.target_entities
            else EvaluationStatus.DISABLED
        )
        if self._evaluator is None:
            error = "evaluation was enabled but no SummaryEvaluator was configured"
            evaluation_status = EvaluationStatus.FAILED
            personalization_status = (
                EvaluationStatus.FAILED
                if state.request.target_entities
                else EvaluationStatus.DISABLED
            )
            evaluation_warnings.append(error)
            self._write_failed_evaluation_artifacts(
                result.final_summary or "",
                error,
                target_entities=tuple(state.request.target_entities),
                execution_stats=(
                    self._evaluator.execution_stats
                    if self._evaluator is not None
                    else {}
                ),
            )
        else:
            try:
                # Evaluation 发生在正式摘要保存后，失败只能增加 warning，不能回滚摘要。
                artifacts = await self._evaluator.evaluate(
                    state, result.final_summary or ""
                )
                evaluation_warnings.extend(artifacts.metrics.warnings)
                if artifacts.personalization is not None:
                    personalization_status = artifacts.personalization.metrics.status
                    if personalization_status == EvaluationStatus.FAILED:
                        evaluation_warnings.extend(
                            f"personalization evaluation failed: {item}"
                            for item in artifacts.personalization.metrics.warnings
                        )
                    elif personalization_status == EvaluationStatus.COMPLETED:
                        evaluation_warnings.extend(
                            artifacts.personalization.metrics.warnings
                        )
                self._write_evaluation_artifacts(artifacts)
            except Exception as exc:
                evaluation_status = EvaluationStatus.FAILED
                personalization_status = (
                    EvaluationStatus.FAILED
                    if state.request.target_entities
                    else EvaluationStatus.DISABLED
                )
                error = f"{type(exc).__name__}: {exc}"
                evaluation_warnings.append(f"evaluation failed: {error}")
                self._write_failed_evaluation_artifacts(
                    result.final_summary or "",
                    error,
                    target_entities=tuple(state.request.target_entities),
                    execution_stats=(
                        self._evaluator.execution_stats
                        if self._evaluator is not None
                        else {}
                    ),
                )

        state.warnings.extend(evaluation_warnings)
        final_warnings = tuple(dict.fromkeys((*result.warnings, *evaluation_warnings)))
        final_status = (
            RunStatus.COMPLETED_WITH_WARNINGS if final_warnings else RunStatus.COMPLETED
        )
        await self._set_status(state, final_status)
        artifact_paths = dict(result.artifact_paths)
        artifact_paths.update(self._evaluation_artifact_paths())
        final_result = result.model_copy(
            update={
                "status": final_status,
                "evaluation_status": evaluation_status,
                "evaluation_metrics_path": str(
                    self._output_dir / "evaluation_metrics.json"
                ),
                "personalization_evaluation_status": personalization_status,
                "personalization_evaluation_metrics_path": str(
                    self._output_dir / "evaluation_personalization_metrics.json"
                ),
                "warnings": final_warnings,
                "artifact_paths": artifact_paths,
            }
        )
        save_working_state_checkpoint(state, self._output_dir)
        self._write_json(
            "summary_run_result.json", final_result.model_dump(mode="json")
        )
        await self._trace.emit(
            "evaluation.finished",
            run_id=state.run_id,
            payload={
                "status": evaluation_status.value,
                "personalization_status": personalization_status.value,
                "formal_summary_preserved": True,
                "evaluated_candidate_index": 0,
            },
        )
        return final_result

    async def _run_lite_evaluation_sidecar(
        self,
        state: SummaryWorkingState,
        result: SummaryRunResult,
    ) -> SummaryRunResult:
        """运行精简评测；失败只影响评测状态，不回滚正式纪要。"""
        await self._set_status(state, RunStatus.EVALUATING)
        running_result = result.model_copy(
            update={
                "evaluation_status": EvaluationStatus.RUNNING,
                "personalization_evaluation_status": EvaluationStatus.DISABLED,
            }
        )
        self._write_json(
            "summary_run_result.json", running_result.model_dump(mode="json")
        )
        await self._trace.emit(
            "evaluation.started",
            run_id=state.run_id,
            payload={"evaluated_candidate_index": 0, "evaluation_mode": "lite"},
        )
        warnings: list[str] = []
        evaluation_status = EvaluationStatus.COMPLETED
        evaluator = self._evaluator
        try:
            if not isinstance(evaluator, LiteSummaryEvaluator):
                raise RuntimeError(
                    "lite evaluation was enabled but LiteSummaryEvaluator was not configured"
                )
            artifacts = await evaluator.evaluate(state, result.final_summary or "")
            warnings.extend(artifacts.warnings)
            for candidate in artifacts.candidates:
                warnings.extend(candidate.warnings)
            write_lite_evaluation_artifacts(
                self._output_dir,
                artifacts,
                model_trace=evaluator.model_trace,
            )
            if artifacts.status == "failed":
                evaluation_status = EvaluationStatus.FAILED
            elif artifacts.status == "completed_with_warnings":
                warnings.append("lite evaluation completed with partial results")
        except Exception as exc:
            evaluation_status = EvaluationStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
            warnings.append(f"lite evaluation failed: {error}")
            write_lite_failure_artifacts(
                self._output_dir,
                error=error,
                stage=(
                    evaluator.last_stage
                    if isinstance(evaluator, LiteSummaryEvaluator)
                    else "not_started"
                ),
                reference=(
                    evaluator.reference
                    if isinstance(evaluator, LiteSummaryEvaluator)
                    else None
                ),
                model_trace=(
                    evaluator.model_trace
                    if isinstance(evaluator, LiteSummaryEvaluator)
                    else ()
                ),
                execution_stats=(
                    evaluator.execution_stats
                    if isinstance(evaluator, LiteSummaryEvaluator)
                    else {}
                ),
            )
        state.warnings.extend(warnings)
        final_warnings = tuple(dict.fromkeys((*result.warnings, *warnings)))
        final_status = (
            RunStatus.COMPLETED_WITH_WARNINGS if final_warnings else RunStatus.COMPLETED
        )
        await self._set_status(state, final_status)
        artifact_paths = dict(result.artifact_paths)
        artifact_paths.update(self._lite_evaluation_artifact_paths())
        final_result = result.model_copy(
            update={
                "status": final_status,
                "evaluation_status": evaluation_status,
                "evaluation_metrics_path": str(
                    self._output_dir / "evaluation_lite_result.json"
                ),
                "personalization_evaluation_status": EvaluationStatus.DISABLED,
                "personalization_evaluation_metrics_path": None,
                "warnings": final_warnings,
                "artifact_paths": artifact_paths,
            }
        )
        save_working_state_checkpoint(state, self._output_dir)
        self._write_json(
            "summary_run_result.json", final_result.model_dump(mode="json")
        )
        await self._trace.emit(
            "evaluation.finished",
            run_id=state.run_id,
            payload={
                "status": evaluation_status.value,
                "personalization_status": EvaluationStatus.DISABLED.value,
                "evaluation_mode": "lite",
                "formal_summary_preserved": True,
                "evaluated_candidate_index": 0,
            },
        )
        return final_result

    async def _finish_failed(
        self,
        state: SummaryWorkingState,
        reason: str,
        *,
        summary_prompt: Optional[str] = None,
        model_input_report: Optional[dict[str, Any]] = None,
    ) -> SummaryRunResult:
        """把阶段错误写入失败产物并返回无正式摘要的 FAILED 结果。"""
        if reason not in state.errors:
            state.errors.append(reason)
        await self._set_status(state, RunStatus.FAILED)
        artifact_paths = self._artifact_paths(
            include_summary=False,
            include_review=False,
            include_reviewed_candidate=False,
            include_rich=False,
        )
        if summary_prompt is not None:
            artifact_paths["multi_agent_summary_prompt.txt"] = str(
                self._output_dir / "multi_agent_summary_prompt.txt"
            )
        if model_input_report is not None:
            artifact_paths["model_input_report.json"] = str(
                self._output_dir / "model_input_report.json"
            )
        result = SummaryRunResult(
            run_id=state.run_id,
            status=RunStatus.FAILED,
            warnings=tuple(dict.fromkeys((*state.warnings, *state.errors))),
            coverage_complete=state.coverage_ledger.is_complete,
            evaluation_enabled=self._config.evaluation_enabled,
            evaluation_mode=self._config.evaluation_mode,
            evaluation_status=(
                EvaluationStatus.FAILED
                if self._config.evaluation_enabled
                else EvaluationStatus.DISABLED
            ),
            personalization_evaluation_status=(
                EvaluationStatus.FAILED
                if (
                    self._config.evaluation_enabled
                    and self._config.evaluation_mode == EvaluationMode.STRICT
                    and state.request.target_entities
                )
                else EvaluationStatus.DISABLED
            ),
            artifact_paths=artifact_paths,
        )
        self._save_artifacts(
            state,
            result,
            summary_prompt=summary_prompt,
            model_input_report=model_input_report,
            review_result=None,
            report_bundle=None,
        )
        await self._trace.emit(
            "pipeline.finished",
            run_id=state.run_id,
            payload={"status": RunStatus.FAILED.value, "reason": reason},
        )
        return result

    def _evaluation_artifact_paths(self) -> dict[str, str]:
        """返回 Evaluation Sidecar 预期生成的产物路径集合。"""
        names = (
            "evaluation_source_snapshot.json",
            "evaluation_source_relations.json",
            "evaluation_source_facts.json",
            "evaluation_summary_facts.json",
            "evaluation_fact_matches.json",
            "evaluation_metrics.json",
            "evaluation_report.md",
            "evaluation_model_trace.json",
            "evaluation_personalization_source_annotations.json",
            "evaluation_personalization_claim_annotations.json",
            "evaluation_personalization_metrics.json",
            "evaluation_personalization_report.md",
            "evaluation_personalization_model_trace.json",
        )
        return {name: str(self._output_dir / name) for name in names}

    def _lite_evaluation_artifact_paths(self) -> dict[str, str]:
        names = (
            "evaluation_lite_reference.json",
            "evaluation_lite_result.json",
            "evaluation_lite_report.md",
            "evaluation_lite_report.html",
            "evaluation_lite_model_trace.json",
        )
        return {name: str(self._output_dir / name) for name in names}

    def _write_evaluation_artifacts(self, artifacts: EvaluationArtifacts) -> None:
        """持久化 Evaluation 的事实、匹配、指标、报告和模型轨迹。"""
        snapshot = (
            self._evaluator.source_snapshot if self._evaluator is not None else None
        )
        self._write_json(
            "evaluation_source_snapshot.json",
            snapshot.model_dump(mode="json") if snapshot is not None else None,
        )
        self._write_json(
            "evaluation_source_relations.json",
            [
                item.model_dump(mode="json")
                for item in (snapshot.fact_relations if snapshot is not None else ())
            ],
        )
        self._write_json(
            "evaluation_source_facts.json",
            [item.model_dump(mode="json") for item in artifacts.source_facts],
        )
        self._write_json(
            "evaluation_summary_facts.json",
            [item.model_dump(mode="json") for item in artifacts.summary_facts],
        )
        self._write_json(
            "evaluation_fact_matches.json",
            [item.model_dump(mode="json") for item in artifacts.matches],
        )
        self._write_json(
            "evaluation_metrics.json", artifacts.metrics.model_dump(mode="json")
        )
        self._write_text("evaluation_report.md", artifacts.report_markdown)
        self._write_json("evaluation_model_trace.json", list(artifacts.model_trace))
        personalization = artifacts.personalization
        if personalization is None:
            summary_path = self._output_dir / "summary_multi_agent.md"
            persisted_summary = (
                summary_path.read_text(encoding="utf-8").rstrip()
                if summary_path.is_file()
                else ""
            )
            personalization = disabled_personalization_artifacts(persisted_summary)
        self._write_personalization_artifacts(personalization)

    def _write_personalization_artifacts(
        self, artifacts: PersonalizationEvaluationArtifacts
    ) -> None:
        """持久化与整体 Evaluation 隔离的目标实体评测产物。"""
        self._write_json(
            "evaluation_personalization_source_annotations.json",
            [item.model_dump(mode="json") for item in artifacts.source_annotations],
        )
        self._write_json(
            "evaluation_personalization_claim_annotations.json",
            [item.model_dump(mode="json") for item in artifacts.claim_annotations],
        )
        self._write_json(
            "evaluation_personalization_metrics.json",
            artifacts.metrics.model_dump(mode="json"),
        )
        self._write_text(
            "evaluation_personalization_report.md", artifacts.report_markdown
        )
        self._write_json(
            "evaluation_personalization_model_trace.json",
            list(artifacts.model_trace),
        )

    def _write_failed_evaluation_artifacts(
        self,
        summary: str,
        error: str,
        *,
        target_entities: tuple[str, ...] = (),
        execution_stats: Optional[dict[str, int]] = None,
    ) -> None:
        """在评测失败时写入空事实集和失败指标，同时保留正式摘要。"""
        metrics = EvaluationMetrics(
            status=EvaluationStatus.FAILED,
            evaluated_summary_sha256=persisted_summary_sha256(summary),
            config={
                "temperature": 0.0,
                "timeout_seconds": self._config.evaluation_timeout_seconds,
                "model_timeout_seconds": (
                    self._config.evaluation_model_timeout_seconds
                ),
                "max_model_calls": self._config.evaluation_max_model_calls,
            },
            execution_stats=dict(execution_stats or {}),
            warnings=(error,),
        )
        self._write_json("evaluation_source_facts.json", [])
        self._write_json("evaluation_source_snapshot.json", None)
        self._write_json("evaluation_source_relations.json", [])
        self._write_json("evaluation_summary_facts.json", [])
        self._write_json("evaluation_fact_matches.json", [])
        self._write_json("evaluation_metrics.json", metrics.model_dump(mode="json"))
        self._write_text(
            "evaluation_report.md",
            f"# 纪要事实级评测\n\n评测失败，但正式摘要保持不变。\n\n错误：{error}\n",
        )
        trace = list(self._evaluator.model_trace) if self._evaluator is not None else []
        self._write_json("evaluation_model_trace.json", trace)
        personalization = (
            failed_personalization_artifacts(
                summary,
                target_entities,
                f"overall evaluation unavailable: {error}",
                execution_stats=(
                    self._evaluator.personalization_execution_stats
                    if self._evaluator is not None
                    else {}
                ),
                model_trace=(
                    self._evaluator.personalization_model_trace
                    if self._evaluator is not None
                    else ()
                ),
            )
            if target_entities
            else disabled_personalization_artifacts(summary)
        )
        self._write_personalization_artifacts(personalization)

    async def _set_status(
        self,
        state: SummaryWorkingState,
        status: RunStatus,
        *,
        stage: Optional[str] = None,
    ) -> None:
        """更新 Pipeline 状态并追加可选阶段名到 Trace。"""
        state.status = status
        payload: dict[str, Any] = {"status": status.value}
        if stage:
            payload["stage"] = stage
        await self._trace.emit(
            "pipeline.status_changed", run_id=state.run_id, payload=payload
        )

    def _artifact_paths(
        self,
        *,
        include_summary: bool,
        include_review: bool,
        include_reviewed_candidate: bool,
        include_rich: bool,
    ) -> dict[str, str]:
        """根据本次运行实际启用阶段列出稳定的审计产物路径。"""
        names = [
            "current_material.txt",
            "raw_material_manifest.json",
            "plan.json",
            "plan_history.json",
            "agent_results.json",
            "artifact_board.json",
            "current_segment_analyses.json",
            "current_fact_map_trace.json",
            "coverage_ledger.json",
            "evidence_index.json",
            "context_compression_report.json",
            "synthesis_context.json",
            "summary_synthesis_trace.json",
            "working_state.json",
            "summary_run_result.json",
            "execution_trace.jsonl",
        ]
        names.extend(
            name
            for name in (
                "planner_prompt.txt",
                "planner_raw_output.txt",
                "planner_validation_report.json",
                "planner_model_trace.json",
                "replanner_prompt.txt",
                "replanner_raw_output.txt",
                "replanner_validation_report.json",
                "replanner_model_trace.json",
            )
            if (self._output_dir / name).exists()
        )
        names.extend(
            name
            for name in (
                "model_call_input_report.json",
                "history_selection_report.json",
            )
            if (self._output_dir / name).exists()
            or name == "history_selection_report.json"
        )
        if include_summary:
            names.extend(
                [
                    "multi_agent_summary_prompt.txt",
                    "model_input_report.json",
                    "summary_multi_agent.md",
                ]
            )
        if include_rich:
            names.extend(
                [
                    "minutes_document.json",
                    "minutes_sources.json",
                    "minutes_report.html",
                    "minutes_report_manifest.json",
                    "minutes_report_bundle.zip",
                ]
            )
        if include_review:
            names.extend(
                [
                    "review_run_result.json",
                    "review_red_reports.json",
                    "review_verification_results.json",
                    "review_blue_results.json",
                    "review_judge_decisions.json",
                    "review_candidate_summaries.json",
                    "review_model_trace.json",
                ]
            )
        if include_reviewed_candidate:
            names.append("summary_multi_agent_reviewed_candidate.md")
        return {name: str(self._output_dir / name) for name in names}

    def _save_artifacts(
        self,
        state: SummaryWorkingState,
        result: SummaryRunResult,
        *,
        summary_prompt: Optional[str],
        model_input_report: Optional[dict[str, Any]],
        review_result: Optional[ReviewRunResult],
        report_bundle: Optional[MinutesReportBundle],
    ) -> Optional[MinutesReportManifest]:
        """保存原文、计划、M4、Evidence、摘要、Review 和运行结果产物。"""
        (self._output_dir / "current_material.txt").write_text(
            state.request.current_material, encoding="utf-8"
        )
        self._write_json(
            "raw_material_manifest.json",
            {
                "schema_version": state.raw_material.schema_version,
                "source_id": state.raw_material.source_id,
                "content_sha256": state.raw_material.content_sha256,
                "total_chars": state.raw_material.total_chars,
                "max_segment_chars": state.raw_material.max_segment_chars,
                "overlap_chars": state.raw_material.overlap_chars,
                "segments": [
                    segment.model_dump(mode="json", exclude={"text"})
                    for segment in state.raw_material.segments
                ],
            },
        )
        self._write_json(
            "plan.json",
            state.plan.model_dump(mode="json") if state.plan else None,
        )
        self._write_json(
            "plan_history.json",
            [item.model_dump(mode="json") for item in state.plan_history],
        )
        self._write_json(
            "agent_results.json",
            {
                task_id: value.model_dump(mode="json")
                for task_id, value in state.agent_results.items()
            },
        )
        self._write_json("artifact_board.json", state.historical_evidence_board)
        self._write_json(
            "current_segment_analyses.json",
            [item.model_dump(mode="json") for item in state.current_segment_analyses],
        )
        self._write_json("current_fact_map_trace.json", list(self._analyzer.traces))
        self._write_json(
            "coverage_ledger.json", state.coverage_ledger.model_dump(mode="json")
        )
        self._write_json(
            "evidence_index.json",
            {
                evidence_id: value.model_dump(mode="json")
                for evidence_id, value in state.evidence_index.items()
            },
        )
        self._write_json(
            "context_compression_report.json",
            state.context_compression_report.model_dump(mode="json")
            if state.context_compression_report
            else None,
        )
        self._write_json(
            "synthesis_context.json",
            state.synthesis_context.model_dump(mode="json")
            if state.synthesis_context
            else None,
        )
        self._write_json(
            "history_selection_report.json",
            build_history_selection_report(state),
        )
        self._write_json(
            "summary_synthesis_trace.json",
            list(self._synthesizer.synthesis_trace),
        )
        if summary_prompt is not None:
            self._write_text("multi_agent_summary_prompt.txt", summary_prompt)
        if model_input_report is not None:
            self._write_json("model_input_report.json", model_input_report)
        if result.final_summary is not None:
            self._write_text("summary_multi_agent.md", result.final_summary)
        report_manifest = None
        if report_bundle is not None:
            report_manifest = persist_report_bundle(self._output_dir, report_bundle)
        if review_result is not None:
            self._write_review_artifacts(state, review_result)
        save_working_state_checkpoint(state, self._output_dir)
        self._write_json("summary_run_result.json", result.model_dump(mode="json"))
        return report_manifest

    def _write_review_artifacts(
        self,
        state: SummaryWorkingState,
        review_result: ReviewRunResult,
    ) -> None:
        """保存 Shadow Review 的轮次、候选、裁决和模型轨迹。"""
        self._write_json(
            "review_run_result.json", review_result.model_dump(mode="json")
        )
        self._write_json(
            "review_red_reports.json",
            [
                item.model_dump(mode="json")
                for item in (
                    self._reviewer.red_reports if self._reviewer is not None else ()
                )
            ],
        )
        self._write_json(
            "review_verification_results.json",
            [
                result.model_dump(mode="json")
                for result in (
                    self._reviewer.verification_results
                    if self._reviewer is not None
                    else ()
                )
            ],
        )
        self._write_json(
            "review_blue_results.json",
            [
                item.model_dump(mode="json")
                for item in (
                    self._reviewer.blue_results if self._reviewer is not None else ()
                )
            ],
        )
        self._write_json(
            "review_judge_decisions.json",
            [
                item.model_dump(mode="json")
                for item in (
                    self._reviewer.judge_decisions if self._reviewer is not None else ()
                )
            ],
        )
        self._write_json(
            "review_candidate_summaries.json",
            [
                {"candidate_index": index, "summary": summary}
                for index, summary in enumerate(state.candidate_summaries)
            ],
        )
        self._write_json(
            "review_model_trace.json",
            self._reviewer.model_traces if self._reviewer is not None else {},
        )
        recommended_index = review_result.recommended_candidate_index
        if 0 < recommended_index < len(state.candidate_summaries):
            self._write_text(
                "summary_multi_agent_reviewed_candidate.md",
                state.candidate_summaries[recommended_index],
            )

    def _write_json(self, name: str, payload: Any) -> None:
        """以稳定 JSON 格式写入 Pipeline 产物。"""
        (self._output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_text(self, name: str, content: str) -> None:
        """写入文本产物并统一补充末尾换行。"""
        (self._output_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")
