"""使用固定 Source Snapshot 对一份纪要执行事实级评分。"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..adapters.model import ModelClient
from ..schemas import RunConfig
from .matcher import EvaluationFactMatcher
from .metrics import calculate_evaluation_metrics
from .model_io import EvaluationModelIO
from .personalization import (
    TargetPersonalizationEvaluator,
    disabled_personalization_artifacts,
    failed_personalization_artifacts,
)
from .reporting import build_evaluation_report
from .schemas import EvaluationArtifacts
from .snapshot import EvaluationSourceSnapshotBundle
from .summary_fact_extractor import EvaluationSummaryFactExtractor


class EvaluationSummaryScorer:
    """在独立预算内抽取 Claim、匹配固定事实并确定性计算指标。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        personalization_model_client: Optional[ModelClient] = None,
        config: RunConfig,
        max_model_calls: Optional[int] = None,
        personalization_max_model_calls: Optional[int] = None,
    ) -> None:
        """初始化单纪要整体与个性化分支的隔离预算。"""
        self._config = config
        self._model_io = EvaluationModelIO(
            model_client,
            max_model_calls=(
                max_model_calls
                if max_model_calls is not None
                else config.evaluation_max_model_calls
            ),
        )
        personalization_config = config.model_copy(
            update={
                "evaluation_personalization_max_model_calls": (
                    personalization_max_model_calls
                    if personalization_max_model_calls is not None
                    else config.evaluation_personalization_max_model_calls
                )
            }
        )
        self._personalization = TargetPersonalizationEvaluator(
            model_client=personalization_model_client or model_client,
            config=personalization_config,
        )

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回单纪要整体评测模型 trace。"""
        return tuple(self._model_io.traces)

    @property
    def personalization_model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回单纪要目标 Claim 标注 trace。"""
        return self._personalization.model_trace

    @property
    def personalization_execution_stats(self) -> dict[str, int]:
        """返回单纪要目标 Claim 标注的调用统计。"""
        return self._personalization.execution_stats

    @property
    def execution_stats(self) -> dict[str, int]:
        """返回整体模型调用与 repair 次数。"""
        return {
            "model_call_count": self._model_io.model_call_count,
            "repair_call_count": self._model_io.repair_call_count,
        }

    async def score(
        self,
        *,
        snapshot_bundle: EvaluationSourceSnapshotBundle,
        summary: str,
        temporal_support: Optional[dict[str, object]] = None,
        worker_results: Optional[dict[str, dict[str, Any]]] = None,
    ) -> EvaluationArtifacts:
        """对一份纪要评分；Snapshot 中的分母和目标源标注保持不变。"""
        snapshot = snapshot_bundle.snapshot
        extractor = EvaluationSummaryFactExtractor(
            model_io=self._model_io,
            config=self._config,
        )
        summary_facts = await extractor.extract(summary)
        matcher = EvaluationFactMatcher(
            model_io=self._model_io,
            config=self._config,
        )
        matches = await matcher.match(
            summary_facts,
            snapshot.source_facts,
            temporal_support=dict(temporal_support or {}),
            fact_relations=snapshot.fact_relations,
        )
        warnings = tuple(
            dict.fromkeys(
                (*snapshot.warnings, *extractor.warnings, *matcher.warnings)
            )
        )
        execution_stats = {
            **self.execution_stats,
            "evaluation_summary_segment_count": extractor.segment_count,
            **matcher.execution_stats,
        }
        metrics = calculate_evaluation_metrics(
            summary=summary,
            current_material=snapshot_bundle.current_material,
            visible_history_text=snapshot_bundle.visible_history_text,
            source_facts=snapshot.source_facts,
            summary_facts=summary_facts,
            matches=matches,
            fact_relations=snapshot.fact_relations,
            config={
                "mode": "fixed_source_snapshot",
                "snapshot_id": snapshot.snapshot_id,
                "temperature": 0.0,
                "segment_max_chars": self._config.evaluation_segment_max_chars,
                "summary_fact_max_tokens": (
                    self._config.evaluation_summary_fact_max_tokens
                ),
                "match_max_tokens": self._config.evaluation_match_max_tokens,
                "repair_attempts": self._config.evaluation_repair_attempts,
                "model_timeout_seconds": self._config.evaluation_model_timeout_seconds,
                "max_model_calls": self._config.evaluation_max_model_calls,
                "match_batch_size": self._config.evaluation_match_batch_size,
                "match_full_scan_threshold": (
                    self._config.evaluation_match_full_scan_threshold
                ),
                "match_candidate_limit_per_fact": (
                    self._config.evaluation_match_candidate_limit_per_fact
                ),
                "match_candidate_union_limit": (
                    self._config.evaluation_match_candidate_union_limit
                ),
            },
            execution_stats=execution_stats,
            warnings=warnings,
        )
        if not snapshot.target_entities:
            personalization = disabled_personalization_artifacts(summary)
        elif source_error := next(
            (
                item
                for item in snapshot.warnings
                if item.startswith("personalization source annotation failed:")
            ),
            None,
        ):
            personalization = failed_personalization_artifacts(
                summary,
                snapshot.target_entities,
                source_error,
                execution_stats=self._personalization.execution_stats,
                model_trace=self.personalization_model_trace,
            )
        else:
            try:
                personalization = await asyncio.wait_for(
                    self._personalization.evaluate(
                        summary=summary,
                        target_entities=snapshot.target_entities,
                        source_facts=snapshot.source_facts,
                        summary_facts=summary_facts,
                        matches=matches,
                        fact_relations=snapshot.fact_relations,
                        source_annotations=snapshot.target_source_annotations,
                        worker_results=dict(worker_results or {}),
                    ),
                    timeout=self._config.evaluation_personalization_timeout_seconds,
                )
            except Exception as exc:
                personalization = failed_personalization_artifacts(
                    summary,
                    snapshot.target_entities,
                    f"{type(exc).__name__}: {exc}",
                    execution_stats=self._personalization.execution_stats,
                    model_trace=self.personalization_model_trace,
                )
        return EvaluationArtifacts(
            source_facts=snapshot.source_facts,
            summary_facts=summary_facts,
            matches=matches,
            metrics=metrics,
            report_markdown=build_evaluation_report(
                snapshot.source_facts,
                summary_facts,
                matches,
                metrics,
                snapshot.fact_relations,
            ),
            model_trace=self.model_trace,
            personalization=personalization,
        )
